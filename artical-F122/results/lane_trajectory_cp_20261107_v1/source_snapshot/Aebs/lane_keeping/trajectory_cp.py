"""Frozen-policy trajectory-level CP for the nominal rendered lane task.

Each independent rollout is ONE calibration sample: max over time AND both
normalized observation coordinates. Models and protocol are bound before
collection; holdout collection starts only after the threshold is frozen.
No unsafe rollout, large residual, or incomplete run may be silently removed.
"""
import argparse
import hashlib
import json
import importlib.metadata as metadata
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import beta

from . import core
from .cp import conformal_radius
from .learned_sbc import sha256
from .policy import load_actor
from .patch_perception import PatchImageObserver
from .renderer import RoadRenderer


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def trajectory_scores(errors):
    """errors[episode,time,coordinate], returns one score per episode."""
    errors=np.asarray(errors,dtype=np.float64)
    if errors.ndim!=3 or errors.shape[2]!=2 or min(errors.shape[:2])<1:
        raise ValueError('nonempty [episodes,time,2] errors required')
    if np.isnan(errors).any():
        raise ValueError('NaN residuals cannot be discarded or calibrated')
    return np.abs(errors).max(axis=(1,2))


def coverage_summary(scores, radius, independent=False):
    scores=np.asarray(scores)
    n=len(scores);covered=int((scores<=radius).sum())
    result=dict(episodes=n,covered=covered,not_covered=n-covered,fraction=covered/n)
    if independent:
        lower=0. if covered==0 else float(beta.ppf(.025,covered,n-covered+1))
        upper=1. if covered==n else float(beta.ppf(.975,covered+1,n-covered))
        result['binomial_two_sided_95_interval']=[lower,upper]
    return result


def initial_draws(seed,n):
    # Separate independent pseudorandom streams for runs; no corners or
    # hand-picked success cases are inserted into the calibration sample.
    children=np.random.SeedSequence(seed).spawn(n)
    initial=[];curvatures=[];seeds=[]
    for child in children:
        rng=np.random.default_rng(child)
        initial.append(rng.uniform(core.INITIAL.low,core.INITIAL.high))
        curvatures.append(rng.choice(core.KAPPAS))
        seeds.append(int(child.generate_state(1,dtype=np.uint64)[0]))
    return np.array(initial),np.array(curvatures),np.array(seeds,dtype=np.uint64)


def make_protocol(args):
    source=Path(__file__).resolve().parent
    return dict(format='lane_trajectory_cp_protocol_v1',date='2026-11-07',
        unit='one complete closed-loop image rollout',horizon=args.horizon,alpha=args.alpha,
        calibration_episodes=args.calibration_episodes,holdout_episodes=args.holdout_episodes,
        calibration_seed=args.calibration_seed,holdout_seed=args.holdout_seed,
        inference_batch_size=args.batch_size,
        environment=dict(device='cpu',dtype='float32 actor and observer; float64 plant',threads=4,
                         versions={name:metadata.version(name) for name in ['numpy','torch','Pillow','scipy']}),
        score='max over 80 (or registered horizon) frames and two normalized absolute observation errors',
        distribution=dict(initial='independent uniform on original initial rectangle',
                          initial_low=core.INITIAL.low.tolist(),initial_high=core.INITIAL.high.tolist(),
                          curvature='independent uniform on registered values, constant within each rollout',
                          kappas=list(core.KAPPAS),lighting=1.,camera_lateral=0.,camera_height=1.25,
                          occluder=False,blur=0.,pixel_noise=0.,image_size=[192,192],
                          speed=core.SPEED,dt=core.DT,wheelbase=core.WHEELBASE,process_noise=0),
        binding=dict(actor_sha256=sha256(args.package/'actor.pt'),
                     observer_sha256=sha256(args.package/'patch_projection.pt'),
                     sbc_sha256=sha256(args.package/'sbc.pt'),dino_weights_sha256=sha256(args.dino_weights),
                     source_sha256={p.name:sha256(p) for p in [source/'trajectory_cp.py',source/'cp.py',
                         source/'core.py',source/'renderer.py',source/'policy.py',source/'patch_perception.py',
                         source.parent/'dino_latent/backbone.py']}),
        rules=dict(threshold_uses_calibration_only=True,holdout_after_threshold_frozen=True,
                   stop_or_drop_unsafe_rollouts=False,retrain_actor_or_observer=False,
                   reuse_previous_12_development_runs=False,pad_last_inference_batch=True),
        scope='IID rollout generator under registered nominal simulator law; arbitrary within-rollout temporal dependence. No claim for changed policy, camera/weather, real world, fixed-state conditional coverage or longer horizon.')


@torch.no_grad()
def collect(args, split, actor, observer, renderer, protocol_hash):
    n=getattr(args,split+'_episodes');seed=getattr(args,split+'_seed')
    combined_path=args.output_dir/f'{split}.npz'
    if combined_path.exists():
        saved=dict(np.load(combined_path,allow_pickle=False))
        if str(saved['protocol_sha256'])!=protocol_hash or len(saved['scores'])!=n:
            raise ValueError('saved split does not match frozen protocol')
        return {k:v for k,v in saved.items() if k!='protocol_sha256'}
    target=args.output_dir/split;target.mkdir(exist_ok=True)
    initial,kappas,episode_seeds=initial_draws(seed,n)
    pieces=[];started=time.monotonic()
    for first in range(0,n,args.batch_size):
        count=min(args.batch_size,n-first)
        path=target/f'batch_{first:04d}.npz'
        if path.exists():
            saved=dict(np.load(path,allow_pickle=False))
            if str(saved['protocol_sha256'])!=protocol_hash:raise ValueError('batch binding mismatch')
            pieces.append(saved);continue
        x=initial[first:first+count].copy();ks=kappas[first:first+count]
        # A fixed inference shape also avoids batch-size-dependent arithmetic
        # between the last calibration and holdout batches. Padding is discarded.
        pad=args.batch_size-count
        if pad:
            x=np.concatenate([x,np.repeat(x[-1:],pad,axis=0)])
            ks=np.concatenate([ks,np.repeat(ks[-1:],pad)])
        state_trace=[x[:count].copy()];obs_trace=[];action_trace=[];error_trace=[]
        digests=[hashlib.sha256() for _ in range(count)]
        for step in range(args.horizon):
            images=np.array([renderer.render(s,float(k),lighting=1.,occluder=False,
                            camera_lateral=0.,camera_height=1.25,blur=0.,noise=0.) for s,k in zip(x,ks)])
            obs=observer(torch.from_numpy(images)).numpy()
            if not np.isfinite(obs).all():raise ValueError('nonfinite observation; collection invalid, do not omit this batch')
            action=actor(torch.tensor(np.c_[obs,ks/core.KAPPA_SCALE],dtype=torch.float32)).numpy().flatten()
            errors=obs-x/[core.E_SCALE,core.THETA_SCALE]
            for i in range(count):digests[i].update(images[i].tobytes())
            x=core.dynamics_batch(x,action*core.PHI_MAX,ks)
            if not np.isfinite(x).all():raise ValueError('nonfinite state; collection invalid')
            state_trace.append(x[:count].copy());obs_trace.append(obs[:count].copy())
            action_trace.append(action[:count].copy());error_trace.append(errors[:count].copy())
            if step%20==0:
                print(json.dumps(dict(split=split,completed=first,total=n,batch_step=step,
                                      seconds=time.monotonic()-started)),flush=True)
        states=np.stack(state_trace,axis=1);errors=np.stack(error_trace,axis=1)
        scores=trajectory_scores(errors)
        saved=dict(states=states,observations=np.stack(obs_trace,axis=1),
                   actions=np.stack(action_trace,axis=1),errors=errors,scores=scores,
                   kappas=ks[:count],episode_seeds=episode_seeds[first:first+count],
                   unsafe=(abs(states/[core.E_SCALE,core.THETA_SCALE])>1).any(axis=(1,2)),
                   images_sha256=np.array([d.hexdigest() for d in digests]),
                   protocol_sha256=np.array(protocol_hash))
        np.savez_compressed(path,**saved);pieces.append(saved)
    combined={key:np.concatenate([p[key] for p in pieces],axis=0)
              for key in pieces[0] if key!='protocol_sha256'}
    np.savez_compressed(combined_path,**combined,protocol_sha256=protocol_hash)
    return combined


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--dino-repo',type=Path,required=True)
    p.add_argument('--dino-weights',type=Path,required=True)
    p.add_argument('--calibration-episodes',type=int,default=199)
    p.add_argument('--holdout-episodes',type=int,default=100)
    p.add_argument('--calibration-seed',type=int,default=2026110701)
    p.add_argument('--holdout-seed',type=int,default=2026110702)
    p.add_argument('--horizon',type=int,default=80)
    p.add_argument('--alpha',type=float,default=.05)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args(argv)
    if min(args.calibration_episodes,args.holdout_episodes,args.horizon,args.batch_size)<1:
        raise ValueError('positive counts required')
    if not 0<args.alpha<1 or args.calibration_seed==args.holdout_seed:
        raise ValueError('valid alpha and distinct split seeds required')
    if args.output_dir.exists() and not args.resume:raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    protocol=make_protocol(args);path=args.output_dir/'protocol.json'
    if path.exists():
        if json.loads(path.read_text())!=protocol:raise ValueError('frozen protocol or model changed')
    else:write_json(path,protocol)
    protocol_hash=sha256(path)
    torch.set_num_threads(4)
    actor=load_actor(args.package/'actor.pt').requires_grad_(False)
    observer=PatchImageObserver(args.package/'patch_projection.pt','cpu',args.dino_repo,args.dino_weights)
    renderer=RoadRenderer(192,192)
    calibration=collect(args,'calibration',actor,observer,renderer,protocol_hash)
    radius,rank=conformal_radius(calibration['scores'],args.alpha)
    cal_report=dict(format='lane_trajectory_split_cp_v1',alpha=args.alpha,
        n=args.calibration_episodes,rank=rank,radius=radius if np.isfinite(radius) else None,
        finite_radius=bool(np.isfinite(radius)),protocol_sha256=protocol_hash,
        calibration_data_sha256=sha256(args.output_dir/'calibration.npz'),
        coverage=coverage_summary(calibration['scores'],radius),
        scores_sorted=np.sort(calibration['scores']).tolist(),
        unsafe_episodes=int(calibration['unsafe'].sum()),
        previous_radius_008_coverage=coverage_summary(calibration['scores'],.08),
        claim='Marginal 1-alpha coverage for a NEW complete registered-horizon rollout under exchangeability of rollout scores; probability over calibration sample and new rollout. Not conditional on one fixed calibration set.')
    cal_path=args.output_dir/'calibration.json'
    if cal_path.exists():
        if json.loads(cal_path.read_text())!=cal_report:raise ValueError('frozen calibration changed')
    else:write_json(cal_path,cal_report)
    threshold_hash=sha256(cal_path)
    print(json.dumps(dict(event='threshold_frozen',radius=cal_report['radius'],rank=rank,
                         n=args.calibration_episodes,sha256=threshold_hash)),flush=True)
    holdout=collect(args,'holdout',actor,observer,renderer,protocol_hash)
    if sha256(cal_path)!=threshold_hash:raise ValueError('threshold modified during holdout')
    report=dict(format='lane_trajectory_cp_experiment_v1',calibration=cal_report,
        holdout=dict(coverage=coverage_summary(holdout['scores'],radius,independent=True),
                     old_radius_008_coverage=coverage_summary(holdout['scores'],.08,independent=True),
                     unsafe_episodes=int(holdout['unsafe'].sum()),
                     frame_coverage=float((abs(holdout['errors']).max(2)<=radius).mean()),
                     max_trajectory_score=float(holdout['scores'].max())),
        frozen_models_unchanged=make_protocol(args)==protocol,
        calibration_frozen_before_holdout=True,calibration_report_sha256=threshold_hash,
        scope=protocol['scope'],certificate_at_new_radius='requires separate numerical audit; not inferred from coverage')
    if not report['frozen_models_unchanged']:raise ValueError('model or code changed during collection')
    write_json(args.output_dir/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='calibration'}),flush=True)


if __name__=='__main__':main()
