"""Frozen-model trajectory CP: one independent complete robot run per score.

Fixed inference batch shape, full 120-step runs including unsafe runs,
calibration threshold frozen before holdout. No SBC pass is inferred.
"""
import argparse,hashlib,json,math,time
from pathlib import Path
import numpy as np
import torch
from Aebs.lane_keeping.cp import conformal_radius
from Aebs.lane_keeping.policy import load_actor
from Aebs.lane_keeping.learned_sbc import sha256
from Aebs.lane_keeping.trajectory_cp import coverage_summary
from . import core
from .renderer import RobotRenderer
from .perception import ImageObserver


def trajectory_scores(errors,shape):
    errors=np.asarray(errors);shape=np.asarray(shape)
    if errors.ndim!=3 or errors.shape[-1]!=3 or not errors.size:raise ValueError('expected [episode,time,3]')
    if shape.shape!=(3,) or not np.isfinite(shape).all() or (shape<=0).any():raise ValueError('positive finite scales required')
    if not np.isfinite(errors).all():raise ValueError('nonfinite errors cannot be discarded')
    return (abs(errors)/shape).max(axis=(1,2))


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


@torch.no_grad()
def collect(actor,observer,args,split,protocol_hash):
    count=getattr(args,split+'_episodes');seed=getattr(args,split+'_seed')
    children=np.random.SeedSequence(seed).spawn(count)
    initial=np.array([core.sample_initial(np.random.default_rng(child),1)[0] for child in children])
    renderer=RobotRenderer();pieces=[];start=time.monotonic();directory=args.output_dir/split;directory.mkdir()
    for first in range(0,count,args.batch_size):
        n=min(count-first,args.batch_size);x=initial[first:first+n].copy()
        if n<args.batch_size:x=np.concatenate([x,np.repeat(x[-1:],args.batch_size-n,axis=0)])
        states=[x[:n].copy()];observations=[];actions=[];errors=[];digests=[hashlib.sha256() for _ in range(n)]
        for step in range(core.HORIZON):
            images=np.array([renderer.render(s) for s in x]);obs=observer(torch.from_numpy(images)).cpu()
            if not torch.isfinite(obs).all():raise ValueError('nonfinite observation')
            u=actor(obs).numpy();errors.append(obs.numpy()[:n]-core.normalize(x[:n]))
            observations.append(obs.numpy()[:n]);actions.append(u[:n])
            for i in range(n):digests[i].update(images[i].tobytes())
            x=core.dynamics(x,u);states.append(x[:n].copy())
            if step%40==0:print(json.dumps(dict(split=split,first=first,step=step,seconds=time.monotonic()-start)),flush=True)
        states=np.stack(states,1);errors=np.stack(errors,1);actions=np.stack(actions,1)
        bad=core.unsafe(states).any(1);reached=core.goal(states).any(1)
        held=core.goal(states[:,-10:]).all(1)&(((actions[:,-10:,0]+1)*core.V_MAX/2).max(1)<.08)
        data=dict(states=states,errors=errors,actions=actions,observations=np.stack(observations,1),
            scores=trajectory_scores(errors,args.shape),unsafe=bad,reached=reached&~bad,held=held&~bad,
            image_sha256=np.array([d.hexdigest() for d in digests]))
        np.savez_compressed(directory/f'batch_{first:04d}.npz',**data,protocol_sha256=protocol_hash);pieces.append(data)
    combined={k:np.concatenate([d[k] for d in pieces]) for k in pieces[0]}
    np.savez_compressed(args.output_dir/f'{split}.npz',**combined,protocol_sha256=protocol_hash)
    return combined


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['actor','observer','dino-repo','dino-weights','output-dir']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--calibration-episodes',type=int,default=39);p.add_argument('--holdout-episodes',type=int,default=20)
    p.add_argument('--calibration-seed',type=int,default=2026110811);p.add_argument('--holdout-seed',type=int,default=2026110812)
    p.add_argument('--shape',type=float,nargs=3,default=[.03,.03,.2]);p.add_argument('--alpha',type=float,default=.05)
    p.add_argument('--batch-size',type=int,default=16);p.add_argument('--device',default='cpu')
    args=p.parse_args(argv)
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    if min(args.calibration_episodes,args.holdout_episodes,args.batch_size)<1 or not 0<args.alpha<1:raise ValueError('invalid counts or alpha')
    if args.calibration_seed==args.holdout_seed:raise ValueError('independent split seeds required')
    trajectory_scores(np.zeros((1,1,3)),args.shape)
    args.output_dir.mkdir(parents=True);torch.set_num_threads(4)
    source=Path(__file__).resolve().parent
    paths=[args.actor,args.observer,args.dino_weights,source/'trajectory_cp.py',source/'core.py',source/'renderer.py',
           source/'perception.py',source.parent/'dino_latent/backbone.py',source.parent/'lane_keeping/policy.py',source.parent/'lane_keeping/cp.py']
    binding={str(path):sha256(path) for path in paths}
    protocol=dict(task=core.config(),alpha=args.alpha,shape=args.shape,batch_size=args.batch_size,device=args.device,
        calibration_episodes=args.calibration_episodes,holdout_episodes=args.holdout_episodes,
        calibration_seed=args.calibration_seed,holdout_seed=args.holdout_seed,binding=binding,
        law='independent uniform initial poses, fixed nominal renderer, deterministic frozen observer and actor',
        score='max over all 120 frames and 3 coordinates of abs(normalized pose error)/fixed shape',
        rules=dict(fixed_inference_batch_shape=True,unsafe_runs_retained=True,holdout_after_threshold=True,model_update=False),
        scope='trajectory-marginal coverage under exchangeability; not fixed-calibration conditional coverage or changed scenes')
    write(args.output_dir/'protocol.json',protocol);protocol_hash=sha256(args.output_dir/'protocol.json')
    actor=load_actor(args.actor).requires_grad_(False);observer=ImageObserver(args.observer,args.dino_repo,args.dino_weights,args.device)
    cal=collect(actor,observer,args,'calibration',protocol_hash);q,rank=conformal_radius(cal['scores'],args.alpha)
    threshold=dict(q=float(q) if np.isfinite(q) else None,finite=bool(np.isfinite(q)),rank=rank,
        radii=(q*np.array(args.shape)).tolist() if np.isfinite(q) else None,shape=args.shape,alpha=args.alpha,
        coverage=coverage_summary(cal['scores'],q),unsafe=int(cal['unsafe'].sum()),
        protocol_sha256=protocol_hash,calibration_sha256=sha256(args.output_dir/'calibration.npz'))
    write(args.output_dir/'calibration.json',threshold);threshold_hash=sha256(args.output_dir/'calibration.json')
    print(json.dumps(dict(event='threshold_frozen',**threshold)),flush=True)
    test=collect(actor,observer,args,'holdout',protocol_hash)
    if binding!={str(path):sha256(path) for path in paths}:raise RuntimeError('bound models or source changed during collection')
    if threshold_hash!=sha256(args.output_dir/'calibration.json'):raise RuntimeError('threshold changed during holdout')
    report=dict(calibration=threshold,holdout=dict(coverage=coverage_summary(test['scores'],q,True),
        unsafe=int(test['unsafe'].sum()),reached=int(test['reached'].sum()),held=int(test['held'].sum())),
        frozen_models_unchanged=True,certificate_passed=False,
        safety_claim='None yet. Separately verify a neural SBC at these exact radii.',
        cp_claim='At least 1-alpha marginal coverage for a new complete 120-step rollout under the registered exchangeable law; randomness includes calibration and new rollout.')
    write(args.output_dir/'report.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
