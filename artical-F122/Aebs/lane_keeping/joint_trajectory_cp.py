"""Fresh trajectory CP for jointly learned visual p and C, then SBC audit.

The latent residual set is {C delta: ||delta||_infinity <= q}. It is not
the axis-aligned raw-latent box [-q,q]^2. Under the explicit inverse actor
interface, the semantic verifier sees precisely the delta box.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from . import core
from .cp import conformal_radius
from .joint_observation import JointImageObserver, DecodedJointObserver, LatentActor
from .learned_sbc import sha256
from .policy import load_actor
from .renderer import RoadRenderer
from .trajectory_cp import collect, coverage_summary, write_json


def freeze_json(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value: raise ValueError('frozen artifact changed: '+str(path))
    else: write_json(path,value)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--joint',type=Path,required=True)
    p.add_argument('--actor',type=Path,required=True)
    p.add_argument('--sbc',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--dino-repo',type=Path,required=True)
    p.add_argument('--dino-weights',type=Path,required=True)
    p.add_argument('--calibration-episodes',type=int,default=199)
    p.add_argument('--holdout-episodes',type=int,default=100)
    p.add_argument('--calibration-seed',type=int,default=202611090101)
    p.add_argument('--holdout-seed',type=int,default=202611090102)
    p.add_argument('--horizon',type=int,default=80)
    p.add_argument('--alpha',type=float,default=.05)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--verify',action='store_true')
    args=p.parse_args(argv)
    if min(args.calibration_episodes,args.holdout_episodes,args.horizon,args.batch_size)<1:
        raise ValueError('positive sample counts and horizon required')
    if not 0<args.alpha<1 or args.calibration_seed==args.holdout_seed:
        raise ValueError('valid alpha and separate split seeds required')
    if args.output_dir.exists() and not args.resume: raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for src,name in [(args.joint,'joint_observation.pt'),(args.actor,'actor.pt'),(args.sbc,'sbc.pt')]:
        target=args.output_dir/name
        if target.exists():
            if sha256(target)!=sha256(src):raise ValueError('frozen model mismatch')
        else:shutil.copyfile(src,target)
    torch.set_num_threads(4)
    observer=JointImageObserver(args.output_dir/'joint_observation.pt','cpu',args.dino_repo,args.dino_weights)
    decoded=DecodedJointObserver(observer)
    actor=load_actor(args.output_dir/'actor.pt').requires_grad_(False)
    C=observer.projection.matrix().detach().cpu()
    latent_actor=LatentActor(actor,C)
    # Numerical regression of the algebraic bridge; not a proof of all floating-point errors.
    generator=torch.Generator().manual_seed(90109)
    x=2*torch.rand(10000,2,generator=generator)-1
    delta=.3*(2*torch.rand(10000,2,generator=generator)-1)
    k=torch.randint(-1,2,(10000,1),generator=generator).float()
    with torch.no_grad():
        a=actor(torch.cat([x+delta,k],dim=1))
        b=latent_actor(torch.cat([(x+delta)@C.T,k],dim=1))
    bridge_error=float((a-b).abs().max())
    if bridge_error>1e-5:raise ValueError('inverse-coordinate actor bridge regression failed')
    source=Path(__file__).resolve().parent
    protocol=dict(format='lane_joint_trajectory_cp_protocol_v1',
        horizon=args.horizon,alpha=args.alpha,calibration_episodes=args.calibration_episodes,
        holdout_episodes=args.holdout_episodes,calibration_seed=args.calibration_seed,
        holdout_seed=args.holdout_seed,batch_size=args.batch_size,device='cpu',threads=4,
        C=C.tolist(),state_scales=[core.E_SCALE,core.THETA_SCALE],
        controller='pi_latent(y,k)=pi_old(solve(C,y),k); coordinate adaptation, no PPO retraining',
        score='max_t ||solve(C,p(DINO(I_t)))-x_t||_infinity',
        latent_set='p(DINO(I))-C*x in {C*delta: ||delta||_infinity<=q}',
        initial_low=core.INITIAL.low.tolist(),initial_high=core.INITIAL.high.tolist(),
        initial_law='independent uniform; curvature independent uniform, constant per episode',
        kappas=list(core.KAPPAS),renderer=dict(size=[192,192],lighting=1.,camera_height=1.25,
            camera_lateral=0.,blur=0.,noise=0.,occluder=False),
        dynamics=dict(speed=core.SPEED,dt=core.DT,wheelbase=core.WHEELBASE,action_limit=core.PHI_MAX),
        bindings={name:sha256(args.output_dir/name) for name in ['joint_observation.pt','actor.pt','sbc.pt']},
        dino_weights_sha256=sha256(args.dino_weights),
        source_sha256={name:sha256(source/name) for name in ['joint_trajectory_cp.py','joint_observation.py',
            'trajectory_cp.py','core.py','renderer.py','policy.py','cp.py']},
        rules=dict(fresh_rollouts=True,holdout_after_threshold_frozen=True,unsafe_rollouts_retained=True,
                   threshold_never_capped=True,model_frozen_before_calibration=True),
        scope='Registered nominal renderer law, fresh 80-step rollouts, marginal split CP; no changed camera, real-world or longer-horizon guarantee')
    freeze_json(args.output_dir/'protocol.json',protocol)
    protocol_hash=sha256(args.output_dir/'protocol.json')
    renderer=RoadRenderer(192,192)
    cal=collect(args,'calibration',actor,decoded,renderer,protocol_hash)
    q,rank=conformal_radius(cal['scores'],args.alpha)
    contract=dict(format='lane_joint_latent_cp_v1',protocol_sha256=protocol_hash,
        calibration_sha256=sha256(args.output_dir/'calibration.npz'),alpha=args.alpha,n=len(cal['scores']),rank=rank,
        radius=float(q) if np.isfinite(q) else None,finite_radius=bool(np.isfinite(q)),
        C=C.tolist(),latent_set=protocol['latent_set'],score=protocol['score'],
        numerical_bridge_max_action_error=bridge_error,coverage=coverage_summary(cal['scores'],q),
        claim='Under exchangeability of frozen-controller rollout scores, marginal probability of a new full registered-horizon rollout lying in this tube is at least 1-alpha.')
    freeze_json(args.output_dir/'perception_contract.json',contract)
    threshold_hash=sha256(args.output_dir/'perception_contract.json')
    print(json.dumps(dict(event='threshold_frozen',q=contract['radius'],rank=rank)),flush=True)
    hold=collect(args,'holdout',actor,decoded,renderer,protocol_hash)
    if sha256(args.output_dir/'perception_contract.json')!=threshold_hash:
        raise ValueError('threshold changed during holdout')
    roundtrip={}
    for name,data in [('calibration',cal),('holdout',hold)]:
        states=data['states'][:,:-1]/[core.E_SCALE,core.THETA_SCALE]
        residual=data['latents']-states@C.numpy().T
        metric=np.linalg.solve(C.double().numpy(),residual.reshape(-1,2).T).T.reshape(residual.shape)
        roundtrip[name]=float(np.max(np.abs(metric-data['errors'])))
        np.savez_compressed(args.output_dir/(name+'_latent_residuals.npz'),latent_residual=residual,
                            decoded_error=data['errors'],C=C.numpy())
    report=dict(format='lane_joint_full_experiment_v1',contract=contract,
        holdout=coverage_summary(hold['scores'],q,independent=True),
        unsafe_calibration=int(cal['unsafe'].sum()),unsafe_holdout=int(hold['unsafe'].sum()),
        raw_latent_to_decoded_error_roundoff=roundtrip,
        certificate_status='pending separate audit',scope=protocol['scope'])
    write_json(args.output_dir/'report.json',report)
    if args.verify and np.isfinite(q):
        audit=args.output_dir/'verification'
        # No selecting a smaller q to make the audit pass.
        if not (audit/'certificate.json').exists():
            command=[sys.executable,'-m','Aebs.lane_keeping.verify_learned',
                '--actor',str(args.output_dir/'actor.pt'),'--sbc',str(args.output_dir/'sbc.pt'),
                '--output-dir',str(audit),'--radius',repr(float(q)),'--grid','40',
                '--depth','18','--error-splits','8','--method','affine','--stop-on-witness']
            subprocess.run(command,check=True)
        cert=json.loads((audit/'certificate.json').read_text())
        if (cert['actor_sha256']!=protocol['bindings']['actor.pt'] or
            cert['sbc_sha256']!=protocol['bindings']['sbc.pt'] or cert['verification']['radius']!=q):
            raise ValueError('audit/model/radius mismatch')
        report['certificate_status']='passed_numerical' if cert['passed'] else 'not_passed'
        report['certificate_sha256']=sha256(audit/'certificate.json')
        report['bridge']='For e_y=C*delta: pi_latent(C*x+e_y,k)=pi_old(x+delta,k); audit uses this exact real-arithmetic identity. Not an outward-rounded floating-point proof.'
        write_json(args.output_dir/'report.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
