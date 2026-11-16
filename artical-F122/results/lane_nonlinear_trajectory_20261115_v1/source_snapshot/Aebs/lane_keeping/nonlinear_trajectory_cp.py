"""Fresh image feedback, trajectory CP and learned SBC for nonlinear h.

The uncertainty set is state dependent: E(x)={h(x+d)-h(x): ||d||inf<=q}.
The explicit inverse gives pi(h^-1(h(x)+e))=pi(x+d) exactly over reals.
This is not certification of a constant raw-latent box or IEEE arithmetic.
"""
import argparse
import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from . import core
from .cp import conformal_radius
from .policy import load_actor
from .renderer import RoadRenderer
from .learned_sbc import sha256
from .trajectory_cp import collect, coverage_summary, write_json
from .joint_trajectory_cp import freeze_json
from .nonlinear_observation import NonlinearImageObserver


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['observer','actor','sbc','dino-repo','dino-weights','output-dir']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--calibration-episodes',type=int,default=199)
    p.add_argument('--holdout-episodes',type=int,default=100)
    p.add_argument('--calibration-seed',type=int,default=202611150101)
    p.add_argument('--holdout-seed',type=int,default=202611150102)
    p.add_argument('--horizon',type=int,default=80)
    p.add_argument('--alpha',type=float,default=.05)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if min(a.calibration_episodes,a.holdout_episodes,a.horizon,a.batch_size)<1 or not 0<a.alpha<1:
        raise ValueError('invalid counts or alpha')
    if a.calibration_seed==a.holdout_seed: raise ValueError('separate split streams required')
    if a.output_dir.exists() and not a.resume: raise FileExistsError(a.output_dir)
    a.output_dir.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    for src,name in [(a.observer,'nonlinear_observation.pt'),(a.actor,'actor.pt'),(a.sbc,'sbc.pt')]:
        target=a.output_dir/name
        if target.exists():
            if sha256(target)!=sha256(src): raise ValueError('changed model')
        else: shutil.copyfile(src,target)
    observer=NonlinearImageObserver(a.output_dir/'nonlinear_observation.pt',a.device,a.dino_repo,a.dino_weights)
    actor=load_actor(a.output_dir/'actor.pt').requires_grad_(False)
    model=copy.deepcopy(observer.projection).cpu().requires_grad_(False)
    torch.manual_seed(1501)
    x=torch.rand(10000,2)*2-1;delta=torch.rand_like(x)*.4-.2
    with torch.no_grad():
        recovered=model.decode(model.nominal(x+delta))
        coordinate_error=float((recovered-(x+delta)).abs().max())
        k=torch.randint(-1,2,(len(x),1)).float()
        action_error=float((actor(torch.cat([recovered,k],1))-actor(torch.cat([x+delta,k],1))).abs().max())
    if max(coordinate_error,action_error)>1e-5: raise ValueError('inverse bridge regression failed')
    src=Path(__file__).parent
    files=['nonlinear_trajectory_cp.py','nonlinear_observation.py','trajectory_cp.py','cp.py',
           'core.py','policy.py','renderer.py','patch_perception.py','joint_observation.py',
           'verify_learned.py','learned_sbc.py','../dino_latent/backbone.py']
    source_hashes={n:sha256(src/n) for n in files}
    bindings={n:sha256(a.output_dir/n) for n in ['nonlinear_observation.pt','actor.pt','sbc.pt']}
    protocol=dict(format='lane_nonlinear_trajectory_protocol_v1',
        horizon=a.horizon,alpha=a.alpha,calibration_episodes=a.calibration_episodes,
        holdout_episodes=a.holdout_episodes,calibration_seed=a.calibration_seed,holdout_seed=a.holdout_seed,
        batch_size=a.batch_size,device=a.device,torch_version=torch.__version__,
        bindings=bindings,source_sha256=source_hashes,dino_weights_sha256=sha256(a.dino_weights),
        score='max_t ||h_inverse(p(DINO(I_t)))-x_t||inf',
        latent_set='{h(x+delta)-h(x): ||delta||inf<=q}',
        controller='original semantic actor composed with explicit nonlinear inverse; no PPO retraining',
        distribution=dict(initial_low=core.INITIAL.low.tolist(),initial_high=core.INITIAL.high.tolist(),
            initial='independent uniform',kappas=list(core.KAPPAS),curvature='uniform and constant per episode',
            speed=core.SPEED,dt=core.DT,wheelbase=core.WHEELBASE,action_limit=core.PHI_MAX,
            scales=[core.E_SCALE,core.THETA_SCALE],image_size=[192,192],lighting=1.,camera_height=1.25,
            camera_lateral=0.,blur=0.,noise=0.,occluder=False,process_noise=0),
        rules=dict(fresh_images=True,unsafe_retained=True,threshold_never_capped=True,holdout_after_threshold=True),
        scope='Registered renderer law and 80 or specified steps; marginal CP plus conditional numerical SBC. No IEEE outward-rounding proof.')
    freeze_json(a.output_dir/'protocol.json',protocol);ph=sha256(a.output_dir/'protocol.json')
    renderer=RoadRenderer(192,192)
    cal=collect(a,'calibration',actor,observer,renderer,ph)
    q,rank=conformal_radius(cal['scores'],a.alpha)
    contract=dict(radius=float(q) if np.isfinite(q) else None,finite_radius=bool(np.isfinite(q)),
        n=len(cal['scores']),alpha=a.alpha,rank=rank,latent_set=protocol['latent_set'],score=protocol['score'],
        calibration_sha256=sha256(a.output_dir/'calibration.npz'),protocol_sha256=ph,
        inverse_coordinate_max_error=coordinate_error,inverse_action_max_error=action_error,
        coverage=coverage_summary(cal['scores'],q))
    freeze_json(a.output_dir/'perception_contract.json',contract);ch=sha256(a.output_dir/'perception_contract.json')
    hold=collect(a,'holdout',actor,observer,renderer,ph)
    diagnostics={}
    for name,data in [('calibration',cal),('holdout',hold)]:
        states=data['states'][:,:-1]/[core.E_SCALE,core.THETA_SCALE]
        with torch.no_grad():
            nominal=model.nominal(torch.tensor(states.reshape(-1,2),dtype=torch.float32)).numpy().reshape(states.shape)
        residual=data['latents']-nominal
        np.savez_compressed(a.output_dir/(name+'_raw_residuals.npz'),raw_latent_error=residual,
            decoded_error=data['errors'],nominal_latent=nominal)
        diagnostics[name]=dict(raw_latent_max_abs=float(abs(residual).max()),
                              decoded_max_abs=float(abs(data['errors']).max()))
    if ch!=sha256(a.output_dir/'perception_contract.json'): raise ValueError('threshold changed')
    if any(sha256(a.output_dir/n)!=h for n,h in bindings.items()): raise ValueError('model changed')
    if any(sha256(src/n)!=h for n,h in source_hashes.items()): raise ValueError('source changed')
    report=dict(contract=contract,holdout=coverage_summary(hold['scores'],q,independent=True),
        unsafe_calibration=int(cal['unsafe'].sum()),unsafe_holdout=int(hold['unsafe'].sum()),
        residual_diagnostics=diagnostics,certificate_status='pending',scope=protocol['scope'])
    write_json(a.output_dir/'report.json',report)
    if np.isfinite(q):
        target=a.output_dir/'verification'
        if not (target/'certificate.json').exists():
            subprocess.run([sys.executable,'-m','Aebs.lane_keeping.verify_learned',
                '--actor',str(a.output_dir/'actor.pt'),'--sbc',str(a.output_dir/'sbc.pt'),
                '--output-dir',str(target),'--radius',repr(float(q)),
                '--grid','40','--depth','18','--error-splits','8','--method','affine','--stop-on-witness'],check=True)
        cert=json.loads((target/'certificate.json').read_text())
        if cert['actor_sha256']!=bindings['actor.pt'] or cert['sbc_sha256']!=bindings['sbc.pt'] or cert['verification']['radius']!=q:
            raise ValueError('certificate binding mismatch')
        if cert['source_sha256']['core.py']!=source_hashes['core.py']: raise ValueError('plant mismatch')
        report['certificate_status']='passed_numerical' if cert['passed'] else 'not_passed'
        report['certificate_sha256']=sha256(target/'certificate.json')
    else: report['certificate_status']='not_passed_infinite_radius'
    report['bridge']='For e=h(x+delta)-h(x), inverse(h(x)+e)=x+delta. The verifier covers all delta in the new calibrated box.'
    write_json(a.output_dir/'report.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
