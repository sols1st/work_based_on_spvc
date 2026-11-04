"""Independent startup-only, group-conditional joint split conformal data.

One sampling unit is one newly drawn initial pose and its image. Fit, calibration,
audit and deterministic center checks are disjoint. No closed-loop frames enter.
"""
import argparse,json,math,time
from pathlib import Path
import numpy as np
import torch
from .core import INITIAL_STATES,C,RENDER_OFFSET
from .ppo_env import FrozenImageSensor
from .train_ppo import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--bound',type=float,default=.005);p.add_argument('--seed',type=int,default=641907)
    p.add_argument('--fit',type=int,default=16);p.add_argument('--calibration',type=int,default=32)
    p.add_argument('--audit',type=int,default=16);p.add_argument('--alpha',type=float,default=.05)
    p.add_argument('--device',default='cuda');p.add_argument('--threads',type=int,default=1)
    a=p.parse_args();out=Path(a.output_dir);root=Path(a.candidate)
    if out.exists() or a.bound<=0 or min(a.fit,a.calibration,a.audit)<1 or not 0<a.alpha<1:p.error('new output and positive valid sample sizes required')
    rank=math.ceil((a.calibration+1)*(1-a.alpha))
    if rank>a.calibration:p.error('not enough calibration units for finite quantile')
    torch.set_num_threads(a.threads);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out.mkdir(parents=True);rng=np.random.default_rng(a.seed)
    states=[];groups=[];splits=[]
    for j in range(15):
        for split,count in [('fit',a.fit),('calibration',a.calibration),('audit',a.audit),('center',1)]:
            x=np.repeat(INITIAL_STATES[j:j+1],count,axis=0)
            if split!='center':x=x+rng.uniform(-a.bound,a.bound,(count,4))
            states.extend(x);groups.extend([j]*count);splits.extend([split]*count)
    states=np.asarray(states);groups=np.asarray(groups);splits=np.asarray(splits)
    cfg=dict(vars(a),projection_sha256=sha(root/'projection.pt'),
             dino_sha256=sha('../external/dinov2_weights/dinov2_vitg14_pretrain.pth'),
             unit='independent initial-state draw and one fresh render',
             distribution='uniform +/- bound around each of 15 registered starts; fixed scene and renderer',
             control_policy_does_not_affect_sampling=True,trajectory_frames_reused=False,
             probability_scope='one future initial draw from its same group protocol; not per-state conditional coverage')
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    sensor=FrozenImageSensor('../VISION-SLS','../external/dinov2','../external/dinov2_weights/dinov2_vitg14_pretrain.pth',root/'projection.pt',device=a.device)
    observations=[];started=time.monotonic()
    try:
        for n,x in enumerate(states):
            observations.append(sensor(x))
            if (n+1)%65==0:print('startup images',n+1,'/',len(states),flush=True)
    finally:sensor.close()
    y=np.asarray(observations);target=(states+RENDER_OFFSET)@C.T;residual=y-target
    biases=[];scales=[];multipliers=[];radii=[];records=[]
    for j in range(15):
        fit=residual[(groups==j)&(splits=='fit')];bias=fit.mean(0)
        scale=np.maximum(np.quantile(np.abs(fit-bias),.9,axis=0),.01)
        cal=residual[(groups==j)&(splits=='calibration')]
        score=np.max(np.abs(cal-bias)/scale,axis=1);q=float(np.sort(score)[rank-1]);r=np.nextafter(q*scale,np.inf)
        audit=residual[(groups==j)&(splits=='audit')];center=residual[(groups==j)&(splits=='center')]
        rows=dict(group=j,rank=rank,calibration_count=len(cal),q=q,radius=r.tolist(),
                  calibration_covered=int((np.abs(cal-bias)<=r).all(1).sum()),
                  audit_count=len(audit),audit_covered=int((np.abs(audit-bias)<=r).all(1).sum()),
                  center_covered=bool((np.abs(center-bias)<=r).all()))
        records.append(rows);biases.append(bias);scales.append(scale);multipliers.append(q);radii.append(r)
    model=dict(schema='parking_startup_group_cp_v1',initial_bound=a.bound,bias=np.asarray(biases).tolist(),
               scale=np.asarray(scales).tolist(),q=multipliers,radius=np.asarray(radii).tolist(),alpha=a.alpha,
               projection_sha256=cfg['projection_sha256'],dino_sha256=cfg['dino_sha256'],
               calibration_rank=rank,calibration_units_per_group=a.calibration,
               initial_states=INITIAL_STATES.tolist(),coverage_requires_matching_startup_protocol=True,
               full_trajectory_cp_claimed=False)
    np.savez_compressed(out/'samples.npz',states=states,observations=y,targets=target,residuals=residual,groups=groups,splits=splits)
    (out/'startup_model.json').write_text(json.dumps(model,indent=2)+'\n')
    metrics=dict(config=cfg,groups=records,audit_covered=sum(r['audit_covered'] for r in records),audit_count=15*a.audit,
                 centers_covered=sum(r['center_covered'] for r in records),runtime_seconds=time.monotonic()-started)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print('center coverage',metrics['centers_covered'],'/15; audit',metrics['audit_covered'],'/',metrics['audit_count'],flush=True)


if __name__=='__main__':main()
