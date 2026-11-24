"""Retarget ONLY our neural barrier learner/auditor to native SPVT physics.

The RGB observer and neural controller stay frozen. No SPVT module is patched.
Existing project learner and adaptive interval auditor are reused unchanged.
"""
import argparse
import json
import math
from pathlib import Path


def configure():
    import numpy as np
    import torch
    from Aebs.lane_keeping import core, learned_sbc, train_joint
    core.SPEED=5.;core.DT=.05;core.WHEELBASE=2.5;core.HORIZON=160
    core.INIT_E=.6;core.INIT_THETA=math.radians(7)
    core.E_MAX=.8;core.THETA_MAX=.30
    # Frozen observer/controller were trained in these coordinates.
    core.E_SCALE=.85;core.THETA_SCALE=.30;core.KAPPAS=(0.,)
    core.INITIAL=core.Box(np.array([-.6,-math.radians(7)]),np.array([.6,math.radians(7)]))
    def unsafe_boxes(grid=16,dtype=torch.float64):
        limit=.8/.85
        regions=[([-1.3,-1.3],[-limit,1.3]),([limit,-1.3],[1.3,1.3]),
                 ([-limit,-1.3],[limit,-1]),([-limit,1],[limit,1.3])]
        pairs=[learned_sbc.grid_boxes(l,h,grid,dtype) for l,h in regions]
        return torch.cat([r[0] for r in pairs]),torch.cat([r[1] for r in pairs])
    learned_sbc.unsafe_boxes=unsafe_boxes;train_joint.unsafe_boxes=unsafe_boxes
    @torch.no_grad()
    def rollout(actor,radius,seed):
        rng=np.random.default_rng(seed)
        x=torch.tensor(core.sample_initial(rng,256)/[.85,.30],dtype=torch.float32)
        x[:4]=torch.tensor([[e,t] for e in [-.6/.85,.6/.85] for t in [-math.radians(7)/.30,math.radians(7)/.30]])
        records=[x]
        from Aebs.lane_keeping.sbc import next_states_for_errors
        for _ in range(160):
            errors=torch.tensor(rng.uniform(-radius,radius,(len(x),1,2)),dtype=x.dtype)
            x=next_states_for_errors(actor,x,torch.zeros(len(x)),errors)[:,0]
            records.append(x)
        return torch.cat(records)
    train_joint.collect_rollout=rollout
    return train_joint


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['train','audit'])
    p.add_argument('--actor',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--radius',type=float,default=.12)
    p.add_argument('--barrier',type=Path)
    p.add_argument('--epochs',type=int,default=4000)
    p.add_argument('--resume',type=Path)
    p.add_argument('--extra-states',type=Path)
    p.add_argument('--hidden',type=int,nargs='+',default=[16,8])
    p.add_argument('--bound-method',choices=['ibp','affine'],default='ibp')
    p.add_argument('--seed',type=int,default=2026112313)
    p.add_argument('--depth',type=int,default=16)
    a=p.parse_args();train=configure()
    if a.phase=='train':
        if a.bound_method=='affine':
            # Reuse the existing stable bound-training mode: tangent search
            # points are constants for gradients, while affine weights train.
            from Aebs.lane_keeping.learned_sbc import raw_bounds as original_bounds
            def fixed_root_bounds(barrier,lo,hi,method='affine'):
                return original_bounds(barrier,lo,hi,method,detach_root=True)
            train.raw_bounds=fixed_root_bounds
        args=['--actor',str(a.actor),'--output-dir',str(a.output),'--radius',str(a.radius),
              '--epochs',str(a.epochs),'--seed',str(a.seed),'--hidden',*[str(v) for v in a.hidden],
              '--error-splits','4','--bound-method',a.bound_method]
        if a.resume:args+=['--resume',str(a.resume),'--learning-rate','.001']
        if a.extra_states:args+=['--extra-states',str(a.extra_states)]
        train.main(args)
    else:
        if a.barrier is None:raise ValueError('barrier required for audit')
        from Aebs.lane_keeping import verify_learned
        verify_learned.main(['--actor',str(a.actor),'--sbc',str(a.barrier),'--output-dir',str(a.output),
            '--radius',str(a.radius),'--grid','40','--depth',str(a.depth),'--error-splits','4','--stop-on-witness'])
    from .proxy_architecture_compare import sha
    record=dict(wrapper_sha256=sha(__file__),phase=a.phase,actor_sha256=sha(a.actor),radius=a.radius,
        plant=dict(speed=5.,dt=.05,wheelbase=2.5,initial_cm=[-60,60],initial_heading_degrees=[-7,7],
                   safety_lateral_m=.8,auxiliary_heading_bound_rad=.30,normalization=[.85,.30],curvature=0),
        native_spvt_changed=False,controller_changed=False,
        scope='our neural barrier only; global numerical verification required before any guarantee')
    (a.output/'task_adapter.json').write_text(json.dumps(record,indent=2)+'\n')
    (a.output/'executed_adapter.py').write_bytes(Path(__file__).read_bytes())


if __name__=='__main__':main()
