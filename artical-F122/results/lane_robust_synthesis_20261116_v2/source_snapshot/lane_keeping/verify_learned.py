"""Continuous-box numerical audit of the AEBS-class learned lane SBC.

Unknown boxes are not violations and are never accepted as proof. Actual
point witnesses and unresolved box centres are saved separately for development.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from . import core
from .policy import load_actor
from .sbc import next_states_for_errors
from .certify_neural import split
from .learned_sbc import (load_sbc_checkpoint, sha256, bounds, grid_boxes,
                         unsafe_boxes, error_boxes, transitions)


@torch.no_grad()
def region_check(barrier, lo, hi, initial, method, depth=12):
    accepted=0;best=0. if initial else float('inf'); history=[]
    original_area=float((hi-lo).prod(1).sum());area=0.
    for level in range(depth+1):
        lows,highs=[],[]
        for l,h in zip(lo.split(2048),hi.split(2048)):
            bl,bh=bounds(barrier,l,h,method)
            good=bh<=1 if initial else bl>=20
            if good.any():
                value=float(bh[good].max()) if initial else float(bl[good].min())
                best=max(best,value) if initial else min(best,value)
            accepted+=int(good.sum());area+=float((h-l).prod(1)[good].sum())
            lows.append(l[~good]);highs.append(h[~good])
        lo,hi=torch.cat(lows),torch.cat(highs)
        history.append(dict(depth=level,unresolved=len(lo),accepted=accepted))
        if len(lo)==0 or level==depth or len(lo)>100000:break
        lo,hi=split(lo,hi)
    area_error=abs(area+float((hi-lo).prod(1).sum())-original_area)
    return dict(passed=len(lo)==0 and area_error<1e-8,unresolved=len(lo),
                bound=best if np.isfinite(best) else None,partition_area_error=area_error,
                history=history), (lo+hi)/2


@torch.no_grad()
def point_witnesses(actor, barrier, states, kappa, radius):
    cuts=[-radius,0.,radius]
    errors=torch.tensor([[a,b] for a in cuts for b in cuts],dtype=states.dtype)
    witnesses=[]
    for x in states.split(1024):
        if not len(x):continue
        b=barrier(x).flatten()
        nx=next_states_for_errors(actor,x,torch.full((len(x),),kappa/core.KAPPA_SCALE,dtype=x.dtype),
                                 errors[None].expand(len(x),-1,-1))
        drift=barrier(nx.flatten(0,1)).reshape(len(x),-1)-b[:,None]
        worst,idx=drift.max(1)
        bad=(b<=20)&(worst>1e-9)
        if bad.any():
            witnesses.append(torch.cat([x[bad],errors[idx[bad]],worst[bad,None]],1))
    return torch.cat(witnesses) if witnesses else torch.empty((0,5),dtype=states.dtype)


@torch.no_grad()
def dynamic_check(actor,barrier,kappa,radius,grid,depth,splits,method,max_pending,stop_on_witness):
    lo,hi=grid_boxes([-1,-1],[1,1],grid)
    errors=error_boxes(radius,splits)
    counts=dict(outside=0,plateau=0,decreasing=0);history=[];area=0.
    witnesses=torch.empty((0,5),dtype=lo.dtype)
    for level in range(depth+1):
        pending_lo,pending_hi=[],[]
        for low,high in zip(lo.split(512),hi.split(512)):
            bl,bh=bounds(barrier,low,high,method)
            outside=bl>20
            counts['outside']+=int(outside.sum());area+=float((high-low).prod(1)[outside].sum())
            low,high,bl=low[~outside],high[~outside],bl[~outside]
            if not len(low):continue
            following=torch.zeros(len(low),dtype=low.dtype)
            for el,eh in errors:
                nl,nh=transitions(actor,low,high,kappa,el,eh,method)
                following=torch.maximum(following,bounds(barrier,nl,nh,method)[1])
            plateau=following==0
            decrease=(~plateau)&(following<=bl)
            good=plateau|decrease
            counts['plateau']+=int(plateau.sum());counts['decreasing']+=int(decrease.sum())
            area+=float((high-low).prod(1)[good].sum())
            pending_lo.append(low[~good]);pending_hi.append(high[~good])
        pl=torch.cat(pending_lo) if pending_lo else torch.empty((0,2),dtype=lo.dtype)
        ph=torch.cat(pending_hi) if pending_hi else pl.clone()
        entry=dict(kappa=kappa,depth=level,unresolved=len(pl),**counts)
        history.append(entry);print(json.dumps(entry),flush=True)
        if level==2 or (level>=4 and level%2==0):
            centres=(pl+ph)/2
            witnesses=point_witnesses(actor,barrier,centres[:30000],kappa,radius)
        if (not len(pl) or level==depth or len(pl)*2>max_pending or
                (stop_on_witness and len(witnesses))):
            error=abs(area+float((ph-pl).prod(1).sum())-4.)
            return dict(kappa=kappa,passed=not len(pl) and error<1e-8,
                        unresolved=len(pl),actual_witnesses=len(witnesses),
                        partition_area_error=error,history=history), (pl+ph)/2,witnesses
        lo,hi=split(pl,ph)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--actor',type=Path,required=True)
    p.add_argument('--sbc',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--radius',type=float,default=.08)
    p.add_argument('--grid',type=int,default=40)
    p.add_argument('--depth',type=int,default=16)
    p.add_argument('--error-splits',type=int,default=4)
    p.add_argument('--method',choices=['affine','ibp'],default='affine')
    p.add_argument('--max-pending',type=int,default=100000)
    p.add_argument('--stop-on-witness',action='store_true')
    args=p.parse_args(argv)
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    if (not np.isfinite(args.radius) or args.radius<0 or min(args.grid,args.error_splits)<1
            or args.depth<0 or args.max_pending<1):
        raise ValueError('invalid audit settings')
    args.output_dir.mkdir(parents=True);torch.set_num_threads(4)
    actor=load_actor(args.actor).double().requires_grad_(False)
    barrier,metadata=load_sbc_checkpoint(args.sbc);barrier=barrier.double().requires_grad_(False)
    il,ih=grid_boxes([-core.INIT_E/core.E_SCALE,-core.INIT_THETA/core.THETA_SCALE],
                    [core.INIT_E/core.E_SCALE,core.INIT_THETA/core.THETA_SCALE],16)
    init,ic=region_check(barrier,il,ih,True,args.method)
    ul,uh=unsafe_boxes(16)
    unsafe,uc=region_check(barrier,ul,uh,False,args.method)
    # Universal action clipping bounds all first exits into this audited domain.
    reach=[1+core.SPEED*core.DT/core.E_SCALE*np.sin(core.THETA_SCALE),
           1+core.SPEED*core.DT/(core.WHEELBASE*core.THETA_SCALE)*np.tan(core.PHI_MAX)
           +core.SPEED*core.DT*max(abs(k) for k in core.KAPPAS)/core.THETA_SCALE]
    assert max(reach)<1.3
    checks=[];centres=[];witnesses=[]
    for k in core.KAPPAS:
        report,points,wit=dynamic_check(actor,barrier,k,args.radius,args.grid,args.depth,
                                       args.error_splits,args.method,args.max_pending,args.stop_on_witness)
        checks.append(report);centres.append(points)
        if len(wit):witnesses.append(torch.cat([wit,torch.full((len(wit),1),k)],1))
    result=dict(format='lane_aebs_learned_sbc_audit_v1',passed=init['passed'] and unsafe['passed'] and all(c['passed'] for c in checks),
                initial=init,unsafe=unsafe,checks=checks,actor_sha256=sha256(args.actor),
                sbc_sha256=sha256(args.sbc),metadata=metadata,verification=vars(args).copy(),
                task=dict(speed=core.SPEED,dt=core.DT,safety_halfwidths=[core.E_MAX,core.THETA_MAX],
                          initial_halfwidths=[core.INIT_E,core.INIT_THETA],kappas=list(core.KAPPAS),
                          wheelbase=core.WHEELBASE,action_limit=core.PHI_MAX,
                          observation_scaling=[core.E_SCALE,core.THETA_SCALE],
                          successor_coordinate_abs_bounds=reach,unsafe_audited_domain=[-1.3,1.3],process_noise=0),
                source_sha256={p.name:sha256(p) for p in [Path(__file__),
                    Path(__file__).with_name('core.py'),Path(__file__).with_name('learned_sbc.py'),
                    Path(__file__).with_name('verify.py'),Path(__file__).with_name('policy.py'),
                    Path(__file__).parent.parent/'vision_sls/sbc_cp.py',
                    Path(__file__).parent.parent/'vision_sls/neural_bounds.py']},
                scope='Conditional on every normalized observation error in the radius box. Continuous-box numerical audit, no outward-rounded IEEE proof, no unconditional CP trajectory probability.')
    result['verification']={k:str(v) if isinstance(v,Path) else v for k,v in result['verification'].items()}
    (args.output_dir/'certificate.json').write_text(json.dumps(result,indent=2))
    np.savez_compressed(args.output_dir/'feedback.npz',states=torch.cat(centres).numpy(),
                        initial_unresolved=ic.numpy(),unsafe_unresolved=uc.numpy(),
                        actual_witnesses=torch.cat(witnesses).numpy() if witnesses else np.empty((0,6)))
    print(json.dumps(dict(passed=result['passed'],initial=init,unsafe=unsafe)),flush=True)


if __name__=='__main__':main()
