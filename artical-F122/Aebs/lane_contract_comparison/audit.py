"""Numerical continuous-domain lane audit with full visual and clipped-action constraints."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from ..lane_keeping import core
from ..lane_keeping.certify_neural import split
from ..lane_keeping.learned_sbc import bounds,error_boxes,grid_boxes,load_sbc_checkpoint,sha256,unsafe_boxes
from ..lane_keeping.policy import load_actor
from ..lane_keeping.verify import dynamics_bounds
from ..lane_keeping.verify_learned import region_check
from ..vision_sls.neural_bounds import tanh_affine_bounds
from .collect import ARMS,MARGINS,ROOT
from ..lane_keeping.robust_trajectory_cp import write_json

PAD=1e-12


def actor_interval(actor,low,high,kappa):
    curvature=low.new_full((len(low),1),kappa/core.KAPPA_SCALE)
    lo,hi=tanh_affine_bounds([*actor.policy_net,actor.action_net],
                            torch.cat([low,curvature],1),torch.cat([high,curvature],1))
    return lo.clamp(-1,1),hi.clamp(-1,1)


def allowed_action(lo,hi,nominal,action_cap,extra):
    """Trim by the calibrated action condition before the optional additive perturbation."""
    if action_cap is not None:
        lo=torch.maximum(lo,nominal[0]-action_cap-PAD)
        hi=torch.minimum(hi,nominal[1]+action_cap+PAD)
    valid=(hi>=lo).flatten()
    # Placeholder interval for infeasible rows, masked out of the successor upper bound.
    lo=torch.where(valid[:,None],lo,0.);hi=torch.where(valid[:,None],hi,0.)
    return (lo-extra-PAD).clamp(-1,1),(hi+extra+PAD).clamp(-1,1),valid


def next_upper(actor,barrier,l,h,kappa,contract,extra,splits):
    nominal=actor_interval(actor,l,h,kappa) if contract['action'] is not None else None
    following=torch.full((len(l),),float('-inf'),dtype=l.dtype)
    width=l.new_tensor(contract['width'])
    for el,eh in error_boxes(1,splits,l.dtype):
        lo,hi=actor_interval(actor,l+width*el,h+width*eh,kappa)
        alo,ahi,valid=allowed_action(lo,hi,nominal,contract['action'],extra)
        nl,nh=dynamics_bounds(l,h,alo,ahi,kappa)
        upper=bounds(barrier,nl-PAD,nh+PAD,'affine')[1]
        following=torch.maximum(following,upper.masked_fill(~valid,float('-inf')))
    if not torch.isfinite(following).all():
        raise RuntimeError('all error cells empty or nonfinite')
    return following


def find_witness(actor,barrier,x,kappa,contract,extra):
    if not len(x):
        return None
    curve=x.new_full((len(x),1),kappa/core.KAPPA_SCALE)
    nominal=actor(torch.cat([x,curve],1)).flatten()
    before=barrier(x).flatten()
    width=x.new_tensor(contract['width'])
    for i in [-1.,-.5,0.,.5,1.]:
        for j in [-1.,-.5,0.,.5,1.]:
            eta=width*x.new_tensor([i,j])
            observed=actor(torch.cat([x+eta,curve],1)).flatten()
            allowed=(torch.ones_like(before,dtype=torch.bool) if contract['action'] is None else
                     (observed-nominal).abs()<=contract['action'])
            for delta in [-extra,extra]:
                actual=(observed+delta).clamp(-1,1)[:,None]
                nxt,_=dynamics_bounds(x,x,actual,actual,kappa)
                after=barrier(nxt).flatten()
                bad=allowed & (before<=20) & (after-before>1e-8)
                if bad.any():
                    index=int(bad.nonzero()[0])
                    return dict(x=x[index].tolist(),kappa=kappa,error=eta.tolist(),additive_action=delta,
                                action_difference=float((observed-nominal)[index]),before=float(before[index]),
                                after=float(after[index]),next_state=nxt[index].tolist())
    return None


@torch.inference_mode()
def dynamic(actor,barrier,kappa,arm,extra,config):
    l,h=grid_boxes([-1,-1],[1,1],config['grid'])
    checked=0;area=0.
    for depth in range(config['max_depth']+1):
        ll=[];hh=[]
        for low,high in zip(l.split(512),h.split(512)):
            checked+=len(low)
            bl,_=bounds(barrier,low,high,'affine')
            outside=bl>20
            area+=float((high-low).prod(1)[outside].sum())
            low,high,bl=low[~outside],high[~outside],bl[~outside]
            if not len(low):
                continue
            upper=next_upper(actor,barrier,low,high,kappa,arm,extra,config['error_subdivisions'])
            good=upper<=bl
            area+=float((high-low).prod(1)[good].sum())
            ll.append(low[~good]);hh.append(high[~good])
        l=torch.cat(ll) if ll else l[:0]
        h=torch.cat(hh) if hh else h[:0]
        err=abs(area+float((h-l).prod(1).sum())-4.)
        if err>1e-8:
            raise RuntimeError('partition area lost')
        if len(l)==0:
            return dict(status='pass',checked=checked,depth=depth,unresolved=0,area_error=err)
        if depth%2==0:
            witness=find_witness(actor,barrier,(l+h)/2,kappa,arm,extra)
            if witness:
                return dict(status='witness',checked=checked,depth=depth,witness=witness)
        if len(l)>config['max_pending'] or depth==config['max_depth']:
            return dict(status='unresolved',checked=checked,depth=depth,unresolved=len(l),area_error=err)
        l,h=split(l,h)


def run(root):
    torch.set_num_threads(2)
    protocol=json.loads((root/'protocol.json').read_text())
    config=protocol['audit']
    contracts=json.loads((root/'contracts.json').read_text())
    if contracts['protocol']!=sha256(root/'protocol.json'):
        raise ValueError('protocol mismatch')
    package=ROOT/'package'
    for name,field in [('actor','actor_sha256'),('sbc','sbc_sha256')]:
        if sha256(package/(name+'.pt'))!=protocol['binding'][field]:
            raise ValueError('changed model')
    actor=load_actor(package/'actor.pt').double().requires_grad_(False)
    barrier=load_sbc_checkpoint(package/'sbc.pt')[0].double().requires_grad_(False)
    il,ih=grid_boxes([-core.INIT_E/core.E_SCALE,-core.INIT_THETA/core.THETA_SCALE],
                     [core.INIT_E/core.E_SCALE,core.INIT_THETA/core.THETA_SCALE],16)
    init,_=region_check(barrier,il,ih,True,'affine')
    unsafe,_=region_check(barrier,*unsafe_boxes(16),False,'affine')
    first_exit=[1+core.SPEED*core.DT/core.E_SCALE*np.sin(core.THETA_SCALE),
                1+core.SPEED*core.DT/core.THETA_SCALE*(np.tan(core.PHI_MAX)/core.WHEELBASE+max(abs(k) for k in core.KAPPAS))]
    if not(init['passed'] and unsafe['passed']) or max(first_exit)>=1.3:
        raise RuntimeError('separation/domain conditions unproved')
    result=dict(contracts_sha256=sha256(root/'contracts.json'),audit_source=sha256(Path(__file__)),
                first_exit_bound=first_exit,initial=init,unsafe=unsafe,runs={},summary={})
    capgrid=config['extra_normalized_input_error']
    if capgrid!=MARGINS:
        raise ValueError('grid differs from registered setup')
    for arm in ARMS:
        status={}
        for margin in capgrid:
            # A point violating at a registered smaller perturbation remains a valid
            # witness at every larger perturbation, along the very same transition.
            previous=next((m for m,r in status.items() if r['status']=='witness'),None)
            if previous is not None:
                status[margin]=dict(status='witness_by_inclusion',from_margin=previous)
                continue
            t=time.monotonic();rows={}
            for k in core.KAPPAS:
                rows[k]=dynamic(actor,barrier,k,contracts['arms'][arm],margin,config)
                if rows[k]['status']=='witness':
                    break
            flags=[row['status'] for row in rows.values()]
            final='witness' if 'witness' in flags else ('unresolved' if 'unresolved' in flags else 'pass')
            status[margin]=dict(status=final,curvatures=rows,seconds=time.monotonic()-t)
            print(arm,margin,final,round(time.monotonic()-t,2),flush=True)
        result['runs'][arm]=status
        passed=[m for m,r in status.items() if r['status']=='pass']
        result['summary'][arm]=dict(largest_grid_pass=max(passed) if passed else None,
                                    statuses={str(m):r['status'] for m,r in status.items()})
        write_json(root/'verification.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('root',type=Path)
    run(p.parse_args().root)
