"""Adaptive continuous-box audit of the learned three-dimensional SBC.

Checks B<=1 initially, B>=20 on all one-step reachable unsafe states,
and B(f(x,pi(x+e)))<=B(x) on safe states with B(x)<=20.
Float64 numerical enclosures; not a machine-checked floating-point proof.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from Aebs.lane_keeping.policy import load_actor
from Aebs.lane_keeping.learned_sbc import bounds,load_sbc_checkpoint,sha256
from . import core
from .bounds import grid_boxes,unsafe_boxes,error_boxes,transitions,bisect,wholly_unsafe


@torch.no_grad()
def region(barrier,lo,hi,kind,depth=12,max_pending=150000):
    history=[];initial_volume=float((hi-lo).prod(1).sum());terminal=0.
    for level in range(depth+1):
        pl=[];ph=[];extreme=-float('inf') if kind=='initial' else float('inf')
        for low,high in zip(lo.split(512),hi.split(512)):
            bl,bh=bounds(barrier,low,high)
            good=bh<=1 if kind=='initial' else bl>=20
            value=bh.max().item() if kind=='initial' else bl.min().item()
            extreme=max(extreme,value) if kind=='initial' else min(extreme,value)
            terminal+=float((high-low).prod(1)[good].sum());pl.append(low[~good]);ph.append(high[~good])
        lo,hi=torch.cat(pl),torch.cat(ph)
        history.append(dict(depth=level,unresolved=len(lo),extreme=extreme))
        print(json.dumps(dict(stage=kind,**history[-1])),flush=True)
        if not len(lo) or level==depth or 2*len(lo)>max_pending:break
        lo,hi=bisect(lo,hi)
    error=abs(initial_volume-terminal-float((hi-lo).prod(1).sum()))
    return dict(passed=not len(lo) and error<1e-9,unresolved=len(lo),partition_volume_error=error,history=history),lo,hi


@torch.no_grad()
def dynamics(actor,barrier,radius,grid=16,depth=12,max_pending=150000,active_level=20.):
    lo,hi=grid_boxes(core.normalize(core.LOW),core.normalize(core.HIGH),grid)
    initial_volume=float((hi-lo).prod(1).sum());terminal=0.;history=[]
    errors=error_boxes(radius,2)
    for level in range(depth+1):
        pl=[];ph=[]
        for low,high in zip(lo.split(256),hi.split(256)):
            bl,bh=bounds(barrier,low,high)
            exempt=(bl>active_level)|wholly_unsafe(low,high)
            terminal+=float((high-low).prod(1)[exempt].sum())
            low,high,bl=low[~exempt],high[~exempt],bl[~exempt]
            if not len(low):continue
            following=torch.zeros(len(low),dtype=low.dtype)
            for el,eh in errors:
                nl,nh=transitions(actor,low,high,el,eh)
                following=torch.maximum(following,bounds(barrier,nl,nh)[1])
            good=following<=bl
            terminal+=float((high-low).prod(1)[good].sum());pl.append(low[~good]);ph.append(high[~good])
        lo=torch.cat(pl) if pl else torch.empty((0,3),dtype=torch.float64)
        hi=torch.cat(ph) if ph else lo.clone()
        history.append(dict(depth=level,unresolved=len(lo)))
        print(json.dumps(dict(stage='dynamics',**history[-1])),flush=True)
        if not len(lo) or level==depth or 2*len(lo)>max_pending:break
        lo,hi=bisect(lo,hi)
    error=abs(initial_volume-terminal-float((hi-lo).prod(1).sum()))
    return dict(passed=not len(lo) and error<1e-9,unresolved=len(lo),partition_volume_error=error,history=history),lo,hi


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--actor',type=Path,required=True);p.add_argument('--sbc',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--radius',type=float,default=.03)
    p.add_argument('--radii',type=float,nargs=3,help='optional separate normalized x,y,heading halfwidths')
    p.add_argument('--grid',type=int,default=16);p.add_argument('--depth',type=int,default=12)
    p.add_argument('--active-level',type=float,default=20.)
    args=p.parse_args(argv)
    radius=args.radii if args.radii is not None else args.radius
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    if not np.isfinite(radius).all() or (np.asarray(radius)<0).any():raise ValueError('finite nonnegative radius required')
    if not 1<=args.active_level<=20:raise ValueError('active level must be between 1 and 20')
    args.output_dir.mkdir(parents=True);torch.set_num_threads(4);start=time.monotonic()
    actor=load_actor(args.actor).double().requires_grad_(False)
    barrier,_=load_sbc_checkpoint(args.sbc);barrier=barrier.double().requires_grad_(False)
    result={};feedback=[]
    for name,pair in [('initial',grid_boxes(core.normalize(core.INITIAL_LOW),core.normalize(core.INITIAL_HIGH),4)),('unsafe',unsafe_boxes())]:
        result[name],lo,hi=region(barrier,*pair,name,args.depth)
        np.savez_compressed(args.output_dir/f'{name}_unresolved.npz',lower=lo.numpy(),upper=hi.numpy())
        # Region samples are not dynamic feedback: do not relabel unsafe states safe.
    result['dynamics'],lo,hi=dynamics(actor,barrier,radius,args.grid,args.depth,active_level=args.active_level)
    np.savez_compressed(args.output_dir/'feedback.npz',states=((lo+hi)/2).numpy(),lower=lo.numpy(),upper=hi.numpy())
    result.update(passed=all(result[k]['passed'] for k in ['initial','unsafe','dynamics']),
        task=core.config(),radius=radius,active_level=args.active_level,actor_sha256=sha256(args.actor),sbc_sha256=sha256(args.sbc),
        seconds=time.monotonic()-start,
        scope='conditional sampled-time safety under all errors in registered box; float64 numerical audit, not formal rounding proof; no CP guarantee unless separately calibrated',
        unresolved_meaning='unproved boxes; not automatically real counterexamples')
    (args.output_dir/'certificate.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(passed=result['passed'],seconds=result['seconds'])),flush=True)


if __name__=='__main__':main()
