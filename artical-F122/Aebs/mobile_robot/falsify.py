"""Projected-gradient search for actual finite-horizon unsafe trajectories.

No counterexample found does not mean verified safe.
"""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from Aebs.lane_keeping.policy import load_actor
from . import core


def margins(x):
    p=x*x.new_tensor(core.SCALE)+x.new_tensor(core.CENTER)
    low=(p-p.new_tensor(core.LOW))/p.new_tensor(core.SCALE)
    high=(p.new_tensor(core.HIGH)-p)/p.new_tensor(core.SCALE)
    obstacle=(torch.linalg.vector_norm(p[...,:2]-p.new_tensor(core.OBSTACLE),dim=-1)-core.EXCLUSION_RADIUS)/2
    return torch.cat([low,high,obstacle[...,None]],-1).amin(-1)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--actor',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--radius',type=float,default=.03);p.add_argument('--iterations',type=int,default=120)
    p.add_argument('--radii',type=float,nargs=3)
    p.add_argument('--episodes',type=int,default=32);p.add_argument('--seed',type=int,default=917)
    args=p.parse_args(argv)
    radii=np.broadcast_to(np.asarray(args.radii if args.radii is not None else args.radius,dtype=float),(3,)).copy()
    if not np.isfinite(radii).all() or (radii<0).any():raise ValueError('invalid error box')
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True);torch.set_num_threads(4);torch.manual_seed(args.seed)
    actor=load_actor(args.actor).requires_grad_(False);rng=np.random.default_rng(args.seed)
    physical_initial=core.sample_initial(rng,args.episodes)
    initial=torch.tensor(core.normalize(physical_initial),dtype=torch.float32)
    limit=torch.tensor(radii,dtype=torch.float32)
    errors=torch.nn.Parameter((torch.rand(core.HORIZON,args.episodes,3)*2-1)*limit)
    optimizer=torch.optim.Adam([errors],lr=max(float(radii.max())/5,1e-4))
    best=torch.full((args.episodes,),float('inf'));best_errors=errors.detach().clone();history=[]
    for iteration in range(args.iterations):
        x=initial;trace=[x]
        for e in errors:x=core.torch_dynamics(x,actor(x+e));trace.append(x)
        trace=torch.stack(trace);margin=margins(trace).amin(0)
        with torch.no_grad():
            update=margin<best;best[update]=margin[update];best_errors[:,update]=errors[:,update]
        loss=margin.mean();optimizer.zero_grad();loss.backward();optimizer.step()
        with torch.no_grad():errors.copy_(torch.maximum(torch.minimum(errors,limit),-limit))
        if (iteration+1)%20==0:
            record=dict(iteration=iteration+1,minimum_margin=float(best.min()),unsafe=int((best<0).sum()))
            history.append(record);print(json.dumps(record),flush=True)
    with torch.no_grad():
        # Confirm using the deployed float64 plant and exact registered box,
        # independently of differentiable float32 optimization dynamics.
        replay_errors=np.clip(best_errors.detach().numpy().astype(np.float64),-radii,radii)
        x=physical_initial.copy();trace=[x.copy()]
        for e in replay_errors:
            u=actor(torch.tensor(core.normalize(x)+e,dtype=torch.float32)).numpy()
            x=core.dynamics(x,u);trace.append(x.copy())
        states=np.array(trace)
    report=dict(radius=radii.tolist(),episodes=args.episodes,iterations=args.iterations,
        unsafe=int(core.unsafe(states).any(0).sum()),minimum_normalized_margin=float(best.min()),
        verified=False,history=history,independent_float64_replay=True,
        unsafe_by_type=dict(position_x=int(((states[:,:,0]<core.LOW[0])|(states[:,:,0]>core.HIGH[0])).any(0).sum()),
            position_y=int(((states[:,:,1]<core.LOW[1])|(states[:,:,1]>core.HIGH[1])).any(0).sum()),
            heading=int(((states[:,:,2]<core.LOW[2])|(states[:,:,2]>core.HIGH[2])).any(0).sum()),
            obstacle=int((np.linalg.norm(states[:,:,:2]-core.OBSTACLE,axis=-1)<=core.EXCLUSION_RADIUS).any(0).sum())))
    np.savez_compressed(args.output_dir/'rollout.npz',states=states,errors=replay_errors)
    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
