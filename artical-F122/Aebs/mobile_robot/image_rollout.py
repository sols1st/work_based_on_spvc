"""Full-horizon image-in-the-loop development rollouts, not final CP."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from Aebs.lane_keeping.policy import load_actor
from Aebs.lane_keeping.learned_sbc import sha256
from . import core
from .perception import ImageObserver
from .renderer import RobotRenderer


@torch.no_grad()
def rollout(actor,observer,n,seed):
    rng=np.random.default_rng(seed);x=core.sample_initial(rng,n);renderer=RobotRenderer()
    states=[x.copy()];observations=[];actions=[];start=time.monotonic()
    for t in range(core.HORIZON):
        images=torch.tensor(np.array([renderer.render(s) for s in x]))
        obs=torch.cat([observer(batch).cpu() for batch in images.split(32)])
        u=actor(obs).numpy();observations.append(obs.numpy());actions.append(u)
        x=core.dynamics(x,u);states.append(x.copy())
        if t%20==0:print(json.dumps(dict(step=t,episodes=n,seconds=time.monotonic()-start)),flush=True)
    states=np.array(states);observations=np.array(observations);actions=np.array(actions)
    errors=observations-core.normalize(states[:-1]);scores=np.abs(errors).max(axis=(0,2))
    bad=core.unsafe(states).any(0);reach=core.goal(states).any(0)
    held=core.goal(states[-10:]).all(0)&(((actions[-10:,:,0]+1)*core.V_MAX/2).max(0)<.08)
    report=dict(episodes=n,unsafe=int(bad.sum()),reached=int((reach&~bad).sum()),reached_and_held=int((held&~bad).sum()),
        normalized_mae=np.abs(errors).mean((0,1)).tolist(),max_abs_by_coordinate=np.abs(errors).max((0,1)).tolist(),
        trajectory_score_max=float(scores.max()),trajectory_score_q95=float(np.quantile(scores,.95)),
        minimum_obstacle_clearance=float(np.linalg.norm(states[:,:,:2]-core.OBSTACLE,axis=-1).min()-core.EXCLUSION_RADIUS),
        final_trajectory_cp=False,certificate=False)
    return report,dict(states=states,observations=observations,actions=actions,errors=errors,scores=scores)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['actor','observer','dino-repo','dino-weights','output-dir']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--episodes',type=int,default=24);p.add_argument('--seed',type=int,default=811)
    p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=4)
    args=p.parse_args(argv)
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True);torch.set_num_threads(args.threads)
    actor=load_actor(args.actor);observer=ImageObserver(args.observer,args.dino_repo,args.dino_weights,args.device)
    report,data=rollout(actor,observer,args.episodes,args.seed)
    report.update(actor_sha256=sha256(args.actor),observer_sha256=sha256(args.observer),seed=args.seed,device=args.device,threads=args.threads)
    np.savez_compressed(args.output_dir/'rollout.npz',**data)
    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
