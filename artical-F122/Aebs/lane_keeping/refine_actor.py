"""AEBS-style SBC-feedback actor refinement, with task retention and rollback.

Frozen learned SBC; update the PPO actor itself, no online action filter.
This stage produces a candidate only. Any changed actor requires re-verification.
"""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch

from . import core
from .policy import load_actor
from .learned_sbc import (load_sbc_checkpoint, save_actor, sha256, raw, bounds,
                         error_boxes, transitions, task_panel, retention_failures)


def feedback_loss(actor, original, barrier, states, kappas, radius, splits=2):
    current=barrier(states).flatten().detach()
    active=current<=20
    next_upper=torch.zeros_like(current)
    for el,eh in error_boxes(radius,splits,states.dtype):
        lo,hi=transitions(actor,states,states,kappas,el,eh)
        next_upper=torch.maximum(next_upper,bounds(barrier,lo,hi,'affine')[1])
    drift=torch.relu(next_upper-current)
    safety=(drift[active].mean()+drift[active].max())/20 if active.any() else current.sum()*0
    obs=torch.cat([states,(kappas/core.KAPPA_SCALE)[:,None]],1)
    imitation=(actor(obs)-original(obs).detach()).square().mean()
    return safety+.02*imitation,safety


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--actor',type=Path,required=True)
    p.add_argument('--sbc',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--feedback',type=Path)
    p.add_argument('--radius',type=float,default=.08)
    p.add_argument('--rounds',type=int,default=8)
    p.add_argument('--steps-per-round',type=int,default=100)
    p.add_argument('--learning-rate',type=float,default=3e-5)
    p.add_argument('--seed',type=int,default=61)
    args=p.parse_args(argv)
    if (args.rounds<1 or args.steps_per_round<1 or not np.isfinite(args.radius)
            or args.radius<0 or args.learning_rate<=0):
        raise ValueError('invalid refinement settings')
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True);torch.set_num_threads(4);torch.manual_seed(args.seed)
    rng=np.random.default_rng(args.seed)
    actor=load_actor(args.actor)
    original=copy.deepcopy(actor).requires_grad_(False)
    barrier,_=load_sbc_checkpoint(args.sbc);barrier.requires_grad_(False)
    states=torch.tensor(rng.uniform(-1,1,(8192,2)),dtype=torch.float32)
    kappas=torch.tensor(rng.choice(core.KAPPAS,len(states)),dtype=torch.float32)
    if args.feedback:
        data=np.load(args.feedback)
        points=data['states']
        if len(points):
            extra=torch.tensor(points,dtype=torch.float32)
            states=torch.cat([states,extra.repeat(len(core.KAPPAS),1)])
            kappas=torch.cat([kappas,torch.tensor(core.KAPPAS,dtype=torch.float32).repeat_interleave(len(extra))])
    with torch.no_grad():
        active=barrier(states).flatten()<=20
        states,kappas=states[active],kappas[active]
    if not len(states):raise ValueError('no active states; train region separation first')
    baseline=task_panel(original,args.radius)
    optimizer=torch.optim.Adam(actor.parameters(),lr=args.learning_rate)
    history=[];accepted=0
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(source_actor_sha256=sha256(args.actor),source_sbc_sha256=sha256(args.sbc),
                  online_filter=False,verification_required_after_actor_change=True,
                  retention_panel='development only; nominal/uniform/alternating, 192 episodes each')
    (args.output_dir/'config.json').write_text(json.dumps(config,indent=2))
    for round_id in range(args.rounds):
        weights=copy.deepcopy(actor.state_dict());moments=copy.deepcopy(optimizer.state_dict())
        for step in range(args.steps_per_round):
            ids=torch.randint(len(states),(128,))
            x,k=states[ids],kappas[ids]
            loss,safety=feedback_loss(actor,original,barrier,x,k,args.radius)
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(actor.parameters(),1.)
            optimizer.step()
        candidate=copy.deepcopy(actor.state_dict())
        panel=task_panel(actor,args.radius);reasons=retention_failures(baseline,panel)
        fraction=1.
        if reasons:
            for fraction in [.5,.25,.125,.0625]:
                actor.load_state_dict({k:weights[k]+fraction*(candidate[k]-weights[k]) for k in weights})
                panel=task_panel(actor,args.radius);reasons=retention_failures(baseline,panel)
                if not reasons:break
            optimizer.load_state_dict(moments)
        if reasons:
            actor.load_state_dict(weights);optimizer.load_state_dict(moments);fraction=0.
        else:accepted+=1
        entry=dict(round=round_id+1,accepted=not reasons,fraction=fraction,
                   retention_failures=reasons,sampled_safety_loss=float(safety),panel=panel)
        history.append(entry);print(json.dumps(entry),flush=True)
        save_actor(args.output_dir/'actor.pt',actor)
        (args.output_dir/'history.json').write_text(json.dumps(history,indent=2))
    changed=any(not torch.equal(v,original.state_dict()[k]) for k,v in actor.state_dict().items())
    report=dict(accepted_rounds=accepted,weights_changed=changed,baseline=baseline,
                final_panel=task_panel(actor,args.radius),certificate=False)
    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
