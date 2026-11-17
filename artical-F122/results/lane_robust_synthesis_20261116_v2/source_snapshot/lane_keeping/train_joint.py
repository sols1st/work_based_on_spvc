"""AEBS-style learned SBC stage for lane keeping, with interval-error loss.

Uses AEBS NonnegativeSBC directly. No geometric certificate, teacher, LQR,
or frozen certificate branch. This stage freezes the actor; refine_actor is
the alternating controller stage. Sampling losses are not safety proofs.
"""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch

from . import core
from .policy import load_actor
from .sbc import next_states_for_errors
from .learned_sbc import (NonnegativeSBC, load_sbc_checkpoint, save_barrier, sha256,
                         raw, raw_bounds, bounds, grid_boxes, unsafe_boxes,
                         error_boxes, transitions)


@torch.no_grad()
def prepare(actor, states, kappas, radius, splits):
    lows, highs = [], []
    for el, eh in error_boxes(radius, splits, states.dtype):
        out = [transitions(actor, x, x, k, el, eh) for x,k in
               zip(states.split(1024), kappas.split(1024))]
        lows.append(torch.cat([p[0] for p in out]))
        highs.append(torch.cat([p[1] for p in out]))
    return torch.stack(lows, 1), torch.stack(highs, 1)


def successor_raw(barrier, lo, hi, method='ibp'):
    return raw_bounds(barrier,lo.flatten(0,1),hi.flatten(0,1),method)[1].reshape(lo.shape[:2]).amax(1)


@torch.no_grad()
def diagnostics(barrier, states, lows, highs, il, ih, ul, uh):
    initial = bounds(barrier,il,ih)[1].max().item()
    unsafe = bounds(barrier,ul,uh)[0].min().item()
    drifts, actives = [], []
    for x,lo,hi in zip(states.split(512),lows.split(512),highs.split(512)):
        b = barrier(x).flatten()
        upper = successor_raw(barrier,lo,hi,'affine').relu()
        drifts.append(upper-b); actives.append(b<=20)
    drift, active = torch.cat(drifts), torch.cat(actives)
    positive = active & (drift>0)
    return dict(initial_upper=initial,unsafe_lower=unsafe,positive=int(positive.sum()),
                max_drift=float(drift[active].max()) if active.any() else None,
                sampled_pass=bool(initial<=1 and unsafe>=20 and not positive.any()),
                certificate=False), drift, active


@torch.no_grad()
def collect_rollout(actor, radius, seed):
    rng = np.random.default_rng(seed)
    x = torch.tensor(core.sample_initial(rng,256)/[core.E_SCALE,core.THETA_SCALE],dtype=torch.float32)
    x[:4] = torch.tensor([[e,t] for e in [-core.INIT_E/core.E_SCALE,core.INIT_E/core.E_SCALE]
                         for t in [-core.INIT_THETA/core.THETA_SCALE,core.INIT_THETA/core.THETA_SCALE]])
    records = [x]
    for step in range(core.HORIZON):
        k = torch.tensor(rng.choice([-1.,0.,1.],len(x)),dtype=x.dtype)
        err = torch.tensor(rng.uniform(-radius,radius,(len(x),1,2)),dtype=x.dtype)
        x = next_states_for_errors(actor,x,k,err)[:,0]
        records.append(x)
    return torch.cat(records)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--actor',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--resume',type=Path)
    p.add_argument('--extra-states',type=Path)
    p.add_argument('--epochs',type=int,default=6000)
    p.add_argument('--seed',type=int,default=41)
    p.add_argument('--radius',type=float,default=.08)
    p.add_argument('--error-splits',type=int,default=2)
    p.add_argument('--learning-rate',type=float,default=.003)
    p.add_argument('--hidden',type=int,nargs='+',default=[16,8])
    p.add_argument('--drift-weight',type=float,default=20.)
    p.add_argument('--bound-method',choices=['ibp','affine'],default='ibp')
    args=p.parse_args(argv)
    if (args.epochs<1 or args.error_splits<1 or not np.isfinite(args.radius)
            or args.radius<0 or args.learning_rate<=0 or any(n<1 for n in args.hidden)):
        raise ValueError('invalid training settings')
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    torch.set_num_threads(4);torch.manual_seed(args.seed)
    rng=np.random.default_rng(args.seed)
    actor=load_actor(args.actor).requires_grad_(False)
    if args.resume: barrier,_=load_sbc_checkpoint(args.resume)
    else:
        barrier=NonnegativeSBC([2,*args.hidden,1],nonnegative_activation='relu')
        # Numerical output scaling only, unrelated to any certificate geometry.
        with torch.no_grad():
            barrier.layers[-1].weight.mul_(20.)
            barrier.layers[-1].bias.mul_(20.)
    optimizer=torch.optim.Adam(barrier.parameters(),lr=args.learning_rate)
    rollout=collect_rollout(actor,args.radius,args.seed+1)
    states=torch.tensor(rng.uniform(-1,1,(12288,2)),dtype=torch.float32)
    states=torch.cat([states,rollout[::5]])
    kappas=torch.tensor(rng.choice(core.KAPPAS,len(states)),dtype=torch.float32)
    if args.extra_states:
        extra=torch.tensor(np.load(args.extra_states)['states'],dtype=torch.float32).reshape(-1,2)
        # Cover each feedback state at ALL registered curvatures rather than
        # randomly replacing the curvature that produced a counterexample.
        states=torch.cat([states,extra.repeat(len(core.KAPPAS),1)])
        kappas=torch.cat([kappas,torch.tensor(core.KAPPAS,dtype=torch.float32).repeat_interleave(len(extra))])
    lows,highs=prepare(actor,states,kappas,args.radius,args.error_splits)
    il,ih=grid_boxes([-core.INIT_E/core.E_SCALE,-core.INIT_THETA/core.THETA_SCALE],
                    [core.INIT_E/core.E_SCALE,core.INIT_THETA/core.THETA_SCALE],12,torch.float32)
    ul,uh=unsafe_boxes(16,torch.float32)
    initial=torch.cat([il,ih,torch.stack([il[:,0],ih[:,1]],1),torch.stack([ih[:,0],il[:,1]],1)])
    unsafe=torch.cat([ul,uh,torch.stack([ul[:,0],uh[:,1]],1),torch.stack([uh[:,0],ul[:,1]],1)])
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(source_actor_sha256=sha256(args.actor),model_class='Aebs.vision_sls.sbc_cp.NonnegativeSBC',
                  features=barrier.features,geometric_teacher=False,actor_updated=False,
                  verification_thresholds=dict(initial=1,unsafe=20,drift=0),
                  training_only_raw_margin=True,seed_role='development; not independent final testing')
    (args.output_dir/'config.json').write_text(json.dumps(config,indent=2))
    best_value=float('inf');best=copy.deepcopy(barrier.state_dict())
    hard=None;history=[];start=time.monotonic()
    for epoch in range(args.epochs):
        ids=torch.randint(len(states),(512,))
        if hard is not None:
            ids=torch.cat([ids[:256],hard[torch.randint(len(hard),(256,))]])
        ri=torch.randint(len(initial),(256,));ui=torch.randint(len(unsafe),(512,))
        bi=raw(barrier,initial[ri])/20;bu=raw(barrier,unsafe[ui])/20
        init_loss=torch.relu(bi+.05).square()
        unsafe_loss=torch.relu(1.25-bu).square()
        safe=raw(barrier,rollout[torch.randint(len(rollout),(256,))])/20
        region=init_loss.mean()+init_loss.max()+unsafe_loss.mean()+unsafe_loss.max()
        region=region+.1*torch.relu(safe+.05).square().mean()
        current=raw(barrier,states[ids])/20
        next_hi=successor_raw(barrier,lows[ids],highs[ids],args.bound_method)/20
        active=current.detach()<=1.1
        # In a ReLU zero region require a small NEGATIVE raw head margin.
        # Actual verifier still checks B(next)<=B(now), without a tolerance.
        drift=torch.relu(next_hi-current.clamp_min(-.025)+.0005)
        dyn=(drift[active].square().mean()+drift[active].square().max()) if active.any() else current.sum()*0
        warm=0 if args.resume else min(1000,args.epochs//4)
        weight=0. if epoch<warm else args.drift_weight
        loss=region+weight*dyn
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(barrier.parameters(),10.)
        optimizer.step()
        if (epoch+1)%250==0 or epoch+1==args.epochs:
            stat,drift,active=diagnostics(barrier,states,lows,highs,il,ih,ul,uh)
            hard=torch.where(active&(drift>0))[0]
            hard=hard[drift[hard].argsort(descending=True)[:2048]] if len(hard) else None
            score=100*(max(stat['initial_upper']-1,0)+max(20-stat['unsafe_lower'],0))+max(stat['max_drift'] or 0,0)
            if score<best_value:best_value=score;best=copy.deepcopy(barrier.state_dict())
            stat.update(epoch=epoch+1,loss=float(loss),seconds=time.monotonic()-start)
            history.append(stat);print(json.dumps(stat),flush=True)
            save_barrier(args.output_dir/'last_sbc.pt',barrier,config)
            current_weights=copy.deepcopy(barrier.state_dict())
            barrier.load_state_dict(best);save_barrier(args.output_dir/'sbc.pt',barrier,config)
            barrier.load_state_dict(current_weights)
            (args.output_dir/'history.json').write_text(json.dumps(history,indent=2))
            if stat['sampled_pass'] and epoch>=warm:break
    barrier.load_state_dict(best)
    stat,drift,active=diagnostics(barrier,states,lows,highs,il,ih,ul,uh)
    (args.output_dir/'report.json').write_text(json.dumps(stat,indent=2))
    np.savez_compressed(args.output_dir/'development_checks.npz',states=states.numpy(),
                        kappas=kappas.numpy(),drift=drift.numpy(),active=active.numpy())


if __name__=='__main__':main()
