"""Train AEBS's neural NonnegativeSBC on the robot closed loop.

Loss sampling is development evidence only; run certify separately.
"""
import argparse,copy,json,time
from pathlib import Path
import numpy as np
import torch
from Aebs.lane_keeping.policy import load_actor
from Aebs.lane_keeping.learned_sbc import NonnegativeSBC,raw,raw_bounds,bounds,save_barrier,sha256,load_sbc_checkpoint
from . import core
from .bounds import grid_boxes,unsafe_boxes,error_boxes,transitions
from Aebs.vision_sls.neural_bounds import tanh_affine_bounds


@torch.no_grad()
def prepare(actor,states,radius):
    lows,highs=[],[]
    for el,eh in error_boxes(radius,2,states.dtype):
        pairs=[transitions(actor,x,x,el,eh) for x in states.split(512)]
        lows.append(torch.cat([p[0] for p in pairs]));highs.append(torch.cat([p[1] for p in pairs]))
    return torch.stack(lows,1),torch.stack(highs,1)


def successor(barrier,lo,hi,method='ibp'):
    if method=='affine' and torch.is_grad_enabled():
        layers=[]
        for layer in barrier.layers[:-1]:layers.extend([layer,torch.nn.Tanh()])
        layers.append(barrier.layers[-1])
        upper=tanh_affine_bounds(layers,lo.flatten(0,1),hi.flatten(0,1),detach_root=True)[1]
        return upper.reshape(lo.shape[:2]).amax(1)
    return raw_bounds(barrier,lo.flatten(0,1),hi.flatten(0,1),method)[1].reshape(lo.shape[:2]).amax(1)


@torch.no_grad()
def diagnostics(barrier,states,lows,highs,il,ih,ul,uh,active_level=20.):
    initial=bounds(barrier,il,ih)[1].max().item();unsafe=bounds(barrier,ul,uh)[0].min().item()
    drift=[];active=[]
    for x,lo,hi in zip(states.split(256),lows.split(256),highs.split(256)):
        b=barrier(x).flatten();drift.append(successor(barrier,lo,hi,'affine').relu()-b);active.append(b<=active_level)
    drift,active=torch.cat(drift),torch.cat(active)
    md=float(drift[active].max()) if active.any() else None
    return dict(initial_upper=initial,unsafe_lower=unsafe,positive=int((active&(drift>0)).sum()),
        max_drift=md,sampled_pass=bool(initial<=1 and unsafe>=20 and md is not None and md<=0),certificate=False),drift,active


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--actor',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--radius',type=float,default=.03);p.add_argument('--epochs',type=int,default=6000)
    p.add_argument('--radii',type=float,nargs=3,help='optional separate normalized x,y,heading halfwidths')
    p.add_argument('--seed',type=int,default=41);p.add_argument('--resume',type=Path)
    p.add_argument('--extra-states',type=Path);p.add_argument('--hidden',type=int,nargs='+',default=[32,32])
    p.add_argument('--active-level',type=float,default=20.)
    p.add_argument('--learning-rate',type=float,default=.003)
    p.add_argument('--bound-method',choices=['ibp','affine'],default='ibp')
    p.add_argument('--drift-weight',type=float,default=20.)
    args=p.parse_args(argv)
    if args.radii is not None:args.radius=np.array(args.radii)
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    if np.any(np.asarray(args.radius)<0) or not np.isfinite(args.radius).all():raise ValueError('invalid radius')
    if not 1<=args.active_level<=20:raise ValueError('active level must contain initial level 1 and exclude unsafe level 20')
    args.output_dir.mkdir(parents=True);torch.set_num_threads(4);torch.manual_seed(args.seed)
    rng=np.random.default_rng(args.seed);actor=load_actor(args.actor).requires_grad_(False)
    if args.resume:barrier,_=load_sbc_checkpoint(args.resume)
    else:
        barrier=NonnegativeSBC([3,*args.hidden,1],nonnegative_activation='relu')
        with torch.no_grad():
            barrier.layers[-1].weight.mul_(20);barrier.layers[-1].bias.mul_(20)
    with torch.no_grad():
        x=torch.tensor(core.normalize(core.sample_initial(rng,192)),dtype=torch.float32);rollout=[x]
        for _ in range(core.HORIZON):
            e=torch.tensor(rng.uniform(-args.radius,args.radius,x.shape),dtype=x.dtype)
            x=core.torch_dynamics(x,actor(x+e));rollout.append(x)
        rollout=torch.cat(rollout)
    samples=rng.uniform(core.LOW,core.HIGH,(16000,3));samples=samples[~core.unsafe(samples)]
    states=torch.cat([torch.tensor(core.normalize(samples),dtype=torch.float32),rollout[::4]])
    if args.extra_states:
        states=torch.cat([states,torch.tensor(np.load(args.extra_states)['states'],dtype=torch.float32).reshape(-1,3)])
    il,ih=grid_boxes(core.normalize(core.INITIAL_LOW),core.normalize(core.INITIAL_HIGH),4,torch.float32)
    ul,uh=unsafe_boxes(torch.float32);initial=torch.cat([il,ih]);unsafe=torch.cat([ul,uh,(ul+uh)/2])
    print(json.dumps(dict(stage='prepare',states=len(states),unsafe_boxes=len(ul))),flush=True)
    lows,highs=prepare(actor,states,args.radius)
    config={k:str(v) if isinstance(v,Path) else v.tolist() if isinstance(v,np.ndarray) else v for k,v in vars(args).items()}
    config.update(task=core.config(),actor_sha256=sha256(args.actor),model_class='Aebs.vision_sls.sbc_cp.NonnegativeSBC',
                  geometric_teacher=False,actor_updated=False,thresholds=dict(initial=1,unsafe=20,drift=0))
    (args.output_dir/'config.json').write_text(json.dumps(config,indent=2))
    optimizer=torch.optim.Adam(barrier.parameters(),lr=args.learning_rate)
    baseline,d0,a0=diagnostics(barrier,states,lows,highs,il,ih,ul,uh,args.active_level)
    best_value=100*(max(baseline['initial_upper']-1,0)+max(20-baseline['unsafe_lower'],0))+max(baseline['max_drift'] or 0,0)
    best=copy.deepcopy(barrier.state_dict());baseline.update(epoch=0,source='initial or resumed candidate')
    print(json.dumps(baseline),flush=True)
    hard=torch.where(a0&(d0>0))[0]
    hard=hard[d0[hard].argsort(descending=True)[:2048]] if len(hard) and args.resume else None
    history=[baseline];start=time.monotonic()
    for epoch in range(args.epochs):
        ids=torch.randint(len(states),(512,))
        if hard is not None:ids=torch.cat([ids[:256],hard[torch.randint(len(hard),(256,))]])
        bi=raw(barrier,initial[torch.randint(len(initial),(256,))])/20
        bu=raw(barrier,unsafe[torch.randint(len(unsafe),(512,))])/20
        safe=raw(barrier,rollout[torch.randint(len(rollout),(256,))])/20
        iloss=(bi+.05).relu().square();uloss=(1.25-bu).relu().square()
        region=iloss.mean()+iloss.max()+uloss.mean()+uloss.max()+.1*(safe+.05).relu().square().mean()
        current=raw(barrier,states[ids])/20;nh=successor(barrier,lows[ids],highs[ids],args.bound_method)/20
        active=current.detach()<=args.active_level*1.1/20;drift=(nh-current.clamp_min(-.025)+.0005).relu()
        dyn=drift[active].square().mean()+drift[active].square().max() if active.any() else current.sum()*0
        warm=0 if args.resume else min(1000,args.epochs//4)
        loss=region+(args.drift_weight if epoch>=warm else 0)*dyn
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss: stop rather than publish a broken checkpoint')
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(barrier.parameters(),10,error_if_nonfinite=True);optimizer.step()
        if (epoch+1)%250==0 or epoch+1==args.epochs:
            stat,d,a=diagnostics(barrier,states,lows,highs,il,ih,ul,uh,args.active_level)
            hard=torch.where(a&(d>0))[0];hard=hard[d[hard].argsort(descending=True)[:2048]] if len(hard) else None
            score=100*(max(stat['initial_upper']-1,0)+max(20-stat['unsafe_lower'],0))+max(stat['max_drift'] or 0,0)
            if score<best_value:best_value=score;best=copy.deepcopy(barrier.state_dict())
            stat.update(epoch=epoch+1,loss=float(loss.detach()),seconds=time.monotonic()-start)
            history.append(stat);print(json.dumps(stat),flush=True)
            save_barrier(args.output_dir/'last_sbc.pt',barrier,config)
            current_weights=copy.deepcopy(barrier.state_dict());barrier.load_state_dict(best)
            save_barrier(args.output_dir/'sbc.pt',barrier,config);barrier.load_state_dict(current_weights)
            (args.output_dir/'history.json').write_text(json.dumps(history,indent=2))
            if stat['sampled_pass'] and epoch>=warm:break
    barrier.load_state_dict(best);stat,d,a=diagnostics(barrier,states,lows,highs,il,ih,ul,uh,args.active_level)
    (args.output_dir/'report.json').write_text(json.dumps(stat,indent=2))
    np.savez_compressed(args.output_dir/'development_checks.npz',states=states.numpy(),drift=d.numpy(),active=a.numpy())


if __name__=='__main__':main()
