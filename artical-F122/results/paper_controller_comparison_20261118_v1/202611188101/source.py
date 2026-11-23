"""Matched controlled controller training; not native cGAN/SPVC/SPVT retraining.

All three arms share an already robust-trained anchor, fixed B and semantic
support. No new camera CP is claimed after changing a policy. Safety objectives
are the ONLY arm-dependent training loss; optimization batches and anchoring
are shared. A common hard-state curriculum is refreshed from the SPVT arm's
development reachability loss so the reachability baseline is not deprived of
its verification-guided training mechanism.
"""
import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import time
import numpy as np

from .tasks import RESULTS,load_task

ARMS=('spvt','spvc','robust')
SEEDS=(202611188101,202611188102,202611188103)
LR=3e-5
UPDATES=240
BATCH=64
CURRICULUM=(4,6,8,10)
TEST_SEED=202611188901


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False))


def successor(x,u,k):
    import torch
    return torch.stack([x[:,0]+.4/.85*torch.sin(.30*x[:,1]),
                        x[:,1]+.4/.30*(torch.tan(.45*u)/2.6-k)],1)


def act(actor,x,k,e=None):
    import torch
    return actor(torch.cat([x if e is None else x+e,(k/.01)[:,None]],1)).flatten()


def spvt_loss(actor,x,k,q,horizon):
    """Eq.(5) style terminal reachability width/location, two safety axes.

    Differentiable full-support affine actor bounds with exact monotone bicycle
    dynamics. Throughout H<=10, even full |theta|<=.3 starts stay within the
    sin-monotone interval under clipped steering, independent of training.
    """
    from Aebs.lane_keeping.learned_sbc import transitions
    lo=hi=x
    for _ in range(horizon):
        lo,hi=transitions(actor,lo,hi,k,x.new_full((2,),-q),x.new_full((2,),q),'affine',detach_root=True)
    loss=(lo.abs()+hi.abs()).mean(1)/(horizon-1)
    priority=loss+(hi.abs()-1).relu().sum(1)+(lo.abs()-1).relu().sum(1)
    return loss,priority


def safety_loss(arm,actor,barrier,x,k,q,horizon):
    import torch
    from Aebs.lane_keeping.learned_sbc import error_boxes,transitions,bounds
    if arm=='spvt':return spvt_loss(actor,x,k,q,horizon)[0].mean()
    current=barrier(x).flatten().detach()
    if arm=='spvc':
        nodes,weights=np.polynomial.legendre.leggauss(5)
        errors=x.new_tensor(list(itertools.product(nodes*q,repeat=2)))
        probabilities=x.new_tensor(np.outer(weights,weights).ravel()/4)
        xx=x[:,None,:].expand(-1,25,-1).reshape(-1,2)
        kk=k[:,None].expand(-1,25).reshape(-1)
        ee=errors[None].expand(len(x),-1,-1).reshape(-1,2)
        nxt=successor(xx,act(actor,xx,kk,ee),kk)
        following=(barrier(nxt).reshape(len(x),25)*probabilities).sum(1)
    else:
        following=torch.zeros_like(current)
        for el,eh in error_boxes(q,2,x.dtype):
            lo,hi=transitions(actor,x,x,k,el,eh,'affine',detach_root=True)
            upper=bounds(barrier,lo,hi,'affine',detach_root=True)[1]
            following=torch.maximum(following,upper)
    residual=(following-current).relu()/20
    return residual.mean()+residual.max()


def common_performance(actor,anchor,x,k):
    import torch
    # Identical nominal short-horizon task term avoids an all-zero safety loss
    # freezing the two B-based arms at their already verified warm start.
    with torch.no_grad():target=act(anchor,x,k)
    anchor_loss=(act(actor,x,k)-target).square().mean()
    state=x;cost=x.new_zeros(len(x))
    for _ in range(4):
        u=act(actor,state,k);state=successor(state,u,k)
        cost+=state[:,0].square()+.2*state[:,1].square()+.005*u.square()
    return .25*anchor_loss+.05*(cost/4).mean(),anchor_loss,(cost/4).mean()


def train(root,seed,updates=UPDATES):
    import torch
    from Aebs.lane_keeping.policy import load_actor
    from Aebs.lane_keeping.learned_sbc import load_sbc_checkpoint,save_actor
    torch.set_num_threads(1);torch.manual_seed(seed)
    out=root/str(seed);out.mkdir(parents=True,exist_ok=False)
    task=load_task('lane');q=float(task.error_hi[0])
    actor_path=Path(task.metadata['actor']['path']);b_path=Path(task.metadata['barrier']['path'])
    anchor=load_actor(actor_path).double().requires_grad_(False)
    barrier=load_sbc_checkpoint(b_path)[0].double().requires_grad_(False)
    actors={arm:copy.deepcopy(anchor).requires_grad_(True) for arm in ARMS}
    opts={arm:torch.optim.Adam(actors[arm].parameters(),lr=LR) for arm in ARMS}
    protocol=dict(type='controlled training objectives; no original image generator retraining',arms=list(ARMS),seed=seed,
       common_initialization='same frozen robust-v2 actor, which already contains prior robust training',
       anchor_sha256=digest(actor_path),barrier_sha256=digest(b_path),adapter_sha256=digest(Path(__file__).with_name('tasks.py')),
       code_sha256=digest(__file__),capacity=sum(p.numel() for p in anchor.parameters()),updates=updates,learning_rate=LR,batch_size=BATCH,
       optimizer='Adam; clip gradient norm 1; final iterate, no test-based selection',dtype='float64',threads=1,
       horizon_curriculum=list(CURRICULUM),anchor_weight=.25,common_four_step_tracking_weight=.05,safety_weight=1.,
       shared_data='same 4096 initial and 4096 active B<20 development states, same minibatch indices in every arm',
       shared_hard_curriculum='top4000 by SPVT current development reachability priority, refreshed each60updates; half of every shared batch from this buffer',
       spvt='Eq5-style terminal abs lower+abs upper, extended to both normalized safety coordinates; H4/6/8/10; differentiable affine bounds',
       spvc='uniform semantic residual support, 5x5 Gauss-Legendre expected B next quadrature; mean+max positive drift/20',
       robust='four full-coverage semantic-error subboxes, affine upper B next; mean+max positive drift/20',
       support_halfwidth=q,cp_scope='original radius reused only as fixed support, not a CP guarantee for changed actors',
       testing='independent synthetic 2000 initial draws with same scenario seeds; not new images',
       equal_steps_not_equal_compute=True)
    save(out/'protocol.json',protocol);shutil.copyfile(__file__,out/'source.py')
    rng=np.random.default_rng(seed)
    initial=rng.uniform(task.initial_lo,task.initial_hi,(4096,2))
    candidate=rng.uniform(-1,1,(32768,2))
    with torch.no_grad():active=barrier(torch.tensor(candidate,dtype=torch.float64)).flatten().numpy()<20
    active_states=candidate[active][:4096]
    if len(active_states)!=4096:raise ValueError('insufficient common active states')
    pool=torch.tensor(np.r_[initial,active_states],dtype=torch.float64)
    curves=torch.tensor(rng.choice(task.contexts,len(pool)),dtype=torch.float64)
    np.savez_compressed(out/'development.npz',states=pool.numpy(),kappas=curves.numpy())
    history=[];seconds={arm:0. for arm in ARMS};begin=time.monotonic();hard=np.arange(len(pool))
    for step in range(updates):
        phase=min(3,step//max(1,updates//4));horizon=CURRICULUM[phase]
        if step%max(1,updates//4)==0:
            priority=[];at=time.monotonic()
            with torch.no_grad():
                for first in range(0,len(pool),256):
                    _,pr=spvt_loss(actors['spvt'],pool[first:first+256],curves[first:first+256],q,horizon)
                    priority.append(pr.numpy())
            hard=np.argsort(np.concatenate(priority))[-4000:]
            save(out/f'hard_phase_{phase}.json',dict(horizon=horizon,source='current SPVT development priorities shared with all arms',
                  indices=hard.tolist(),seconds=time.monotonic()-at))
        ids=np.r_[rng.integers(len(pool),size=BATCH//2),rng.choice(hard,size=BATCH//2)]
        x,k=pool[ids],curves[ids]
        record=dict(update=step+1,horizon=horizon,batch_indices=ids.tolist())
        for arm in ARMS:
            at=time.monotonic();actor=actors[arm]
            safe=safety_loss(arm,actor,barrier,x,k,q,horizon)
            common,anchor_loss,tracking=common_performance(actor,anchor,x,k)
            loss=safe+common
            if not torch.isfinite(loss):raise FloatingPointError(f'{arm} nonfinite loss')
            opts[arm].zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(actor.parameters(),1.,error_if_nonfinite=True);opts[arm].step()
            seconds[arm]+=time.monotonic()-at
            record[arm]=dict(loss=float(loss.detach()),safety=float(safe.detach()),anchor=float(anchor_loss.detach()),tracking=float(tracking.detach()))
        history.append(record)
        if (step+1)%20==0:print(json.dumps(dict(seed=seed,update=step+1,horizon=horizon,elapsed=time.monotonic()-begin,
                                               losses={arm:record[arm]['loss'] for arm in ARMS})),flush=True)
    for arm in ARMS:
        target=out/arm;target.mkdir()
        save_actor(target/'actor.pt',actors[arm])
        shadow=target/'shadow/results/lane_robust_synthesis_20261116_v2'
        (shadow/'package').mkdir(parents=True);(shadow/'final_cp').mkdir()
        shutil.copyfile(target/'actor.pt',shadow/'package/actor.pt');shutil.copyfile(b_path,shadow/'package/sbc.pt')
        shutil.copyfile(task.metadata['contract']['path'],shadow/'final_cp/calibration.json')
        change=sum(float((p-r).square().sum()) for p,r in zip(actors[arm].parameters(),anchor.parameters()))**.5
        save(target/'frozen.json',dict(seed=seed,arm=arm,actor_sha256=digest(target/'actor.pt'),common_barrier_sha256=digest(b_path),
            actor_weight_l2_change=change,updates=updates,training_seconds_excluding_shared_curriculum=seconds[arm],
            total_shared_run_seconds=time.monotonic()-begin,protocol_sha256=digest(out/'protocol.json'),
            cp_recalibrated=False,guarantee='conditional residual support only'))
    save(out/'training_history.json',history)
    print(json.dumps(dict(seed=seed,stage='frozen',seconds=time.monotonic()-begin,arms=list(ARMS))),flush=True)


def synthetic_metrics(task,n=2000,seed=TEST_SEED):
    rng=np.random.default_rng(seed);initial=task.sample_initial(n,seed);k=rng.choice(task.contexts,n)
    errors=dict(iid=rng.uniform(task.error_lo,task.error_hi,(n,80,2)))
    errors['persistent']=np.repeat(rng.uniform(task.error_lo,task.error_hi,(n,1,2)),80,axis=1)
    signs=rng.choice([-1.,1.],(n,1,2));errors['alternating']=signs*task.error_hi*((-1.)**np.arange(80))[None,:,None]
    from Aebs.lane_keeping.policy import load_actor
    import torch
    actor=load_actor(task.metadata['actor']['path']).double().requires_grad_(False)
    rows={};summaries={}
    for mode,disturbance in errors.items():
        x=initial.copy();history=[x];actions=[]
        for t in range(80):
            with torch.no_grad():u=act(actor,torch.tensor(x),torch.tensor(k),torch.tensor(disturbance[:,t])).numpy()
            following=np.empty_like(x)
            for context in task.contexts:
                ids=np.isclose(k,context);following[ids]=task.step(x[ids],disturbance[ids,t],context=context)
            x=following;history.append(x);actions.append(u*.45)
        states=np.stack(history,axis=1);actions=np.stack(actions,axis=1)
        unsafe=(np.abs(states)>1).any(axis=(1,2));e=np.abs(states[:,:,0])*.85;theta=np.abs(states[:,:,1])*.30
        rows[mode]=dict(unsafe=unsafe,mean_abs_e_m=e.mean(1),peak_abs_e_m=e.max(1),mean_abs_heading_rad=theta.mean(1),
                        mean_steering_change_rad=np.abs(np.diff(actions,axis=1)).mean(1))
        summaries[mode]=dict(n=n,safe=int((~unsafe).sum()),unsafe=int(unsafe.sum()),
            **{key:float(value.mean()) for key,value in rows[mode].items() if key!='unsafe'})
    return summaries,rows


def evaluate(root,seed,arm,budget):
    import torch
    from .spvt_reachability import evaluate_task,ReachabilityConfig
    torch.set_num_threads(1);target=root/str(seed)/arm
    task=load_task('lane',root=target/'shadow')
    task.metadata.update(changed_controller=True,cp_recalibrated=False,frozen_model=digest(target/'actor.pt'))
    fresh=task.current_verify(budget);fresh['metadata']=task.metadata
    save(target/'current_conditional.json',fresh)
    print(json.dumps(dict(seed=seed,arm=arm,current=fresh.get('status','pass' if fresh.get('passed') else 'unknown'))),flush=True)
    metrics,rows=synthetic_metrics(task)
    save(target/'synthetic_performance.json',dict(seed=TEST_SEED,initial_n=2000,metrics=metrics,
         distribution='synthetic residuals, not actual rendered images',cp_guarantee=False))
    np.savez_compressed(target/'synthetic_per_trajectory.npz',**{f'{mode}__{key}':value for mode,row in rows.items() for key,value in row.items()})
    cfg=ReachabilityConfig(timeout_seconds=1.,max_frontier=256,max_depth=16,falsification_paths=8,seed=TEST_SEED)
    reach=evaluate_task(task,n=2000,seed=TEST_SEED,horizons=[4,10,20,40,80],config=cfg,total_timeout_seconds=budget,screening_pass=True)
    save(target/'spvt_conditional.json',reach)
    print(json.dumps(dict(seed=seed,arm=arm,stage='evaluation_complete',summary=reach['contexts'][0]['summary'])),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['train','evaluate'])
    p.add_argument('--root',type=Path,default=RESULTS/'paper_controller_comparison_20261118_v1')
    p.add_argument('--seed',type=int,default=SEEDS[0]);p.add_argument('--arm',choices=ARMS)
    p.add_argument('--updates',type=int,default=UPDATES);p.add_argument('--budget',type=float,default=300.)
    args=p.parse_args()
    if args.stage=='train':train(args.root,args.seed,args.updates)
    else:
        if not args.arm:p.error('--arm required')
        evaluate(args.root,args.seed,args.arm,args.budget)


if __name__=='__main__':main()
