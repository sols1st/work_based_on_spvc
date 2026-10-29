"""Train the original nonnegative SPVC MLP against a CP observation tube.

Explicit adaptation: stopped AEBS semantics, nonincrease for safety, positive
1e-4 training margin (default), unchanged B(init)<=1 / B(unsafe)>=20. This
stage freezes its supplied PPO and vision; actor refinement is a separate stage.
Tube robustness is conditional; marginal CP does not certify the image tail.
"""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
from Aebs.VT.utils import MLP
from Aebs.system.env import Aebs
from .deploy import VisionSLSController
from .conformal import envelope
from .cp_ncbf import action_bounds,successors
from .sbc_cp import barrier_bounds, successor_upper, NonnegativeSBC
from .audit_sbc_cp import region_grid
from .run import sha256


def region_samples(n,seed):
    rng=np.random.default_rng(seed)
    return tuple(torch.tensor(rng.uniform(lo,hi,size=(n,2)),dtype=torch.float32)
                 for lo,hi in [([15,2.5],[16,3]),([5,.5],[6,3]),([6.000001,.000001],[16,3])])


def prepare(policy,contract,states):
    r=envelope(states[:,0],contract['coefficients'],contract['calibration']['q_hat'])
    return successors(states,action_bounds(policy,states,r))


@torch.no_grad()
def check(model,scale,states,low,high,epsilon,region_size=48):
    divisor=torch.tensor([scale,1.])
    current=model(states/divisor).flatten()
    nhi=successor_upper(model,low/divisor,high/divisor)
    active=current<=20
    drift=nhi-current
    regions={}
    for name,lo,hi in [('initial',[15,2.5],[16,3]),('unsafe',[5,.5],[6,3])]:
        gl,gu=region_grid(lo,hi,region_size)
        lb,ub=barrier_bounds(model,gl/divisor,gu/divisor)
        regions[name]=dict(lower=float(lb.min()),upper=float(ub.max()))
    initial=regions['initial']['upper'];unsafe=regions['unsafe']['lower']
    return dict(regions=regions,active=int(active.sum()),positive=int((active&(drift>0)).sum()),
                training_band_positive=int(((current<=22)&(drift>0)).sum()),
                positive_training_margin=int((active&(drift+epsilon>0)).sum()),
                verification_condition='nonnegative supermartingale: B(next)<=B(now); positive epsilon is a training margin only',
                max_drift=float(drift[active].max()) if active.any() else None,
                region_pass=initial<=1 and unsafe>=20,
                sampled_tube_pass=initial<=1 and unsafe>=20 and not (active&(drift>0)).any().item(),
                formal_certificate=False),drift,active


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact-dir',type=Path,required=True)
    p.add_argument('--perception-contract',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=12000)
    p.add_argument('--seed',type=int,default=41)
    p.add_argument('--epsilon',type=float,default=.0001)
    p.add_argument('--learning-rate',type=float,default=.003)
    p.add_argument('--resume',type=Path)
    p.add_argument('--predecessors',type=Path,help='Development counterexamples with admissible unsafe trajectories')
    p.add_argument('--drift-weight',type=float,default=100.)
    p.add_argument('--preserve-regions',action='store_true',help='Reject updates that lose already verified region constraints')
    p.add_argument('--nonnegative-activation',choices=['softplus','relu'],default='softplus')
    p.add_argument('--counterexample-feedback',action='store_true')
    p.add_argument('--region-grid',type=int,default=48)
    p.add_argument('--drift-loss',choices=['squared','linear'],default='squared')
    p.add_argument('--extra-states',type=Path,help='Actual interval-verifier counterexamples, development only')
    args=p.parse_args(argv)
    if args.output_dir.exists():p.error('choose a new output directory')
    if args.epsilon<=0 or args.epochs<1:p.error('positive epsilon and training budget required')
    args.output_dir.mkdir(parents=True)
    torch.set_num_threads(1);torch.manual_seed(args.seed)
    contract=json.loads(args.perception_contract.read_text())
    source=args.artifact_dir
    for field,name in [('source_policy_sha256','policy.pt'),('source_observation_sha256','observation.pt')]:
        if contract[field]!=sha256(source/name):raise ValueError('model identity mismatch')
    controller=VisionSLSController(source);policy=controller.policy
    scale=float(Aebs(.05).std1);divisor=torch.tensor([scale,1.])
    initial,unsafe,states=region_samples(16384,args.seed)
    safe_progress=None
    if args.predecessors:
        predecessor=np.load(args.predecessors)
        states=torch.tensor(predecessor['states'],dtype=torch.float32)
        bad=torch.tensor(predecessor['unsafe_witness'])
        unsafe=torch.cat([unsafe,states[bad]])
        safe_progress=states[~bad & ~torch.tensor(predecessor['timeout'])]
    # Closed-region boundaries are essential: random draws almost never visit
    # the critical unsafe corner (6m, .5m/s).
    ilo,ihi=region_grid([15,2.5],[16,3],16)
    ulo,uhi=region_grid([5,.5],[6,3],16)
    # Opposite vertices alone omit the cross-corner (6m,.5m/s) that is
    # critical for AEBS. Include all four vertices of each box.
    initial_boundary=torch.cat([ilo,ihi,torch.stack([ilo[:,0],ihi[:,1]],1),torch.stack([ihi[:,0],ilo[:,1]],1)])/divisor
    unsafe_boundary=torch.cat([ulo,uhi,torch.stack([ulo[:,0],uhi[:,1]],1),torch.stack([uhi[:,0],ulo[:,1]],1)])/divisor
    low,high=prepare(policy,contract,states)
    _,_,evaluation=region_samples(8192,8801)
    if args.extra_states:
        extra=torch.tensor(np.load(args.extra_states)['states'],dtype=torch.float32).reshape(-1,2)
        evaluation=torch.cat([evaluation,extra])
    eval_low,eval_high=prepare(policy,contract,evaluation)
    model=NonnegativeSBC(nonnegative_activation=args.nonnegative_activation)
    if args.resume:model.load_state_dict(torch.load(args.resume,map_location='cpu',weights_only=True))
    optimizer=torch.optim.Adam(model.parameters(),lr=args.learning_rate)
    def regions_hold():
        with torch.no_grad():
            li,ui=region_grid([15,2.5],[16,3],args.region_grid);lu,uu=region_grid([5,.5],[6,3],args.region_grid)
            _,ib=barrier_bounds(model,li/divisor,ui/divisor)
            ub,_=barrier_bounds(model,lu/divisor,uu/divisor)
            return bool(ib.max()<=1 and ub.min()>=20)
    if args.preserve_regions and not regions_hold():
        raise ValueError('region-preserving training requires an initially passing checkpoint')
    accepted_weights=copy.deepcopy(model.state_dict());accepted_optimizer=copy.deepcopy(optimizer.state_dict())
    rejected_region_blocks=0
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(backend='original nonnegative VT.utils.MLP',source_policy_sha256=sha256(source/'policy.pt'),
                  source_observation_sha256=sha256(source/'observation.pt'),
                  barrier_distance_scale_m=scale,reach_probability=.95,
                  actor_changed=False,vision_changed=False,terminal='absorb at original safe/unsafe events; zero drift after stopping',
                  support_assumption='all visual observations inside fitted CP tube; not established by marginal CP',
                  mathematical_scope='stopped-process SBC adaptation, not literal original strict drift after termination')
    (args.output_dir/'config.json').write_text(json.dumps(config,indent=2))
    best=None;best_value=float('inf');history=[];start=time.monotonic()
    hard_indices=None
    for epoch in range(args.epochs):
        ids=torch.randint(len(states),(512,));rid=torch.randint(len(initial),(256,))
        uid=torch.randint(len(unsafe),(256,))
        bi=model(initial[rid]/divisor).flatten();bu=model(unsafe[uid]/divisor).flatten()
        # Region margins are stronger than the unchanged verification thresholds.
        boundary_initial=model(initial_boundary).flatten()
        boundary_unsafe=model(unsafe_boundary).flatten()
        init_target=0. if args.nonnegative_activation=='relu' else .4
        region_loss=(bi-init_target).square().mean()+torch.relu(25-bu).square().mean()
        region_loss=region_loss+torch.relu(boundary_initial-.6).square().max()+torch.relu(25-boundary_unsafe).square().max()
        if safe_progress is not None:
            examples=safe_progress[torch.randint(len(safe_progress),(512,))]
            progress_target=torch.zeros_like(examples[:,0]) if args.nonnegative_activation=='relu' else .1+.02*examples[:,0]
            region_loss=region_loss+(model(examples/divisor).flatten()-progress_target).square().mean()*.1
        current_states=states[ids];next_low=low[ids];next_high=high[ids]
        if hard_indices is not None:
            chosen=hard_indices[torch.randint(len(hard_indices),(256,))]
            current_states=torch.cat([current_states[:256],evaluation[chosen]])
            next_low=torch.cat([next_low[:256],eval_low[chosen]])
            next_high=torch.cat([next_high[:256],eval_high[chosen]])
        b=model(current_states/divisor).flatten()
        upper=successor_upper(model,next_low/divisor,next_high/divisor,subdivisions=8)
        active=(b.detach()<=22)
        violation=torch.relu(upper+args.epsilon-b)
        penalty=violation if args.drift_loss=='linear' else violation.square()
        drift_loss=(penalty[active].mean()+penalty[active].max()) if active.any() else b.sum()*0
        # Establish region separation first; then add the dynamics constraint.
        weight=0. if epoch<min(1000,args.epochs//5) and args.resume is None else args.drift_weight
        loss=region_loss+weight*drift_loss
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),20.);optimizer.step()
        if args.preserve_regions and (epoch+1)%50==0:
            if regions_hold():
                accepted_weights=copy.deepcopy(model.state_dict());accepted_optimizer=copy.deepcopy(optimizer.state_dict())
            else:
                model.load_state_dict(accepted_weights);optimizer.load_state_dict(accepted_optimizer)
                rejected_region_blocks+=1
        if (epoch+1)%500==0 or epoch+1==args.epochs:
            result,residual,checked=check(model,scale,evaluation,eval_low,eval_high,args.epsilon,args.region_grid)
            if args.counterexample_feedback:
                with torch.no_grad():training_band=model(evaluation/divisor).flatten()<=22
                indices=torch.where(training_band&(residual>0))[0]
                if len(indices):hard_indices=indices[residual[indices].argsort(descending=True)[:1024]]
                else:hard_indices=None
            result.update(epoch=epoch+1,loss=float(loss),seconds=time.monotonic()-start,
                          rejected_region_blocks=rejected_region_blocks,
                          feedback_samples=0 if hard_indices is None else len(hard_indices))
            history.append(result)
            region_penalty=max(result['regions']['initial']['upper']-1,0)+max(20-result['regions']['unsafe']['lower'],0)
            objective=region_penalty*100+max(result['max_drift'] or 0,0)
            if objective<best_value:best_value=objective;best=copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(),args.output_dir/'last_sbc.pt')
            torch.save(best,args.output_dir/'sbc.pt')
            (args.output_dir/'history.json').write_text(json.dumps(history,indent=2))
            print(json.dumps(result),flush=True)
            if result['sampled_tube_pass'] and result['training_band_positive']==0:break
    model.load_state_dict(best)
    result,residual,active=check(model,scale,evaluation,eval_low,eval_high,args.epsilon,args.region_grid)
    result.update(source_policy_unchanged=sha256(source/'policy.pt')==config['source_policy_sha256'],
                  vision_unchanged=sha256(source/'observation.pt')==config['source_observation_sha256'],
                  final_image_certificate=False,conditional_expectation_assumption_established=False)
    np.savez(args.output_dir/'development_checks.npz',states=evaluation.numpy(),residual=residual.numpy(),active=active.numpy())
    (args.output_dir/'report.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
