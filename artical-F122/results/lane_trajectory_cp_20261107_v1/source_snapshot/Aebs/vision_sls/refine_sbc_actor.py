"""Original SPVC actor update driven by a fixed nonnegative SBC.

No deployment filter. Train the PPO network itself, protect existing task
success, and keep observation/plant uncertainty in the SBC loss.
"""
import argparse
import copy
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .deploy import VisionSLSController
from .sbc_cp import NonnegativeSBC,successor_upper
from .cp_ncbf import sequential_bounds,successors
from .conformal import envelope
from .train_sbc_cp import region_samples,prepare,check
from .training import TrainingTaskPanel,retention_failures
from .run import sha256


def differentiable_actions(policy,states,radii,subdivisions=16):
    actor=nn.Sequential(*policy.controller_net.mlp_extractor,policy.controller_net.action_net)
    edges=states[:,0,None]+radii[:,None]*torch.linspace(-1,1,subdivisions+1)
    lo=torch.stack([edges[:,:-1]/policy.distance_scale_m,states[:,1,None].expand(-1,subdivisions)],-1).reshape(-1,2)
    hi=torch.stack([edges[:,1:]/policy.distance_scale_m,states[:,1,None].expand(-1,subdivisions)],-1).reshape(-1,2)
    lower,upper=sequential_bounds(actor,lo,hi)
    return torch.stack([lower.reshape(len(states),-1).amin(1).clamp(-3,3),upper.reshape(len(states),-1).amax(1).clamp(-3,3)],1)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact-dir',type=Path,required=True)
    p.add_argument('--barrier-dir',type=Path,required=True)
    p.add_argument('--perception-contract',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--rounds',type=int,default=80)
    p.add_argument('--steps-per-round',type=int,default=100)
    p.add_argument('--learning-rate',type=float,default=.00003)
    p.add_argument('--imitation-weight',type=float,default=.02)
    p.add_argument('--line-search',action='store_true')
    args=p.parse_args(argv)
    if args.output_dir.exists():p.error('choose a new output directory')
    args.output_dir.mkdir(parents=True);out=args.output_dir
    torch.set_num_threads(1);torch.manual_seed(61)
    controller=VisionSLSController(args.artifact_dir);policy=controller.policy
    original=copy.deepcopy(policy).eval().requires_grad_(False)
    cfg=json.loads((args.barrier_dir/'config.json').read_text())
    barrier=NonnegativeSBC(nonnegative_activation=cfg['nonnegative_activation'])
    barrier.load_state_dict(torch.load(args.barrier_dir/'sbc.pt',weights_only=True));barrier.eval().requires_grad_(False)
    scale=cfg['barrier_distance_scale_m'];divisor=torch.tensor([scale,1.])
    contract=json.loads(args.perception_contract.read_text())
    for field,name in [('source_policy_sha256','policy.pt'),('source_observation_sha256','observation.pt')]:
        if contract[field]!=sha256(args.artifact_dir/name):raise ValueError('source contract mismatch')
    _,_,states=region_samples(16384,61);_,_,evaluation=region_samples(8192,8801)
    radii=torch.tensor(envelope(states[:,0],contract['coefficients'],contract['calibration']['q_hat']),dtype=torch.float32)
    with torch.no_grad():b=barrier(states/divisor).flatten()
    active_ids=torch.where(b<20)[0]
    cache=np.load(args.artifact_dir/'observations.npz');split=np.load(args.artifact_dir/'splits.npz')['train']
    panel=TrainingTaskPanel(policy.distance_scale_m,cache['distance_m'][split],cache['observation'][split])
    baseline=panel(policy)
    # Imitation includes the actual train-image paths, not just uniform states.
    from .rollout import rollout
    trace=rollout(policy,policy.distance_scale_m,panel.initial,panel.distances,panel.observations,record_trajectories=True)
    imitation=torch.tensor([[obs,v] for step in trace['trajectories'] for obs,v in zip(step['observation'],step['speed_mps'])],dtype=torch.float32)
    with torch.no_grad():targets=original.forward_observation(imitation[:,:1],imitation[:,1:])
    policy.controller_net.requires_grad_(True)
    optimizer=torch.optim.Adam(policy.controller_net.parameters(),lr=args.learning_rate)
    history=[];best=None;best_value=float('inf')
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(barrier_distance_scale_m=scale,nonnegative_activation=cfg['nonnegative_activation'],
                  source_policy_sha256=sha256(args.artifact_dir/'policy.pt'),
                  source_barrier_sha256=sha256(args.barrier_dir/'sbc.pt'),
                  source_observation_sha256=sha256(args.artifact_dir/'observation.pt'),
                  offline_actor_training=True,deployment_action_filter=False,task_panel_uses_train_images_only=True)
    (out/'config.json').write_text(json.dumps(config,indent=2))
    for iteration in range(args.rounds):
        weights=copy.deepcopy(policy.state_dict());moments=copy.deepcopy(optimizer.state_dict())
        for step in range(args.steps_per_round):
            ids=active_ids[torch.randint(len(active_ids),(128,))]
            actions=differentiable_actions(policy,states[ids],radii[ids])
            lower,upper=successors(states[ids],actions)
            value=successor_upper(barrier,lower/divisor,upper/divisor,subdivisions=8)
            violations=torch.relu(value-b[ids]+.001)
            ii=torch.randint(len(imitation),(256,))
            actual=policy.forward_observation(imitation[ii,:1],imitation[ii,1:])
            loss=violations.mean()+violations.max()+args.imitation_weight*(actual-targets[ii]).square().mean()
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(policy.controller_net.parameters(),1.);optimizer.step()
        proposed=panel(policy);failures=retention_failures(baseline,proposed)
        fraction=1.
        if failures and args.line_search:
            candidate=copy.deepcopy(policy.state_dict())
            for fraction in [.5,.25,.125,.0625,.03125,.015625]:
                mixed={k:weights[k]+fraction*(candidate[k]-weights[k]) for k in weights}
                policy.load_state_dict(mixed)
                proposed=panel(policy);failures=retention_failures(baseline,proposed)
                if not failures:
                    optimizer.load_state_dict(moments)
                    break
        if failures:policy.load_state_dict(weights);optimizer.load_state_dict(moments)
        low,high=prepare(policy,contract,evaluation)
        result,delta,active=check(barrier,scale,evaluation,low,high,.0001)
        result.update(round=iteration,accepted=not failures,rejection_reasons=failures,
                      step_fraction=fraction if not failures else 0.,
                      train_outcomes={k:v['counts'] for k,v in proposed.items()})
        history.append(result)
        objective=max(result['max_drift'] or 0,0)+result['positive']*1e-6
        if objective<best_value:best_value=objective;best=copy.deepcopy(policy.state_dict())
        (out/'history.json').write_text(json.dumps(history,indent=2))
        print(json.dumps(result),flush=True)
        if result['sampled_tube_pass']:break
    policy.load_state_dict(best)
    saved=torch.load(args.artifact_dir/'policy.pt',weights_only=False)
    saved['state_dict']=policy.state_dict();saved['metadata']['sbc_refinement']='offline tube-aware SPVC actor update; no deployment QP'
    torch.save(saved,out/'policy.pt')
    torch.save(barrier.state_dict(),out/'sbc.pt')
    for name in ['observation.pt','observations.npz','splits.npz']:
        shutil.copy2(args.artifact_dir/name,out/name)
    contract['source_policy_sha256']=sha256(out/'policy.pt')
    contract['actor_rebinding_note']='observation model and image-only calibration unchanged; original actor updated using the reference tube'
    (out/'perception_contract.json').write_text(json.dumps(contract,indent=2))
    low,high=prepare(policy,contract,evaluation)
    result,delta,active=check(barrier,scale,evaluation,low,high,.0001)
    result.update(actor_changed=any(not torch.equal(v,original.state_dict()[k]) for k,v in policy.state_dict().items()),
                  source_policy_unchanged=sha256(args.artifact_dir/'policy.pt')==config['source_policy_sha256'],
                  deployment_action_filter=False,full_image_certificate=False)
    np.savez(out/'development_checks.npz',states=evaluation.numpy(),residual=delta.numpy(),active=active.numpy())
    (out/'report.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
