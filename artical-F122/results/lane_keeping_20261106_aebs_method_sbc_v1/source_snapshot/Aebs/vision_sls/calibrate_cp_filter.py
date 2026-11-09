"""Fresh state and trajectory calibration of the fixed CP-NCBF filter.

Trajectory guarantee is CONDITIONAL ON THE FIXED SAVED-IMAGE REPLAY WORLD:
IID initial states, deterministic nearest-image renderer, fixed plant/policy,
400 steps. It does not transfer to live CARLA, new images, or process noise.
"""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch
from .conformal import cp_quantile,envelope
from .cp_ncbf import sample_panel,action_bounds,successors,scores
from .deploy_cp import VisionSLSCPController
from .rollout import rollout,random_initial,initial_grid
from .run import sha256


def trajectory_panel(policy,library,initial):
    n=len(initial)
    barrier=copy.deepcopy(policy.barrier).double()
    with torch.no_grad():
        maximum=(-barrier(torch.tensor(initial,dtype=torch.float64))).clamp(min=0).numpy()
    infeasible=np.zeros(n,dtype=int);interventions=np.zeros(n,dtype=int)
    outside=np.zeros(n,dtype=int);uncovered=np.zeros(n,dtype=int)
    def monitor(ids,current,next_state,action,observation,outcome):
        with torch.no_grad():
            h=barrier(torch.tensor(current,dtype=torch.float64))
            hn=barrier(torch.tensor(next_state,dtype=torch.float64))
            terminal=np.isin(outcome,['success','stopped_safe_outside_goal'])
            hn[torch.tensor(terminal)]=32.
            q=((1-policy.decay)*h-hn).numpy()
            q[np.isin(outcome,['unsafe','out_of_domain'])]=np.maximum(q[np.isin(outcome,['unsafe','out_of_domain'])],1.)
            maximum[ids]=np.maximum(maximum[ids],q)
            info=policy.last_info
            infeasible[ids]+=(~info['feasible']).cpu().numpy()
            interventions[ids]+=info['intervention'].cpu().numpy()
            outside[ids]+=(~info['observation_in_contract_domain']).cpu().numpy()
            b=policy.radius(torch.tensor(current[:,0],dtype=torch.float32)).numpy()
            uncovered[ids]+=abs(observation*policy.distance_scale_m-current[:,0])>b+1e-6
    result=rollout(policy,policy.distance_scale_m,initial,library['distance_m'],library['observation'],monitor=monitor)
    result.update(trajectory_score_max=maximum.tolist(),qp_infeasible_steps=int(infeasible.sum()),
                  intervention_steps=int(interventions.sum()),observation_domain_misses=int(outside.sum()),
                  perception_tube_miss_steps=int(uncovered.sum()),
                  perception_tube_miss_episodes=int((uncovered>0).sum()))
    return result,maximum


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-dir',type=Path,required=True)
    parser.add_argument('--certificate-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.output_dir.exists(): parser.error('choose a new output directory')
    args.output_dir.mkdir(parents=True);out=args.output_dir
    torch.set_num_threads(4)
    controller=VisionSLSCPController(args.artifact_dir,args.certificate_dir)
    policy=controller.policy
    contract=json.loads((args.certificate_dir/'perception_contract.json').read_text())
    source_hash=sha256(args.artifact_dir/'policy.pt');ncbf_hash=sha256(args.certificate_dir/'ncbf.pt')
    # Register protocol before generating final calibration/test states.
    protocol=dict(source_policy_sha256=source_hash,ncbf_sha256=ncbf_hash,
                  fixed_before_calibration=True,no_updates_after_calibration=True,
                  state_calibration_seed=9011,state_audit_seed=9012,state_samples=8192,
                  trajectory_calibration_seed=9051,trajectory_test_seed=9052,
                  trajectory_calibration_count=512,trajectory_test_count=1000,horizon=400,
                  renderer='nearest distance over all 400 frozen images, deterministic argmin tie-breaking',
                  image_library_sha256=sha256(args.artifact_dir/'observations.npz'),
                  initial_distribution='uniform [15,16] metres x [2.5,3] m/s',
                  confidence_failure=.01,live_carla=False)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    def state_scores(seed):
        initial,unsafe,state=sample_panel(8192,seed)
        radius=envelope(state[:,0],contract['coefficients'],contract['calibration']['q_hat'])
        original=action_bounds(policy.base_policy,state,radius)
        # For every covered y: inferred lower distance <= true distance.
        # Required braking decreases with distance. Thus this is an offline
        # lower bound for ALL deployed filtered actions, without oracle input.
        with torch.no_grad():
            required,feasible=policy.required_action(state[:,0],state[:,1])
            lower=torch.maximum(original[:,0],required).clamp(-3,3)
            _,next_high=successors(state,torch.stack([lower,torch.full_like(lower,3.)],1))
            values=scores(policy.barrier,initial,unsafe,state,next_high).numpy()
            active=(policy.barrier(state)>=0)&(state[:,0]>6)&(state[:,1]>1e-6)
        return values,int((~feasible&active).sum())
    cal,cal_bad=state_scores(9011);test,test_bad=state_scores(9012)
    state_cp=cp_quantile(cal,.01)
    library=np.load(args.artifact_dir/'observations.npz')
    calibration,cal_scores=trajectory_panel(policy,library,random_initial(512,9051))
    # Max order statistic, the distribution-free one-sided tolerance choice.
    trajectory_cp=cp_quantile(cal_scores,1/513,.01)
    trajectory_cp['guarantee_scope']='one IID initial-state draw in the fixed replay world; score is the maximum over the stopped 400-step trajectory'
    test_rollout,test_scores=trajectory_panel(policy,library,random_initial(1000,9052))
    grid_rollout,grid_scores=trajectory_panel(policy,library,initial_grid(21))
    result=dict(state_constraint_cp=state_cp,state_calibration_positive=int((cal>0).sum()),
                state_audit_positive=int((test>0).sum()),state_active_infeasible=[cal_bad,test_bad],
                trajectory_cp=trajectory_cp,
                finite_replay_trajectory_certificate=bool(trajectory_cp['q_hat']<=0),
                claim_scope='conditional finite fixed-library replay model, random initial state, 400-step safety until existing terminal event',
                probability_lower_for_fixed_replay=(trajectory_cp['coverage_lower_at_confidence'] if trajectory_cp['q_hat']<=0 else None),
                confidence=.99,original_spvc_certificate=False,real_world_image_certificate=False,
                calibration=calibration,test=test_rollout,grid=grid_rollout,
                formal_continuous_state_verification=False,
                limitations=['not a new-image or live-CARLA certificate','not an infinite-horizon SBC',
                             'confidence bound covers finite-horizon safety, not goal-reaching probability',
                             'pointwise perception CP was a reused-data reference; trajectory scores calibrate the final fixed replay system directly',
                             'no claim of deterministic correctness at every state; state CP remains distributional'])
    (out/'report.json').write_text(json.dumps(result,indent=2))
    np.savez(out/'scores.npz',state_calibration=cal,state_audit=test,trajectory_calibration=cal_scores,trajectory_test=test_scores,grid=grid_scores)
    assert sha256(args.artifact_dir/'policy.pt')==source_hash and sha256(args.certificate_dir/'ncbf.pt')==ncbf_hash
    summary={k:v for k,v in result.items() if k not in ['calibration','test','grid']}
    summary['rollouts']={name:{k:record[k] for k in ['counts','qp_infeasible_steps','intervention_steps','perception_tube_miss_episodes']} for name,record in [('calibration',calibration),('test',test_rollout),('grid',grid_rollout)]}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
