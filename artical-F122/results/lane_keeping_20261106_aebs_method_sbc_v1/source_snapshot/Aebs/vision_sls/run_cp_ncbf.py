"""VISION-SLS scale -> perception CP -> discrete CP-NCBF prototype.

Frozen existing vision/actor. Development margins never reuse the final NCBF
calibration panel. Current 400-image data support a reference CP experiment,
not independently collected trajectory-level coverage or a global certificate.
"""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch
from .conformal import fit_scale,envelope,cp_quantile
from .cp_ncbf import MonotoneNCBF,action_bounds,successors,raw_constraints,scores,sample_panel
from .deploy import VisionSLSController
from .run import sha256


def serialize(obj):
    if isinstance(obj,dict): return {k:serialize(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)): return [serialize(v) for v in obj]
    if isinstance(obj,(float,np.floating)) and not np.isfinite(obj): return 'infinity' if obj>0 else '-infinity'
    if isinstance(obj,np.generic): return obj.item()
    return obj


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--epochs-per-round',type=int,default=1500)
    parser.add_argument('--rounds',type=int,default=3)
    parser.add_argument('--perception-alpha',type=float,default=.05)
    parser.add_argument('--ncbf-alpha',type=float,default=.01)
    args=parser.parse_args(argv)
    if args.output_dir.exists(): parser.error('choose a new output directory')
    if args.epochs_per_round<1 or args.rounds<1: parser.error('positive training budgets required')
    out=args.output_dir; out.mkdir(parents=True)
    torch.set_num_threads(4); torch.manual_seed(73)
    source=args.artifact_dir
    controller=VisionSLSController(source)
    policy=controller.policy
    ckpt=torch.load(source/'observation.pt',map_location='cpu',weights_only=False)
    cached=np.load(source/'observations.npz'); split=np.load(source/'splits.npz')
    d=cached['distance_m']; estimated=cached['observation']*ckpt['distance_scale_m']; residual=abs(estimated-d)
    coeff,fit=fit_scale(d[split['train']],residual[split['train']])
    # This image library has been inspected in prior experiments. Mark it as
    # a development reference; do not relabel it independent new evidence.
    order=np.random.default_rng(20261028).permutation(split['test'])
    calibration_ids,audit_ids=np.array_split(order,2)
    percept=cp_quantile(residual[calibration_ids]/envelope(d[calibration_ids],coeff),args.perception_alpha)
    if not np.isfinite(percept['q_hat']): raise ValueError('insufficient image calibration samples for finite radius')
    q=percept['q_hat']
    coverage={name:dict(samples=len(ids),covered=int((residual[ids]<=envelope(d[ids],coeff,q)+1e-8).sum()),
                       mean_radius_m=float(envelope(d[ids],coeff,q).mean()),max_error_m=float(residual[ids].max()))
              for name,ids in [('fit',split['train']),('calibration_reference',calibration_ids),('audit_reference',audit_ids)]}
    contract=dict(format='vision_sls_state_cp_reference_v1',coefficients=coeff.tolist(),fit=fit,calibration=percept,
                  coverage=coverage,source_observation_sha256=sha256(source/'observation.pt'),
                  source_policy_sha256=sha256(source/'policy.pt'),
                  calibration_ids=calibration_ids.tolist(),audit_ids=audit_ids.tolist(),
                  input='true distance used offline; not a deployment state estimator',
                  statistical_status='reference only: reused finite image library, acquisition exchangeability unverified',
                  independent_trajectory_calibration=False,formal_image_certificate=False)
    (out/'perception_contract.json').write_text(json.dumps(serialize(contract),indent=2))
    print('Perception CP reference',percept,coverage,flush=True)
    def make_panel(n,seed):
        initial,unsafe,states=sample_panel(n,seed)
        radius=envelope(states[:,0].numpy(),coeff,q)
        bounds=action_bounds(policy,states,radius)
        low,high=successors(states,bounds)
        return initial,unsafe,states,high
    train=make_panel(8192,301); dev=make_panel(4096,302)
    model=MonotoneNCBF(); optimizer=torch.optim.Adam(model.parameters(),lr=.02)
    margin=0.; history=[]; best_value=float('inf'); best=None
    for round_index in range(args.rounds):
        for epoch in range(args.epochs_per_round):
            ids=torch.randint(len(train[0]),(1024,))
            panel=tuple(a[ids] for a in train)
            constraints=raw_constraints(model,*panel)
            initial_exact=model(torch.tensor([[15.,3.]]))
            loss=torch.relu(.1-initial_exact).square().mean()*20
            # Excluded states have q3=0 by definition: do not penalize those
            # zeros for a positive robustness margin (an impossible objective).
            h=model(panel[2]); active=(panel[2][:,0]>6)&(panel[2][:,1]>1e-6)&(h>=0)
            if active.any():
                violation=torch.relu(constraints[active,2]+margin)
                loss=loss+violation.mean()+violation.max()*2
            loss=loss+torch.relu(constraints[:,0]+margin).mean()
            optimizer.zero_grad();loss.backward();optimizer.step()
            if (epoch+1)%250==0:
                with torch.no_grad():
                    dev_score=scores(model,*dev).numpy()
                    dq=cp_quantile(dev_score,args.ncbf_alpha)['q_hat']
                    penalty=max(.1-model.exact_region_bounds()['initial_min'],0.)
                    objective=dq+penalty
                    if objective<best_value:
                        best_value=objective;best=copy.deepcopy(model.state_dict())
        with torch.no_grad():
            dev_scores=scores(model,*dev).numpy()
        dq=cp_quantile(dev_scores,args.ncbf_alpha)['q_hat']
        margin=max(margin,max(dq,0.))
        history.append(dict(round=round_index,development_q_hat=dq,next_margin=margin,
                            exact_regions=model.exact_region_bounds(),final_calibration_used=False))
        print('NCBF development',history[-1],flush=True)
    if best is not None: model.load_state_dict(best)
    model.eval().requires_grad_(False)
    # Only now create fresh IID state triples. No retraining/selection follows.
    final=make_panel(8192,7001); audit=make_panel(8192,7002)
    with torch.no_grad():
        cal_scores=scores(model,*final).numpy(); audit_scores=scores(model,*audit).numpy()
    final_cp=cp_quantile(cal_scores,args.ncbf_alpha)
    region=model.exact_region_bounds()
    result=dict(experiment='VISION_SLS_CP_discrete_NCBF_fixed_actor',perception_contract='perception_contract.json',
                final_constraint_cp=final_cp,exact_regions=region,
                calibrated_score_nonpositive=final_cp['q_hat']<=0,
                calibration_positive_scores=int((cal_scores>0).sum()),
                audit_positive_scores=int((audit_scores>0).sum()),
                audit_max_violation=float(audit_scores.max()),
                calibration_scope='IID triples: initial rectangle, unsafe rectangle, uniform full domain; fixed nominal actor and fitted observation tube',
                development_history=history,source_policy_sha256=sha256(source/'policy.pt'),
                actor_changed=False,vision_changed=False,
                ncbf='structured monotone ReLU neural barrier; discrete stopped-system h_next >= (1-.05)h',
                uncertainty='all observation values in state-dependent CP tube propagated through frozen actor via interval bounds',
                model_scope='AEBS deterministic plant with perception uncertainty; NOT legacy uniform additive state noise',
                trajectory_safety_probability=None,original_spvc_certificate=False,
                end_to_end_image_certificate=False,
                limitations=['image CP uses reused static data, not independent on-policy trajectories',
                             'random-state constraint coverage is not uniform-in-state or trajectory safety',
                             'new discrete NCBF is not the original nonnegative SBC or continuous-time QP theorem'])
    torch.save(dict(format='vision_sls_discrete_ncbf_v1',state_dict=model.state_dict(),knots=12,delta=model.delta,
                    perception_contract_sha256=sha256(out/'perception_contract.json'),
                    source_policy_sha256=sha256(source/'policy.pt')),out/'ncbf.pt')
    np.savez(out/'constraint_scores.npz',calibration=cal_scores,audit=audit_scores)
    (out/'report.json').write_text(json.dumps(serialize(result),indent=2))
    print(json.dumps(serialize(result),indent=2),flush=True)


if __name__=='__main__': main()
