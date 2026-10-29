"""Keep original SPVC SBC/actor and audit the new CP observation interface.

This diagnostic never declares a new certificate from marginal CP coverage.
It checks the original nonnegative network, region thresholds, strict drift,
and records unresolved transition-law and domain assumptions explicitly.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from Aebs.VT.utils import MLP
from Aebs.system.env import Aebs
from .deploy import VisionSLSController
from .conformal import envelope
from .run import sha256
from .sbc_cp import barrier_bounds, sbc_observation_tube_audit


def region_grid(low, high, size=32):
    edges=[torch.linspace(a,b,size+1) for a,b in zip(low,high)]
    return (torch.cartesian_prod(edges[0][:-1], edges[1][:-1]),
            torch.cartesian_prod(edges[0][1:], edges[1][1:]))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-dir',type=Path,required=True)
    parser.add_argument('--perception-contract',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.output_dir.exists(): parser.error('choose a new output directory')
    contract=json.loads(args.perception_contract.read_text())
    source=args.artifact_dir
    if (contract['source_policy_sha256']!=sha256(source/'policy.pt') or
        contract['source_observation_sha256']!=sha256(source/'observation.pt')):
        parser.error('perception contract does not match the frozen vision/actor')
    args.output_dir.mkdir(parents=True)
    torch.set_num_threads(4)
    model_hashes={name:sha256(source/name) for name in ['policy.pt','sbc.pt','observation.pt']}
    controller=VisionSLSController(source)
    barrier=MLP([2,16,8,1],activation='tanh',square_output=True)
    barrier.load_state_dict(torch.load(source/'sbc.pt',map_location='cpu',weights_only=True))
    barrier.eval().requires_grad_(False)
    barrier_scale=Aebs(.05).std1
    scale=torch.tensor([barrier_scale,1.])
    regions={}
    with torch.no_grad():
        for name,low,high in [('initial',[15.,2.5],[16.,3.]),('unsafe',[5.,.5],[6.,3.])]:
            lo,hi=region_grid(low,high)
            lb,ub=barrier_bounds(barrier,lo/scale,hi/scale)
            regions[name]=dict(lower=float(lb.min()),upper=float(ub.max()),cells=len(lo))
        # Entire grid, plus zero-speed states: do not hide fixed points/terminal
        # states to make the original strict-drift condition appear satisfied.
        states=torch.cat([torch.cartesian_prod(torch.linspace(5,16,101),torch.linspace(0,3,101)),
                          torch.tensor([[15.,0.],[10.,0.],[6.,0.]])])
        radii=envelope(states[:,0],contract['coefficients'],contract['calibration']['q_hat'])
        values=sbc_observation_tube_audit(barrier,controller.policy,states,radii,barrier_scale)
    active=values['active']; robust=values['robust_residual']; nominal=values['nominal_residual']
    ids=torch.nonzero(active).flatten()
    worst=ids[torch.argsort(robust[ids],descending=True)[:10]]
    witnesses=[dict(state_physical=states[i].tolist(),B=float(values['current'][i]),
                    robust_upper_residual=float(robust[i]),nominal_residual=float(nominal[i])) for i in worst]
    result=dict(backend='original_SPVC_SBC',architecture='original VT.utils.MLP [2,16,8,1], tanh, softplus output',
                source_hashes=model_hashes,actor_changed=False,barrier_changed=False,vision_changed=False,
                ncbf_used=False,action_filter_used=False,reach_probability_target=.95,epsilon=.1,
                domain_nonnegative_by_construction=True,regions=regions,
                initial_upper_at_most_one=regions['initial']['upper']<=1,
                unsafe_lower_at_least_twenty=regions['unsafe']['lower']>=20,
                region_separation=regions['initial']['upper']<regions['unsafe']['lower'],
                states=len(states),checked_states=int(active.sum()),
                unresolved_robust_drift_states=int((active&(robust>0)).sum()),
                nominal_strict_drift_witnesses=int((active&(nominal>1e-5)).sum()),
                successor_domain_exit_states=int((active&values['successor_domain_exit']).sum()),
                witnesses=witnesses,
                cp_role='state-dependent observation radius only; not a conditional disturbance probability law',
                expectation_assumption_established=False,
                reference_tube_sufficient_condition='sup_y B(f(x,pi(y,v))) + epsilon <= B(x), conditional on full support inside the tube',
                original_spvc_certificate=False,
                limitations=['marginal image CP does not establish the conditional expectation required by SBC',
                             'no claimed tail bound outside the CP tube',
                             'grid states do not cover the full continuous drift domain',
                             'strict drift is checked literally, including safe terminal states; stopping semantics need explicit theorem treatment',
                             'numerical IBP is not a floating-point formal proof'])
    for name,digest in model_hashes.items(): assert sha256(source/name)==digest
    np.savez(args.output_dir/'checks.npz',states=states.numpy(),**{k:v.numpy() for k,v in values.items()})
    (args.output_dir/'report.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
