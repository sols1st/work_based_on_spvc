"""Read-only audit of the nominal SPVC certificate and disturbance semantics.

Monte Carlo is diagnostic only. It stops on the existing episode events and
records domain exits separately; it cannot establish an infinite-horizon SBC.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from collections import Counter
import numpy as np
import torch
from auto_LiRPA import BoundedModule
from Aebs.VT.utils import MLP
from Aebs.VT.verify import VTVerifier
from Aebs.semantic_spvc.corrected_verify import CorrectedIntervalVTVerifier
from Aebs.system.env import Aebs
from Aebs.system.outcomes import classify_terminal_outcome
from .deploy import VisionSLSController
from .run import sha256


@torch.no_grad()
def stochastic_rollout(policy, env, distribution, n=10000, seed=1701):
    rng = np.random.default_rng(seed)
    state = rng.uniform([15.,2.5],[16.,3.],size=(n,2))
    scale = policy.distance_scale_m
    noise_amplitude = np.asarray(env.noise)*[env.std1,1.]
    outcomes = np.full(n,'',dtype=object)
    for _ in range(400):
        ids = np.flatnonzero(outcomes=='')
        if not len(ids): break
        s = torch.tensor(state[ids],dtype=torch.float32)
        action = policy.forward_observation(s[:,:1]/scale,s[:,1:]).cpu().numpy().reshape(-1)
        next_state = np.column_stack([state[ids,0]-.05*state[ids,1],np.clip(state[ids,1]-.05*action,0,3)])
        if distribution=='uniform': noise = rng.uniform(-1,1,size=next_state.shape)
        elif distribution=='triangular': noise = rng.triangular(-1,0,1,size=next_state.shape)
        elif distribution=='none': noise = np.zeros_like(next_state)
        else: raise ValueError('unknown noise distribution')
        next_state += noise*noise_amplitude
        for j,i in enumerate(ids):
            d,v = next_state[j]
            outcome = classify_terminal_outcome(d,v)
            if outcome is None and (v<0 or v>3): outcome = 'velocity_domain_exit'
            if outcome is not None: outcomes[i] = outcome
        state[ids] = next_state
    outcomes[outcomes==''] = 'timeout'
    return dict(episodes=n,counts=dict(Counter(outcomes)),seed=seed,
                noise_amplitude_physical=noise_amplitude.tolist(),distribution=distribution,
                semantics='v_next then additive noise; original terminal events, explicit velocity domain exits',
                formal_certificate=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.output_dir.exists(): parser.error('choose a new output directory')
    args.output_dir.mkdir(parents=True)
    torch.set_num_threads(4); torch.manual_seed(19)
    env=Aebs(.05)
    policy=VisionSLSController(args.artifact_dir).policy
    policy.set_state_distance_scale(env.std1)
    barrier=MLP([2,16,8,1],activation='tanh',square_output=True)
    barrier.load_state_dict(torch.load(args.artifact_dir/'sbc.pt',weights_only=True))
    barrier.eval().requires_grad_(False)
    bounded=BoundedModule(barrier,torch.zeros(1,2),device='cpu')
    learner=SimpleNamespace(p_net=policy,l_model=barrier)
    old=VTVerifier(learner,env,bounded,2048,.9,False)
    corrected=CorrectedIntervalVTVerifier(learner,env,bounded,2048,.9,False)
    regions={}
    for name,method in [('initial','compute_bound_init'),('unsafe','compute_bound_unsafe'),('domain','compute_bound_domain')]:
        regions[name]={'legacy':getattr(old,method)(env.space_split),
                       'corrected_interval':getattr(corrected,method)(env.space_split)}
    lower=torch.tensor([[15./env.std1,2.5]],dtype=torch.float32)
    upper=torch.tensor([[16./env.std1,3.]],dtype=torch.float32)
    with torch.no_grad():
        actual=barrier(torch.cat([lower,upper,(lower+upper)/2])).flatten().tolist()
        lo,hi=old.l_ibp.compute_bounds((lower,upper),method='IBP')
        correct_lo,correct_hi=corrected._interval_bounds(lower,upper)
    interval=dict(endpoint_and_center_values=actual,legacy=[float(lo),float(hi)],
                  corrected=[float(correct_lo),float(correct_hi)],
                  legacy_encloses_samples=min(actual)>=float(lo)-1e-6 and max(actual)<=float(hi)+1e-6)
    result=dict(source_policy_sha256=sha256(args.artifact_dir/'policy.pt'),source_barrier_sha256=sha256(args.artifact_dir/'sbc.pt'),
                regions=regions,interval_witness=interval,training_noise='triangular',verification_noise='uniform',
                noise_model_consistent=False,policy_frozen=True,
                stochastic_diagnostics={name:stochastic_rollout(policy,env,name) for name in ['none','triangular','uniform']},
                image_residual_in_verifier=False,certificate_established=False)
    (args.output_dir/'audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
