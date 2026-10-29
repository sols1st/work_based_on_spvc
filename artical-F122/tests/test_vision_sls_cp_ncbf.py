import numpy as np
import pytest
import torch
from torch import nn
from scipy.stats import beta
from Combined_network.model import CombinedPolicyNetwork
from Aebs.vision_sls.model import VisionSLSPolicy
from Aebs.vision_sls.conformal import cp_quantile,fit_scale,envelope
from Aebs.vision_sls.cp_ncbf import MonotoneNCBF,action_bounds,successors,raw_constraints


def test_cp_order_statistic_and_infinity_convention():
    result=cp_quantile(np.arange(40),.05)
    assert result['rank']==39 and result['q_hat']==38
    assert result['coverage_lower_at_confidence']==pytest.approx(beta.ppf(.01,39,2))
    assert np.isinf(cp_quantile(np.arange(40),.001)['q_hat'])
    assert cp_quantile(np.zeros(40),.05)['q_hat']==0


def test_positive_state_envelope_fit_and_domain_rejection():
    d=np.linspace(5,16,40); residual=.02+.01*(d-5)
    coeff,_=fit_scale(d,residual,degree=2)
    assert (envelope(np.linspace(5,16,1001),coeff)>0).all()
    assert np.max(residual-envelope(d,coeff))<1e-6
    with pytest.raises(ValueError): envelope([17],coeff)
    with pytest.raises(ValueError): envelope([7],coeff,np.inf)


def test_actor_interval_encloses_observation_scenarios():
    torch.manual_seed(12)
    actor=CombinedPolicyNetwork(nn.Sequential(nn.Linear(2,8),nn.Tanh()),nn.Linear(8,1))
    policy=VisionSLSPolicy(actor,3.2)
    s=torch.tensor([[6.,.4],[10.,2.],[15.,3.]])
    radii=np.array([.3,.5,1.])
    intervals=action_bounds(policy,s,radii,subdivisions=8)
    for i in range(len(s)):
        observation=torch.linspace(float(s[i,0]-radii[i]),float(s[i,0]+radii[i]),101)[:,None]/3.2
        samples=policy.forward_observation(observation,s[i,1].expand(101,1)).flatten()
        assert samples.min()>=intervals[i,0]-1e-6
        assert samples.max()<=intervals[i,1]+1e-6
    lower,upper=successors(s,intervals)
    assert torch.equal(lower[:,0],upper[:,0])
    assert torch.allclose(lower[:,0],s[:,0]-.05*s[:,1])
    assert (lower<=upper).all()


def test_monotone_barrier_has_exact_region_and_successor_bounds():
    model=MonotoneNCBF()
    d=torch.linspace(5,6,41); v=torch.linspace(.5,3,41)
    unsafe=torch.cartesian_prod(d,v)
    assert model(unsafe).max()<=-model.delta+1e-6
    assert model.exact_region_bounds()['unsafe_max']==pytest.approx(-model.delta,abs=1e-6)
    s=torch.tensor([[7.,.8],[10.,2.]])
    a=torch.tensor([[-1.,2.],[.2,3.]])
    lower,upper=successors(s,a)
    middle=(lower+upper)/2
    assert (model(upper)<=model(middle)+1e-6).all()
    assert (model(middle)<=model(lower)+1e-6).all()


def test_invariance_violation_is_not_hidden_by_initial_and_unsafe_terms():
    model=MonotoneNCBF()
    initial=torch.tensor([[15.,3.]])
    unsafe=torch.tensor([[5.5,2.]])
    current=torch.tensor([[8.,1.]])
    bad_next=torch.tensor([[5.8,2.]])
    constraints=raw_constraints(model,initial,unsafe,current,bad_next)
    assert constraints[0,0]<0 and constraints[0,1]<0
    assert constraints[0,2]>0


def make_filter():
    from Aebs.vision_sls.cp_filter import CPNCBFFilter
    actor=nn.Linear(2,1)
    with torch.no_grad():
        actor.weight.zero_();actor.bias.fill_(-1.)
    return CPNCBFFilter(VisionSLSPolicy(actor,3.2),MonotoneNCBF(),[.05,.16,.32,.17],2.5)


def test_inverse_tube_contains_every_sampled_covered_distance():
    policy=make_filter()
    d=torch.linspace(5,16,101)
    for factor in torch.linspace(-1,1,11):
        observed=d+factor*policy.radius(d)
        low,high,valid=policy.distance_interval(observed)
        assert valid.all()
        assert (low<=d+2e-6).all() and (high>=d-2e-6).all()
    _,_,valid=policy.distance_interval(torch.tensor([-100.,100.]))
    assert not valid.any()


def test_filter_projects_to_minimal_feasible_action_and_reports_failure():
    policy=make_filter()
    d=torch.tensor([6.,8.,10.,15.,16.]);v=torch.tensor([3.,.8,2.,3.,2.5])
    required,feasible=policy.required_action(d,v)
    assert feasible.any() and (~feasible).any()
    assert (policy.constraint(d[feasible],v[feasible],required[feasible])>=0).all()
    interior=feasible&(required>-2.99)
    assert (policy.constraint(d[interior],v[interior],required[interior]-.01)<policy.numerical_margin).all()
    # Maximum braking on failure is explicitly flagged, never silently certified.
    assert (required[~feasible]==3).all()
    observed=(d+policy.radius(d))/policy.distance_scale_m
    actual=policy.forward_observation(observed[:,None],v).flatten()
    torch.testing.assert_close(actual,torch.maximum(required,torch.full_like(required,-1.)),atol=2e-4,rtol=0)
    assert torch.equal(policy.last_info['feasible'],feasible)


def test_filtered_action_lower_bound_covers_uncertain_observations():
    policy=make_filter()
    d=torch.tensor([7.,9.,12.,15.]);v=torch.tensor([.8,1.4,2.2,2.8])
    required,_=policy.required_action(d,v)
    for factor in torch.linspace(-1,1,21):
        observed=d+factor*policy.radius(d)
        action=policy.forward_observation(observed[:,None]/policy.distance_scale_m,v).flatten()
        assert (action>=required-2e-4).all()
        good=policy.last_info['feasible']
        assert (policy.constraint(d[good],v[good],action[good])>=-2e-6).all()


def test_trajectory_score_detects_unsafe_and_monitor_preserves_rollout():
    from Aebs.vision_sls.rollout import rollout
    from Aebs.vision_sls.calibrate_cp_filter import trajectory_panel
    policy=make_filter()
    initial=np.array([[5.1,3.],[15.,2.5]])
    library={'distance_m':np.array([5.,16.]),'observation':np.array([5.,16.])/3.2}
    expected=rollout(policy,3.2,initial,library['distance_m'],library['observation'])
    actual,score=trajectory_panel(policy,library,initial)
    for key,value in expected.items(): assert actual[key]==value
    assert actual['outcomes'][0]=='unsafe' and score[0]>=1.


def test_loader_rejects_changed_perception_before_loading_encoder(tmp_path):
    import json
    from Aebs.vision_sls.deploy_cp import VisionSLSCPController
    from Aebs.vision_sls.run import sha256
    (tmp_path/'policy.pt').write_bytes(b'actor')
    (tmp_path/'observation.pt').write_bytes(b'changed observer')
    (tmp_path/'perception_contract.json').write_text(json.dumps({'source_observation_sha256':'original observer'}))
    torch.save({'perception_contract_sha256':sha256(tmp_path/'perception_contract.json'),
                'source_policy_sha256':sha256(tmp_path/'policy.pt')},tmp_path/'ncbf.pt')
    with pytest.raises(ValueError,match='identity mismatch'):
        VisionSLSCPController(tmp_path,tmp_path)
