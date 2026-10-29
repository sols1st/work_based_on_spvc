"""Observation semantics, deployment parity, and real VT parameter updates."""
import copy
import numpy as np
import pytest
import torch
from torch import nn
from Aebs.vision_sls.model import ObservationProjection, VisionSLSPolicy, load_projection
from Aebs.vision_sls.env import ObservationEnv


def test_nominal_and_observation_paths_agree_after_scale_conversion():
    actor = nn.Linear(2,1,bias=False)
    actor.weight.data[:] = torch.tensor([[.4,.3]])
    policy = VisionSLSPolicy(actor,2.)
    policy.set_state_distance_scale(4.)
    state = torch.tensor([[3.,2.]])
    assert torch.allclose(policy.controller_input(None,state),torch.tensor([[6.,2.]]))
    assert torch.allclose(policy(None,state),policy.forward_observation(torch.tensor([[6.]]),torch.tensor([2.])))
    assert torch.equal(policy(torch.randn(1,4),state),policy(torch.zeros(1,4),state))
    policy(None,state).sum().backward()
    assert actor.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError):
        policy.set_state_distance_scale(float('nan'))


def test_image_observation_not_true_distance_and_seeded_reset():
    env = ObservationEnv(2.,[5.,16.],[100.,200.])
    assert np.allclose(env.reset_to(15.,2.),[200.,2.])
    nominal = ObservationEnv(2.)
    assert np.allclose(nominal.reset_to(15.,2.),[7.5,2.])
    assert np.array_equal(env.reset(seed=42)[0],env.reset(seed=42)[0])


def test_projection_is_observation_not_arbitrary_latent(tmp_path):
    model = ObservationProjection()
    assert torch.equal(model.nominal(torch.tensor([[2.,99.]])),torch.tensor([[2.]]))
    path = tmp_path/'observation.pt'
    torch.save(dict(format='vision_sls_aebs_v1',distance_scale_m=2.,feature_dim=384,
                    hidden_dim=128,projection=model.state_dict()),path)
    loaded,_ = load_projection(path)
    features = torch.randn(3,384)
    assert torch.allclose(model(features),loaded(features))
    assert not any(p.requires_grad for p in loaded.parameters())


def test_vt_bound_wrapper_does_not_disable_barrier_learning():
    from Aebs.system.env import Aebs
    from Aebs.VT.train import VTLearner
    torch.manual_seed(1)
    env = Aebs(.05)
    actor = nn.Sequential(nn.Linear(2,8),nn.Tanh(),nn.Linear(8,1))
    policy = VisionSLSPolicy(actor,env.std1)
    learner = VTLearner([2,16,8,1],env,2.,4.,.1,1.,.95,p_net=policy)
    before = copy.deepcopy(learner.l_model.state_dict())
    learner.create_bounded_module(learner.l_model)
    assert all(p.requires_grad for p in learner.l_model.parameters())
    states = learner.sample_init(4,16)
    metrics = learner.train_step_l(torch.zeros(16,4),states,.001,np.array([.01,.01]))
    assert np.isfinite(list(metrics.values())).all()
    assert any(not torch.equal(v,learner.l_model.state_dict()[k]) for k,v in before.items())


def test_grid_intersection_includes_containing_cells():
    from Aebs.VT.verify import v_intersect
    from Aebs.system.env import Box
    box = Box(np.array([5.,.5]),np.array([6.,3.]))
    low = np.array([[4.,0.],[6.,1.],[6.1,0.],[4.,3.1]])
    high = np.array([[7.,4.],[7.,2.],[7.,4.],[7.,4.]])
    assert v_intersect(box,low,high).tolist() == [True,True,False,False]


def test_batched_rollout_matches_original_environment_step_semantics():
    from Aebs.vision_sls.rollout import rollout
    for acc in (-3.,0.,.7,3.):
        actor = nn.Linear(2,1)
        with torch.no_grad():
            actor.weight.zero_(); actor.bias.fill_(acc)
        policy = VisionSLSPolicy(actor,3.2378063201904297)
        initial = np.array([[15.,2.5],[5.5,.2],[5.1,2.5],[5.01,.4],[15.8,0.]])
        result = rollout(policy,policy.distance_scale_m,initial,max_steps=400)
        for i,(d,v) in enumerate(initial):
            env = ObservationEnv(policy.distance_scale_m)
            env.reset_to(d,v)
            total = 0.
            for step in range(400):
                _,reward,done,truncated,info = env.step(np.array([acc],dtype=np.float32))
                total += reward
                if done or truncated:
                    break
            assert result['outcomes'][i] == info['outcome']
            assert result['steps'][i] == step+1
            assert result['returns'][i] == pytest.approx(total,abs=1e-5)
            np.testing.assert_allclose(result['terminal_states'][i],
                                       [env.base.state[0]*env.base.std1,env.base.state[1]],atol=1e-5)


def test_task_retention_checks_each_case_not_only_success_count():
    from Aebs.vision_sls.training import retention_failures
    before = {'nominal':{'outcomes':['success','timeout','unsafe']}}
    swapped = {'nominal':{'outcomes':['timeout','success','unsafe']}}
    assert retention_failures(before,swapped) == ['nominal:lost_success']
    assert retention_failures(before,{'nominal':{'outcomes':['success','unsafe','unsafe']}}) == ['nominal:new_unsafe_episode']
    assert retention_failures(before,{'nominal':{'outcomes':['success','success','success']}}) == []


@pytest.mark.parametrize('reject',[True,False])
def test_actor_guard_restores_weights_and_adam_state_on_rejection(monkeypatch,reject):
    from Aebs.vision_sls.training import TaskPreservingLearner
    from Aebs.VT.train import VTLearner
    from Aebs.system.env import Aebs
    env = Aebs(.05)
    actor = nn.Linear(2,1)
    policy = VisionSLSPolicy(actor,env.std1)
    calls = []
    def panel(policy):
        calls.append(1)
        return {'nominal':{'outcomes':['success' if len(calls)==1 or not reject else 'unsafe']}}
    learner = TaskPreservingLearner([2,16,8,1],env,2.,4.,.1,1.,.95,
                                    p_net=policy,task_panel=panel)
    def real_optimizer_step(self,*args,**kwargs):
        self.p_optimizer.zero_grad()
        sum(p.square().sum() for p in self.p_net.parameters()).backward()
        self.p_optimizer.step()
        return [{'loss':1.}]
    real_optimizer_step(learner)
    weights = copy.deepcopy(policy.state_dict())
    optimizer = copy.deepcopy(learner.p_optimizer.state_dict())
    monkeypatch.setattr(VTLearner,'train_epoch',real_optimizer_step)
    learner.train_epoch(train_fn='p')
    assert learner.task_update_history[-1]['accepted'] is (not reject)
    if reject:
        for key,value in weights.items():
            torch.testing.assert_close(policy.state_dict()[key],value,rtol=0,atol=0)
        actual = learner.p_optimizer.state_dict()
        assert actual['param_groups'] == optimizer['param_groups']
        for key,items in optimizer['state'].items():
            for name,value in items.items():
                torch.testing.assert_close(actual['state'][key][name],value,rtol=0,atol=0)
    else:
        assert any(not torch.equal(value,policy.state_dict()[key]) for key,value in weights.items())
