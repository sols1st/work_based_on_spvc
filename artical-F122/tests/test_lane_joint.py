"""Safety-critical regression checks for the learned AEBS-style lane pipeline."""
from pathlib import Path
import numpy as np
import torch

from Aebs.vision_sls.sbc_cp import NonnegativeSBC, load_sbc_checkpoint
from Aebs.lane_keeping.learned_sbc import (raw_bounds, bounds, transitions, save_barrier,
    save_actor, retention_failures)
from Aebs.lane_keeping.policy import load_actor
from Aebs.lane_keeping.refine_actor import feedback_loss
from Aebs.lane_keeping.verify_learned import region_check, dynamic_check

ROOT=Path(__file__).resolve().parents[1]
ACTOR=ROOT/'results/lane_keeping_20261105/ppo_image/actor.pt'
torch.set_num_threads(2)


def test_shared_aebs_model_roundtrip(tmp_path):
    torch.manual_seed(19)
    model=NonnegativeSBC([2,16,8,1],nonnegative_activation='relu')
    path=tmp_path/'sbc.pt'
    save_barrier(path,model,{'purpose':'unit regression'})
    restored,metadata=load_sbc_checkpoint(path)
    assert type(restored) is NonnegativeSBC
    x=torch.randn(100,2)
    torch.testing.assert_close(model(x),restored(x),rtol=0,atol=0)
    assert metadata['nonnegative_activation']=='relu'
    actor=load_actor(ACTOR);save_actor(tmp_path/'actor.pt',actor)
    other=load_actor(tmp_path/'actor.pt')
    obs=torch.randn(100,3)
    torch.testing.assert_close(actor(obs),other(obs),rtol=0,atol=0)


def test_learned_bounds_enclose_random_and_corner_values():
    torch.manual_seed(17)
    model=NonnegativeSBC([2,16,8,1],nonnegative_activation='relu').double()
    lo=torch.rand(50,2,dtype=torch.float64)*2-1
    hi=lo+torch.rand_like(lo)*.2
    for method in ['affine','ibp']:
        bl,bh=bounds(model,lo,hi,method)
        for _ in range(20):
            x=lo+torch.rand_like(lo)*(hi-lo)
            b=model(x).flatten()
            assert (b>=bl-1e-12).all() and (b<=bh+1e-12).all()


def test_actor_feedback_has_finite_nonzero_gradients_and_frozen_barrier():
    torch.manual_seed(22)
    actor=load_actor(ACTOR)
    original=load_actor(ACTOR).requires_grad_(False)
    barrier=NonnegativeSBC([2,16,8,1],nonnegative_activation='relu').requires_grad_(False)
    with torch.no_grad():barrier.layers[-1].bias.fill_(2.)
    x=torch.rand(128,2)*1.8-.9
    loss,safety=feedback_loss(actor,original,barrier,x,torch.zeros(len(x)),.08)
    assert safety>0
    loss.backward()
    grads=[p.grad for p in actor.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert sum(float(g.abs().sum()) for g in grads)>0
    assert all(p.grad is None for p in barrier.parameters())


def test_constant_zero_cannot_pass_unsafe_and_large_constant_cannot_pass_initial():
    barrier=NonnegativeSBC([2,16,8,1],nonnegative_activation='relu').double()
    with torch.no_grad():
        for p in barrier.parameters():p.zero_()
    lo=torch.tensor([[1.,-1.3]],dtype=torch.float64)
    hi=torch.tensor([[1.3,1.3]],dtype=torch.float64)
    result,_=region_check(barrier,lo,hi,False,'affine',depth=1)
    assert not result['passed'] and result['unresolved']>0
    with torch.no_grad():barrier.layers[-1].bias.fill_(30.)
    result,_=region_check(barrier,-torch.ones(1,2,dtype=torch.float64)*.1,
                          torch.ones(1,2,dtype=torch.float64)*.1,True,'ibp',depth=1)
    assert not result['passed']


def test_unknown_dynamics_not_accepted_and_boundary_not_excluded():
    # B=20+10*tanh(tanh(theta)) touches 20 at theta=0; it is not all >20.
    barrier=NonnegativeSBC([2,1,1,1],nonnegative_activation='relu').double()
    with torch.no_grad():
        for p in barrier.parameters():p.zero_()
        barrier.layers[0].weight[0,1]=1
        barrier.layers[1].weight.fill_(1)
        barrier.layers[2].weight.fill_(10)
        barrier.layers[2].bias.fill_(20)
    actor=load_actor(ACTOR).double()
    result,_,_=dynamic_check(actor,barrier,0.,.08,2,0,1,'affine',1000,False)
    assert not result['passed'] and result['unresolved']>0
    assert result['partition_area_error']<1e-12


def test_task_retention_detects_safety_tracking_and_oscillation():
    base={'nominal':{'unsafe_episodes':0,'mean_abs_e':.03,'mean_action_change':.1}}
    assert not retention_failures(base,base)
    bad={'nominal':{'unsafe_episodes':1,'mean_abs_e':.08,'mean_action_change':.5}}
    assert len(retention_failures(base,bad))==3


def test_refinement_rejects_task_regression_and_restores_actor(tmp_path,monkeypatch):
    from Aebs.lane_keeping import refine_actor
    torch.manual_seed(22)
    barrier=NonnegativeSBC([2,16,8,1],nonnegative_activation='relu')
    with torch.no_grad():barrier.layers[-1].bias.fill_(2.)
    save_barrier(tmp_path/'sbc.pt',barrier,{})
    calls=[]
    def panel(actor,radius):
        calls.append(1)
        return {'nominal':dict(unsafe_episodes=0 if len(calls)==1 else 1,
                              mean_abs_e=.03,mean_action_change=.1)}
    monkeypatch.setattr(refine_actor,'task_panel',panel)
    refine_actor.main(['--actor',str(ACTOR),'--sbc',str(tmp_path/'sbc.pt'),
                       '--output-dir',str(tmp_path/'run'),'--rounds','1','--steps-per-round','2'])
    original=load_actor(ACTOR);restored=load_actor(tmp_path/'run/actor.pt')
    for key,value in original.state_dict().items():
        torch.testing.assert_close(value,restored.state_dict()[key],rtol=0,atol=0)
    import json
    result=json.loads((tmp_path/'run/report.json').read_text())
    assert result['accepted_rounds']==0 and not result['weights_changed']
