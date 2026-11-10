"""Guard the learned-C observation semantics and certificate bridge."""
import numpy as np
import pytest
import torch
from torch import nn

from Aebs.lane_keeping.joint_observation import JointPatchProjection, LatentActor, require_invertible, load_joint


def test_both_branches_receive_gradients_and_C_is_normalized():
    torch.manual_seed(77)
    model=JointPatchProjection(feature_dim=3,hidden=4,grid=2)
    with torch.no_grad():model.raw_C.copy_(torch.tensor([[1.,.5],[-.2,1.]]))
    before=model.matrix().detach().clone()
    optimizer=torch.optim.Adam(model.parameters(),lr=.01)
    features=torch.randn(8,4,3);x=torch.randn(8,2)
    loss=(model(features)-model.nominal(x)).norm(dim=1).mean()
    loss.backward()
    assert model.raw_C.grad.norm()>0
    assert model.visual.conv[0].weight.grad.norm()>0
    optimizer.step()
    assert not torch.equal(before,model.matrix())
    torch.testing.assert_close(model.matrix().norm(dim=1),torch.ones(2))


def test_latent_actor_and_parallelogram_residual_match_semantic_loop():
    C=torch.tensor([[1.,.4],[-.2,1.]],dtype=torch.float64)
    C=C/C.norm(dim=1,keepdim=True)
    torch.manual_seed(88)
    actor=nn.Sequential(nn.Linear(3,6),nn.Tanh(),nn.Linear(6,1)).double()
    adapted=LatentActor(actor,C)
    x=torch.randn(30,2,dtype=torch.float64)
    delta=.09*torch.rand(30,2,dtype=torch.float64)
    k=torch.randn(30,1,dtype=torch.float64)
    latent=x@C.T+delta@C.T
    torch.testing.assert_close(adapted(torch.cat([latent,k],1)),actor(torch.cat([x+delta,k],1)))
    recovered=torch.linalg.solve(C,(latent-x@C.T).T).T
    torch.testing.assert_close(recovered,delta)
    # A raw-latent infinity box is not the same uncertainty set.
    assert not np.isclose(float((delta@C.T).abs().max()),float(delta.abs().max()))


@pytest.mark.parametrize('C',[torch.zeros(2,2),torch.tensor([[1.,0.],[1.,0.]]),
    torch.tensor([[1.,0.],[1.,.001]]),torch.tensor([[2.,0.],[0.,2.]])])
def test_invalid_or_degenerate_C_rejected(C):
    with pytest.raises(ValueError):require_invertible(C)


def test_checkpoint_matrix_tampering_is_rejected(tmp_path):
    model=JointPatchProjection(3,4,2)
    path=tmp_path/'model.pt'
    saved=dict(format='lane_joint_observation_v1',feature_dim=3,hidden=4,grid=2,
               state_dict=model.state_dict(),C=torch.eye(2),max_condition=10.)
    torch.save(saved,path)
    loaded,_=load_joint(path)
    torch.testing.assert_close(loaded.matrix(),torch.eye(2))
    saved['C']=torch.tensor([[0.,1.],[1.,0.]])
    torch.save(saved,path)
    with pytest.raises(ValueError,match='differs'):load_joint(path)


def test_aebs_joint_gradient_and_measured_speed_decoder():
    from Aebs.vision_sls.joint_observation import JointCLSProjection,JointLatentPolicy
    model=JointCLSProjection(3,8).double()
    x=torch.tensor([[3.,0.],[3.,3.]],dtype=torch.float64)
    feat=torch.ones(2,3,dtype=torch.float64)
    loss=(model(feat)-model.nominal(x)).abs().mean();loss.backward()
    assert model.raw_C.grad.norm()>0
    C=model.matrix().detach()
    # Decoder must subtract C_v v, including when C_v is not small.
    class Echo(nn.Module):
        def forward_observation(self,d,v):return d
    actor=JointLatentPolicy(Echo(),C)
    y=x@C.T
    torch.testing.assert_close(actor.forward_observation(y,x[:,1:]),x[:,:1])


def test_aebs_replay_score_uses_latent_error_and_visual_scale(monkeypatch):
    from types import SimpleNamespace
    import Aebs.vision_sls.joint_replay_cp as module
    C=torch.tensor([[.8,.6]])
    def fake(policy,scale,initial,distances,observations,record_trajectories):
        assert scale==7.  # deliberately differs from visual scale=3
        return dict(trajectories=[dict(indices=[0,1],observation=[4.,4.],
            distance_m=[12.,11.7],speed_mps=[1.,0.])],
            outcomes=['unsafe','timeout'],steps=[1,1],lookup_max_error_m=.3)
    monkeypatch.setattr(module,'rollout',fake)
    library=dict(weather=np.array(['sun']),color=np.array(['red']),
        distance_m=np.array([12.]),observation=np.array([4.]))
    result,_=module.collect(SimpleNamespace(C=C),3.,7.,library,2,41)
    expected=3*(4-np.array([12.,11.7])/3*.8-.6*np.array([1.,0.]))/.8
    np.testing.assert_allclose(result['scores'],expected,rtol=1e-6)
    assert result['outcomes'].tolist()==['unsafe','timeout']
