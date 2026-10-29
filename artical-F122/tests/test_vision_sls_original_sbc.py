"""Keep the actual SPVC SBC and distinguish bounds from certificates."""
import numpy as np
import torch
from torch import nn
from Aebs.VT.utils import MLP
from Combined_network.model import CombinedPolicyNetwork
from Aebs.vision_sls.model import VisionSLSPolicy
from Aebs.vision_sls.sbc_cp import barrier_bounds, sbc_observation_tube_audit


def test_original_softplus_barrier_intervals_contain_actual_network_outputs():
    torch.manual_seed(14)
    barrier=MLP([2,16,8,1],activation='tanh',square_output=True)
    low=torch.tensor([[1.,0.],[4.,2.5]])
    high=torch.tensor([[2.,.5],[5.,3.]])
    lb,ub=barrier_bounds(barrier,low,high)
    samples=low[:,None,:]+torch.rand(2,300,2)*(high-low)[:,None,:]
    actual=barrier(samples.reshape(-1,2)).reshape(2,300)
    assert (actual>=lb[:,None]-1e-6).all() and (actual<=ub[:,None]+1e-6).all()
    assert (lb>=0).all()
    exact_low,exact_high=barrier_bounds(barrier,low,low)
    torch.testing.assert_close(exact_low,barrier(low).flatten())
    torch.testing.assert_close(exact_high,exact_low)


def test_original_strict_drift_includes_safe_fixed_points_without_changing_models():
    barrier=MLP([2,16,8,1],activation='tanh',square_output=True)
    actor=CombinedPolicyNetwork(nn.Sequential(nn.Linear(2,8),nn.Tanh()),nn.Linear(8,1))
    with torch.no_grad():
        for p in barrier.parameters(): p.zero_()
        for p in actor.parameters(): p.zero_()
        actor.action_net.bias.fill_(3.)
    policy=VisionSLSPolicy(actor,3.2)
    before=[p.detach().clone() for p in list(barrier.parameters())+list(policy.parameters())]
    state=torch.tensor([[10.,0.],[15.,2.5],[5.5,2.]])
    result=sbc_observation_tube_audit(barrier,policy,state,np.full(3,.5),4.)
    assert result['active'].tolist()==[True,True,False]
    torch.testing.assert_close(result['nominal_residual'],torch.full((3,),.1))
    torch.testing.assert_close(result['robust_residual'],torch.full((3,),.1))
    for saved,current in zip(before,list(barrier.parameters())+list(policy.parameters())):
        assert torch.equal(saved,current)


def test_monotonic_successor_bound_contains_dense_values_and_derivatives():
    from Aebs.vision_sls.sbc_cp import velocity_derivative_bounds,successor_upper
    torch.manual_seed(25)
    barrier=MLP([2,16,8,1],activation='tanh',square_output=True).double()
    low=torch.tensor([[1.8,0.],[2.2,.4],[4.,2.]],dtype=torch.float64)
    high=low+torch.tensor([[0.,.5],[0.,1.],[0.,.8]],dtype=torch.float64)
    dl,du=velocity_derivative_bounds(barrier,low,high)
    sample=(low[:,None,:]+(high-low)[:,None,:]*torch.linspace(0,1,101,dtype=torch.float64)[None,:,None]).reshape(-1,2).requires_grad_()
    values=barrier(sample).flatten()
    derivative=torch.autograd.grad(values.sum(),sample)[0][:,1].reshape(3,101)
    assert (derivative>=dl[:,None]-1e-12).all() and (derivative<=du[:,None]+1e-12).all()
    upper=successor_upper(barrier,low,high)
    assert (values.reshape(3,101)<=upper[:,None]+1e-12).all()
    upper.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in barrier.parameters())


def test_relu_sbc_keeps_nonnegativity_and_refined_bounds_differentiable():
    from Aebs.vision_sls.sbc_cp import NonnegativeSBC,successor_upper,monotone_box_bounds
    torch.manual_seed(31)
    model=NonnegativeSBC(nonnegative_activation='relu').double()
    with torch.no_grad():model.layers[-1].bias.fill_(.05)
    low=torch.tensor([[1.5,.1],[2.,.4],[3.,1.]],dtype=torch.float64)
    high=low+torch.tensor([0.,.8],dtype=torch.float64)
    upper=successor_upper(model,low,high,subdivisions=4)
    upper.sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    wide=high+torch.tensor([.3,0.],dtype=torch.float64)
    lb,ub=monotone_box_bounds(model,low,wide)
    samples=low[:,None,:]+(wide-low)[:,None,:]*torch.rand(3,500,2,dtype=torch.float64)
    values=model(samples.reshape(-1,2)).reshape(3,500)
    assert (values>=0).all()
    assert (values>=lb[:,None]-1e-12).all() and (values<=ub[:,None]+1e-12).all()


def test_sbc_checkpoint_records_activation_and_rejects_mismatch(tmp_path):
    import pytest
    from Aebs.vision_sls.sbc_cp import NonnegativeSBC,load_sbc_checkpoint
    model=NonnegativeSBC(nonnegative_activation='relu')
    with torch.no_grad():
        for p in model.parameters():p.zero_()
        model.layers[-1].bias.fill_(1.)
    path=tmp_path/'sbc.pt'
    torch.save({'format':'vision_sls_nonnegative_sbc_v2','features':[2,16,8,1],
                'metadata':{'nonnegative_activation':'relu','barrier_distance_scale_m':3.2},
                'state_dict':model.state_dict()},path)
    loaded,_=load_sbc_checkpoint(path)
    torch.testing.assert_close(loaded(torch.tensor([[4.,2.]])),torch.tensor([[1.]]))
    with pytest.raises(ValueError,match='activation'):
        load_sbc_checkpoint(path,{'nonnegative_activation':'softplus'})


def test_continuous_state_box_encloses_real_observation_scenarios():
    from Aebs.vision_sls.sbc_cp import NonnegativeSBC
    from Aebs.vision_sls.verify_sbc_cells import state_boxes
    from Aebs.vision_sls.conformal import envelope
    torch.manual_seed(37)
    actor=CombinedPolicyNetwork(nn.Sequential(nn.Linear(2,8),nn.Tanh()),nn.Linear(8,1)).double()
    policy=VisionSLSPolicy(actor,3.2)
    barrier=NonnegativeSBC(nonnegative_activation='relu').double()
    with torch.no_grad():barrier.layers[-1].bias.fill_(1.)
    low=torch.tensor([[6.1,.6],[9.,1.5],[15.,2.7]],dtype=torch.float64);high=low+torch.tensor([.05,.04])
    contract={'coefficients':[.05,.16,.32,.17],'calibration':{'q_hat':2.5}}
    bl,bu,nhi=state_boxes(policy,barrier,3.5,contract,low,high)
    states=low[:,None,:]+(high-low)[:,None,:]*torch.rand(3,100,2,dtype=torch.float64)
    flat=states.reshape(-1,2)
    radius=torch.tensor(envelope(flat[:,0],contract['coefficients'],2.5),dtype=torch.float64)
    obs=(flat[:,0]+radius*torch.linspace(-1,1,len(flat),dtype=torch.float64))/3.2
    action=policy.forward_observation(obs[:,None],flat[:,1:]).flatten()
    nxt=torch.stack([flat[:,0]-.05*flat[:,1],(flat[:,1]-.05*action).clamp(0,3)],1)
    div=torch.tensor([3.5,1.],dtype=torch.float64)
    current=barrier(flat/div).reshape(3,100);future=barrier(nxt/div).reshape(3,100)
    assert (current>=bl[:,None]-1e-10).all() and (current<=bu[:,None]+1e-10).all()
    assert (future<=nhi[:,None]+1e-10).all()


def test_packager_refuses_unresolved_verification(tmp_path):
    import json,pytest
    from Aebs.vision_sls.package_sbc import main
    source=tmp_path/'source';source.mkdir();(source/'policy.pt').write_bytes(b'p');(source/'sbc.pt').write_bytes(b'b')
    report=tmp_path/'report.json';report.write_text(json.dumps({'continuous_state_numerical_interval_pass':False}))
    output=tmp_path/'published'
    with pytest.raises(ValueError,match='incomplete or failed'):
        main(['--artifact-dir',str(source),'--barrier-dir',str(source),
              '--tube-verification',str(report),'--replay-verification',str(report),'--output-dir',str(output)])
    assert not output.exists()


def test_evaluation_refuses_to_reuse_checks_after_model_change(tmp_path):
    import json,pytest
    from Aebs.vision_sls.package_sbc import verified_bundle_status
    from Aebs.vision_sls.run import sha256
    (tmp_path/'policy.pt').write_bytes(b'old controller')
    digest=sha256(tmp_path/'policy.pt')
    (tmp_path/'summary.json').write_text(json.dumps({'status':'continuous_state_numerical_interval_checks_passed',
                                                   'hashes':{'policy.pt':digest}}))
    (tmp_path/'policy.pt').write_bytes(b'changed controller')
    with pytest.raises(ValueError,match='artifact changed'):
        verified_bundle_status(tmp_path)
