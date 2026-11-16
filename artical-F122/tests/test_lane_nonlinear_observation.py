import torch
from Aebs.lane_keeping.nonlinear_observation import NonlinearPatchProjection


def model():
    torch.manual_seed(15)
    m=NonlinearPatchProjection(3,4,2).double()
    with torch.no_grad():
        m.raw_C.copy_(torch.tensor([[.8,.6],[-.6,.8]]))
        for n in (m.f,m.g): n[2].weight.normal_(0,.1)
    m.constrain()
    return m


def test_global_nonlinear_inverse_and_uncertainty_identity():
    m=model();x=torch.randn(200,2,dtype=torch.float64)*3
    torch.testing.assert_close(m.decode(m.nominal(x)),x,atol=1e-12,rtol=1e-12)
    delta=torch.randn_like(x)*.1
    residual=m.nominal(x+delta)-m.nominal(x)
    torch.testing.assert_close(m.decode(m.nominal(x)+residual)-x,delta,atol=1e-12,rtol=1e-12)
    # A linear C-inverse is no longer the correct runtime decoder.
    assert (torch.linalg.solve(m.matrix(),m.nominal(x).T).T-x).abs().max()>.001


def test_both_visual_and_nonlinear_proxy_receive_gradients():
    m=model();patches=torch.randn(8,4,3,dtype=torch.float64);x=torch.randn(8,2,dtype=torch.float64)
    loss=(m(patches)-m.nominal(x)).square().mean();loss.backward()
    assert m.visual.out.weight.grad.abs().sum()>0
    assert m.raw_C.grad.abs().sum()>0
    assert m.f[2].weight.grad.abs().sum()>0 and m.g[2].weight.grad.abs().sum()>0


def test_nonlinear_jacobian_has_no_rank_collapse():
    m=model();x=torch.tensor([.3,-.2],dtype=torch.float64)
    jac=torch.autograd.functional.jacobian(lambda x:m.nominal(x[None])[0],x)
    torch.testing.assert_close(torch.linalg.det(jac),torch.linalg.det(m.matrix()),atol=1e-12,rtol=1e-12)
