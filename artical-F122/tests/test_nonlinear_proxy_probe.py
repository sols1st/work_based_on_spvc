import numpy as np
import pytest
import torch

from Aebs.vision_sls.nonlinear_proxy_probe import ResidualProxy


def test_residual_initialization_preserves_linear_proxy_and_can_learn():
    torch.manual_seed(1)
    model = ResidualProxy([[1., .01]])
    x = torch.randn(30, 2)
    torch.testing.assert_close(model(x), x @ model.C.T)
    loss = (model(x) - (x[:, :1] + .2*x[:, :1].sin())).square().mean()
    loss.backward()
    assert model.net[-1].weight.grad.abs().max() > 0


def test_spectral_constraint_retains_distance_informativeness():
    torch.manual_seed(2)
    model = ResidualProxy([[.9, .01]])
    with torch.no_grad():
        model.net[-1].weight.fill_(30)
    model.constrain()
    assert model.slope_bound() <= .500001
    x = torch.randn(500, 2, requires_grad=True)
    slopes = torch.autograd.grad(model(x).sum(), x)[0][:, 0]
    assert slopes.min() >= .9-.500001


def test_proxy_residual_alone_is_not_total_observation_error():
    # A perfect proxy fit can have zero residual yet a biased controller input.
    scale, cd = 3.2, .8
    nominal = np.array([2., 3.])
    latent = nominal + .2
    h = latent.copy()
    residual = scale/cd*(latent-h)
    bias = scale/cd*(h-nominal)
    total = scale/cd*(latent-nominal)
    np.testing.assert_allclose(residual+bias, total)
    assert np.all(residual == 0) and np.all(total > 0)


def test_degenerate_anchor_rejected():
    with pytest.raises(ValueError):
        ResidualProxy([[.1, 1.]])
