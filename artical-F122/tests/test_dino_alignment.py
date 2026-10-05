import pytest
import torch
from Aebs.dino_latent.align_representation import alignment_stats
from Aebs.dino_latent.models import SafetyProjection, PhysicalToLatent


def test_relative_alignment_error_does_not_reward_uniform_shrinkage():
    z = torch.arange(128, dtype=torch.float32).reshape(4, 32)
    q = z + 2
    a = alignment_stats(z, q)
    b = alignment_stats(z * 0.1, q * 0.1)
    assert b['latent_rmse'] < a['latent_rmse']
    assert b['relative_latent_rmse'] == pytest.approx(a['relative_latent_rmse'], rel=1e-5)
    assert alignment_stats(torch.zeros(4, 32), torch.zeros(4, 32))['collapsed']


def test_alignment_gradient_reaches_both_networks():
    p, q = SafetyProjection(), PhysicalToLatent()
    loss = (p(torch.randn(4, 384)) - q(torch.randn(4, 1))).square().mean()
    loss.backward()
    for net in (p, q):
        assert sum(float(x.grad.abs().sum()) for x in net.parameters()) > 0
