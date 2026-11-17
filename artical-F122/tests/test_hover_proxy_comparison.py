"""Check the nonlinear raw residual and equivalent paired initialization."""
import numpy as np
import torch

from Aebs.visual_hover import plant
from Aebs.visual_hover.compare_proxies import geometry, make_pair
from Aebs.visual_hover.evaluate import collect


def test_same_start_but_nonlinear_latent_residual_uses_full_proxy():
    torch.manual_seed(11)
    ridge=torch.eye(4,3)
    models=make_pair(3,ridge,11)
    sample=torch.tensor([[.4,.2,-.1]])
    assert torch.equal(models['linear'].nominal(sample),models['neural'].nominal(sample))
    model=models['neural']
    with torch.no_grad():
        model.nets[0][2].weight.fill_(.3)
    fake_encoder=lambda images:torch.tensor([[.1,.2,-.3]]).repeat(len(images),1)
    front=dict(mean=torch.zeros(3),basis=torch.eye(3),std=torch.ones(3))
    records=collect(plant.scenarios(2,17),dict(l=.7,m=1.,s_limit=.63),
                    fake_encoder,front,model,2,record_observations=True)
    x=torch.from_numpy(records['states'][:,:-1,:3]).float().reshape(-1,3)
    latent=records['latent'].reshape(-1,3)
    with torch.no_grad():
        expected=latent-model.nominal(x).numpy()
    np.testing.assert_allclose(records['raw'].reshape(-1,3),expected,atol=1e-6)
    assert np.max(np.abs(expected - (latent-x.numpy())))>1e-2
    assert not model.bias.requires_grad
    g=geometry(model)
    assert g['affine_max']>1e-3 and g['inverse_max']<1e-12
