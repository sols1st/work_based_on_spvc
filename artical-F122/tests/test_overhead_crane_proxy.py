import hashlib

import numpy as np
import torch

from Aebs.overhead_crane import plant
from Aebs.overhead_crane_proxy.evaluate import ImageDigest
from Aebs.overhead_crane_proxy.train import ProxyObservation


def test_physical_interface_with_nonidentity_rotation_and_neural_shears():
    torch.manual_seed(156)
    model = ProxyObservation('neural')
    with torch.no_grad():
        model.skew.copy_(torch.randn_like(model.skew)*.2)
        for shear in model.shears:
            torch.nn.init.normal_(shear[2].weight, std=.4)
    states, _, _ = plant.scenarios(17,11)
    z = torch.tensor(plant.observation(states),dtype=torch.float32)
    decoded_error = torch.sin(z)*.08
    actual = model.nominal(z+decoded_error)
    assert (actual-model.nominal(z)-decoded_error).abs().max() > .001
    recovered = model.decode(actual)
    torch.testing.assert_close(recovered-z,decoded_error,rtol=1e-4,atol=5e-7)
    gains = [1.7103,2.3227,-2.4583,-.4332]
    ref = plant.control((z+decoded_error).numpy(),states,gains)
    np.testing.assert_allclose(plant.control(recovered.detach().numpy(),states,gains),ref,atol=1e-6)


def test_digest_ignores_padded_rows_and_tracks_each_full_trajectory():
    batch,count=2,3
    digest=ImageDigest(lambda rows:len(rows),batch,count)
    hashes=[hashlib.sha256() for _ in range(count)]
    for start in (0,2):
        for step in range(plant.HORIZON):
            images=[np.array([start+i,step],dtype=np.uint8) for i in range(batch)]
            assert digest(images)==batch
            for i in range(min(batch,count-start)):
                hashes[start+i].update(images[i].tobytes())
    assert [h.digest() for h in hashes]==[h.digest() for h in digest.hashers]
