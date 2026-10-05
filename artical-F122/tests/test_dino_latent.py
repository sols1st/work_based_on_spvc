import torch

from Aebs.dino_latent.backbone import FrozenDinoV2
from Aebs.dino_latent.models import LatentController, PhysicalToLatent, SafetyProjection
from Aebs.dino_latent.spvc_policy import DinoLatentSPVCPolicy
from Combined_network.model import CombinedPolicyNetwork


def test_dino_preprocess_maps_grayscale_to_rgb_224():
    output = FrozenDinoV2.preprocess(torch.zeros(2, 32, 32))
    assert output.shape == (2, 3, 224, 224)
    assert torch.isfinite(output).all()


def test_safety_projection_preserves_32_dimensional_latent():
    projection = SafetyProjection()
    latent = projection(torch.randn(5, 384))
    assert latent.shape == (5, 32)


def test_physical_surrogate_and_controller_interfaces():
    q_model = PhysicalToLatent()
    controller = LatentController()
    latent = q_model(torch.randn(7, 1))
    action = controller(latent, torch.rand(7))
    assert latent.shape == (7, 32)
    assert action.shape == (7, 1)
    assert torch.all(action >= -3.0) and torch.all(action <= 3.0)


def test_spvc_policy_converts_physical_grid_to_33_inputs():
    q_model = PhysicalToLatent()
    controller = CombinedPolicyNetwork(
        torch.nn.Sequential(torch.nn.Linear(33, 8), torch.nn.Tanh()),
        torch.nn.Linear(8, 1),
    )
    policy = DinoLatentSPVCPolicy(q_model, controller, 3.0, 32)
    state = torch.tensor([[2.0, 1.5], [3.0, 2.5]])
    controller_input = policy.controller_input(torch.zeros(2, 4), state)
    assert controller_input.shape == (2, 33)
    assert policy(torch.zeros(2, 4), state).shape == (2, 1)
