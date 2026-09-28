"""Runtime image-to-latent and physical-state-to-latent interfaces."""

import torch
import torch.nn as nn

from Aebs.dino_latent.backbone import FrozenDinoV2
from Aebs.dino_latent.models import PhysicalToLatent, SafetyProjection


class DinoSafetyLatentEncoder(nn.Module):
    """Frozen DINOv2 -> frozen trained 32-D safety projection."""

    def __init__(self, checkpoint_path, device):
        super().__init__()
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.backbone = FrozenDinoV2(
            checkpoint["dino_repo"], checkpoint["dino_weights"], device,
            model_name=checkpoint["dino_model"],
        )
        self.projection = SafetyProjection(
            latent_dim=int(checkpoint["latent_dimension"])
        ).to(device)
        self.projection.load_state_dict(checkpoint["projection_state_dict"])
        self.projection.eval()
        for parameter in self.projection.parameters():
            parameter.requires_grad_(False)
        self.register_buffer("feature_mean", checkpoint["feature_mean"].reshape(1, -1))
        self.register_buffer("feature_std", checkpoint["feature_std"].reshape(1, -1))
        self.latent_dimension = int(checkpoint["latent_dimension"])

    def forward(self, image):
        feature = self.backbone(image)
        return self.projection((feature - self.feature_mean) / self.feature_std)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        self.projection.eval()
        return self


class PhysicalLatentSurrogate(nn.Module):
    """Load q_psi for offline verification and CP residual construction."""

    def __init__(self, checkpoint_path, device):
        super().__init__()
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model = PhysicalToLatent(
            physical_dim=1, latent_dim=int(checkpoint["latent_dimension"])
        ).to(device)
        self.model.load_state_dict(checkpoint["physical_to_latent_state_dict"])
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.distance_scale_m = float(checkpoint["distance_scale_m"])

    def forward(self, distance_m):
        if distance_m.ndim == 1:
            distance_m = distance_m.unsqueeze(1)
        return self.model(distance_m / self.distance_scale_m)

