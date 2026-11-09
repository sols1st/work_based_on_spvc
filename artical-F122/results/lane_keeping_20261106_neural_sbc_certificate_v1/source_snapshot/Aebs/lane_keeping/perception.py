"""Frozen DINOv2 -> learned 2-D lane observation ``(e, theta)``.

The projection is trained on scene-group training splits and then frozen.  It
outputs normalized coordinates so the same box radius can be used for both
components before rescaling by ``E_SCALE`` / ``THETA_SCALE``.
"""
import json
from pathlib import Path
import torch
from torch import nn

from ..dino_latent.backbone import FrozenDinoV2


class DINOProjection(nn.Module):
    def __init__(self, feature_dim=384, hidden=128, output_dim=2):
        super().__init__()
        self.register_buffer('feature_mean', torch.zeros(1, feature_dim))
        self.register_buffer('feature_std', torch.ones(1, feature_dim))
        self.net = nn.Sequential(nn.Linear(feature_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, output_dim))

    def forward(self, features):
        return self.net((features - self.feature_mean) / self.feature_std)


class ImageObserver(nn.Module):
    """Deployment path: uint8/float images -> normalized (e, theta)."""

    def __init__(self, checkpoint_path, device='cpu'):
        super().__init__()
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if checkpoint.get('format') != 'lane_keeping_dino_projection_v1':
            raise ValueError('unexpected lane-keeping projection checkpoint')
        self.projection = DINOProjection(checkpoint['feature_dim'], checkpoint['hidden'],
                                         checkpoint['output_dim'])
        self.projection.load_state_dict(checkpoint['projection'])
        self.projection.feature_mean.copy_(torch.as_tensor(checkpoint['feature_mean']))
        self.projection.feature_std.copy_(torch.as_tensor(checkpoint['feature_std']))
        self.backbone = FrozenDinoV2(checkpoint['dino_repo'], checkpoint['dino_weights'],
                                     device, model_name=checkpoint['dino_model'])
        self.scales = dict(checkpoint['scales'])
        self.metadata = checkpoint
        self.eval().requires_grad_(False)

    def forward(self, images):
        if images.dtype == torch.uint8:
            images = images.float() / 255.
        if images.ndim == 4 and images.shape[-1] == 3:
            images = images.permute(0, 3, 1, 2)
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError('images must be [B,H,W,3] or [B,3,H,W]')
        if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
            raise ValueError('images must be uint8 or floats in [0,1]')
        features = self.backbone(images)
        return self.projection(features)

    def physical(self, observation):
        """Normalized observation -> physical (e, theta) metres/radians."""
        observation = torch.as_tensor(observation)
        return observation * observation.new_tensor([self.scales['e'], self.scales['theta']])

    def train(self, mode=True):
        super().train(False)
        return self


def load_observer(path, device='cpu'):
    return ImageObserver(path, device)


def save_checkpoint(path, projection, feature_mean, feature_std, dino_repo, dino_weights,
                    dino_model, scales, metadata):
    torch.save(dict(
        format='lane_keeping_dino_projection_v1',
        feature_dim=int(projection.net[0].in_features),
        hidden=int(projection.net[0].out_features),
        output_dim=int(projection.net[-1].out_features),
        projection=projection.state_dict(),
        feature_mean=torch.as_tensor(feature_mean).clone(),
        feature_std=torch.as_tensor(feature_std).clone(),
        dino_repo=str(dino_repo), dino_weights=str(dino_weights), dino_model=str(dino_model),
        scales=dict(scales), metadata=metadata), path)
