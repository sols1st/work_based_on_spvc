"""Frozen DINO -> learned low-dimensional observation y ~= Cx.

AEBS has x=[distance/scale, speed]. Only distance is visually labelled, so
C=[1,0] is fixed, unit-norm and non-collapsing. Speed remains proprioceptive.
This adapts VISION-SLS's observation matching, not its robot-specific MPC.
"""
import math
import hashlib
from pathlib import Path
import torch
from torch import nn
from Aebs.dino_latent.backbone import FrozenDinoV2


def positive_scale(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('distance scale must be positive and finite')
    return value


class ObservationProjection(nn.Module):
    def __init__(self, feature_dim=384, hidden_dim=128):
        super().__init__()
        self.register_buffer('C', torch.tensor([[1., 0.]]))
        self.register_buffer('feature_mean', torch.zeros(1, feature_dim))
        self.register_buffer('feature_std', torch.ones(1, feature_dim))
        self.net = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.GELU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                 nn.Linear(hidden_dim, 1))

    def forward(self, features):
        return self.net((features-self.feature_mean)/self.feature_std)

    def nominal(self, normalized_state):
        return normalized_state @ self.C.T


def load_projection(path, device='cpu'):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get('format') != 'vision_sls_aebs_v1':
        raise ValueError('expected an AEBS VISION-SLS checkpoint, not an R2/parking model')
    positive_scale(checkpoint['distance_scale_m'])
    model = ObservationProjection(checkpoint['feature_dim'], checkpoint['hidden_dim']).to(device)
    model.load_state_dict(checkpoint['projection'])
    if not torch.equal(model.C.cpu(), torch.tensor([[1., 0.]])):
        raise ValueError('AEBS requires C=[1,0]')
    return model.eval().requires_grad_(False), checkpoint


class ImageObservation(nn.Module):
    """Deployment path. Inputs: raw uint8 or float [0,1], BCHW/BHW images."""
    def __init__(self, checkpoint_path, device='cpu', dino_repo=None, dino_weights=None):
        super().__init__()
        self.projection, checkpoint = load_projection(checkpoint_path, device)
        weights = Path(dino_weights or checkpoint['dino_weights'])
        if hashlib.sha256(weights.read_bytes()).hexdigest() != checkpoint['provenance']['dino_weights_sha256']:
            raise ValueError('DINO weights differ from projection training weights')
        self.backbone = FrozenDinoV2(dino_repo or checkpoint['dino_repo'],
                                    weights, device)
        self.distance_scale_m = checkpoint['distance_scale_m']

    def forward(self, images):
        if images.dtype == torch.uint8:
            images = images.float()/255.
        if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
            raise ValueError('images must be uint8 or floats in [0,1]')
        return self.projection(self.backbone(images))

    def train(self, mode=True):
        super().train(False)
        return self


class VisionSLSPolicy(nn.Module):
    """Original VT interface on nominal Cx; image observations at deployment.

No cGAN, reconstruction network, learned state-to-latent proxy, or true
physical distance is consulted by forward_observation/forward_image.
The nominal VT branch does NOT certify image observation residuals.
"""
    def __init__(self, controller_net, distance_scale_m):
        super().__init__()
        self.controller_net = controller_net
        self.distance_scale_m = positive_scale(distance_scale_m)
        self.state_distance_scale_m = self.distance_scale_m
        self.register_buffer('C', torch.tensor([[1., 0.]]))

    def set_state_distance_scale(self, scale):
        self.state_distance_scale_m = positive_scale(scale)

    def controller_input(self, z, state):
        if state.ndim != 2 or state.shape[1] != 2:
            raise ValueError('state must be [batch,2]')
        adapted = torch.cat([state[:, :1] * (self.state_distance_scale_m/self.distance_scale_m),
                             state[:, 1:2]], dim=1)
        return torch.cat([adapted @ self.C.T, adapted[:, 1:2]], dim=1)

    def forward(self, z, state):
        return self.controller_net(self.controller_input(z, state)).clamp(-3., 3.)

    def forward_observation(self, observation, speed):
        if speed.ndim == 1:
            speed = speed[:, None]
        if observation.ndim != 2 or observation.shape[1] != 1 or speed.shape != observation.shape:
            raise ValueError('observation and speed must be [batch,1]')
        return self.controller_net(torch.cat([observation, speed], dim=1)).clamp(-3., 3.)

    def forward_image(self, encoder, images, speed):
        return self.forward_observation(encoder(images), speed)

    def metadata(self):
        return dict(frontend='vision_sls', C=[[1., 0.]], visual_dimension=1,
                    distance_scale_m=self.distance_scale_m,
                    state_distance_scale_m=self.state_distance_scale_m,
                    deployment='image -> frozen DINOv2 -> projection + measured speed -> actor',
                    verification='nominal Cx only; image residual not certified')
