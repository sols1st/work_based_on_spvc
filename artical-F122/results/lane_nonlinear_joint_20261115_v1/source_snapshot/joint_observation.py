"""Joint VISION-SLS observation matching, isolated from fixed-state heads.

State coordinates here are x=(e/0.85, theta/0.30). The visual head returns
latent y, not a state estimate. An explicit inverse interface permits reuse of
the nominal actor: pi_latent(y,k)=pi_state(C^{-1}y,k). This is a coordinate
adaptation, not PPO retraining and not permission to reuse a CP threshold.
"""
import torch
from torch import nn

from .patch_perception import PatchProjection
from ..dino_latent.backbone import FrozenDinoV2
from .learned_sbc import sha256


class JointPatchProjection(nn.Module):
    def __init__(self, feature_dim=384, hidden=64, grid=16):
        super().__init__()
        self.visual = PatchProjection(feature_dim, hidden, grid)
        self.raw_C = nn.Parameter(torch.eye(2))

    def matrix(self):
        return self.raw_C / self.raw_C.norm(dim=1, keepdim=True).clamp_min(1e-12)

    def forward(self, patches):
        return self.visual(patches)

    def nominal(self, normalized_state):
        return normalized_state @ self.matrix().T

    def decode(self, latent):
        return torch.linalg.solve(self.matrix(), latent.T).T


def require_invertible(C, max_condition=10.):
    if C.shape != (2, 2) or not torch.isfinite(C).all():
        raise ValueError('finite 2x2 C required')
    sv = torch.linalg.svdvals(C.double())
    if sv[-1] <= 0 or sv[0] / sv[-1] > max_condition:
        raise ValueError('C is rank deficient or exceeds registered condition limit')
    if not torch.allclose(C.norm(dim=1), torch.ones(2, dtype=C.dtype, device=C.device), atol=1e-6):
        raise ValueError('rows of C must have unit norm')
    return sv


def load_joint(path, device='cpu'):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved.get('format') != 'lane_joint_observation_v1':
        raise ValueError('expected a jointly learned lane observation')
    model = JointPatchProjection(saved['feature_dim'], saved['hidden'], saved['grid'])
    model.load_state_dict(saved['state_dict'])
    C = model.matrix().detach()
    require_invertible(C, saved['max_condition'])
    if not torch.equal(C, saved['C']):
        raise ValueError('saved C differs from trainable checkpoint matrix')
    return model.to(device).eval().requires_grad_(False), saved


class JointImageObserver(nn.Module):
    """Public forward is the true visual latent p(DINO(image))."""
    def __init__(self, path, device='cpu', dino_repo=None, dino_weights=None):
        super().__init__()
        self.projection, saved = load_joint(path, device)
        weights = dino_weights or saved['dino_weights']
        if sha256(weights) != saved['dino_weights_sha256']:
            raise ValueError('DINO weight binding mismatch')
        self.backbone = FrozenDinoV2(dino_repo or saved['dino_repo'], weights,
                                     device, saved['dino_model'])
        self.device = device

    @torch.no_grad()
    def forward(self, images):
        images = images.to(self.device)
        if images.dtype == torch.uint8:
            images = images.float() / 255.
        if images.ndim == 4 and images.shape[-1] == 3:
            images = images.permute(0, 3, 1, 2)
        if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
            raise ValueError('images must be finite RGB values in [0,1]')
        patches = self.backbone.backbone.forward_features(
            self.backbone.preprocess(images))['x_norm_patchtokens']
        return self.projection(patches)


class DecodedJointObserver(nn.Module):
    """Explicit adapter for the existing semantic-input actor and verifier."""
    def __init__(self, observer):
        super().__init__()
        self.observer = observer

    @torch.no_grad()
    def forward_with_latent(self, images):
        latent = self.observer(images)
        return self.observer.projection.decode(latent), latent

    def forward(self, images):
        return self.forward_with_latent(images)[0]


class LatentActor(nn.Module):
    def __init__(self, semantic_actor, C):
        super().__init__()
        require_invertible(C)
        self.semantic_actor = semantic_actor
        self.register_buffer('C', C.detach().clone())

    def forward(self, latent_and_curvature):
        decoded = torch.linalg.solve(self.C, latent_and_curvature[:, :2].T).T
        return self.semantic_actor(torch.cat([decoded, latent_and_curvature[:, 2:]], dim=1))
