"""DINOv2 patch-token spatial readout for the lane observation.

A global CLS feature loses fine lateral position.  This head keeps the frozen
16x16 patch grid and uses a spatial soft-argmax, then a small linear map to
normalized (e, theta).  It is still a low-dimensional observation of the same
frozen backbone, not an end-to-end controller.
"""
import torch
from torch import nn

from ..dino_latent.backbone import FrozenDinoV2


class PatchProjection(nn.Module):
    def __init__(self, feature_dim=384, hidden=64, grid=16):
        super().__init__()
        self.grid = int(grid)
        self.conv = nn.Sequential(
            nn.Conv2d(feature_dim, hidden, 1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, 1, 1))
        self.out = nn.Linear(2, 2)
        axis = torch.linspace(-1., 1., self.grid)
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        self.register_buffer('coords', torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1))

    def forward(self, patches):
        # patches: [B, grid*grid, feature_dim]
        batch = patches.shape[0]
        x = patches.reshape(batch, self.grid, self.grid, -1).permute(0, 3, 1, 2)
        logits = self.conv(x).reshape(batch, 1, -1)
        weights = torch.softmax(logits, dim=2)
        expected = (weights @ self.coords).reshape(batch, 2)
        return self.out(expected)


class PatchImageObserver(nn.Module):
    def __init__(self, checkpoint_path, device='cpu', dino_repo=None, dino_weights=None):
        super().__init__()
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if checkpoint.get('format') != 'lane_keeping_dino_patch_projection_v1':
            raise ValueError('unexpected patch projection checkpoint')
        self.projection = PatchProjection(checkpoint['feature_dim'], checkpoint['hidden'],
                                          checkpoint['grid'])
        self.projection.load_state_dict(checkpoint['projection'])
        self.backbone = FrozenDinoV2(dino_repo or checkpoint['dino_repo'],
                                     dino_weights or checkpoint['dino_weights'],
                                     device, model_name=checkpoint['dino_model'])
        self.scales = dict(checkpoint['scales'])
        self.eval().requires_grad_(False)

    @torch.no_grad()
    def features(self, images):
        if images.dtype == torch.uint8:
            images = images.float() / 255.
        if images.ndim == 4 and images.shape[-1] == 3:
            images = images.permute(0, 3, 1, 2)
        return self.backbone.backbone.forward_features(
            self.backbone.preprocess(images))['x_norm_patchtokens']

    def forward(self, images):
        return self.projection(self.features(images))


def save_patch_checkpoint(path, projection, dino_repo, dino_weights, dino_model, scales, metadata):
    torch.save(dict(
        format='lane_keeping_dino_patch_projection_v1',
        feature_dim=384, hidden=int(projection.conv[0].out_channels), grid=int(projection.grid),
        projection=projection.state_dict(),
        dino_repo=str(dino_repo), dino_weights=str(dino_weights), dino_model=str(dino_model),
        scales=dict(scales), metadata=metadata), path)
