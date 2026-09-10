from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticEncoder(nn.Module):
    """Predict normalized distance and an image-dependent uncertainty scale."""

    def __init__(self, input_dim: int = 1024, hidden_dims: Tuple[int, ...] = (256, 64), min_scale: float = 1e-3):
        super().__init__()
        layers = []
        previous = input_dim
        for width in hidden_dims:
            layers.extend((nn.Linear(previous, width), nn.LayerNorm(width), nn.ReLU()))
            previous = width
        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(previous, 1)
        self.scale_head = nn.Linear(previous, 1)
        self.min_scale = float(min_scale)

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(image.reshape(image.shape[0], -1))
        mean = self.mean_head(features)
        scale = F.softplus(self.scale_head(features)) + self.min_scale
        return mean, scale

    def predict_interval(self, image: torch.Tensor, conformal_quantile: float):
        mean, scale = self(image)
        radius = float(conformal_quantile) * scale
        return mean, mean - radius, mean + radius
