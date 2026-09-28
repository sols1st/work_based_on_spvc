"""Small verification-friendly networks around the frozen DINO embedding."""

import torch
import torch.nn as nn


class SafetyProjection(nn.Module):
    """Project 384-D DINO features to a 32-D safety-relevant latent."""

    def __init__(self, feature_dim=384, hidden_dim=128, latent_dim=32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, feature):
        return self.network(feature)


class SafetyDecoder(nn.Module):
    """Auxiliary-only head; it is not the deployed semantic representation."""

    def __init__(self, latent_dim=32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, latent):
        return self.network(latent)


class PhysicalToLatent(nn.Module):
    """Verification surrogate q_psi: normalized physical distance -> latent."""

    def __init__(self, physical_dim=1, latent_dim=32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(physical_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )

    def forward(self, physical_state):
        return self.network(physical_state)


class LatentController(nn.Module):
    """Controller interface for [32-D safety latent, speed]."""

    def __init__(self, latent_dim=32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim + 1, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1), nn.Tanh(),
        )

    def forward(self, latent, speed):
        if speed.ndim == 1:
            speed = speed.unsqueeze(1)
        return 3.0 * self.network(torch.cat([latent, speed], dim=1))
