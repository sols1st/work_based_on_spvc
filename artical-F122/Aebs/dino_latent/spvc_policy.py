"""Adapter from the frozen-DINO latent PPO to the original SPVC interface."""

import math
import torch
import torch.nn as nn
from stable_baselines3 import PPO

from Aebs.dino_latent.models import PhysicalToLatent
from Combined_network.model import CombinedPolicyNetwork


class DinoLatentSPVCPolicy(nn.Module):
    """q_psi(distance)+speed -> latent PPO for physical-grid SBC training."""

    def __init__(self, q_model, controller_net, distance_scale_m, latent_dimension):
        super().__init__()
        self.q_model = q_model
        self.controller_net = controller_net
        self.distance_scale_m = float(distance_scale_m)
        self.state_distance_scale_m = self.distance_scale_m
        self.latent_dimension = int(latent_dimension)
        for parameter in self.q_model.parameters():
            parameter.requires_grad_(False)
        self.q_model.eval()

    @classmethod
    def from_checkpoints(cls, representation_checkpoint, controller_checkpoint, device):
        checkpoint = torch.load(representation_checkpoint, map_location=device)
        latent_dimension = int(checkpoint["latent_dimension"])
        q_model = PhysicalToLatent(physical_dim=1, latent_dim=latent_dimension).to(device)
        q_model.load_state_dict(checkpoint["physical_to_latent_state_dict"])
        q_model.eval()
        ppo = PPO.load(controller_checkpoint, device=device)
        ppo.policy.eval()
        controller_net = CombinedPolicyNetwork(
            ppo.policy.mlp_extractor.policy_net,
            ppo.policy.action_net,
        ).to(device)
        return cls(
            q_model=q_model,
            controller_net=controller_net,
            distance_scale_m=float(checkpoint["distance_scale_m"]),
            latent_dimension=latent_dimension,
        ).to(device)

    def train(self, mode=True):
        super().train(mode)
        self.q_model.eval()
        return self

    def controller_input(self, z, physical_state):
        del z
        if physical_state.ndim != 2 or physical_state.shape[1] != 2:
            raise ValueError("physical_state must have shape [batch, 2]")
        # Original VT coordinates use env.std1; q uses its own checkpoint scale.
        q_distance = physical_state[:, 0:1] * (
            self.state_distance_scale_m / self.distance_scale_m
        )
        latent = self.q_model(q_distance)
        return torch.cat([latent, physical_state[:, 1:2]], dim=1)

    def set_state_distance_scale(self, scale_m):
        scale_m = float(scale_m)
        if not math.isfinite(scale_m) or scale_m <= 0:
            raise ValueError("state distance scale must be positive and finite")
        if not math.isfinite(self.distance_scale_m) or self.distance_scale_m <= 0:
            raise ValueError("representation distance scale must be positive and finite")
        self.state_distance_scale_m = scale_m

    def forward(self, z, physical_state):
        return self.controller_net(self.controller_input(z, physical_state))

    def forward_latent(self, latent, speed):
        if speed.ndim == 1:
            speed = speed.unsqueeze(1)
        return self.controller_net(torch.cat([latent, speed], dim=1))

    def metadata(self):
        return {
            "variant": "dino_safety_latent_spvc_no_cp",
            "vt_path": "physical distance -> q_psi latent -> PPO",
            "deployment_path": "image -> frozen DINO -> projection -> PPO",
            "latent_dimension": self.latent_dimension,
            "distance_scale_m": self.distance_scale_m,
            "state_distance_scale_m": self.state_distance_scale_m,
            "q_input_scale_ratio": self.state_distance_scale_m / self.distance_scale_m,
            "excluded": ["conformal prediction", "recoverable set", "new verifier"],
        }
