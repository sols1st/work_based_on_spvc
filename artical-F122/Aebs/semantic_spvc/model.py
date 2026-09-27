"""Semantic-only observation front end for the original SPVC pipeline.

The original AEBS policy is

    (state, latent) -> cGAN image -> state_net -> PPO actor.

This minimal variant is

    real image -> semantic encoder -> PPO actor.

The original VT learner and verifier already operate on a two-dimensional
semantic state grid.  Their ``forward(z, state)`` path therefore consumes that
semantic state directly and ignores the obsolete cGAN latent ``z``.  No SBC
loss, transition noise, grid, threshold, or verifier setting is changed.
"""

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from stable_baselines3 import PPO

from Aebs.semantic.model import SemanticEncoder
from Combined_network.model import CombinedPolicyNetwork


class SemanticSPVCPolicy(nn.Module):
    """Frozen image-to-distance encoder followed by the original PPO actor."""

    def __init__(
        self,
        semantic_encoder: SemanticEncoder,
        controller_net: CombinedPolicyNetwork,
        distance_scale_m: float,
    ) -> None:
        super().__init__()
        self.semantic_encoder = semantic_encoder
        self.controller_net = controller_net
        self.distance_scale_m = float(distance_scale_m)
        for parameter in self.semantic_encoder.parameters():
            parameter.requires_grad = False
        self.semantic_encoder.eval()

    @classmethod
    def from_checkpoints(
        cls,
        semantic_checkpoint: str,
        controller_checkpoint: str,
        device: torch.device,
    ) -> "SemanticSPVCPolicy":
        checkpoint = torch.load(semantic_checkpoint, map_location=device)
        encoder = SemanticEncoder().to(device)
        encoder.load_state_dict(checkpoint["model_state_dict"])
        encoder.eval()

        ppo = PPO.load(controller_checkpoint, device=device)
        ppo.policy.eval()
        controller = CombinedPolicyNetwork(
            ppo.policy.mlp_extractor.policy_net,
            ppo.policy.action_net,
        ).to(device)
        return cls(
            semantic_encoder=encoder,
            controller_net=controller,
            distance_scale_m=float(checkpoint["distance_scale_m"]),
        ).to(device)

    def train(self, mode: bool = True):
        super().train(mode)
        # The version requested here changes only the observation front end;
        # the already-trained semantic converter remains fixed.
        self.semantic_encoder.eval()
        return self

    def controller_input(self, z: torch.Tensor, semantic_state: torch.Tensor) -> torch.Tensor:
        """Return ``[normalized semantic distance, speed]`` for VT code.

        ``z`` is retained only to preserve the original SPVC call signature.
        It represented cGAN appearance noise and is intentionally unused after
        cGAN removal.
        """
        del z
        if semantic_state.ndim != 2 or semantic_state.shape[1] != 2:
            raise ValueError("semantic_state must have shape [batch, 2]")
        return semantic_state

    def forward(self, z: torch.Tensor, semantic_state: torch.Tensor) -> torch.Tensor:
        return self.controller_net(self.controller_input(z, semantic_state))

    def encode_image(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert a real 32x32 image to normalized distance and scale."""
        return self.semantic_encoder(image)

    def forward_image(self, image: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
        """Run the deployable image -> semantics -> PPO path."""
        mean, _ = self.encode_image(image)
        if speed.ndim == 1:
            speed = speed.unsqueeze(1)
        if speed.ndim != 2 or speed.shape[1] != 1:
            raise ValueError("speed must have shape [batch] or [batch, 1]")
        observation = torch.cat([mean, speed], dim=1)
        return self.controller_net(observation)

    def metadata(self) -> dict:
        return {
            "variant": "spvc_semantic_only",
            "observation_path": "real_image -> semantic_encoder -> original_ppo",
            "removed_modules": ["cgan", "legacy_state_net"],
            "unchanged_modules": [
                "ppo_architecture",
                "sbc_architecture",
                "sbc_losses",
                "transition_noise",
                "vt_grid",
                "ibp_verifier",
                "verification_thresholds",
            ],
            "distance_scale_m": self.distance_scale_m,
        }
