"""PPO actor followed by the existing differentiable SBC-QP output layer."""

import weakref

import torch
from torch import nn

from Aebs.semantic_spvc.qp_constraint import DiscreteSBCConstraint
from Aebs.semantic_spvc.qp_layer import ScalarSBCQP


class SemanticSPVCQPPolicy(nn.Module):
    """Keep the original actor trainable while using the current SBC in QP.

    The SBC is held by weak reference, not registered as a child of the actor:
    this prevents its weights from entering the PPO optimizer/state_dict.
    """

    def __init__(self, base_policy, barrier, env, slack_weight=100.0):
        super().__init__()
        self.base_policy = base_policy
        object.__setattr__(self, "_barrier_ref", weakref.ref(barrier))
        constraint = DiscreteSBCConstraint(barrier, env, noise_bins=10).to(next(barrier.parameters()).device)
        object.__setattr__(self, "_constraint", constraint)
        self.qp_layer = ScalarSBCQP(
            float(env.action_space.low[0]), float(env.action_space.high[0]), slack_weight
        )

    @property
    def controller_net(self):
        return self.base_policy.controller_net

    def controller_input(self, z, state):
        return self.base_policy.controller_input(z, state)

    def nominal_action(self, z, state):
        return self.base_policy(z, state)

    def forward(self, z, state):
        if self._barrier_ref() is None:
            raise RuntimeError("SBC reference was released")
        nominal = self.nominal_action(z, state)
        affine = self._constraint.linearize(state, nominal)
        return self.qp_layer(nominal, affine.coefficient, affine.right_hand_side).action

    def forward_image(self, image, speed):
        mean, _ = self.base_policy.encode_image(image)
        if speed.ndim == 1:
            speed = speed.unsqueeze(1)
        state = torch.cat([mean, speed], dim=1)
        latent = torch.zeros((len(state), 4), dtype=state.dtype, device=state.device)
        return self.forward(latent, state)

    def metadata(self):
        result = self.base_policy.metadata()
        result["output_layer"] = "differentiable_scalar_sbc_qp"
        result["qp_constraint"] = "10x10_midpoint_linearization_soft_slack"
        return result
