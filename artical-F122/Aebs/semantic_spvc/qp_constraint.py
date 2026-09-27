"""A local affine approximation of the original discrete AEBS SBC condition.

The midpoint expectation below uses the same uniform disturbance box as the
original VT verifier. It is differentiable, but it is a quadrature estimate,
not the verifier's interval upper bound. Always check a QP action again using
the unlinearized expectation and, when required, the original IBP upper bound.
"""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class AffineSBCConstraint:
    coefficient: torch.Tensor
    right_hand_side: torch.Tensor
    current_value: torch.Tensor
    nominal_expected_value: torch.Tensor


class DiscreteSBCConstraint(nn.Module):
    def __init__(self, barrier: nn.Module, env, noise_bins: int = 10, margin: float = 0.0):
        super().__init__()
        if noise_bins < 1:
            raise ValueError("noise_bins must be positive")
        if margin < 0:
            raise ValueError("margin must be nonnegative")
        self.barrier = barrier
        self.env = env
        self.margin = float(margin)
        low = torch.as_tensor(env.noise_bounds[0], dtype=torch.float32)
        high = torch.as_tensor(env.noise_bounds[1], dtype=torch.float32)
        axes = [
            low[i] + (torch.arange(noise_bins, dtype=torch.float32) + 0.5)
            * (high[i] - low[i]) / noise_bins
            for i in range(2)
        ]
        mesh = torch.meshgrid(*axes, indexing="ij")
        self.register_buffer("noise_midpoints", torch.stack(mesh, dim=-1).reshape(-1, 2))

    @staticmethod
    def _validate(states: torch.Tensor, actions: torch.Tensor) -> None:
        if states.ndim != 2 or states.shape[1] != 2:
            raise ValueError("states must have shape [batch, 2]")
        if actions.ndim != 2 or actions.shape != (states.shape[0], 1):
            raise ValueError("actions must have shape [batch, 1]")

    def expected_next_value(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Differentiable uniform-noise midpoint quadrature, shape [batch, 1]."""
        self._validate(states, actions)
        next_state = self.env.v_next(states, actions)
        offsets = self.noise_midpoints.to(device=states.device, dtype=states.dtype)
        disturbed = next_state[:, None, :] + offsets[None, :, :]
        values = self.barrier(disturbed.reshape(-1, 2))
        return values.reshape(states.shape[0], -1).mean(dim=1, keepdim=True)

    def residual(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Positive means the unlinearized midpoint decrease condition fails."""
        self._validate(states, actions)
        return self.expected_next_value(states, actions) - self.barrier(states) + self.margin

    def linearize(self, states: torch.Tensor, nominal_actions: torch.Tensor) -> AffineSBCConstraint:
        """Return c*a <= rhs, linearized at the nominal action.

        create_graph keeps gradients through c and rhs available to later
        actor/SBC training. The QP code does not treat this approximation as
        a formal upper bound on the nonlinear expectation.
        """
        self._validate(states, nominal_actions)
        with torch.enable_grad():
            low = float(self.env.action_space.low[0]) + 1e-4
            high = float(self.env.action_space.high[0]) - 1e-4
            reference = nominal_actions.clamp(min=low, max=high)
            if not reference.requires_grad:
                reference = reference.detach().requires_grad_(True)
            nominal_expected = self.expected_next_value(states, reference)
            coefficient = torch.autograd.grad(
                nominal_expected.sum(), reference, create_graph=True
            )[0]
            current = self.barrier(states)
            rhs = current - self.margin - nominal_expected + coefficient * reference
        return AffineSBCConstraint(coefficient, rhs, current, nominal_expected)
