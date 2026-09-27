"""Small data-free checks for the first scalar SBC-QP implementation."""

from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from Aebs.semantic_spvc.qp_constraint import DiscreteSBCConstraint
from Aebs.semantic_spvc.qp_layer import ScalarSBCQP


def test_scalar_qp_matches_brute_force_candidates():
    qp = ScalarSBCQP(slack_weight=25.0)
    examples = [
        (1.0, 1.0, 0.0),
        (0.0, -1.0, -0.5),
        (4.0, 2.0, -2.0),
        (-4.0, -2.0, -2.0),
        (0.7, 0.0, -0.4),
        (0.7, 0.0, 0.4),
    ]
    grid = np.linspace(-3.0, 3.0, 60001)
    for nominal, coefficient, rhs in examples:
        result = qp(
            torch.tensor([[nominal]]), torch.tensor([[coefficient]]),
            torch.tensor([[rhs]])
        )
        brute_cost = 0.5 * (grid - nominal) ** 2 + 12.5 * np.maximum(
            coefficient * grid - rhs, 0.0
        ) ** 2
        assert float(result.objective.item()) <= float(brute_cost.min()) + 1e-5
        assert -3.0 <= float(result.action.item()) <= 3.0
        assert float(result.linear_residual.item()) <= 1e-6


def test_scalar_qp_backpropagates_through_active_constraint():
    qp = ScalarSBCQP(slack_weight=10.0)
    nominal = torch.tensor([[1.0]], requires_grad=True)
    coefficient = torch.tensor([[1.0]], requires_grad=True)
    rhs = torch.tensor([[0.0]], requires_grad=True)
    result = qp(nominal, coefficient, rhs)
    gradients = torch.autograd.grad(result.action.sum(), (nominal, coefficient, rhs))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert np.isclose(float(gradients[0].item()), 1.0 / 11.0, atol=1e-6)


class QuadraticBarrier(nn.Module):
    def forward(self, states):
        return states[:, 1:2].square()


class TinyEnvironment:
    noise_bounds = (np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32))
    action_space = SimpleNamespace(
        low=np.array([-3.0], dtype=np.float32),
        high=np.array([3.0], dtype=np.float32),
    )

    @staticmethod
    def v_next(states, actions):
        return torch.cat([
            states[:, 0:1] - 0.05 * states[:, 1:2],
            states[:, 1:2] - 0.05 * actions,
        ], dim=1)


def test_discrete_constraint_linearization_matches_local_derivative():
    constraint = DiscreteSBCConstraint(
        QuadraticBarrier(), TinyEnvironment(), noise_bins=2
    )
    states = torch.tensor([[2.0, 1.0]])
    nominal = torch.tensor([[0.5]])
    affine = constraint.linearize(states, nominal)
    step = 1e-3
    plus = constraint.expected_next_value(states, nominal + step)
    minus = constraint.expected_next_value(states, nominal - step)
    finite_difference = (plus - minus) / (2 * step)
    assert torch.allclose(affine.coefficient, finite_difference, atol=1e-4)
    at_reference = affine.coefficient * nominal - affine.right_hand_side
    assert torch.allclose(at_reference, constraint.residual(states, nominal), atol=1e-6)
