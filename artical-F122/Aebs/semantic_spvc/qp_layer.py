"""Exact, piecewise differentiable solution of the one-action soft SBC QP.

min 0.5*(a-a_nom)^2 + 0.5*rho*slack^2
s.t. coefficient*a - slack <= rhs, action_low <= a <= action_high,
     slack >= 0.

This is a scalar backend for AEBS. The interface can later be replaced by a
multi-action differentiable QP solver without changing constraint generation.
"""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class QPSolution:
    action: torch.Tensor
    slack: torch.Tensor
    objective: torch.Tensor
    linear_residual: torch.Tensor


class ScalarSBCQP(nn.Module):
    def __init__(self, action_low: float = -3.0, action_high: float = 3.0,
                 slack_weight: float = 100.0, slope_tolerance: float = 1e-9):
        super().__init__()
        if action_low >= action_high:
            raise ValueError("action_low must be less than action_high")
        if slack_weight <= 0:
            raise ValueError("slack_weight must be positive")
        self.action_low = float(action_low)
        self.action_high = float(action_high)
        self.slack_weight = float(slack_weight)
        self.slope_tolerance = float(slope_tolerance)

    def forward(self, nominal: torch.Tensor, coefficient: torch.Tensor,
                rhs: torch.Tensor) -> QPSolution:
        if nominal.ndim != 2 or nominal.shape[1] != 1:
            raise ValueError("nominal must have shape [batch, 1]")
        if coefficient.shape != nominal.shape or rhs.shape != nominal.shape:
            raise ValueError("coefficient and rhs must match nominal shape")
        if not (torch.isfinite(nominal).all() and torch.isfinite(coefficient).all()
                and torch.isfinite(rhs).all()):
            raise ValueError("QP inputs must be finite")

        low, high = self.action_low, self.action_high
        active_slope = coefficient.abs() > self.slope_tolerance
        safe_coefficient = torch.where(active_slope, coefficient, torch.ones_like(coefficient))
        boundary = rhs / safe_coefficient

        # On the no-slack side, the optimum is nominal projected to the
        # intersection of the action interval and coefficient*a <= rhs.
        feasible_low = torch.where(coefficient < -self.slope_tolerance,
                                   boundary, torch.full_like(boundary, low))
        feasible_high = torch.where(coefficient > self.slope_tolerance,
                                    boundary, torch.full_like(boundary, high))
        feasible_low = feasible_low.clamp(min=low)
        feasible_high = feasible_high.clamp(max=high)
        zero_slope_feasible = active_slope | (rhs >= 0)
        no_slack_valid = zero_slope_feasible & (feasible_low <= feasible_high)
        no_slack = torch.minimum(torch.maximum(nominal, feasible_low), feasible_high)

        # On the active-slack side, differentiate the quadratic objective.
        slack_stationary = (
            nominal + self.slack_weight * coefficient * rhs
        ) / (1.0 + self.slack_weight * coefficient.square())
        slack_stationary = slack_stationary.clamp(min=low, max=high)

        candidates = torch.cat([
            nominal.clamp(min=low, max=high),
            no_slack,
            slack_stationary,
            boundary.clamp(min=low, max=high),
            torch.full_like(nominal, low),
            torch.full_like(nominal, high),
        ], dim=1)
        candidate_slack = torch.relu(coefficient * candidates - rhs)
        costs = 0.5 * (candidates - nominal).square() + (
            0.5 * self.slack_weight * candidate_slack.square()
        )
        validity = torch.cat([
            torch.ones_like(no_slack_valid), no_slack_valid,
            *[torch.ones_like(no_slack_valid) for _ in range(4)],
        ], dim=1)
        costs = torch.where(validity, costs, torch.full_like(costs, float("inf")))
        selected = costs.argmin(dim=1, keepdim=True)
        action = candidates.gather(1, selected)
        slack = torch.relu(coefficient * action - rhs)
        objective = 0.5 * (action - nominal).square() + 0.5 * self.slack_weight * slack.square()
        return QPSolution(action, slack, objective, coefficient * action - rhs - slack)
