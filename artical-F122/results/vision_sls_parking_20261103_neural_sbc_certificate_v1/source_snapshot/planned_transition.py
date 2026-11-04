"""Differentiable closed-loop transition for future finite-horizon SBC work.

Continuous state is [physical x(4), observer estimate(4)], with discrete
reference index and time supplied separately. This is NOT a verifier/certificate.
Initial image-dependent reference-selection guards must also be verified.
"""
import copy
import torch
from torch import nn
from .core import C, RENDER_OFFSET, DT, HORIZON, ACTION_HIGH
from .planning_feedback import RESIDUAL_SCALE


class PlannedTransition(nn.Module):
    def __init__(self, bank, policy=None):
        super().__init__()
        for name, value in [('reference_states', bank.states), ('reference_actions', bank.actions),
                            ('feedback', bank.k), ('observer', bank.l), ('goals', bank.goals),
                            ('measurement', C), ('offset', RENDER_OFFSET), ('action_limits', ACTION_HIGH),
                            ('residual_scale', RESIDUAL_SCALE)]:
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float64))
        self.actor = None if policy is None else nn.Sequential(
            copy.deepcopy(policy.policy.mlp_extractor.policy_net), copy.deepcopy(policy.policy.action_net))

    def _value(self, name, state):
        return getattr(self, name).to(dtype=state.dtype, device=state.device)

    def action(self, state, reference, time):
        if reference not in range(15) or time not in range(HORIZON):
            raise ValueError('valid discrete reference and time required')
        estimate = state[:, 4:]
        ref = self._value('reference_states', state)[reference]
        limits = self._value('action_limits', state)
        base = self._value('reference_actions', state)[reference, time]-(
            estimate-ref[time])@self._value('feedback', state)[reference, time].T
        base = torch.maximum(torch.minimum(base, limits), -limits)
        if self.actor is None:
            return base
        features = torch.cat([estimate-ref[time], ref[min(time+4, HORIZON), :2]-estimate[:, :2],
                              ref[min(time+8, HORIZON), :2]-estimate[:, :2], base/limits,
                              self._value('goals', state)[reference]-estimate[:, :2],
                              state.new_full((len(state), 1), time/HORIZON)], dim=1)
        parameter = next(self.actor.parameters())
        residual = self.actor(features.to(dtype=parameter.dtype)).to(dtype=state.dtype).clamp(-1, 1)
        return torch.maximum(torch.minimum(base+residual*self._value('residual_scale', state), limits), -limits)

    @staticmethod
    def dynamics(x, u, disturbance=None):
        def ode(q):
            velocity = torch.stack([q[:, 3]*torch.cos(q[:, 2]), q[:, 3]*torch.sin(q[:, 2]), u[:, 0], u[:, 1]], 1)
            return velocity if disturbance is None else velocity+.01*disturbance
        k1 = ode(x); k2 = ode(x+DT*k1/2); k3 = ode(x+DT*k2/2); k4 = ode(x+DT*k3)
        return x+DT*(k1+2*k2+2*k3+k4)/6

    def forward(self, state, reference, time, observation_error=None, process_noise=None):
        if state.ndim != 2 or state.shape[1] != 8:
            raise ValueError('batch of physical state plus observer memory required')
        u = self.action(state, reference, time)
        x, estimate = state[:, :4], state[:, 4:]
        error = state.new_zeros((len(state), 3)) if observation_error is None else observation_error
        if error.shape != (len(state), 3):
            raise ValueError('one three-dimensional visual error per state required')
        # Both visual measurements refer to x_t, then the estimate is predicted
        # with u_t; using x_(t+1) here would silently remove the one-frame delay.
        innovation = (x-estimate)@self._value('measurement', state).T+error
        corrected = estimate+innovation@self._value('observer', state)[reference, time+1].T
        if process_noise is not None:
            if process_noise.shape != x.shape:
                raise ValueError('one four-dimensional process disturbance per state required')
        physical_next = self.dynamics(x, u, process_noise)
        return torch.cat([physical_next, self.dynamics(corrected, u)], dim=1)
