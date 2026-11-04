"""Time-conditioned nonnegative SBC candidates for the 8D planned closed loop.

Unsafe lower separation is enforced geometrically, rather than learned at
random unsafe points. The remaining initial/drift conditions need verification.
"""
import numpy as np
import torch
from torch import nn
from .core import STATE_LOW, STATE_HIGH, OBSTACLES, HORIZON


def risk(state):
    """At least 20 on the original unsafe set, including its tolerances.

    Margin widths only shape the certificate; they do not change task sets.
    Includes all four state bounds and sampled point-center circular obstacles.
    """
    x = state[..., :4]
    low, high = x.new_tensor(STATE_LOW)-.001, x.new_tensor(STATE_HIGH)+.001
    widths = x.new_tensor([.01, .01, .05, .02])
    bounds = torch.cat([(x-low)/widths, (high-x)/widths], dim=-1)
    circles = x.new_tensor(OBSTACLES)
    distance_squared = ((x[..., None, :2]-circles[:, :2])**2).sum(-1)
    collision = (distance_squared-circles[:, 2]**2-.0001)/(2*circles[:, 2]*.01+.01**2)
    margin = torch.cat([bounds, collision], dim=-1).min(dim=-1, keepdim=True).values
    return 20*torch.relu(1-margin)


class PlannedBarrier(nn.Module):
    def __init__(self, bank):
        super().__init__()
        self.register_buffer('references', torch.tensor(bank.states, dtype=torch.float32))
        self.net = nn.Sequential(nn.Linear(24, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(),
                                 nn.Linear(64, 1), nn.Softplus())
        nn.init.constant_(self.net[-2].bias, -2.)

    def features(self, state, reference, time):
        ref = self.references[reference, time].to(state)
        scale = state.new_tensor([1., 1., np.pi, 1.5]*2)
        center = torch.cat([ref, ref], dim=-1)
        onehot = torch.nn.functional.one_hot(reference, 15).to(state)
        return torch.cat([(state-center)/scale, time[:, None].to(state)/HORIZON, onehot], dim=-1)

    def forward(self, state, reference, time):
        if (state.ndim != 2 or state.shape[1] != 8 or reference.shape != (len(state),)
                or time.shape != (len(state),) or (reference < 0).any() or (reference >= 15).any()
                or (time < 0).any() or (time > HORIZON).any()):
            raise ValueError('8D state, valid reference index and finite-horizon time required')
        # A finite-horizon clock budget permits strict slack along safe paths.
        # It stays nonnegative through H and does not weaken unsafe separation.
        return risk(state)+self.net(self.features(state, reference, time))+.01*(HORIZON-time[:,None]).to(state)


@torch.no_grad()
def finite_horizon_barrier(transition, state, reference, time):
    """Exact deterministic lookahead formula, numerically evaluated here.

    B_t(s)=max_{k=t..H} risk(x_k). Analytically nonnegative, >=20 unsafe,
    and nonincreasing along this SAME deterministic F, by suffix inclusion.
    This does not certify a nonzero observation-error model or its initial set.
    """
    if state.ndim != 2 or state.shape[1] != 8 or reference not in range(15) or time not in range(HORIZON+1):
        raise ValueError('8D state and valid finite-horizon discrete state required')
    value = risk(state)
    for t in range(time, HORIZON):
        state = transition(state, reference, t)
        value = torch.maximum(value, risk(state))
    return value
