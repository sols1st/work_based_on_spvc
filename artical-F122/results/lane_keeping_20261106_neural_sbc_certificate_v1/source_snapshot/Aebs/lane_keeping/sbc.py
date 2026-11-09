"""Nonnegative neural SBC for lane keeping and its training losses.

This module trains a candidate over the original lane domain
``|e| <= 0.85 m``, ``|theta| <= 0.30 rad``. Training alone does not certify
invariance. The verified constructive ReLU certificate is implemented
separately in ``certify_neural.py``; this Tanh candidate remains unverified.

The actor sees a noisy observation ``(e_hat, theta_hat, kappa)``; the SBC is a
function of the true normalized state and is checked for all observation
errors in the calibrated box.  The SBC is never used as an online filter.
"""
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import core

# Full safe domain in normalized coordinates.  The safe region is not a box:
# the SBC itself carves out the states from which the fixed policy preserves
# safety; B >= 20 on the true unsafe boundary and B <= 1 on the initial box.
D_E = 1.0
D_THETA = 1.0
INIT_E_N = core.INIT_E / core.E_SCALE
INIT_THETA_N = core.INIT_THETA / core.THETA_SCALE
UNSAFE_LEVEL = 20.
INIT_LEVEL = 1.
INIT_TARGET = .5
UNSAFE_TARGET = 25.
DECREASE_MARGIN = 0.0


class LaneSBC(nn.Module):
    def __init__(self, features=(2, 16, 16, 1)):
        super().__init__()
        self.features = tuple(int(v) for v in features)
        self.layers = nn.ModuleList([nn.Linear(a, b) for a, b in zip(features[:-1], features[1:])])

    def forward(self, state):
        for layer in self.layers[:-1]:
            state = torch.tanh(layer(state))
        return F.relu(self.layers[-1](state))


def save_sbc(path, model, metadata):
    torch.save(dict(format='lane_keeping_sbc_v1', features=model.features,
                    state_dict=model.state_dict(), metadata=metadata), path)


def load_sbc(path, device='cpu'):
    saved = torch.load(path, map_location=device, weights_only=False)
    if saved.get('format') != 'lane_keeping_sbc_v1':
        raise ValueError('unexpected lane-keeping SBC checkpoint')
    model = LaneSBC(saved['features'])
    model.load_state_dict(saved['state_dict'])
    return model.to(device).eval(), saved['metadata']


def initial_mask(states):
    return (np.abs(states[:, 0]) <= INIT_E_N) & (np.abs(states[:, 1]) <= INIT_THETA_N)


def inside_domain(states):
    return (np.abs(states[:, 0]) <= D_E) & (np.abs(states[:, 1]) <= D_THETA)


def outside_domain(states):
    return ~inside_domain(states)


def sample_domain(rng, n):
    return rng.uniform([-D_E, -D_THETA], [D_E, D_THETA], size=(int(n), 2))


def sample_unsafe(rng, n, width=.05):
    """Points inside the unsafe boundary strips (|e_n| > 1 or |theta_n| > 1)."""
    n = int(n)
    side = rng.integers(0, 4, size=n)
    points = np.empty((n, 2), dtype=np.float64)
    points[:, 0] = rng.uniform(-1.05, 1.05, n)
    points[:, 1] = rng.uniform(-1.05, 1.05, n)
    vertical = side < 2
    points[vertical, 0] = rng.uniform(1.0, 1.0 + width, int(vertical.sum())) * np.where(side[vertical] == 0, 1., -1.)
    points[~vertical, 1] = rng.uniform(1.0, 1.0 + width, int((~vertical).sum())) * np.where(side[~vertical] == 2, 1., -1.)
    return points


def sample_initial(rng, n):
    return rng.uniform([-INIT_E_N, -INIT_THETA_N], [INIT_E_N, INIT_THETA_N], size=(int(n), 2))


def sample_outside(rng, n, limit=1.05):
    """Uniform points outside the initial box (used only for diagnostics)."""
    points = rng.uniform([-limit, -limit], [limit, limit], size=(int(n), 2))
    return points


def sample_errors(rng, n, radius, corners=True):
    """Interior samples plus the four corners; shape [n, k, 2]."""
    errors = [rng.uniform(-radius, radius, size=(int(n), 2))]
    if corners:
        signs = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=np.float64)
        errors.extend(np.broadcast_to(radius * signs[i], (int(n), 2)) for i in range(4))
    return np.stack(errors, axis=1)


def next_states_for_errors(actor, states, kappas, errors):
    """Actor + true dynamics for every observation-error sample.

    ``states`` and ``errors`` are normalized; ``kappas`` are normalized
    curvature inputs.  Returns normalized next states [n, k, 2].
    """
    n, k, _ = errors.shape
    observed = states[:, None, :] + errors
    observed = torch.cat([observed, kappas[:, None, None].expand(n, k, 1)], dim=2)
    actions = actor(observed.reshape(n * k, 3)).reshape(n, k)
    e = states[:, 0, None] * core.E_SCALE
    theta = states[:, 1, None] * core.THETA_SCALE
    phi = actions.clamp(-1., 1.) * core.PHI_MAX
    kappa = kappas[:, None] * core.KAPPA_SCALE
    e_next = e + core.SPEED * core.DT * torch.sin(theta)
    theta_next = (theta + core.SPEED * core.DT / core.WHEELBASE * torch.tan(phi)
                  - core.SPEED * core.DT * kappa)
    e_next = e_next.expand_as(theta_next)
    return torch.stack([e_next / core.E_SCALE, theta_next / core.THETA_SCALE], dim=2)


