"""LQR + bounded neural residual, certified by the quadratic SBC.

The residual action is ``eps * clamp(B(x), 0, 1) * tanh(r_theta(x, kappa))``.
Because the residual is bounded by ``eps * B(x)``, a quadratic SBC that
decreases for the LQR controller with a margin can absorb the whole residual
class; the certificate does not depend on the trained residual weights, only
on ``|tanh| <= 1``.  This makes the certified object a neural policy class
rather than the linear baseline.
"""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov

from . import core


def lqr_gains():
    a = core.SPEED * core.DT * core.THETA_SCALE / core.E_SCALE
    b = core.SPEED * core.DT * core.PHI_MAX / (core.WHEELBASE * core.THETA_SCALE)
    A = np.array([[1., a], [0., 1.]])
    B = np.array([[0.], [b]])
    R = np.array([[.1]])
    P_are = solve_discrete_are(A, B, np.eye(2), R)
    K = np.linalg.solve(R + B.T @ P_are @ B, B.T @ P_are @ A).ravel()
    A_cl = A - B @ K.reshape(1, 2)
    P = solve_discrete_lyapunov(A_cl.T, np.eye(2))
    initial = np.array([[e, t] for e in (-core.INIT_E / core.E_SCALE, core.INIT_E / core.E_SCALE)
                        for t in (-core.INIT_THETA / core.THETA_SCALE, core.INIT_THETA / core.THETA_SCALE)])
    peak = float(np.max(np.einsum('ij,jk,ik->i', initial, P, initial)))
    return K, P / peak * .99


class ResidualNet(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 1), nn.Tanh())

    def forward(self, observation):
        return self.net(observation)


class ResidualActor(nn.Module):
    """Deterministic deployment policy for the certified residual class."""

    def __init__(self, K, P, residual, eps=.5, flat_c=None):
        super().__init__()
        self.register_buffer('K', torch.as_tensor(K, dtype=torch.float32).reshape(1, 2))
        self.register_buffer('P', torch.as_tensor(P, dtype=torch.float32))
        self.residual = residual
        self.eps = float(eps)
        self.flat_c = None if flat_c is None else float(flat_c)

    def forward(self, observation):
        state = observation[:, :2]
        kappa = observation[:, 2:3] * core.KAPPA_SCALE
        u_lqr = -(state @ self.K.T) + torch.arctan(core.WHEELBASE * kappa) / core.PHI_MAX
        B = torch.einsum('ij,jk,ik->i', state, self.P, state)
        if self.flat_c is not None:
            B = (B - self.flat_c).clamp_min(0.)
        B = B.clamp(0., 1.)
        residual = self.residual(observation).flatten()
        return (u_lqr.flatten() + self.eps * B * residual).clamp(-1., 1.)


def lqr_only(observation, K):
    state = observation[:, :2]
    kappa = observation[:, 2:3] * core.KAPPA_SCALE
    u = -(state @ torch.as_tensor(K, dtype=torch.float32).reshape(1, 2).T) \
        + torch.arctan(core.WHEELBASE * kappa) / core.PHI_MAX
    return u.flatten().clamp(-1., 1.)


def save_residual(path, actor, metadata):
    torch.save(dict(format='lane_keeping_lqr_residual_v1',
                    K=actor.K.cpu(), P=actor.P.cpu(), eps=actor.eps,
                    flat_c=actor.flat_c, residual=actor.residual.state_dict(),
                    metadata=metadata), path)


def load_residual(path, device='cpu'):
    saved = torch.load(path, map_location=device, weights_only=False)
    if saved.get('format') != 'lane_keeping_lqr_residual_v1':
        raise ValueError('unexpected residual checkpoint')
    hidden = saved['residual']['net.0.weight'].shape[0]
    residual = ResidualNet(hidden)
    residual.load_state_dict(saved['residual'])
    actor = ResidualActor(saved['K'], saved['P'], residual, saved['eps'],
                          saved.get('flat_c')).to(device).eval()
    return actor, saved['metadata']
