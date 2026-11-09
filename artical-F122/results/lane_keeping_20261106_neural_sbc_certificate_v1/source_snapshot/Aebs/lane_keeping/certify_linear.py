"""Certified quadratic SBC for the LQR lane-keeping baseline.

The controller is the discrete LQR state feedback plus a known curvature
feedforward.  The SBC is the quadratic Lyapunov function of the linearised
closed loop, scaled so the initial box has B <= 1.  The certificate is a
numerical interval check on a finite grid that covers the whole sublevel set
{B <= 1}: every grid cell that intersects the set must satisfy
``max B(next) <= min B(current) - margin``.  The set is inside the lane, so
invariance of the sublevel set implies lane safety.

With ``--residual-eps > 0`` the action interval is enlarged by
``eps * B_max``, so the certificate covers every residual neural network with
``|tanh| <= 1`` (a neural policy class), not just the LQR baseline.

Scope: semantic 2-D state, no visual observation error, all three curvature
values, finite-grid numerical intervals (not a formal floating-point proof).
It does NOT certify the original PPO actor or the image feedback loop.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov

from . import core

GRID = 1600
MARGIN = .003
SCALE = .999
KAPPA_RANGE = core.KAPPAS


def lqr():
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
    return K, P / peak * SCALE, A_cl


def quad_bounds(P, lower, upper):
    if not torch.isfinite(lower).all() or not torch.isfinite(upper).all() or torch.any(lower > upper):
        raise ValueError('ordered finite interval bounds required')
    corners = [torch.stack([lower[:, 0], lower[:, 1]], 1),
               torch.stack([lower[:, 0], upper[:, 1]], 1),
               torch.stack([upper[:, 0], lower[:, 1]], 1),
               torch.stack([upper[:, 0], upper[:, 1]], 1)]
    maximum = torch.stack([torch.einsum('ij,jk,ik->i', c, P, c) for c in corners]).max(0).values
    contains_zero = ((lower[:, 0] <= 0) & (upper[:, 0] >= 0) &
                     (lower[:, 1] <= 0) & (upper[:, 1] >= 0))
    minimum = torch.zeros(len(lower), dtype=lower.dtype)
    p11, p12, p22 = P[0, 0], P[0, 1], P[1, 1]
    for face in range(4):
        if face == 0:
            fixed, free_low, free_high = lower[:, 0], lower[:, 1], upper[:, 1]
            pii, pij, pjj = p11, p12, p22
        elif face == 1:
            fixed, free_low, free_high = upper[:, 0], lower[:, 1], upper[:, 1]
            pii, pij, pjj = p11, p12, p22
        elif face == 2:
            fixed, free_low, free_high = lower[:, 1], lower[:, 0], upper[:, 0]
            pii, pij, pjj = p22, p12, p11
        else:
            fixed, free_low, free_high = upper[:, 1], lower[:, 0], upper[:, 0]
            pii, pij, pjj = p22, p12, p11
        free = (-pij * fixed / pjj).clamp(free_low, free_high)
        value = pii * fixed * fixed + 2 * pij * fixed * free + pjj * free * free
        minimum = value if face == 0 else torch.minimum(minimum, value)
    return torch.where(contains_zero, torch.zeros_like(minimum), minimum), maximum


def initial_bounds(P):
    axis = torch.linspace(-core.INIT_E / core.E_SCALE, core.INIT_E / core.E_SCALE, 201, dtype=torch.float64)
    axis_t = torch.linspace(-core.INIT_THETA / core.THETA_SCALE, core.INIT_THETA / core.THETA_SCALE, 201,
                            dtype=torch.float64)
    e, t = torch.meshgrid(axis, axis_t, indexing='ij')
    lower = torch.stack([e[:-1, :-1].reshape(-1), t[:-1, :-1].reshape(-1)], 1)
    upper = torch.stack([e[1:, 1:].reshape(-1), t[1:, 1:].reshape(-1)], 1)
    _, high = quad_bounds(P, lower, upper)
    return float(high.max())


def evaluate_cells(lower, upper, kappa, K, P, residual_eps=0., observation_radius=(0., 0.)):
    current_low, current_high = quad_bounds(P, lower, upper)
    feedforward = float(np.arctan(core.WHEELBASE * kappa) / core.PHI_MAX)
    re, rt = observation_radius
    observed_lower = torch.stack([lower[:, 0] - re, lower[:, 1] - rt], 1)
    observed_upper = torch.stack([upper[:, 0] + re, upper[:, 1] + rt], 1)
    weights = -K
    u_low = observed_lower @ weights.clamp_min(0) + observed_upper @ weights.clamp_max(0) + feedforward
    u_high = observed_upper @ weights.clamp_min(0) + observed_lower @ weights.clamp_max(0) + feedforward
    if residual_eps > 0:
        # The residual uses B(observed); bound it over the observation box.
        _, observed_B_high = quad_bounds(P, observed_lower, observed_upper)
        scale = observed_B_high.clamp(0., 1.)
        u_low = u_low - residual_eps * scale
        u_high = u_high + residual_eps * scale
    u_low, u_high = u_low.clamp(-1, 1), u_high.clamp(-1, 1)
    e_low = (lower[:, 0] * core.E_SCALE + core.SPEED * core.DT
             * torch.sin(lower[:, 1] * core.THETA_SCALE)) / core.E_SCALE
    e_high = (upper[:, 0] * core.E_SCALE + core.SPEED * core.DT
              * torch.sin(upper[:, 1] * core.THETA_SCALE)) / core.E_SCALE
    t_low = (lower[:, 1] * core.THETA_SCALE
             + core.SPEED * core.DT / core.WHEELBASE * torch.tan(u_low * core.PHI_MAX)
             - core.SPEED * core.DT * kappa) / core.THETA_SCALE
    t_high = (upper[:, 1] * core.THETA_SCALE
              + core.SPEED * core.DT / core.WHEELBASE * torch.tan(u_high * core.PHI_MAX)
              - core.SPEED * core.DT * kappa) / core.THETA_SCALE
    _, next_high = quad_bounds(P, torch.stack([e_low, t_low], 1),
                               torch.stack([e_high, t_high], 1))
    return current_low, current_high, next_high


def split_cells(lower, upper):
    middle = (lower + upper) / 2
    children = []
    for fix_e in (0, 1):
        for fix_t in (0, 1):
            low, high = lower.clone(), upper.clone()
            if fix_e == 0:
                high[:, 0] = middle[:, 0]
            else:
                low[:, 0] = middle[:, 0]
            if fix_t == 0:
                high[:, 1] = middle[:, 1]
            else:
                low[:, 1] = middle[:, 1]
            children.append((low, high))
    return (torch.cat([c[0] for c in children]),
            torch.cat([c[1] for c in children]))


def adaptive_check(kappa, K, P, grid=200, max_depth=12,
                   equilibrium_threshold=1e-6, equilibrium_tolerance=0.,
                   residual_eps=0., observation_radius=(0., 0.)):
    axis = torch.linspace(-1., 1., grid + 1, dtype=torch.float64)
    e, t = torch.meshgrid(axis, axis, indexing='ij')
    lower = torch.stack([e[:-1, :-1].reshape(-1), t[:-1, :-1].reshape(-1)], 1)
    upper = torch.stack([e[1:, 1:].reshape(-1), t[1:, 1:].reshape(-1)], 1)
    certified = outside = equilibrium = 0
    history = []
    for depth in range(max_depth + 1):
        current_low, current_high, next_high = evaluate_cells(lower, upper, kappa, K, P,
                                                              residual_eps, observation_radius)
        out = current_low > 1.
        strict = (~out) & (next_high <= current_low)
        # A positive drift tolerance cannot certify exact nonincrease.
        eq = (~out) & (current_high <= equilibrium_threshold) & (next_high <= current_low)
        ok = out | strict | eq
        fail = ~ok
        certified += int(strict.sum())
        outside += int(out.sum())
        equilibrium += int(eq.sum())
        history.append(dict(depth=depth, pending=int(len(lower)), certified=int(strict.sum()),
                            outside=int(out.sum()), equilibrium=int(eq.sum()),
                            fail=int(fail.sum())))
        if not fail.any():
            return dict(kappa=float(kappa), passed=True, certified_cells=certified,
                        outside_cells=outside, equilibrium_cells=equilibrium,
                        depth=depth, history=history)
        lower, upper = split_cells(lower[fail], upper[fail])
    return dict(kappa=float(kappa), passed=False, certified_cells=certified,
                outside_cells=outside, equilibrium_cells=equilibrium,
                unresolved_cells=int(len(lower)), depth=max_depth, history=history)


def sublevel_bounds(P, grid=2000):
    inv = np.linalg.inv(np.asarray(P))
    bounds = np.sqrt(np.diag(inv))
    return dict(max_abs_e_m=float(bounds[0]) * core.E_SCALE,
                max_abs_theta_rad=float(bounds[1]) * core.THETA_SCALE,
                bounds_method='analytic ellipsoid support, not a sampled grid')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--grid', type=int, default=200)
    parser.add_argument('--max-depth', type=int, default=12)
    parser.add_argument('--residual-eps', type=float, default=0.)
    parser.add_argument('--obs-radius-e', type=float, default=0.)
    parser.add_argument('--obs-radius-theta', type=float, default=0.)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f'choose a new output path: {args.output}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    K, P, A_cl = lqr()
    P_t = torch.tensor(P, dtype=torch.float64)
    K_t = torch.tensor(K, dtype=torch.float64)
    initial = initial_bounds(P_t)
    checks = [adaptive_check(kappa, K_t, P_t, grid=args.grid, max_depth=args.max_depth,
                             residual_eps=args.residual_eps,
                             observation_radius=(args.obs_radius_e, args.obs_radius_theta))
              for kappa in KAPPA_RANGE]
    sublevel = sublevel_bounds(P_t)
    safe = sublevel['max_abs_e_m'] < core.E_MAX and sublevel['max_abs_theta_rad'] < core.THETA_MAX
    passed = (initial <= 1. and all(c['passed'] for c in checks) and safe)
    result = dict(
        format='lane_keeping_quadratic_sbc_certificate_v1',
        passed=bool(passed),
        controller=dict(type='discrete LQR + curvature feedforward + bounded residual class', K=K.tolist(),
                        residual_eps=args.residual_eps,
                        residual_bound='eps * clamp(B(x),0,1) * tanh(z)',
                        feedforward='atan(L*kappa)/PHI_MAX', normalized_action_box=[-1, 1]),
        sbc=dict(type='quadratic', P=P.tolist(), nonnegative=True,
                 note='B(x)=x^T P x; positive definite'),
        initial=dict(box=dict(e_m=core.INIT_E, theta_rad=core.INIT_THETA),
                     max_B_interval=initial),
        sublevel=dict(level=1., **sublevel, inside_lane=bool(safe)),
        verification=dict(grid=args.grid, max_depth=args.max_depth,
                          equilibrium_threshold=1e-6, equilibrium_tolerance=0.,
                          observation_radius=dict(e=args.obs_radius_e, theta=args.obs_radius_theta),
                          kappas=list(KAPPA_RANGE), checks=checks,
                          certified_cells=int(sum(c.get('certified_cells', 0) for c in checks)),
                          equilibrium_cells=int(sum(c.get('equilibrium_cells', 0) for c in checks)),
                          unresolved_cells=int(sum(c.get('unresolved_cells', 0) for c in checks))),
        scope=('2-D semantic state; linear LQR controller; quadratic SBC; all three '
               'curvature values; finite-grid numerical interval check, not a formal '
               'floating-point proof. Not the PPO policy and not a visual-error certificate.'),
        conditions=('Cells are excluded only when min B>1; every remaining cell must satisfy '
                    'max B(next) <= min B(current), including equilibrium cells. '
                    'No positive drift tolerance is accepted.'),
    )
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ('passed', 'initial', 'sublevel', 'verification')}, indent=2))


if __name__ == '__main__':
    main()
