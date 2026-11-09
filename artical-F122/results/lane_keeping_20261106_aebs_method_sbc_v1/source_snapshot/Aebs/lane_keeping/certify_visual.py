"""Visual-error certificate for the LQR + bounded neural residual class.

SBC: flat-bottom quadratic
    B(x) = relu(x^T P x - c),  c = 0.5
so the origin is an exact zero plateau; observation noise cannot create a
spurious B increase there.  Certified set: {B <= initial_level}.  The
controller is LQR plus eps * clamp(B(observed),0,1) * tanh(residual), eps=.5.
Observation errors are bounded by the patch-DINO split-CP box.

The check is a numerical adaptive interval audit over the true 2-D state and
the observation-error box.  It is not a formal floating-point proof.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from . import core
from .certify_linear import lqr, quad_bounds, initial_bounds

C = .5
EPS = .5
INITIAL_LEVEL = .49  # initial B_quad <= .99 => B_flat <= .49
GRID = 120
MAX_DEPTH = 12


def flat_bounds(quad_low, quad_high, c=C):
    return (quad_low - c).clamp_min(0.), (quad_high - c).clamp_min(0.)


def evaluate(lower, upper, kappa, K, P, radius, eps=EPS, c=C):
    quad_low, quad_high = quad_bounds(P, lower, upper)
    flat_low, flat_high = flat_bounds(quad_low, quad_high, c)
    re, rt = radius
    observed_lower = torch.stack([lower[:, 0] - re, lower[:, 1] - rt], 1)
    observed_upper = torch.stack([upper[:, 0] + re, upper[:, 1] + rt], 1)
    feedforward = float(np.arctan(core.WHEELBASE * kappa) / core.PHI_MAX)
    weights = -K
    u_low = observed_lower @ weights.clamp_min(0) + observed_upper @ weights.clamp_max(0) + feedforward
    u_high = observed_upper @ weights.clamp_min(0) + observed_lower @ weights.clamp_max(0) + feedforward
    if eps > 0:
        observed_quad_high = quad_bounds(P, observed_lower, observed_upper)[1]
        scale = (observed_quad_high - c).clamp(0., 1.)
        u_low = u_low - eps * scale
        u_high = u_high + eps * scale
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
    next_quad_low, next_quad_high = quad_bounds(
        P, torch.stack([e_low, t_low], 1), torch.stack([e_high, t_high], 1))
    next_flat_low, next_flat_high = flat_bounds(next_quad_low, next_quad_high, c)
    return flat_low, flat_high, next_flat_high


def split(lower, upper):
    middle = (lower + upper) / 2
    out = []
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
            out.append((low, high))
    return torch.cat([o[0] for o in out]), torch.cat([o[1] for o in out])


def adaptive_check(kappa, K, P, radius, grid=GRID, max_depth=MAX_DEPTH,
                   eps=EPS, c=C, initial_level=INITIAL_LEVEL):
    axis = torch.linspace(-1., 1., grid + 1, dtype=torch.float64)
    e, t = torch.meshgrid(axis, axis, indexing='ij')
    lower = torch.stack([e[:-1, :-1].reshape(-1), t[:-1, :-1].reshape(-1)], 1)
    upper = torch.stack([e[1:, 1:].reshape(-1), t[1:, 1:].reshape(-1)], 1)
    certified = outside = plateau = 0
    history = []
    for depth in range(max_depth + 1):
        flat_low, flat_high, next_flat_high = evaluate(lower, upper, kappa, K, P, radius, eps, c)
        out = flat_low >= initial_level
        ok = (~out) & (next_flat_high <= flat_low)
        plateau_mask = (~out) & (flat_high <= 0.) & (next_flat_high <= 0.)
        fail = ~(out | ok | plateau_mask)
        certified += int(ok.sum())
        outside += int(out.sum())
        plateau += int(plateau_mask.sum())
        history.append(dict(depth=depth, pending=int(len(lower)), certified=int(ok.sum()),
                            outside=int(out.sum()), plateau=int(plateau_mask.sum()),
                            fail=int(fail.sum())))
        if not fail.any():
            return dict(kappa=float(kappa), passed=True, certified_cells=certified,
                        outside_cells=outside, plateau_cells=plateau, depth=depth,
                        history=history)
        lower, upper = split(lower[fail], upper[fail])
    return dict(kappa=float(kappa), passed=False, certified_cells=certified,
                outside_cells=outside, plateau_cells=plateau,
                unresolved_cells=int(len(lower)), depth=max_depth, history=history)


def sublevel_bounds(P, c=C, level=INITIAL_LEVEL, grid=2000):
    bounds = np.sqrt((c+level) * np.diag(np.linalg.inv(np.asarray(P))))
    return dict(max_abs_e_m=float(bounds[0]) * core.E_SCALE,
                max_abs_theta_rad=float(bounds[1]) * core.THETA_SCALE,
                bounds_method='analytic ellipsoid support')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--radius-e', type=float, default=.08)
    parser.add_argument('--radius-theta', type=float, default=.08)
    parser.add_argument('--grid', type=int, default=GRID)
    parser.add_argument('--max-depth', type=int, default=MAX_DEPTH)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f'choose a new output path: {args.output}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    K, P, _ = lqr()
    Kt = torch.tensor(K, dtype=torch.float64)
    Pt = torch.tensor(P, dtype=torch.float64)
    initial_quad_max = initial_bounds(Pt)
    initial_flat_max = initial_quad_max - C
    checks = [adaptive_check(kappa, Kt, Pt, (args.radius_e, args.radius_theta),
                             grid=args.grid, max_depth=args.max_depth)
              for kappa in core.KAPPAS]
    sublevel = sublevel_bounds(Pt)
    safe = (sublevel['max_abs_e_m'] < core.E_MAX
            and sublevel['max_abs_theta_rad'] < core.THETA_MAX)
    passed = all(c['passed'] for c in checks) and safe and initial_flat_max <= INITIAL_LEVEL
    result = dict(
        format='lane_keeping_visual_sbc_certificate_v1',
        passed=bool(passed),
        controller=dict(type='LQR + bounded neural residual', K=K.tolist(), eps=EPS,
                        residual_bound='eps * clamp(B(obs),0,1) * tanh(z)'),
        sbc=dict(type='flat-bottom quadratic', P=P.tolist(), c=C,
                 formula='relu(x^T P x - c)', nonnegative=True),
        initial=dict(max_quad=initial_quad_max, max_flat=initial_flat_max,
                     level=INITIAL_LEVEL),
        certified_set=dict(level=INITIAL_LEVEL, **sublevel, inside_lane=bool(safe)),
        observation=dict(radius_e=args.radius_e, radius_theta=args.radius_theta,
                         source='patch-DINO split CP 95% radius 0.072 (calibration); 0.08 used for margin'),
        verification=dict(grid=args.grid, max_depth=args.max_depth,
                          kappas=list(core.KAPPAS), checks=checks,
                          certified_cells=int(sum(c['certified_cells'] for c in checks)),
                          plateau_cells=int(sum(c['plateau_cells'] for c in checks)),
                          unresolved_cells=int(sum(c.get('unresolved_cells', 0) for c in checks))),
        scope=('semantic true state + observation-error box; LQR + bounded residual class; '
               'three registered kappas; finite-grid numerical intervals; not a formal '
               'floating-point proof.'),
    )
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ('passed', 'initial', 'certified_set', 'verification')}, indent=2))


if __name__ == '__main__':
    main()
