"""Interval audit for the lane-keeping SBC.

Propagates state boxes and the observation-error box through the fixed actor,
the true dynamics and the nonnegative SBC.  The default grid audit is a
numerical interval check on a finite partition, not a formal continuous proof;
report it as such.  Set ``--observation-radius 0`` for the semantic controller.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from . import core, sbc
from .policy import load_actor
from .sbc import load_sbc


def linear_bounds(lower, upper, layer):
    center = (lower + upper) / 2
    radius = (upper - lower) / 2
    out_center = center @ layer.weight.T + layer.bias
    out_radius = radius @ layer.weight.abs().T
    return out_center - out_radius, out_center + out_radius


def actor_bounds(actor, lower, upper):
    for module in actor.policy_net:
        if isinstance(module, torch.nn.Linear):
            lower, upper = linear_bounds(lower, upper, module)
        elif isinstance(module, torch.nn.Tanh):
            lower, upper = torch.tanh(lower), torch.tanh(upper)
        elif isinstance(module, torch.nn.ReLU):
            lower, upper = lower.clamp_min(0), upper.clamp_min(0)
        else:
            raise TypeError(f'unsupported actor layer {type(module)}')
    lower, upper = linear_bounds(lower, upper, actor.action_net)
    return lower.clamp(-1, 1), upper.clamp(-1, 1)


def sbc_bounds(barrier, lower, upper):
    for index, layer in enumerate(barrier.layers):
        lower, upper = linear_bounds(lower, upper, layer)
        if index < len(barrier.layers) - 1:
            lower, upper = torch.tanh(lower), torch.tanh(upper)
        else:
            lower, upper = lower.clamp_min(0), upper.clamp_min(0)
    return lower.flatten(), upper.flatten()


def dynamics_bounds(lower, upper, action_low, action_high, kappa):
    # e' = e + v dt sin(theta); theta' = theta + v dt/L tan(phi) - v dt kappa
    sin_low = torch.sin(lower[:, 1] * core.THETA_SCALE)
    sin_high = torch.sin(upper[:, 1] * core.THETA_SCALE)
    e_low = lower[:, 0] * core.E_SCALE + core.SPEED * core.DT * sin_low
    e_high = upper[:, 0] * core.E_SCALE + core.SPEED * core.DT * sin_high
    phi_low = action_low.flatten() * core.PHI_MAX
    phi_high = action_high.flatten() * core.PHI_MAX
    theta_low = (lower[:, 1] * core.THETA_SCALE
                 + core.SPEED * core.DT / core.WHEELBASE * torch.tan(phi_low)
                 - core.SPEED * core.DT * kappa)
    theta_high = (upper[:, 1] * core.THETA_SCALE
                  + core.SPEED * core.DT / core.WHEELBASE * torch.tan(phi_high)
                  - core.SPEED * core.DT * kappa)
    return (torch.stack([e_low / core.E_SCALE, theta_low / core.THETA_SCALE], dim=1),
            torch.stack([e_high / core.E_SCALE, theta_high / core.THETA_SCALE], dim=1))


def state_grid(grid, low=(-1., -1.), high=(1., 1.)):
    axis_e = torch.linspace(low[0], high[0], grid + 1, dtype=torch.float32)
    axis_t = torch.linspace(low[1], high[1], grid + 1, dtype=torch.float32)
    e, t = torch.meshgrid(axis_e, axis_t, indexing='ij')
    low_cells = torch.stack([e[:-1, :-1].reshape(-1), t[:-1, :-1].reshape(-1)], dim=1)
    high_cells = torch.stack([e[1:, 1:].reshape(-1), t[1:, 1:].reshape(-1)], dim=1)
    return low_cells, high_cells


def region_bounds(barrier, low, high, grid=41):
    lower, upper = state_grid(grid, low, high)
    with torch.no_grad():
        b_low, b_high = sbc_bounds(barrier, lower, upper)
    return dict(lower=float(b_low.min()), upper=float(b_high.max()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--actor', type=Path, required=True)
    parser.add_argument('--sbc', type=Path, required=True)
    parser.add_argument('--observation-radius', type=float, default=0.)
    parser.add_argument('--grid', type=int, default=60)
    parser.add_argument('--error-splits', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu')
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f'choose a new output path: {args.output}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    actor = load_actor(args.actor, args.device)
    barrier, metadata = load_sbc(args.sbc, args.device)
    radius = float(args.observation_radius)
    if radius < 0 or not np.isfinite(radius):
        raise ValueError('nonnegative finite observation radius required')
    lower, upper = state_grid(args.grid)
    lower, upper = lower.to(args.device), upper.to(args.device)
    # Error-box subcells: sound cover of [-radius, radius]^2.
    cuts = np.linspace(-radius, radius, args.error_splits + 1)
    error_boxes = [(float(cuts[i]), float(cuts[i + 1])) for i in range(args.error_splits)]
    checked = active = violations = excluded = 0
    worst = -float('inf')
    for kappa_physical in core.KAPPAS:
        kappa = torch.full((len(lower), 1), kappa_physical / core.KAPPA_SCALE,
                           dtype=torch.float32, device=args.device)
        with torch.no_grad():
            current_low, current_high = sbc_bounds(barrier, lower, upper)
        active_mask = current_low < sbc.UNSAFE_LEVEL
        for el, eu in error_boxes:
            for tl, tu in error_boxes:
                observed_low = torch.cat([lower + torch.tensor([el, tl], device=args.device), kappa], dim=1)
                observed_high = torch.cat([upper + torch.tensor([eu, tu], device=args.device), kappa], dim=1)
                with torch.no_grad():
                    action_low, action_high = actor_bounds(actor, observed_low, observed_high)
                    next_low, next_high = dynamics_bounds(lower, upper, action_low, action_high,
                                                          torch.full((len(lower),), kappa_physical,
                                                                     device=args.device))
                    next_low_b, next_high_b = sbc_bounds(barrier, next_low, next_high)
                residual = next_high_b - current_low + sbc.DECREASE_MARGIN
                checked += len(lower)
                active += int(active_mask.sum())
                excluded += int((~active_mask).sum())
                bad = active_mask & (residual > 0)
                violations += int(bad.sum())
                if active_mask.any():
                    worst = max(worst, float(residual[active_mask].max()))
    initial = region_bounds(barrier, [-sbc.INIT_E_N, -sbc.INIT_THETA_N],
                            [sbc.INIT_E_N, sbc.INIT_THETA_N])
    unsafe_low = min(region_bounds(barrier, [.98, -1.05], [1.05, 1.05], grid=21)['lower'],
                     region_bounds(barrier, [-1.05, -1.05], [-.98, 1.05], grid=21)['lower'],
                     region_bounds(barrier, [-1.05, .98], [1.05, 1.05], grid=21)['lower'],
                     region_bounds(barrier, [-1.05, -1.05], [1.05, -.98], grid=21)['lower'])
    result = dict(actor=str(args.actor), sbc=str(args.sbc), observation_radius=radius,
                  grid=args.grid, error_splits=args.error_splits,
                  region=dict(initial=initial, unsafe_lower=float(unsafe_low),
                              initial_pass=initial['upper'] <= sbc.INIT_LEVEL,
                              unsafe_pass=float(unsafe_low) >= sbc.UNSAFE_LEVEL),
                  decrease=dict(checked_cells=checked, active_cells=active,
                                excluded_cells=excluded, violations=violations,
                                worst_residual=float(worst) if np.isfinite(worst) else None,
                                pass_=violations == 0),
                  continuous_interval_pass=bool(initial['upper'] <= sbc.INIT_LEVEL
                                                and float(unsafe_low) >= sbc.UNSAFE_LEVEL
                                                and violations == 0),
                  guarantee_scope=('finite grid interval check; not a continuous-domain '
                                   'or formal floating-point proof'))
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
