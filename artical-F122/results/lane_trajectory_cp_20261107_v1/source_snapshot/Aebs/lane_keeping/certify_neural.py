"""ReLU polyhedral SBC for the exact exported PPO actor, without a filter.

The certificate is B(x)=scale*relu(max_i A_i x - plateau).
It is a finite ReLU network (max(a,b)=relu(a-b)+b), with learned PPO
weights kept unchanged.  A quadratic Lyapunov template seeds the facets;
the nonlinear/noisy PPO closed loop, not the template LQR, is verified.

Float64 numerical continuous-box audit. This is NOT a formal proof of
all library/hardware floating-point errors or a CP trajectory guarantee.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import core
from .certify_linear import lqr
from .policy import load_actor


class PolyhedralSBC(nn.Module):
    def __init__(self, facets, plateau=1., scale=200.):
        super().__init__()
        self.register_buffer('facets', torch.as_tensor(facets, dtype=torch.float64))
        self.plateau = float(plateau)
        self.scale = float(scale)

    def forward(self, state):
        terms = state @ self.facets.T
        value = terms[..., 0]
        for index in range(1, self.facets.shape[0]):
            value = torch.relu(value - terms[..., index]) + terms[..., index]
        return (self.scale * torch.relu(value - self.plateau)).unsqueeze(-1)


def load_certificate(path):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved.get('format') != 'lane_polyhedral_relu_sbc_v1':
        raise ValueError('unexpected certificate format')
    return PolyhedralSBC(saved['facets'], saved['plateau'], saved['scale']).eval()


def affine_extrema(coef, bias, lower, upper):
    pos, neg = coef.clamp_min(0), coef.clamp_max(0)
    return ((pos * lower[:, None] + neg * upper[:, None]).sum(-1) + bias,
            (pos * upper[:, None] + neg * lower[:, None]).sum(-1) + bias)


def actor_affine_bounds(actor, lower, upper):
    """Parallel-line tanh relaxation retaining input correlations.

    For slope a, extrema of tanh(t)-a*t are at endpoints or at
    +/-acosh(1/sqrt(a)). Enumerating those gives an enclosing affine strip.
    Layer composition uses weight signs. Small padding is numerical, not
    a claimed complete floating-point error theorem.
    """
    if torch.any(lower > upper) or not torch.isfinite(lower).all() or not torch.isfinite(upper).all():
        raise ValueError('finite ordered bounds required')
    n, d = lower.shape
    lc = torch.eye(d, dtype=lower.dtype).expand(n, d, d).clone()
    uc = lc.clone()
    lb = torch.zeros_like(lower)
    ub = lb.clone()
    for layer in list(actor.policy_net) + [actor.action_net]:
        if isinstance(layer, nn.Linear):
            wp, wn = layer.weight.clamp_min(0), layer.weight.clamp_max(0)
            lc, uc = (torch.einsum('oi,nij->noj', wp, lc) + torch.einsum('oi,nij->noj', wn, uc),
                      torch.einsum('oi,nij->noj', wp, uc) + torch.einsum('oi,nij->noj', wn, lc))
            lb, ub = lb @ wp.T + ub @ wn.T + layer.bias, ub @ wp.T + lb @ wn.T + layer.bias
        elif isinstance(layer, nn.Tanh):
            lo = affine_extrema(lc, lb, lower, upper)[0] - 1e-12
            hi = affine_extrema(uc, ub, lower, upper)[1] + 1e-12
            width = hi - lo
            slope = torch.where(width > 1e-8, (torch.tanh(hi) - torch.tanh(lo)) / width,
                                1. - torch.tanh((hi + lo) / 2).square()).clamp(1e-15, 1.)
            root = torch.acosh(torch.rsqrt(slope))
            points = torch.stack([lo, hi, root.clamp(lo, hi), (-root).clamp(lo, hi)])
            intercepts = torch.tanh(points) - slope * points
            lb = slope * lb + intercepts.amin(0) - 1e-12
            ub = slope * ub + intercepts.amax(0) + 1e-12
            lc, uc = slope[..., None] * lc, slope[..., None] * uc
        else:
            raise TypeError(type(layer))
    lo = affine_extrema(lc, lb, lower, upper)[0].flatten() - 1e-12
    hi = affine_extrema(uc, ub, lower, upper)[1].flatten() + 1e-12
    return lo.clamp(-1, 1), hi.clamp(-1, 1)


def poly_bounds(facets, lower, upper):
    pos, neg = facets.clamp_min(0), facets.clamp_max(0)
    return ((lower @ pos.T + upper @ neg.T).amax(1) - 1e-12,
            (upper @ pos.T + lower @ neg.T).amax(1) + 1e-12)


def transition_bounds(actor, lower, upper, kappa, error_lower, error_upper, method='affine'):
    curve = torch.full((len(lower), 1), kappa / core.KAPPA_SCALE, dtype=lower.dtype)
    obs_lo = torch.cat([lower + error_lower, curve], 1)
    obs_hi = torch.cat([upper + error_upper, curve], 1)
    if method == 'affine':
        u_lo, u_hi = actor_affine_bounds(actor, obs_lo, obs_hi)
    elif method == 'ibp':
        from .verify import actor_bounds
        u_lo, u_hi = actor_bounds(actor, obs_lo, obs_hi)
        u_lo, u_hi = u_lo.flatten()-1e-12, u_hi.flatten()+1e-12
        u_lo, u_hi = u_lo.clamp(-1, 1), u_hi.clamp(-1, 1)
    else:
        raise ValueError(method)
    # Monotone trig ranges are valid on registered state/action domain.
    if (lower[:, 1].abs().max() * core.THETA_SCALE > np.pi / 2 or
            upper[:, 1].abs().max() * core.THETA_SCALE > np.pi / 2):
        raise ValueError('heading interval outside monotone sine branch')
    elo = lower[:, 0] + core.SPEED * core.DT / core.E_SCALE * torch.sin(lower[:, 1] * core.THETA_SCALE)
    ehi = upper[:, 0] + core.SPEED * core.DT / core.E_SCALE * torch.sin(upper[:, 1] * core.THETA_SCALE)
    tlo = lower[:, 1] + core.SPEED * core.DT / (core.WHEELBASE * core.THETA_SCALE) * torch.tan(u_lo * core.PHI_MAX) - core.SPEED * core.DT * kappa / core.THETA_SCALE
    thi = upper[:, 1] + core.SPEED * core.DT / (core.WHEELBASE * core.THETA_SCALE) * torch.tan(u_hi * core.PHI_MAX) - core.SPEED * core.DT * kappa / core.THETA_SCALE
    return torch.stack([elo, tlo], 1) - 1e-12, torch.stack([ehi, thi], 1) + 1e-12


def split(lower, upper):
    # Binary split reduces transient memory compared with four children.
    dim = (upper - lower).argmax(1)
    middle = (lower + upper) / 2
    rows = torch.arange(len(lower))
    h1, l2 = upper.clone(), lower.clone()
    h1[rows, dim], l2[rows, dim] = middle[rows, dim], middle[rows, dim]
    return torch.cat([lower, l2]), torch.cat([h1, upper])


@torch.no_grad()
def audit(actor, barrier, radius=.08, grid=80, depth=16, error_splits=2,
          batch=2048, max_pending=200000, method='affine'):
    a = torch.linspace(-1., 1., grid + 1, dtype=torch.float64)
    e, t = torch.meshgrid(a, a, indexing='ij')
    original_lo = torch.stack([e[:-1, :-1].flatten(), t[:-1, :-1].flatten()], 1)
    original_hi = torch.stack([e[1:, 1:].flatten(), t[1:, 1:].flatten()], 1)
    cuts = np.linspace(-radius, radius, error_splits + 1)
    errors = [(torch.tensor([cuts[i], cuts[j]]), torch.tensor([cuts[i+1], cuts[j+1]]))
              for i in range(error_splits) for j in range(error_splits)]
    checks = []
    threshold = barrier.plateau + 20. / barrier.scale
    for k in core.KAPPAS:
        lo, hi = original_lo.clone(), original_hi.clone()
        counts = dict(outside=0, plateau=0, decreasing=0)
        terminal_area = 0.
        history = []
        for level in range(depth + 1):
            pending_lo, pending_hi = [], []
            for start in range(0, len(lo), batch):
                low, high = lo[start:start+batch], hi[start:start+batch]
                current_lo, current_hi = poly_bounds(barrier.facets, low, high)
                outside = current_lo > threshold
                terminal_area += float((high-low).prod(1)[outside].sum())
                counts['outside'] += int(outside.sum())
                low, high, current_lo = low[~outside], high[~outside], current_lo[~outside]
                if not len(low):
                    continue
                following = torch.full((len(low),), -float('inf'), dtype=torch.float64)
                for el, eh in errors:
                    nl, nh = transition_bounds(actor, low, high, k, el, eh, method)
                    following = torch.maximum(following, poly_bounds(barrier.facets, nl, nh)[1])
                # Zero plateau is proved invariant, without positive tolerance.
                plateau = following <= barrier.plateau
                decrease = (~plateau) & (following <= current_lo)
                counts['plateau'] += int(plateau.sum())
                counts['decreasing'] += int(decrease.sum())
                bad = ~(plateau | decrease)
                terminal_area += float((high-low).prod(1)[~bad].sum())
                pending_lo.append(low[bad])
                pending_hi.append(high[bad])
            pl = torch.cat(pending_lo) if pending_lo else torch.empty((0, 2), dtype=torch.float64)
            ph = torch.cat(pending_hi) if pending_hi else pl.clone()
            history.append(dict(depth=level, input_cells=len(lo), unresolved=len(pl), **counts))
            print(json.dumps(dict(kappa=k, **history[-1])), flush=True)
            if not len(pl) or level == depth or len(pl) * 2 > max_pending:
                coverage_error = abs(terminal_area + float((ph-pl).prod(1).sum()) - 4.)
                checks.append(dict(kappa=k, passed=not len(pl) and coverage_error < 1e-9,
                                   unresolved=len(pl), partition_area_error=coverage_error,
                                   terminal_area=terminal_area,
                                   history=history, unresolved_examples=torch.stack([pl[:8], ph[:8]], 1).tolist()))
                break
            lo, hi = split(pl, ph)
    return checks


def geometry(facets, plateau, scale):
    """Analytic facet witnesses enclosing the sublevel, not a sampled grid."""
    witnesses = []
    # Two nonnegative facet multipliers reproduce each coordinate direction.
    for dim in range(2):
        for sign in [-1., 1.]:
            target = np.zeros(2)
            target[dim] = sign
            best = None
            for i in range(len(facets)):
                for j in range(i+1, len(facets)):
                    mat = facets[[i, j]].T
                    if abs(np.linalg.det(mat)) < 1e-10:
                        continue
                    weights = np.linalg.solve(mat, target)
                    if np.min(weights) >= 0 and (best is None or weights.sum() < best['sum']):
                        best = dict(dim=dim, sign=sign, rows=[i, j], weights=weights.tolist(),
                                    sum=float(weights.sum()), residual=float(abs(mat @ weights-target).max()))
            if best is None:
                raise ValueError('facets do not enclose all coordinate directions')
            witnesses.append(best)
    min_norm_unsafe = min(1. / w['sum'] for w in witnesses) - 1e-10
    init = np.array([[x, y] for x in [-core.INIT_E/core.E_SCALE, core.INIT_E/core.E_SCALE]
                     for y in [-core.INIT_THETA/core.THETA_SCALE, core.INIT_THETA/core.THETA_SCALE]])
    max_init = float((init @ facets.T).max()) + 1e-10
    return dict(initial_norm_upper=max_init, initial_B_upper=scale*max(0., max_init-plateau),
                unsafe_norm_lower=min_norm_unsafe,
                unsafe_B_lower=scale*max(0., min_norm_unsafe-plateau),
                facet_witnesses=witnesses,
                active_sublevel_coordinate_bounds=[max(w['sum'] for w in witnesses if w['dim']==d)*
                                                   (plateau+20/scale) for d in range(2)])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--actor', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--radius', type=float, default=.08)
    p.add_argument('--grid', type=int, default=80)
    p.add_argument('--depth', type=int, default=16)
    p.add_argument('--error-splits', type=int, default=2)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--method', choices=['affine', 'ibp'], default='affine')
    args = p.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.radius < 0 or args.grid < 1 or args.depth < 0 or args.error_splits < 1:
        raise ValueError('invalid verification parameters')
    args.output_dir.mkdir(parents=True)
    torch.set_num_threads(args.threads)
    actor = load_actor(args.actor).double()
    _, P, _ = lqr()
    angles = np.arange(32) * 2*np.pi/32
    facets = np.c_[np.cos(angles), np.sin(angles)] @ np.linalg.cholesky(P).T
    barrier = PolyhedralSBC(facets)
    geo = geometry(facets, barrier.plateau, barrier.scale)
    torch.save(dict(format='lane_polyhedral_relu_sbc_v1', facets=barrier.facets,
                    plateau=barrier.plateau, scale=barrier.scale), args.output_dir/'sbc.pt')
    checks = audit(actor, barrier, args.radius, args.grid, args.depth, args.error_splits,
                   method=args.method)
    result = dict(format='lane_ppo_relu_sbc_audit_v1',
                  passed=all(c['passed'] for c in checks) and geo['initial_B_upper']<=1
                         and geo['unsafe_B_lower']>=20
                         and max(geo['active_sublevel_coordinate_bounds'])<1,
                  actor=str(args.actor), actor_sha256=hashlib.sha256(args.actor.read_bytes()).hexdigest(),
                  radius=args.radius, geometry=geo, checks=checks,
                  verification=dict(method=args.method, grid=args.grid, depth=args.depth,
                                    error_splits=args.error_splits),
                  task=dict(speed=core.SPEED, dt=core.DT, wheelbase=core.WHEELBASE,
                            safety_halfwidths=[core.E_MAX, core.THETA_MAX],
                            initial_halfwidths=[core.INIT_E, core.INIT_THETA],
                            action_limit=core.PHI_MAX, kappas=list(core.KAPPAS),
                            observation_scaling=[core.E_SCALE, core.THETA_SCALE],
                            process_noise=0),
                  controller='original PPO deterministic actor with final Tanh restored; no online filter',
                  certificate='finite affine/ReLU max-facet network; geometrically synthesized weights, not a trained Tanh SBC',
                  scope='float64 numerical continuous-box audit; registered three kappas; every observation inside radius box; no claim of unconditional CP trajectory or real-world safety')
    (args.output_dir/'certificate.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(passed=result['passed'], geometry=geo)), flush=True)


if __name__ == '__main__':
    main()
