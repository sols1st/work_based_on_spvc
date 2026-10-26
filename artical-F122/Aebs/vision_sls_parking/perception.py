"""Frozen upstream 1536 -> 512 -> 512 -> 3 observation and audit helpers.

Projection architecture adapted from VISION-SLS/perception_utils/networks.py;
MIT Copyright (c) 2026 Antoine Leeman and Shuyu Zhan, see UPSTREAM_LICENSE.
Imports of Torch are lazy to keep NumPy-only interface tests lightweight.
"""
import numpy as np
from .core import C, RENDER_OFFSET


def load_projection(path, device='cpu'):
    import torch
    from torch import nn

    class Projection(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(1536, 512), nn.GELU(approximate='tanh'),
                                     nn.Linear(512, 512), nn.GELU(approximate='tanh'),
                                     nn.Linear(512, 3))
            self.C = nn.Parameter(torch.empty(3, 4))

        def forward(self, features):
            return self.net(features)

    state = torch.load(path, map_location='cpu')
    model = Projection()
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    c = model.C.detach().numpy().astype(np.float64)
    norms = np.linalg.norm(c, axis=1, keepdims=True)
    if not np.isfinite(c).all() or np.any(norms <= 0):
        raise ValueError('invalid learned observation matrix')
    normalized = c/norms
    gap = float(np.max(np.abs(normalized-C)))
    if gap > 1e-6:
        raise ValueError('checkpoint normalized C differs from scene C: %.9g' % gap)
    return model.to(device), normalized, dict(normalized_C_max_gap=gap,
                                             projection_frozen=True, output_dimension=3)


def fitted_bound(world_states, coefficients, bias):
    """Evaluate EXACT polynomial order hardcoded in upstream get_disturbance.

    This empirical bound is not a CP contract. Coordinate shift applied once.
    """
    s = np.asarray(world_states, dtype=np.float64)
    if s.ndim != 2 or s.shape[1] != 4 or not np.isfinite(s).all():
        raise ValueError('finite [N,4] states required')
    coeff = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    if coeff.shape != (10,) or not np.isfinite(coeff).all() or not np.isfinite(bias) or bias < 0:
        raise ValueError('10 finite coefficients and nonnegative finite bias required')
    x, y = s[:, 0]+3., s[:, 1]
    mons = np.column_stack([np.ones(len(s)), y, y*y, y**3, x, x*y, x*y*y,
                            x*x, x*x*y, x**3])
    return np.maximum(mons @ coeff, bias)


def residual_summary(predicted, world_states, c=C, bounds=None):
    y, s, c = np.asarray(predicted, dtype=np.float64), np.asarray(world_states, dtype=np.float64), np.asarray(c)
    if y.ndim != 2 or y.shape[1] != 3 or s.shape != (len(y), 4) or len(y) == 0:
        raise ValueError('nonempty paired [N,3] observations and [N,4] states required')
    if not np.isfinite(y).all() or not np.isfinite(s).all() or c.shape != (3, 4):
        raise ValueError('finite inputs and [3,4] C required')
    target = (s+RENDER_OFFSET) @ c.T
    residual = y-target
    linf = np.abs(residual).max(axis=1)
    # Algebraic diagnostic, NOT a trained distance head or policy observation.
    if np.linalg.matrix_rank(c[:, :3]) != 3:
        raise ValueError('pose block of C must be invertible for diagnostic')
    inferred = y @ np.linalg.inv(c[:, :3]).T
    pose_error = inferred-(s+RENDER_OFFSET)[:, :3]
    angle_error = (pose_error[:, 2]+np.pi) % (2*np.pi)-np.pi
    centered_rms = float(np.sqrt(np.mean((target-target.mean(axis=0))**2)))
    result = dict(samples=len(y), latent_linf_mean=float(linf.mean()),
                  latent_linf_p95=float(np.quantile(linf, .95)), latent_linf_max=float(linf.max()),
                  latent_rmse=float(np.sqrt(np.mean(residual**2))),
                  target_centered_rms=centered_rms,
                  relative_latent_rmse=(float(np.sqrt(np.mean(residual**2)))/centered_rms
                                       if centered_rms > 1e-12 else None),
                  pose_xy_euclidean_mae_m=float(np.linalg.norm(pose_error[:, :2], axis=1).mean()),
                  pose_x_mae_m=float(np.abs(pose_error[:, 0]).mean()),
                  pose_y_mae_m=float(np.abs(pose_error[:, 1]).mean()),
                  pose_heading_wrapped_mae_rad=float(np.abs(angle_error).mean()),
                  position_readout='algebraic inverse of fixed C, evaluation only; NOT deployed',
                  model_coordinate='world position + [3,0,0]; heading unchanged')
    if bounds is not None:
        b = np.asarray(bounds, dtype=np.float64)
        if b.shape != linf.shape or not np.isfinite(b).all() or np.any(b < 0):
            raise ValueError('finite nonnegative [N] bounds required')
        result.update(empirical_bound_coverage=float((linf <= b).mean()),
                      empirical_bound_misses=int((linf > b).sum()),
                      signed_max_instead_of_abs_coverage=float((residual.max(axis=1) <= b).mean()),
                      bound_mean=float(b.mean()), max_exceedance=float(np.maximum(linf-b, 0).max()),
                      formal_contract=False)
    return result


def identifiability_witness(states, images, targets):
    """Identical rendered RGB + different Cx => unavoidable pointwise error.

    For each repeated-image group any deterministic image function must incur
    L-infinity error at least half the target coordinate range at one member.
    This is a finite-panel witness, not a global impossibility claim.
    """
    import hashlib
    groups = {}
    for i, image in enumerate(images):
        groups.setdefault(hashlib.sha256(np.asarray(image, dtype=np.uint8).tobytes()).hexdigest(), []).append(i)
    witnesses = []
    for digest, ids in groups.items():
        if len(ids) < 2:
            continue
        t = np.asarray(targets)[ids]
        axis = int(np.argmax(np.ptp(t, axis=0)))
        lo, hi = ids[int(np.argmin(t[:, axis]))], ids[int(np.argmax(t[:, axis]))]
        lower = float(np.ptp(t[:, axis])*.5)
        if lower > 1e-8:
            witnesses.append(dict(image_sha256=digest, identical_image_members=len(ids),
                                  any_single_image_estimator_minimum_worst_linf_error=lower,
                                  pair_indices=[int(lo), int(hi)],
                                  pair_states=np.asarray(states)[[lo, hi]].tolist()))
    return sorted(witnesses, key=lambda w: -w['any_single_image_estimator_minimum_worst_linf_error'])
