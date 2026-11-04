"""Positive three-output state-dependent scales and joint split-CP scores.

Calibration requires independent/exchangeable sampling units. This module
does not assume that image frames from one trajectory are independent.
"""
import itertools
import math
import numpy as np
from scipy.optimize import linprog
from scipy import sparse


def basis(states, low, high):
    """Trilinear Bernstein basis on a registered (px, py, heading) box."""
    x = np.asarray(states, dtype=np.float64)
    low, high = np.asarray(low, dtype=float), np.asarray(high, dtype=float)
    if (x.ndim != 2 or x.shape[1] not in (3, 4) or low.shape != (3,) or high.shape != (3,)
            or not np.isfinite(x).all() or not np.isfinite(low).all()
            or not np.isfinite(high).all() or (high <= low).any()):
        raise ValueError('finite states and ordered 3D domain required')
    t = (x[:, :3]-low)/(high-low)
    if (t < -1e-7).any() or (t > 1+1e-7).any():
        raise ValueError('state outside registered visual error domain')
    t = np.clip(t, 0, 1)
    return np.column_stack([np.prod(np.where(np.asarray(bits), t, 1-t), axis=1)
                            for bits in itertools.product((0, 1), repeat=3)])


def fit_scale(states, residuals, low, high, floor=.01, slack_penalty=20.):
    m = basis(states, low, high)
    r = np.abs(np.asarray(residuals, dtype=np.float64))
    if r.shape != (len(m), 3) or not len(m) or not np.isfinite(r).all():
        raise ValueError('nonempty finite [N,3] residuals required')
    if not np.isfinite([floor, slack_penalty]).all() or min(floor, slack_penalty) <= 0:
        raise ValueError('positive floor and slack penalty required')
    n = len(m)
    a = sparse.hstack([-sparse.csr_matrix(m), -sparse.eye(n)], format='csr')
    cost = np.r_[m.mean(0), np.full(n, slack_penalty/n)]
    coefficients = []
    for j in range(3):
        fit = linprog(cost, A_ub=a, b_ub=-r[:, j],
                      bounds=[(floor, None)]*8+[(0, None)]*n, method='highs')
        if not fit.success:
            raise RuntimeError(fit.message)
        coefficients.append(fit.x[:8])
    return dict(schema='parking_joint_error_v1', low=np.asarray(low).tolist(),
                high=np.asarray(high).tolist(), coefficients=np.asarray(coefficients).T.tolist(),
                q=1., floor=floor, degree=1, outputs=3, units='unnormalized latent Cx units',
                guarantee='none until a valid calibration protocol is established')


def scale(states, model):
    c = np.asarray(model['coefficients'], dtype=float)
    if c.shape != (8, 3) or not np.isfinite(c).all() or (c <= 0).any():
        raise ValueError('positive finite [8,3] Bernstein coefficients required')
    return basis(states, model['low'], model['high'])@c


def joint_scores(states, residuals, model):
    b = scale(states, model)
    r = np.asarray(residuals, dtype=float)
    if r.shape != b.shape or not np.isfinite(r).all():
        raise ValueError('finite paired vector residuals required')
    return np.max(np.abs(r)/b, axis=1)


def calibrate(states, residuals, model, alpha=.05, groups=None):
    """One joint score per sample, or maximum per complete trajectory/group.

    A missing finite order statistic returns q=inf; no fallback to a maximum
    that falsely claims the requested coverage at an inadequate sample size.
    """
    if not 0 < alpha < 1:
        raise ValueError('alpha must be in (0,1)')
    scores = joint_scores(states, residuals, model)
    if groups is not None:
        groups = np.asarray(groups)
        if groups.shape != scores.shape:
            raise ValueError('one sampling-unit identifier per frame required')
        scores = np.array([scores[groups == g].max() for g in np.unique(groups)])
    if not len(scores):
        raise ValueError('nonempty calibration units required')
    rank = math.ceil((len(scores)+1)*(1-alpha))
    q = float(np.sort(scores)[rank-1]) if rank <= len(scores) else float('inf')
    return dict(q=q, rank=rank, n_units=len(scores), alpha=alpha,
                unit='group maximum' if groups is not None else 'individual observation',
                finite=bool(np.isfinite(q)), conditional_on_exchangeability=True)


def radius(states, model):
    q = float(model['q'])
    if not np.isfinite(q) or q < 0:
        raise ValueError('nonfinite or negative calibrated radius multiplier')
    return q*scale(states, model)
