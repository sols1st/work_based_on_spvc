"""Joint trajectory split conformal boxes for signed, decoded perception errors."""
import math

import numpy as np


METHODS = ('fixed_uniform', 'fixed_axis', 'fixed_signed', 'state_bins', 'state_signed',
           'state_interp', 'state_grid', 'state_loglinear')


def _data(features, errors):
    f, e = np.asarray(features, dtype=float), np.asarray(errors, dtype=float)
    if f.ndim != 3 or f.shape[-1] != 2 or e.ndim != 3 or e.shape[:2] != f.shape[:2]:
        raise ValueError('expected paired [trajectory,time,feature] and [trajectory,time,error] arrays')
    if not f.size or not e.size or not (np.isfinite(f).all() and np.isfinite(e).all()):
        raise ValueError('finite nonempty trajectory observations required')
    return f, e


def _scales(e, signed):
    if signed:
        return np.stack([np.quantile(np.maximum(-e, 0), .9, axis=0),
                         np.quantile(np.maximum(e, 0), .9, axis=0)])
    both = np.quantile(np.abs(e), .9, axis=0)
    return np.stack([both, both])


def _edges(x, quantiles):
    q = np.unique(np.quantile(x, quantiles))
    return q[(q > x.min()) & (q < x.max())]


def fit_shapes(features, errors):
    """Fit relative half-widths only on the designated development trajectories."""
    f, e = _data(features, errors)
    f, e = f.reshape(-1, 2), e.reshape(-1, e.shape[-1])
    floor = np.maximum(0.10 * np.quantile(np.abs(e), .9, axis=0), 1e-6)
    shapes = {}
    for name in METHODS:
        signed = name in ('fixed_signed', 'state_signed')
        base = np.maximum(_scales(e, signed), floor)
        xcuts = ycuts = np.array([], dtype=float)
        if name in ('state_bins', 'state_signed', 'state_interp'):
            xcuts = _edges(f[:, 0], [.25, .5, .75])
        elif name == 'state_grid':
            xcuts = _edges(f[:, 0], [1/3, 2/3])
            ycuts = _edges(f[:, 1], [1/3, 2/3])
        xi = np.searchsorted(xcuts, f[:, 0], side='right')
        yi = np.searchsorted(ycuts, f[:, 1], side='right')
        shape = np.empty((len(xcuts)+1, len(ycuts)+1, *base.shape))
        counts = np.zeros(shape.shape[:2], dtype=int)
        for i, j in np.ndindex(counts.shape):
            sample = e[(xi == i) & (yi == j)]
            counts[i, j] = len(sample)
            if name == 'fixed_uniform':
                shape[i, j] = 1.
            elif name.startswith('fixed_') or not len(sample):
                shape[i, j] = base
            else:
                # Shrink sparsely populated bins toward the common scale, then floor.
                weight = len(sample) / (len(sample) + 100.)
                shape[i, j] = np.maximum(weight*_scales(sample, signed) + (1-weight)*base, floor)
        regression = {}
        if name == 'state_loglinear':
            low, high = np.quantile(f, [.05, .95], axis=0)
            span = np.maximum(high-low, 1e-10)
            u = (np.clip(f, low, high) - low) / span - .5
            design = np.column_stack([np.ones(len(f)), u])
            target = np.log(np.maximum(np.abs(e), floor) / base[0])
            penalty = np.diag([0., .02 * len(f), .02 * len(f)])
            coef = np.linalg.solve(design.T @ design + penalty, design.T @ target)
            regression = dict(low=low.tolist(), high=high.tolist(), coefficients=coef.tolist(),
                              base=base[0].tolist())
        centers = []
        if name == 'state_interp':
            populated = [i for i in range(len(xcuts)+1) if (xi == i).any()]
            centers = [float(np.median(f[xi == i, 0])) for i in populated]
            shape = shape[populated]
            counts = counts[populated]
        shapes[name] = dict(name=name, xcuts=xcuts.tolist(), ycuts=ycuts.tolist(),
                            centers=centers, halfwidths=shape.tolist(), regression=regression, development_frames=counts.tolist())
    return shapes


def cuts(shape):
    """Physical switching boundaries or interpolation knots along each feature."""
    if shape['name'] == 'state_loglinear':
        reg = shape['regression']
        return tuple(sorted(set([lo, hi])) for lo, hi in zip(reg['low'], reg['high']))
    return (shape['centers'] if shape['name'] == 'state_interp' else shape['xcuts'], shape['ycuts'])


def widths(shape, features):
    f = np.asarray(features, dtype=float)
    if f.shape[-1] != 2 or not np.isfinite(f).all():
        raise ValueError('two finite verification-side state features required')
    if shape['name'] == 'state_loglinear':
        r = shape['regression']
        low, high, coef = (np.asarray(r[k]) for k in ('low', 'high', 'coefficients'))
        u = (np.clip(f, low, high) - low) / np.maximum(high-low, 1e-10) - .5
        w = np.exp(np.clip(coef[0] + u @ coef[1:], -3., 2.)) * r['base']
        return w, w
    scales = np.asarray(shape['halfwidths'])
    if shape['name'] == 'state_interp':
        knots = np.asarray(shape['centers'])
        x = f[..., 0]
        out = np.stack([np.interp(x, knots, scales[:, 0, s, d])
                        for s in range(2) for d in range(scales.shape[-1])], axis=-1)
        out = out.reshape((*x.shape, 2, scales.shape[-1]))
    else:
        x = np.searchsorted(shape['xcuts'], f[..., 0], side='right')
        y = np.searchsorted(shape['ycuts'], f[..., 1], side='right')
        out = scales[x, y]
    return out[..., 0, :], out[..., 1, :]


def width_bounds(shape, feature_lo, feature_hi):
    """Enclose every half-width on the closed feature rectangle; tails extend flat."""
    lo, hi = np.asarray(feature_lo, dtype=float), np.asarray(feature_hi, dtype=float)
    if lo.shape != (2,) or hi.shape != (2,) or np.isnan([lo, hi]).any() or (lo > hi).any():
        raise ValueError('ordered two-dimensional feature bounds required')
    if shape['name'] == 'state_loglinear':
        r = shape['regression']
        # A clipped affine log scale is coordinatewise monotone in each feature.
        a, b = np.clip(lo, r['low'], r['high']), np.clip(hi, r['low'], r['high'])
        corners = [[x, y] for x in (a[0], b[0]) for y in (a[1], b[1])]
        w = widths(shape, corners)[0].max(axis=0)
        return w, w
    s = np.asarray(shape['halfwidths'])
    if shape['name'] == 'state_interp':
        nodes = np.asarray(shape['centers'])
        x = np.r_[np.clip(lo[0], nodes[0], nodes[-1]), nodes[(nodes>lo[0]) & (nodes<hi[0])],
                  np.clip(hi[0], nodes[0], nodes[-1])]
        neg, pos = widths(shape, np.column_stack([x, np.zeros_like(x)]))
        return neg.max(axis=0), pos.max(axis=0)
    box = []
    for a, edges in enumerate((shape['xcuts'], shape['ycuts'])):
        first, last = np.searchsorted(edges, [lo[a], hi[a]], side='right')
        box.append(slice(first, last+1))
    result = s[tuple(box)].max(axis=(0,1))
    return result[0], result[1]


def trajectory_scores(shape, features, errors):
    f, e = _data(features, errors)
    neg, pos = widths(shape, f)
    if e.shape != neg.shape or (np.minimum(neg, pos) <= 0).any():
        raise ValueError('strictly positive matching half-widths required')
    return np.maximum(-e / neg, e / pos).max(axis=(1, 2))


def calibrate(shape, features, errors, alpha=.05):
    scores = trajectory_scores(shape, features, errors)
    if not 0 < alpha < 1:
        raise ValueError('alpha must lie strictly between zero and one')
    n = len(scores)
    rank = math.ceil((n+1) * (1-alpha))
    if rank > n:
        raise ValueError('too few independent trajectories for a finite conformal threshold')
    return dict(q=float(np.sort(scores)[rank-1]), rank=rank, n=n)
