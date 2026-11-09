"""Numerical affine bounds for Tanh MLPs (shared, task-independent).

Preserves correlations to the original box inputs. This is float64/float32
arithmetic with small padding, NOT a complete outward-rounded proof.
"""
import torch
from torch import nn


def extrema(coef, bias, lower, upper):
    positive, negative = coef.clamp_min(0), coef.clamp_max(0)
    return ((positive * lower[:, None] + negative * upper[:, None]).sum(-1) + bias,
            (positive * upper[:, None] + negative * lower[:, None]).sum(-1) + bias)


def tanh_affine_bounds(layers, lower, upper, padding=1e-12):
    if lower.shape != upper.shape or lower.ndim != 2 or torch.any(lower > upper):
        raise ValueError('matching ordered boxes required')
    if not torch.isfinite(lower).all() or not torch.isfinite(upper).all():
        raise ValueError('finite boxes required')
    n, d = lower.shape
    lc = torch.eye(d, dtype=lower.dtype, device=lower.device).expand(n, d, d)
    uc = lc
    lb = torch.zeros_like(lower)
    ub = lb
    for layer in layers:
        if isinstance(layer, nn.Linear):
            wp, wn = layer.weight.clamp_min(0), layer.weight.clamp_max(0)
            lc, uc = (torch.einsum('oi,nij->noj', wp, lc) + torch.einsum('oi,nij->noj', wn, uc),
                      torch.einsum('oi,nij->noj', wp, uc) + torch.einsum('oi,nij->noj', wn, lc))
            lb, ub = lb @ wp.T + ub @ wn.T + layer.bias, ub @ wp.T + lb @ wn.T + layer.bias
        elif isinstance(layer, nn.Tanh):
            lo = extrema(lc, lb, lower, upper)[0] - padding
            hi = extrema(uc, ub, lower, upper)[1] + padding
            width = hi - lo
            slope = torch.where(width > 1e-8,
                                (torch.tanh(hi)-torch.tanh(lo))/width.clamp_min(1e-15),
                                1-torch.tanh((hi+lo)/2).square()).clamp(1e-15, 1)
            root = torch.acosh(torch.rsqrt(slope))
            points = torch.stack([lo, hi, root.clamp(lo, hi), (-root).clamp(lo, hi)])
            offsets = torch.tanh(points)-slope*points
            lb, ub = slope*lb+offsets.amin(0)-padding, slope*ub+offsets.amax(0)+padding
            lc, uc = slope[..., None]*lc, slope[..., None]*uc
        else:
            raise TypeError(type(layer))
    return (extrema(lc, lb, lower, upper)[0]-padding,
            extrema(uc, ub, lower, upper)[1]+padding)
