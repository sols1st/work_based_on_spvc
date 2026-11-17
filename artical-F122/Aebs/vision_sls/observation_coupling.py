"""Invertible additive coupling for learned low-dimensional camera coordinates."""
import torch
from torch import nn


def make_shears(dim, hidden=12):
    """Each coordinate is translated using only the other current coordinates."""
    if dim < 2:
        raise ValueError("Additive cross-coordinate coupling needs at least two dimensions")
    nets = nn.ModuleList([
        nn.Sequential(nn.Linear(dim - 1, hidden), nn.Tanh(), nn.Linear(hidden, 1), nn.Tanh())
        for _ in range(dim)
    ])
    for net in nets:
        nn.init.zeros_(net[2].weight)
        nn.init.zeros_(net[2].bias)
    return nets


def apply_shears(z, nets, inverse=False):
    """Reverse the order and subtract the same shifts to recover the input."""
    if not len(nets):
        return z
    if z.ndim != 2 or z.shape[1] != len(nets):
        raise ValueError("The coordinate count must match the number of shear networks")
    columns = list(z.split(1, dim=1))
    order = reversed(range(len(nets))) if inverse else range(len(nets))
    scale = -0.25 if inverse else 0.25
    for i in order:
        others = torch.cat(columns[:i] + columns[i + 1:], dim=1)
        columns[i] = columns[i] + scale * nets[i](others)
    return torch.cat(columns, dim=1)
