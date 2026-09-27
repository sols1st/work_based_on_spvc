"""Separate read-only VT verifier with a correctly perturbed interval input.

The original Aebs.VT.verify module is intentionally left untouched so old
results remain reproducible. This class repairs only its two auto_LiRPA input
calls; all other VT grid, noise, region, and threshold code is inherited.
"""

import numpy as np
import torch
from auto_LiRPA import BoundedTensor, PerturbationLpNorm

from Aebs.VT.verify import VTVerifier


class CorrectedIntervalVTVerifier(VTVerifier):
    def _interval_bounds(self, lower, upper):
        if lower.shape != upper.shape or lower.ndim != 2:
            raise ValueError("Interval endpoints must be matching [batch, dimension] tensors")
        if not torch.all(lower <= upper):
            raise ValueError("Interval lower bound exceeds upper bound")
        center = (lower + upper) / 2
        perturbation = PerturbationLpNorm(norm=np.inf, x_L=lower, x_U=upper)
        bounded_input = BoundedTensor(center, perturbation)
        return self.l_ibp.compute_bounds(x=(bounded_input,), method="IBP")

    def compute_bounds_on_set(self, grid_lb, grid_ub):
        if not len(grid_lb):
            raise ValueError("Bound set is empty")
        global_min = float("inf")
        global_max = float("-inf")
        with torch.no_grad():
            for start in range(0, len(grid_lb), self.batch_size):
                end = start + self.batch_size
                lower = torch.as_tensor(grid_lb[start:end], dtype=torch.float32,
                                        device=self.device)
                upper = torch.as_tensor(grid_ub[start:end], dtype=torch.float32,
                                        device=self.device)
                batch_lower, batch_upper = self._interval_bounds(lower, upper)
                global_min = min(global_min, float(batch_lower.min().item()))
                global_max = max(global_max, float(batch_upper.max().item()))
        return global_min, global_max

    def compute_expected_l(self, s, a, pmass, batched_grid_lb, batched_grid_ub):
        deterministic_next = self.env.v_next(s, a)
        cells = batched_grid_lb.shape[0]
        lower = (deterministic_next[:, None, :] + batched_grid_lb[None, :, :]).reshape(-1, 2)
        upper = (deterministic_next[:, None, :] + batched_grid_ub[None, :, :]).reshape(-1, 2)
        with torch.no_grad():
            _, cell_upper = self._interval_bounds(lower, upper)
            upper_by_state = cell_upper.reshape(len(s), cells)
            return (upper_by_state * pmass.reshape(1, cells)).sum(dim=1)
