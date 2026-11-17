"""State-indexed decoded-error boxes in the original five-state braking barrier."""
import numpy as np

from ..conformal.trajectory_boxes import cuts, width_bounds
from .box_trajectories import BASE
from .plant import SCALE
from .verify import verify_checkpoint


def features(states):
    """True obstacle distance and absolute heading are available in the offline proof."""
    states = np.asarray(states)
    return np.stack([states[..., 2], np.abs(states[..., 1])], axis=-1)


def audit(shape, q):
    """Upper-bound perceived station over the full safe sublevel, not trajectory samples."""
    if not np.isfinite(q) or q < 0:
        raise ValueError('nonnegative finite radius required')
    original = verify_checkpoint(BASE/'barrier.pt', 0.)
    # These bounds come from the four facets of the original B(x)<20 sublevel.
    d_lo, d_hi, max_heading = 1.92, 6.15, .19
    knots, _ = cuts(shape)
    grid = np.unique([d_lo, *[d for d in knots if d_lo < d < d_hi], d_hi])
    candidates = []
    # On an open distance cell, signed halfwidths are constant, or affine for
    # state_interp. The latter has no heading partition. In either case the
    # supremum of d+2*q*w_plus on the closed face/one-sided limit is obtained
    # at an endpoint. Explicitly include boundary values for right-open bins.
    for d in grid:
        _, positive = width_bounds(shape, [d, 0.], [d, max_heading])
        candidates.append((float(d + SCALE[2]*q*positive[2]), [float(d), float(d)]))
    if shape['name'] != 'state_interp':
        for left, right in zip(grid, grid[1:]):
            middle = (left+right)/2
            _, positive = width_bounds(shape, [middle, 0.], [middle, max_heading])
            candidates.append((float(right + SCALE[2]*q*positive[2]), [float(left),float(right)]))
    upper, active = max(candidates, key=lambda pair: pair[0])
    if not np.isfinite(upper):
        raise ValueError('nonfinite reachable perceived distance bound')
    _, global_positive = width_bounds(shape,[d_lo,0.],[d_hi,max_heading])
    global_slack = float(7.50 - (d_hi + SCALE[2]*q*global_positive[2]))
    slack = float(7.50-upper)
    return dict(passed=bool(original['passed'] and slack>1e-9),
                certification_metric=slack, unit='m', direction='larger',
                name='guaranteed spare positive camera-distance bias below brake guard',
                maximum_perceived_distance_upper_m=float(upper), active_distance_cell_m=active,
                global_envelope_slack_m=global_slack, cells=len(grid)-1,
                nominal_checkpoint_passed=original['passed'],
                other_unchanged_checks={k:v for k,v in original['checks'].items()
                                        if k not in ('finite_radius','brake_floor_guard_slack_m')},
                condition='All states in the safe sublevel, both curvatures and arbitrary allowed errors. '
                          'Measured d<=7.50 ensures commanded brake >=1.20 m/s^2; remaining component errors '
                          'enter only through steering/brake outputs confined by actuator bounds.')
