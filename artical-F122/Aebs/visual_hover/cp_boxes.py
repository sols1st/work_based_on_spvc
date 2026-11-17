"""Continuous-domain support of state-conditioned perception boxes for the frozen hover PD."""
from bisect import bisect_right
from fractions import Fraction as F
from itertools import product

import numpy as np

from ..conformal.trajectory_boxes import cuts
from . import plant
from .certificate import verify

# The original gain search is frozen for all policies and boxes in this study.
DESIGN = {'l': .7, 'm': 1., 's_limit': .63}


def features(states):
    p, v = np.split(np.asarray(states), 2, axis=-1)
    return np.stack((p[..., 2], np.abs(v + DESIGN['l'] * p).max(axis=-1)), axis=-1)


def exact(x):
    return F.from_float(float(x))


def regions(lo, hi, knots, extra):
    points = sorted(set([lo, hi] + [k for k in knots + extra if lo < k < hi]))
    cells = list(zip(points[:-1], points[1:]))
    # A right-continuous bin can begin exactly at the endpoint of the physical domain.
    if hi in knots:
        cells.append((hi, hi))
    return cells


def interpolation(knots, values, x):
    if x <= knots[0]:
        return values[0, 0]
    if x >= knots[-1]:
        return values[-1, 0]
    i = bisect_right(knots, x) - 1
    weight = (x - knots[i]) / (knots[i + 1] - knots[i])
    return (1 - weight)*values[i, 0] + weight*values[i + 1, 0]


def cell_max_widths(kind, z, t, knots, values):
    # All breakpoints occur on cell boundaries. A step's open cell is constant;
    # taking its closure is conservative, and the adjoining cell covers the boundary itself.
    a, b = knots
    if kind == 'state_interp':
        return np.maximum(interpolation(a, values, z[0]), interpolation(a, values, z[1]))
    return values[bisect_right(a, sum(z)/2), bisect_right(b, sum(t)/2)]


def audit(shape, q):
    """Lower bound on tolerable wind from the actual box on each height/speed region."""
    if not np.isfinite(q) or q < 0:
        raise ValueError('Invalid CP radius')
    l, m, S = (F(str(DESIGN[k])) for k in ['l', 'm', 's_limit'])
    dt, c, w0, accel = (F(str(x)) for x in [plant.DT, plant.POS_LIMIT, plant.WIND, plant.ACCEL])
    alpha = abs(1-dt*m)
    knots = tuple([exact(x) for x in k] for k in cuts(shape))
    values = np.array([exact(x) for x in np.asarray(shape['halfwidths']).flat], dtype=object)
    values = values.reshape(np.shape(shape['halfwidths'])) * exact(q)
    if values.shape[2:] != (2, 3) or any(x < 0 for x in values.flat):
        raise ValueError('Expected finite nonnegative signed widths on three hover axes')
    zcells = regions(-c, c, knots[0], [F(-1),F(0),F(1)])
    tcells = regions(F(0), c*S, knots[1], [S])
    cap = [None]*3
    culprit = [None]*3
    headroom = None
    for z,t in product(zcells,tcells):
        neg,pos = cell_max_widths(shape['name'], z, t, knots, values)
        height_min = max(F(0), z[0], -z[1])
        # r=max(1,N) >= max(1, |height|, t/S). This convex piecewise-linear
        # wind allowance attains its minimum at the clipped kink, or at an endpoint.
        kink = S * max(F(1),height_min)
        tcrit = min(max(kink,t[0]),t[1])
        wind_budget = (max(kink,tcrit)-alpha*tcrit)/dt
        for i in range(3):
            w = wind_budget-l*m*max(neg[i],pos[i])
            if cap[i] is None or w < cap[i]:
                cap[i] = w
                culprit[i] = {'height':list(map(float,z)),'transformed_speed':list(map(float,t)),
                              'critical_speed':float(tcrit),'signed_error_bound':[float(neg[i]),float(pos[i])]}
            plo, phi = z if i==2 else (-c,c)
            # u=l² p-(l+m)s-lm e must be inside the no-clipping interval on every cell.
            umax = (l+m)*t[1] + max(l*l*phi+l*m*neg[i], -l*l*plo+l*m*pos[i])
            slack = accel-umax
            headroom = slack if headroom is None else min(headroom,slack)
    base = verify(DESIGN,0.)  # initial, unsafe and p-update are independent of perception
    fixed_keys = ['initial_position','initial_transformed_velocity','unsafe_velocity_separation','position_update']
    fixed_ok = all(F(base['exact_margins'][k]) >= 0 for k in fixed_keys)
    w = min(cap)
    return dict(passed=bool(fixed_ok and headroom>=0 and w>=w0),
                certification_metric=float(w) if headroom>=0 and fixed_ok and w>=0 else 0.,
                unit='normalized acceleration per axis (s^-2)',
                meaning='sufficient total symmetric componentwise wind bound; original task wind is 0.08',
                exact_wind_bound=str(w), wind_per_axis=list(map(float,cap)),
                wind_margin_above_nominal=float(w-w0), thrust_headroom=float(headroom),
                exact_thrust_headroom=str(headroom), original_wind=float(w0),
                state_cells=len(zcells)*len(tcells), limiting_cells=culprit,
                independent_geometry={k:base['exact_margins'][k] for k in fixed_keys},
                design=DESIGN, proof='rational cell bounds for continuous p,v, signed errors and wind on original real model')
