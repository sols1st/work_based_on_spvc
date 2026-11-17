"""Synthesize a bounded PD policy and polyhedral barrier for the ideal outer loop."""
from fractions import Fraction
from itertools import product

import numpy as np

from . import plant


def level(state, design):
    p, v = np.split(state, 2, axis=-1)
    s = v + design['l'] * p
    return np.maximum(np.abs(p).max(axis=-1), np.abs(s).max(axis=-1) / design['s_limit'])


def barrier(state, design):
    return 20 * np.maximum(0, (level(state, design) - 1) / (plant.POS_LIMIT - 1))


def verify(design, q):
    """Exact scalar checks for a real-valued model with decimal design parameters.

    On N<=r, for 1<=r<=c, bound the induced row sums of the p and s updates.
    If N<=1 take r=1. Otherwise take r=N. Both give B(next)<=B(now).
    """
    if not np.isfinite(q) or q < 0:
        raise ValueError('finite nonnegative radius required')
    def decimal(value):
        return Fraction(str(value))
    l, m, S = (decimal(design[k]) for k in ('l', 'm', 's_limit'))
    if min(l, m, S) <= 0:
        raise ValueError('positive feedback rates and barrier width required')
    dt, c, wind, accel = map(decimal, (plant.DT, plant.POS_LIMIT, plant.WIND, plant.ACCEL))
    # The recorded float radius is taken exactly, including a possible upper rounding bit.
    radius = Fraction.from_float(float(q))
    margins = {
        'initial_position': 1 - decimal(plant.INIT_POS),
        'initial_transformed_velocity': S - (decimal(plant.INIT_VEL) + l * decimal(plant.INIT_POS)),
        'unsafe_velocity_separation': decimal(plant.VEL_LIMIT) - c * (l + S),
        'actuator_no_clipping': accel - (c * (l*l + (l+m)*S) + l*m*radius),
        'position_update': 1 - (abs(1-dt*l) + dt*S),
        'transformed_velocity_update': S - (abs(1-dt*m)*S + dt*(l*m*radius + wind)),
    }
    return dict(passed=all(x >= 0 for x in margins.values()),
                exact_margins={k: str(v) for k,v in margins.items()},
                margins={k: float(v) for k,v in margins.items()},
                q=float(q), design=dict(design), levels=[0,20],
                method='rational inequalities on real discrete-time model',
                floating_point_implementation_certified=False)


def synthesize():
    """Pick gains on development trajectories, before seeing any CP scores."""
    q_design = .35
    starts, winds, _ = plant.scenarios(128, 170633)
    rng = np.random.default_rng(170634)
    # Include nominal observations plus opposite persistent corner biases on the same starts.
    bias = rng.choice([-q_design, q_design], size=(len(starts), 3))
    errors = np.concatenate((np.zeros_like(bias), bias, -bias))
    history = []
    for l, m, ratio in product([.7,.9,1.1,1.3], [1.,1.4,1.8], [.62,.76,.9]):
        candidate = dict(l=l, m=m, s_limit=round(l*ratio, 3))
        audit = verify(candidate, q_design)
        if not audit['passed']:
            continue
        state = np.tile(starts, (3,1))
        objective = np.zeros(len(state))
        for k in range(plant.HORIZON):
            action = plant.control(state[:,:3] + errors, state[:,3:], candidate)
            state = plant.step(state, action, np.tile(winds[:,k], (3,1)))
            objective += (np.sum(state[:,:3]**2 + .15*state[:,3:]**2,axis=1)
                          + .02*np.sum(action**2,axis=1))
        robust_cost = float(objective.reshape(3,-1).max(axis=0).mean())
        history.append(dict(design=candidate, cost=robust_cost))
    if not history:
        raise RuntimeError('no feasible design on development set')
    chosen = min(history, key=lambda row: (row['cost'], -row['design']['s_limit']))
    return dict(selected=chosen['design'], design_radius=q_design, candidates=36,
                feasible=len(history), development_evaluation=history)
