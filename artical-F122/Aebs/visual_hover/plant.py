"""Outer-loop translational model, in normalized position and velocity units."""
import numpy as np

SCALE = np.array([0.75, 0.65, 0.50])  # metres per normalized position unit
HEIGHT = 2.0
DT = 0.25
HORIZON = 24
WIND = 0.08
ACCEL = 4.4
INIT_POS = 0.56
INIT_VEL = 0.16
POS_LIMIT = 1.25
VEL_LIMIT = 2.70
TARGET_POS = 0.18
TARGET_VEL = 0.14
DWELL = 4


def scenarios(n, seed):
    rng = np.random.default_rng(seed)
    position = rng.uniform(-INIT_POS, INIT_POS, (n, 3))
    velocity = rng.uniform(-INIT_VEL, INIT_VEL, (n, 3))
    # Each trajectory draws an independent wind bias and gust sequence.
    bias = rng.uniform(-0.06, 0.06, (n, 1, 3))
    gust = rng.uniform(-0.02, 0.02, (n, HORIZON, 3))
    return np.column_stack((position, velocity)), bias + gust, rng.integers(0, 3, n)


def control(observation, velocity, design, use_vision=True):
    l, m = design['l'], design['m']
    accel = -(l + m) * velocity
    if use_vision:
        accel = accel - l * m * observation
    return np.clip(accel, -ACCEL, ACCEL)


def step(state, action, wind):
    p, v = np.split(state, 2, axis=-1)
    return np.concatenate((p + DT * v, v + DT * (action + wind)), axis=-1)


def outcomes(states):
    p, v = np.split(np.abs(states), 2, axis=-1)
    safe = (p.max(axis=(1, 2)) < POS_LIMIT) & (v.max(axis=(1, 2)) < VEL_LIMIT)
    dwell = ((p[:, -DWELL:].max(axis=(1, 2)) <= TARGET_POS) &
             (v[:, -DWELL:].max(axis=(1, 2)) <= TARGET_VEL))
    final_distance = np.linalg.norm(states[:, -1, :3] * SCALE, axis=1)
    return safe, safe & dwell, final_distance
