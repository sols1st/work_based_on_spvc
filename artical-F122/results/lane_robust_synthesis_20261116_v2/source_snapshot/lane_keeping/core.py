"""Lane-keeping task: 2-D bicycle error model, normalized network coordinates.

Physical state is ``(e, theta)``: lateral offset from the lane centre and
heading error relative to the road tangent.  The road curvature ``kappa`` is a
known exogenous input.  Network coordinates map ``e``, ``theta`` and ``kappa``
to roughly [-1, 1] so a small MLP can learn and verify.

This module deliberately contains no reward, no policy and no certificate; it
only defines the plant, the safety set and coordinate conversions.
"""
from dataclasses import dataclass
import numpy as np

DT = .1
SPEED = 4.
WHEELBASE = 2.6
HORIZON = 80

E_MAX = .85
THETA_MAX = .30
PHI_MAX = .45
KAPPA_SCALE = .01
KAPPAS = (-.01, 0., .01)

E_SCALE = E_MAX
THETA_SCALE = THETA_MAX

INIT_E = .25
INIT_THETA = .06


@dataclass(frozen=True)
class Box:
    low: np.ndarray
    high: np.ndarray

    def __post_init__(self):
        low = np.asarray(self.low, dtype=np.float64)
        high = np.asarray(self.high, dtype=np.float64)
        if low.shape != high.shape or not np.isfinite(low).all() or not np.isfinite(high).all():
            raise ValueError('finite matching box bounds required')
        if np.any(low > high):
            raise ValueError('box low must not exceed high')
        object.__setattr__(self, 'low', low)
        object.__setattr__(self, 'high', high)

    def sample(self, rng, n):
        return rng.uniform(self.low, self.high, size=(n, len(self.low)))

    def contains(self, points):
        points = np.asarray(points, dtype=np.float64)
        return np.all((points >= self.low) & (points <= self.high), axis=-1)


DOMAIN = Box(np.array([-E_MAX, -THETA_MAX]), np.array([E_MAX, THETA_MAX]))
INITIAL = Box(np.array([-INIT_E, -INIT_THETA]), np.array([INIT_E, INIT_THETA]))
# The unsafe set is a thin strip at the domain boundary so grid cells can
# intersect it; the task's violation test uses the same thresholds.
UNSAFE = [
    Box(np.array([-E_MAX - .05, -THETA_MAX - .05]), np.array([-E_MAX + .02, THETA_MAX + .05])),
    Box(np.array([E_MAX - .02, -THETA_MAX - .05]), np.array([E_MAX + .05, THETA_MAX + .05])),
    Box(np.array([-E_MAX - .05, -THETA_MAX - .05]), np.array([E_MAX + .05, -THETA_MAX + .02])),
    Box(np.array([-E_MAX - .05, THETA_MAX - .02]), np.array([E_MAX + .05, THETA_MAX + .05])),
]


def dynamics(state, action, kappa=0., dt=DT, disturbance=None):
    """One RK-Euler step of the lateral bicycle model (physical units).

    ``action`` is the front-wheel angle in radians.  ``disturbance`` is an
    optional [de, dtheta] perturbation used for robustness training; it is not
    the visual error, which enters through the policy observation instead.
    """
    state = np.asarray(state, dtype=np.float64).reshape(2)
    action = np.clip(np.asarray(action, dtype=np.float64).reshape(-1)[0], -PHI_MAX, PHI_MAX)
    kappa = float(kappa)
    if not np.isfinite(state).all() or not np.isfinite([action, kappa, dt]).all() or dt <= 0:
        raise ValueError('finite state, action, curvature and positive dt required')
    e, theta = state
    e_next = e + SPEED * dt * np.sin(theta)
    theta_next = theta + SPEED * dt / WHEELBASE * np.tan(action) - SPEED * dt * kappa
    if disturbance is not None:
        disturbance = np.asarray(disturbance, dtype=np.float64).reshape(2)
        e_next += disturbance[0]
        theta_next += disturbance[1]
    return np.array([e_next, theta_next])


def dynamics_batch(states, actions, kappas, dt=DT):
    states = np.asarray(states, dtype=np.float64)
    actions = np.clip(np.asarray(actions, dtype=np.float64).reshape(-1), -PHI_MAX, PHI_MAX)
    kappas = np.asarray(kappas, dtype=np.float64).reshape(-1)
    e, theta = states[:, 0], states[:, 1]
    return np.stack([e + SPEED * dt * np.sin(theta),
                     theta + SPEED * dt / WHEELBASE * np.tan(actions) - SPEED * dt * kappas],
                    axis=1)


def to_net(state, kappa):
    """Physical (e, theta) and curvature -> normalized network input [3]."""
    state = np.asarray(state, dtype=np.float64).reshape(2)
    return np.array([state[0] / E_SCALE, state[1] / THETA_SCALE, float(kappa) / KAPPA_SCALE])


def to_net_batch(states, kappas):
    states = np.asarray(states, dtype=np.float64)
    kappas = np.asarray(kappas, dtype=np.float64).reshape(-1)
    return np.stack([states[:, 0] / E_SCALE, states[:, 1] / THETA_SCALE, kappas / KAPPA_SCALE], axis=1)


def state_to_net(state):
    state = np.asarray(state, dtype=np.float64).reshape(2)
    return np.array([state[0] / E_SCALE, state[1] / THETA_SCALE])


def action_to_net(action):
    return float(np.clip(action, -PHI_MAX, PHI_MAX) / PHI_MAX)


def action_from_net(action):
    return float(np.clip(action, -1., 1.)) * PHI_MAX


def unsafe(state):
    state = np.asarray(state, dtype=np.float64)
    return bool(abs(state[0]) > E_MAX or abs(state[1]) > THETA_MAX)


def margin(state):
    state = np.asarray(state, dtype=np.float64)
    return float(min(E_MAX - abs(state[0]), THETA_MAX - abs(state[1])))


def reward(state, action, next_state, kappa=0.):
    """Dense lane-centering reward with a small action penalty."""
    e, theta = np.asarray(next_state, dtype=np.float64)
    return float(1. - .5 * (e / E_SCALE) ** 2 - .2 * (theta / THETA_SCALE) ** 2
                 - .02 * (float(action) / PHI_MAX) ** 2)


def sample_initial(rng, n):
    return INITIAL.sample(rng, n)


def sample_kappa(rng, n):
    return rng.choice(np.asarray(KAPPAS), size=n)


def sample_observation_error(rng, n, radius):
    """Uniform box sample in normalized observation coordinates."""
    return rng.uniform(-radius, radius, size=(n, 2))
