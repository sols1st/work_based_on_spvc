"""Parking task constants and RK4 dynamics adapted from VISION-SLS.

Source: dyn/unicycle.py, dyn/model.py, expe/parking_lot/main_parking_lot.py.
Copyright (c) 2026 Antoine Leeman and Shuyu Zhan, MIT; see UPSTREAM_LICENSE.
No clipping of physical state, no added noise law, no safety certificate.
"""
import numpy as np

DT = 0.15
HORIZON = 30
STATE_LOW = np.array([-3.5, -2., -2 * np.pi, -1.])
STATE_HIGH = np.array([.5, 1.9, 2 * np.pi, 2.])
ACTION_LOW = np.array([-np.pi, -4.])
ACTION_HIGH = -ACTION_LOW
RENDER_OFFSET = np.array([3., 0., 0., 0.])
C = np.array([[.97613788, .13243951, -.17208897, 0.],
              [-.21648932, .66782951, -.7121349, 0.],
              [.02088745, .73288924, .68002731, 0.]])
OBSTACLES = np.array([[-3.55, .7, .4], [-.6, 1.15, .35], [-1.2, -1.2, .4]])
INITIAL_STATES = np.array([
    [-2.1, -1.75, np.pi/2, 0.], [-2.5, -1.5, 0., 0.], [-2.5, -1., 0., 0.],
    [-2., 1.5, 0., 0.], [-1.25, 1.2, 0., 0.], [-.75, 1.75, -np.pi/2, 0.],
    [-1.5, -1.75, np.pi/2, 0.], [-2.5, 0., np.pi*.35, 0.],
    [-2.5, .5, np.pi/4, 0.], [-2.5, 1., 0., 0.], [-2.75, 1.25, np.pi/4, 0.],
    [-2., -1.5, 0., 0.], [-2.5, -1.5, 0., 0.], [-2.25, -.75, 0., 0.],
    [-2.2, -.6, .6, 0.],
])
GOAL_LOW = np.array([[0., .35, -2*np.pi, -1.]] +
                    [[-.2, -.2, -2*np.pi, -1.]]*6 +
                    [[-.2, 1.3, -2*np.pi, -1.]]*4 +
                    [[-.2, -1.45, -2*np.pi, -1.]]*3 +
                    [[-.2, -.2, -2*np.pi, -1.]])
GOAL_HIGH = np.array([[.45, .65, 2*np.pi, 1.]] +
                     [[.2, .2, 2*np.pi, 1.]]*6 +
                     [[.2, 1.7, 2*np.pi, 1.]]*4 +
                     [[.2, -1.05, 2*np.pi, 1.]]*3 +
                     [[.2, .2, 2*np.pi, 1.]])


def vector(value, size, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError('%s must be a finite vector of length %d' % (name, size))
    return result.copy()


def dynamics(state, action, disturbance=None, dt=DT):
    """Original RK4 unicycle ODE; E=.01 I acts INSIDE the ODE.

    Action is [yaw_rate rad/s, acceleration m/s^2], not AEBS braking.
    Disturbance is explicitly supplied and constant over the RK4 step.
    This function deliberately does not impose action/state clipping.
    """
    x, u = vector(state, 4, 'state'), vector(action, 2, 'action')
    w = np.zeros(4) if disturbance is None else vector(disturbance, 4, 'disturbance')
    if not np.isfinite(dt) or dt <= 0 or np.max(np.abs(w)) > 1.:
        raise ValueError('positive finite dt and disturbance in [-1,1]^4 required')

    def ode(z):
        return np.array([z[3]*np.cos(z[2]), z[3]*np.sin(z[2]), u[0], u[1]]) + .01*w

    k1 = ode(x)
    k2 = ode(x + dt*.5*k1)
    k3 = ode(x + dt*.5*k2)
    k4 = ode(x + dt*k3)
    return x + dt*(k1 + 2*k2 + 2*k3 + k4)/6.


def nominal_observation(state):
    """Match upstream: C @ (world_state + [3,0,0,0]). No F(x) yet."""
    return C @ (vector(state, 4, 'state') + RENDER_OFFSET)


def constraints(state, scenario=0):
    """Original sampled point-center collision test, not swept car-body collision."""
    x = vector(state, 4, 'state')
    if not isinstance(scenario, (int, np.integer)) or not 0 <= scenario < 15:
        raise ValueError('scenario must be 0..14')
    distances = np.sum((OBSTACLES[:, :2] - x[:2])**2, axis=1)
    return dict(collision=bool(np.any(distances <= OBSTACLES[:, 2]**2 + 1e-4)),
                in_bounds=bool(np.all(x >= STATE_LOW-1e-3) and np.all(x <= STATE_HIGH+1e-3)),
                in_goal=bool(np.all(x >= GOAL_LOW[scenario]-1e-3) and
                             np.all(x <= GOAL_HIGH[scenario]+1e-3)),
                in_goal_xy=bool(np.all(x[:2] >= GOAL_LOW[scenario, :2]-1e-3) and
                                np.all(x[:2] <= GOAL_HIGH[scenario, :2]+1e-3)))
