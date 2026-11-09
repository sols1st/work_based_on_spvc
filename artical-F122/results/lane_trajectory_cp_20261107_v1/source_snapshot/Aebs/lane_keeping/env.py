"""Gymnasium environments for semantic and error-aware lane keeping.

The plant is always the true 2-D state.  ``LaneKeepingEnv`` exposes the true
state (oracle); ``NoisyLaneEnv`` corrupts the observed state with a box sample
from a calibrated visual-error model.  The image policy is evaluated by
replacing the observation with a frozen DINO projection; training it on the
calibrated error model avoids rendering every PPO step.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from . import core


class LaneKeepingEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, kappa=None, horizon=core.HORIZON, randomize_lighting=False,
                 renderer=None, seed=None, initial_low=None, initial_high=None):
        super().__init__()
        if kappa is not None and not np.isfinite(kappa):
            raise ValueError('finite curvature or None required')
        self.fixed_kappa = kappa
        self.horizon = int(horizon)
        self.randomize_lighting = bool(randomize_lighting)
        self.renderer = renderer
        self.action_space = spaces.Box(-1., 1., (1,), dtype=np.float32)
        self.observation_space = spaces.Box(-2., 2., (3,), dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self.initial_low = (core.INITIAL.low.copy() if initial_low is None
                            else np.asarray(initial_low, dtype=np.float64).reshape(2))
        self.initial_high = (core.INITIAL.high.copy() if initial_high is None
                             else np.asarray(initial_high, dtype=np.float64).reshape(2))
        if np.any(self.initial_low > self.initial_high):
            raise ValueError('initial low must not exceed high')
        self.state = np.zeros(2)
        self.kappa = 0.
        self.steps = 0
        self.unsafe_seen = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = self.rng.uniform(self.initial_low, self.initial_high)
        self.kappa = self.fixed_kappa if self.fixed_kappa is not None else float(
            self.rng.choice(np.asarray(core.KAPPAS)))
        self.steps = 0
        self.unsafe_seen = False
        return self._observation(), {}

    def _observation(self):
        return core.to_net(self.state, self.kappa).astype(np.float32)

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(-1)[0]
        phi = core.action_from_net(action)
        previous = self.state.copy()
        self.state = core.dynamics(previous, phi, self.kappa)
        self.steps += 1
        self.unsafe_seen = self.unsafe_seen or core.unsafe(self.state)
        terminated = bool(self.unsafe_seen)
        truncated = self.steps >= self.horizon and not terminated
        reward = core.reward(previous, phi, self.state, self.kappa)
        if terminated:
            reward -= 10.
        info = dict(success=bool(not self.unsafe_seen and self.steps >= self.horizon),
                    unsafe=bool(self.unsafe_seen), margin=core.margin(self.state),
                    kappa=self.kappa)
        return self._observation(), float(reward), terminated, truncated, info


class NoisyLaneEnv(LaneKeepingEnv):
    """Observation = true normalized state + a box visual-error sample.

    ``error_model`` must provide ``sample(rng, n)`` returning normalized
    [de, dtheta] samples.  A global box is the default; a state-dependent model
    can be passed later without changing the environment.
    """

    def __init__(self, error_model, **kwargs):
        super().__init__(**kwargs)
        self.error_model = error_model

    def _observation(self):
        observation = core.to_net(self.state, self.kappa)
        observation[:2] += self.error_model.sample(self.rng, 1)[0]
        return observation.astype(np.float32)


class BoxErrorModel:
    """Global normalized box, symmetric in e and theta."""

    def __init__(self, radius):
        radius = float(radius)
        if not np.isfinite(radius) or radius < 0:
            raise ValueError('nonnegative finite radius required')
        self.radius = radius

    def sample(self, rng, n):
        return rng.uniform(-self.radius, self.radius, size=(int(n), 2))

    def to_dict(self):
        return dict(kind='global_box', radius=self.radius)

    @classmethod
    def from_dict(cls, value):
        if value.get('kind') != 'global_box':
            raise ValueError('unknown error model kind')
        return cls(value['radius'])


def rollout(policy, env, episodes=20, seed=0, deterministic=True):
    """Run episodes and return per-episode outcomes and state traces."""
    rng = np.random.default_rng(seed)
    records = []
    for episode in range(int(episodes)):
        observation, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        done = False
        states, actions = [], []
        while not done:
            if hasattr(policy, 'predict'):
                action, _ = policy.predict(observation, deterministic=deterministic)
            else:
                import torch
                with torch.no_grad():
                    action = policy(torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0))
                    action = action.detach().numpy().reshape(-1)
                    action = np.clip(action, env.action_space.low, env.action_space.high)
            observation, reward, terminated, truncated, info = env.step(action)
            states.append(env.state.copy())
            actions.append(float(np.asarray(action).reshape(-1)[0]))
            done = terminated or truncated
        states = np.asarray(states)
        records.append(dict(episode=episode, success=bool(info['success']),
                            unsafe=bool(info['unsafe']),
                            margin=float(np.min([core.margin(s) for s in states])),
                            mean_abs_e=float(np.mean(np.abs(states[:, 0]))),
                            max_abs_e=float(np.max(np.abs(states[:, 0]))),
                            kappa=float(info['kappa']), states=states.tolist(),
                            actions=actions))
    return records


def summarise(records):
    return dict(episodes=len(records),
                success=int(sum(r['success'] for r in records)),
                unsafe=int(sum(r['unsafe'] for r in records)),
                mean_margin=float(np.mean([r['margin'] for r in records])),
                mean_abs_e=float(np.mean([r['mean_abs_e'] for r in records])),
                max_abs_e=float(np.max([r['max_abs_e'] for r in records])))


class ResidualLaneEnv(gym.Env):
    """PPO environment whose action is the bounded residual r in [-1, 1].

    The executed normalized action is
        u = clip(u_lqr(x) + eps * clamp(B(x), 0, 1) * r)
    where B is the quadratic SBC.  The certificate covers every residual net
    with |tanh| <= 1 and the chosen eps, so training cannot invalidate it.
    """

    def __init__(self, K, P, eps=.5, flat_c=None, **kwargs):
        super().__init__()
        self.base = LaneKeepingEnv(**kwargs)
        self.K = np.asarray(K, dtype=np.float64).reshape(1, 2)
        self.P = np.asarray(P, dtype=np.float64)
        self.eps = float(eps)
        self.flat_c = None if flat_c is None else float(flat_c)
        self.action_space = spaces.Box(-1., 1., (1,), dtype=np.float32)
        self.observation_space = self.base.observation_space

    def reset(self, seed=None, options=None):
        return self.base.reset(seed=seed, options=options)

    def step(self, action):
        observation = self.base._observation()
        state = np.asarray(observation[:2], dtype=np.float64)
        kappa = float(observation[2]) * core.KAPPA_SCALE
        u_lqr = -(self.K @ state)[0] + np.arctan(core.WHEELBASE * kappa) / core.PHI_MAX
        B = float(state @ self.P @ state)
        if self.flat_c is not None:
            B = max(B - self.flat_c, 0.)
        scale = min(max(B, 0.), 1.)
        residual = float(np.clip(np.asarray(action, dtype=np.float64).reshape(-1)[0], -1., 1.))
        u = float(np.clip(u_lqr + self.eps * scale * residual, -1., 1.))
        observation, reward, terminated, truncated, info = self.base.step(np.array([u]))
        info['residual'] = residual
        info['executed_action'] = u
        return observation, reward, terminated, truncated, info
