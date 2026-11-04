"""Gym wrapper for bounded PPO residuals around a visual planning controller."""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .core import ACTION_HIGH, INITIAL_STATES, GOAL_LOW, GOAL_HIGH, constraints
from .ppo_env import ParkingPPOEnv
from .planning_feedback import ReferenceBank, PlannedController


def perturbed_initial_state(scenario, rng, jitter):
    if not np.isfinite(jitter) or jitter < 0:
        raise ValueError('nonnegative finite initial jitter required')
    for _ in range(100):
        x = INITIAL_STATES[scenario]+rng.uniform(-jitter, jitter, 4)
        c = constraints(x, scenario)
        if c['in_bounds'] and not c['collision']:
            return x
    raise RuntimeError('no safe perturbed initial state sampled')


class PlannedParkingEnv(gym.Env):
    metadata = {'render_modes': []}
    def __init__(self, references, sensor=None, mode='feedback', initial_jitter=0.):
        super().__init__()
        if not np.isfinite(initial_jitter) or initial_jitter < 0:
            raise ValueError('nonnegative finite initial jitter required')
        self.bank = references if isinstance(references, ReferenceBank) else ReferenceBank(references)
        self.controller = PlannedController(self.bank, mode=mode)
        self.inner = ParkingPPOEnv(image_sensor=sensor, reward_mode='safety_v3')
        self.initial_jitter = initial_jitter
        self.action_space = spaces.Box(-1., 1., (2,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (13,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = dict(options or {})
        scenario = int(options['scenario'] if 'scenario' in options else self.np_random.integers(15))
        options['scenario'] = scenario
        if self.initial_jitter and 'state' not in options:
            options['state'] = perturbed_initial_state(scenario, self.np_random, self.initial_jitter)
        self.inner.reset(seed=seed, options=options)
        goal = (GOAL_LOW[scenario, :2]+GOAL_HIGH[scenario, :2])/2
        # Only the measured y and commanded goal cross into the controller.
        obs = self.controller.reset(self.inner.measurements[-1], goal)
        return obs, dict(selected_reference=self.controller.reference)

    def step(self, residual):
        action = self.controller.action(residual)
        _, reward, terminated, truncated, info = self.inner.step(action/ACTION_HIGH)
        obs = self.controller.receive(self.inner.measurements[-1])
        info.update(selected_reference=self.controller.reference, estimated_state=self.controller.estimate.copy())
        return obs, reward, terminated, truncated, info

    def close(self):
        self.inner.close()


def evaluate_planned(references, policy=None, sensor=None, mode='feedback', jitter=0., seed=701):
    env = PlannedParkingEnv(references, sensor=sensor, mode=mode, initial_jitter=jitter)
    rows, paths, actions, estimates, observations = [], [], [], [], []
    try:
        for scenario in range(15):
            obs, initial_info = env.reset(seed=seed+scenario, options={'scenario': scenario})
            states = [env.inner.task.state.copy()]
            belief = [env.controller.estimate.copy()]
            measurements = [env.inner.measurements[-1].copy()]; commands = []
            for _ in range(30):
                residual = np.zeros(2) if policy is None else policy.predict(obs, deterministic=True)[0]
                obs, _, _, done, info = env.step(residual)
                states.append(env.inner.task.state.copy()); belief.append(env.controller.estimate.copy())
                measurements.append(env.inner.measurements[-1].copy()); commands.append(info['action_applied'])
            rows.append(dict(scenario=scenario, success=info['episode_success'],
                             collision=info['collision_seen'], out_of_bounds=info['out_of_bounds_seen'],
                             final_goal_distance=info['final_goal_distance'],
                             selected_reference=initial_info['selected_reference']))
            paths.append(states); actions.append(commands); estimates.append(belief); observations.append(measurements)
    finally:
        env.close()
    return dict(episodes=len(rows), success_count=sum(r['success'] for r in rows),
                collision_count=sum(r['collision'] for r in rows),
                out_of_bounds_count=sum(r['out_of_bounds'] for r in rows),
                mean_final_goal_distance=float(np.mean([r['final_goal_distance'] for r in rows])),
                reference_selection_correct_count=sum(r['selected_reference']==r['scenario'] for r in rows),
                per_scenario=rows), dict(states=np.asarray(paths), actions=np.asarray(actions),
                                        estimates=np.asarray(estimates), observations=np.asarray(observations))
