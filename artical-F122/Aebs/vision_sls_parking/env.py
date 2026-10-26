"""Low-level reset/step/render interface; NOT yet a PPO Gym environment.

No reward or policy is introduced in this connection-only stage. Physical
state in diagnostics is privileged; future image PPO must not consume it.
"""
import numpy as np
from .core import (DT, HORIZON, INITIAL_STATES, ACTION_LOW, ACTION_HIGH,
                   vector, dynamics, constraints, nominal_observation)


class ParkingEnv:
    def __init__(self, renderer=None, observation_delay=1, horizon=HORIZON):
        if observation_delay not in (0, 1) or not isinstance(horizon, int) or horizon < 1:
            raise ValueError('delay 0/1 and positive integer horizon required')
        self.renderer, self.delay, self.horizon = renderer, observation_delay, horizon
        self.history = []

    def reset(self, scenario=0, state=None):
        if not isinstance(scenario, (int, np.integer)) or not 0 <= scenario < 15:
            raise ValueError('scenario must be 0..14')
        self.scenario, self.step_index = int(scenario), 0
        self.state = vector(INITIAL_STATES[scenario] if state is None else state, 4, 'state')
        self.history = [self.state.copy()]
        self.unsafe_seen = not self._safe(self.state)
        self.done = False
        return self.observe()

    def _safe(self, state):
        flags = constraints(state, self.scenario)
        return flags['in_bounds'] and not flags['collision']

    def observe(self):
        if not self.history:
            raise RuntimeError('reset first')
        observed_index = max(0, self.step_index-self.delay)
        capture_state = self.history[observed_index].copy()
        rgb = None if self.renderer is None else self.renderer.render(capture_state)
        return dict(rgb=rgb, observation_state=capture_state,
                    observation_index=observed_index, current_state=self.state.copy(),
                    step=self.step_index, nominal_observation=nominal_observation(capture_state),
                    privileged_state_for_diagnostic_only=True)

    def step(self, action, disturbance=None):
        if not self.history or self.done:
            raise RuntimeError('reset before step / after horizon')
        requested = vector(action, 2, 'action')
        applied = np.clip(requested, ACTION_LOW, ACTION_HIGH)
        self.state = dynamics(self.state, applied, disturbance, DT)
        self.history.append(self.state.copy())
        self.step_index += 1
        self.unsafe_seen = self.unsafe_seen or not self._safe(self.state)
        self.done = self.step_index >= self.horizon
        flags = constraints(self.state, self.scenario)
        # As upstream, evaluate the fixed horizon instead of stopping on goal.
        outcome = None
        if self.done:
            outcome = ('constraint_violation' if self.unsafe_seen or not flags['in_goal']
                       else 'success')
        info = dict(flags, action_requested=requested, action_applied=applied,
                    action_clipped=bool(np.any(applied != requested)),
                    unsafe_seen=self.unsafe_seen, outcome=outcome,
                    upstream_arrival_success_at_horizon=bool(
                        self.done and flags['in_goal_xy'] and
                        not any(constraints(s, self.scenario)['collision'] for s in self.history)))
        return self.observe(), self.done, info

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
