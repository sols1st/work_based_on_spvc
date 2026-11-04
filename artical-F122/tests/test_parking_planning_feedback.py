import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from Aebs.vision_sls_parking.core import INITIAL_STATES, GOAL_LOW, GOAL_HIGH, dynamics, nominal_observation, ACTION_HIGH
from Aebs.vision_sls_parking.planning_feedback import ReferenceBank, PlannedController, feedback_gains
from Aebs.vision_sls_parking.planned_env import PlannedParkingEnv, perturbed_initial_state
from Aebs.vision_sls_parking.planned_transition import PlannedTransition
from Aebs.vision_sls_parking.ppo_env import ParkingPPOEnv


class PlanningFeedbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        actions = np.random.default_rng(92).uniform(-.1, .1, (15, 30, 2))
        paths, gains, observers = [], [], []
        for initial, commands in zip(INITIAL_STATES, actions):
            states = [initial]
            for u in commands:
                states.append(dynamics(states[-1], u))
            k, l = feedback_gains(states, commands)
            paths.append(states); gains.append(k); observers.append(l)
        cls.path = Path(cls.directory.name)/'bank.npz'
        np.savez(cls.path, states=paths, actions=actions, feedback=gains, observer=observers,
                 goals=(GOAL_LOW[:, :2]+GOAL_HIGH[:, :2])/2)
        cls.bank = ReferenceBank(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_nominal_observer_exact_over_delayed_episode(self):
        for scenario in range(15):
            env = PlannedParkingEnv(self.bank)
            try:
                obs, info = env.reset(options={'scenario': scenario})
                self.assertEqual(info['selected_reference'], scenario)
                for t in range(30):
                    obs, _, terminated, done, _ = env.step(np.zeros(2))
                    self.assertEqual(obs.shape, (13,))
                    self.assertFalse(terminated); self.assertEqual(done, t == 29)
                    np.testing.assert_allclose(env.controller.estimate, env.inner.task.state, atol=1e-11)
            finally:
                env.close()

    def test_repeated_initial_frame_is_not_incorporated_twice(self):
        controller = PlannedController(self.bank)
        controller.reset(nominal_observation(INITIAL_STATES[0]), self.bank.goals[0])
        before = controller.estimate.copy(); action = controller.action()
        controller.receive(np.array([100., -100., 50.]))
        np.testing.assert_allclose(controller.estimate, dynamics(before, action), atol=1e-12)

    def test_torch_transition_matches_actual_delayed_sensor_and_residual_actor(self):
        error = np.array([.07, -.04, .05])
        class Sensor:
            def reset(self):
                pass
            def __call__(self, x):
                return nominal_observation(x)+error
        env = PlannedParkingEnv(self.bank, sensor=Sensor(), initial_jitter=.04)
        torch.manual_seed(49)
        policy = PPO('MlpPolicy', env, n_steps=32, batch_size=32, device='cpu')
        transition = PlannedTransition(self.bank, policy)
        try:
            obs, info = env.reset(seed=703, options={'scenario': 2})
            reference = info['selected_reference']
            for t in range(30):
                state = torch.tensor(np.r_[env.inner.task.state, env.controller.estimate][None], requires_grad=True)
                predicted = transition(state, reference, t, torch.tensor(error[None]))
                obs, _, _, _, _ = env.step(policy.predict(obs, deterministic=True)[0])
                np.testing.assert_allclose(predicted.detach().numpy()[0],
                    np.r_[env.inner.task.state, env.controller.estimate], atol=2e-7, rtol=1e-7)
                predicted.sum().backward()
                self.assertTrue(torch.isfinite(state.grad).all())
                self.assertTrue((np.abs(env.controller.last_action) <= ACTION_HIGH).all())
        finally:
            env.close()

    def test_jitter_matches_legacy_baseline_and_observer_does_not_get_true_state(self):
        env = PlannedParkingEnv(self.bank, initial_jitter=.04)
        baseline = ParkingPPOEnv()
        try:
            for scenario in range(15):
                env.reset(seed=701+scenario, options={'scenario': scenario})
                x = perturbed_initial_state(scenario, np.random.default_rng(701+scenario), .04)
                baseline.reset(options=dict(scenario=scenario, state=x))
                np.testing.assert_array_equal(env.inner.task.state, baseline.task.state)
                # Unknown initial velocity remains a prior, not true speed.
                self.assertEqual(env.controller.estimate[3], self.bank.states[env.controller.reference, 0, 3])
        finally:
            env.close(); baseline.close()

    def test_reference_selection_rejects_unknown_goal_and_invalid_measurement(self):
        with self.assertRaises(ValueError):
            self.bank.select([0, 0, 0], [99, 99])
        with self.assertRaises(ValueError):
            self.bank.select([np.nan, 0, 0], self.bank.goals[0])

    def test_process_disturbance_is_integrated_inside_rk4(self):
        x = np.array([[-1.4, .7, .8, .6]])
        u = np.array([[.4, -.3]])
        w = np.array([[.9, -.7, .5, .8]])
        actual = PlannedTransition.dynamics(torch.tensor(x), torch.tensor(u), torch.tensor(w))
        np.testing.assert_allclose(actual.numpy()[0], dynamics(x[0], u[0], w[0]), atol=1e-12)


if __name__ == '__main__':
    unittest.main()
