import unittest
import numpy as np
import torch
from Aebs.vision_sls_parking.ppo_env import ParkingPPOEnv
from Aebs.vision_sls_parking.spvc_backend import (augment, physical_from_augmented,
                                                ParkingSPVCSystem, ParkingSPVCPolicy)
from Aebs.vision_sls_parking.core import ACTION_HIGH, dynamics, nominal_observation, constraints


class ParkingSPVCAdapterTests(unittest.TestCase):
    def setUp(self):
        self.env = ParkingPPOEnv(scenario=2)
        obs, _ = self.env.reset(seed=7)
        self.s = augment(self.env.task.state, obs)
        self.system = ParkingSPVCSystem(self.s[None], self.s[None])

    def test_transition_matches_full_history_not_only_physics(self):
        rng = np.random.default_rng(7)
        s = self.s.copy()
        for _ in range(30):
            normalized = rng.uniform(-1, 1, 2)
            a = normalized*ACTION_HIGH
            predicted = self.system.v_next(torch.tensor(s[None]), torch.tensor(a[None])).numpy()[0]
            obs, _, _, _, _ = self.env.step(normalized)
            actual = augment(self.env.task.state, obs)
            np.testing.assert_allclose(predicted, actual, atol=2e-6, rtol=2e-6)
            s = actual

    def test_delayed_measurement_is_old_current_not_successor(self):
        s = torch.tensor(self.s[None], dtype=torch.float64)
        a = torch.tensor([[1., 2.]], dtype=torch.float64)
        n = self.system.v_next(s, a).numpy()[0]
        old = physical_from_augmented(self.s)
        np.testing.assert_allclose(n[10:13], nominal_observation(old)/3, atol=1e-7)
        self.assertGreater(np.max(np.abs(n[10:13]-nominal_observation(dynamics(old, a.numpy()[0]))/3)), .01)

    def test_actor_export_and_no_privileged_state_input(self):
        from stable_baselines3 import PPO
        model = PPO('MlpPolicy', self.env, n_steps=32, batch_size=16, device='cpu', seed=7,
                    policy_kwargs=dict(net_arch=dict(pi=[8], vf=[8])))
        policy = ParkingSPVCPolicy(model)
        s = torch.tensor(np.repeat(self.s[None], 2, 0))
        s[1, :4] += 10
        with torch.no_grad():
            a = policy(torch.zeros(2, 4), s).numpy()
        np.testing.assert_array_equal(a[0], a[1])
        old, _ = model.predict(s[:, 4:].numpy(), deterministic=True)
        np.testing.assert_allclose(a, old*ACTION_HIGH, atol=1e-6)
        policy.export_actor(model)

    def test_rk4_gradient_reaches_actor_action(self):
        a = torch.tensor([[.2, .4]], requires_grad=True)
        nxt = self.system.v_next(torch.tensor(self.s[None]), a)
        nxt[:, :4].square().sum().backward()
        self.assertTrue(torch.isfinite(a.grad).all())
        self.assertGreater(float(a.grad.abs().sum()), 0)

    def test_unsafe_samples_are_actually_unsafe(self):
        s = self.system.sample_unsafe(101, 1024)
        for x in physical_from_augmented(s):
            f = constraints(x)
            self.assertTrue(f['collision'] or not f['in_bounds'])

    def test_initial_distribution_and_zero_noise(self):
        np.testing.assert_array_equal(self.system.sample_initial(7, 12), np.repeat(self.s[None], 12, 0))
        self.assertEqual(np.count_nonzero(self.system.noise), 0)

    def test_backend_training_methods_are_original_not_copied_losses(self):
        from Aebs.VT.train import VTLearner
        from Aebs.vision_sls_parking.spvc_backend import make_original_learner
        from stable_baselines3 import PPO
        model = PPO('MlpPolicy', self.env, n_steps=32, batch_size=16, device='cpu', seed=7,
                    policy_kwargs=dict(net_arch=dict(pi=[8], vf=[8])))
        learner = make_original_learner(self.system, ParkingSPVCPolicy(model))
        for name in ('train_step_l', 'train_step_p', 'train_epoch'):
            self.assertIs(getattr(learner, name).__func__, getattr(VTLearner, name))
        self.assertEqual(learner.p_optimizer.param_groups[0]['lr'], .05)
        self.assertEqual(learner.l_optimizer.param_groups[0]['lr'], .003)


if __name__ == '__main__':
    unittest.main()
