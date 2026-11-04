import unittest
import tempfile
from pathlib import Path
import numpy as np
import torch
from Aebs.vision_sls_parking.ppo_env import ParkingPPOEnv
from Aebs.vision_sls_parking.spvc_backend import augment, ParkingSPVCSystem
from Aebs.vision_sls_parking.perception_error import basis, fit_scale, calibrate, radius, joint_scores


class ParkingMigrationTests(unittest.TestCase):
    def test_expanded_perception_domain_retains_spatial_split_isolation(self):
        from Aebs.vision_sls_parking.adapt_projection import make_split_plan, cell_ids, historical_replay_indices
        low, high = [-3.5, -2], [.5, 1.9]
        x, cells, splits = make_split_plan(3, domain_low=low, domain_high=high, heading_limit=2.8)
        np.testing.assert_array_equal(cell_ids(x, low, high), cells)
        self.assertTrue((x[:, 0] > 0).any())
        self.assertTrue((x[:, 1] < -1.5).any())
        ids = historical_replay_indices(x, cells, splits, domain_low=low, domain_high=high)
        self.assertTrue(np.isin(cells[ids], cells[splits == 'train']).all())
        with self.assertRaises(ValueError):
            make_split_plan(domain_low=[0, 0], domain_high=[-1, 1])

    def test_reward_change_preserves_task_sensor_and_termination(self):
        old, new = ParkingPPOEnv(), ParkingPPOEnv(reward_mode='safety_v2')
        rng = np.random.default_rng(3)
        try:
            for scenario in range(15):
                a, _ = old.reset(options={'scenario': scenario})
                b, _ = new.reset(options={'scenario': scenario})
                np.testing.assert_array_equal(a, b)
                for t in range(30):
                    u = rng.uniform(-1, 1, 2)
                    a, _, term_a, done_a, i = old.step(u)
                    b, _, term_b, done_b, j = new.step(u)
                    np.testing.assert_array_equal(a, b)
                    np.testing.assert_array_equal(old.task.state, new.task.state)
                    self.assertEqual((term_a, done_a), (term_b, done_b))
                    self.assertEqual(done_a, t == 29)
                    for key in ('collision', 'in_bounds', 'episode_success', 'unsafe_seen'):
                        self.assertEqual(i[key], j[key])
        finally:
            old.close(); new.close()

    def test_stronger_clearance_reward_preserves_physics_and_goal_flags(self):
        a = ParkingPPOEnv(reward_mode='safety_v2')
        b = ParkingPPOEnv(reward_mode='safety_v3', scenario_weights=np.arange(1, 16))
        try:
            a.reset(options={'scenario': 9}); b.reset(options={'scenario': 9})
            for u in np.random.default_rng(8).uniform(-1, 1, (30, 2)):
                oa, _, _, da, ia = a.step(u); ob, _, _, db, ib = b.step(u)
                np.testing.assert_array_equal(oa, ob)
                self.assertEqual((da, ia['episode_success'], ia['unsafe_seen']),
                                 (db, ib['episode_success'], ib['unsafe_seen']))
        finally:
            a.close(); b.close()

    def test_latent_error_enters_only_new_delayed_measurement_once(self):
        env = ParkingPPOEnv(scenario=2)
        o, _ = env.reset()
        z = augment(env.task.state, o)[None]
        system = ParkingSPVCSystem(z, z)
        s = torch.tensor(z, dtype=torch.float64)
        u = torch.tensor([[.3, 1.2]], dtype=torch.float64)
        r = torch.tensor([[.3, .6, .9]], dtype=torch.float64)
        nominal = system.v_next(s, u)
        lo, hi = system.successor_observation_box(s, u, r)
        expected = np.zeros((1, 20)); expected[:, 10:13] = r.numpy()/3
        np.testing.assert_allclose((hi-nominal).numpy(), expected, atol=1e-12)
        np.testing.assert_allclose((nominal-lo).numpy(), expected, atol=1e-12)
        for _ in range(10):
            error = (2*torch.rand_like(r)-1)*r
            v = system.v_next(s, u, error)
            self.assertTrue(torch.all(v >= lo) and torch.all(v <= hi))
        with self.assertRaises(ValueError):
            system.successor_observation_box(s, u, -r)

    def test_positive_multidimensional_scale_and_domain_rejection(self):
        rng = np.random.default_rng(9)
        x = rng.uniform(0, 1, (50, 3))
        b = basis(x, [0]*3, [1]*3)
        np.testing.assert_allclose(b.sum(1), 1)
        self.assertTrue((b >= 0).all())
        r = rng.normal(size=(50, 3))*.1
        m = fit_scale(x, r, [0]*3, [1]*3)
        self.assertTrue((radius(x, m) > 0).all())
        with self.assertRaises(ValueError):
            radius([[2, 0, 0]], m)

    def test_joint_score_and_group_count_do_not_treat_frames_as_trajectories(self):
        m = dict(low=[0]*3, high=[1]*3, coefficients=np.ones((8, 3)).tolist(), q=1.)
        x = np.zeros((40, 3))
        residual = np.column_stack([np.arange(40), np.zeros(40), np.ones(40)*2])
        np.testing.assert_array_equal(joint_scores(x, residual, m), np.maximum(np.arange(40), 2))
        c = calibrate(x, residual, m)
        self.assertEqual((c['rank'], c['q']), (39, 38.))
        grouped = calibrate(x, residual, m, groups=np.repeat([0, 1], 20))
        self.assertEqual(grouped['n_units'], 2)
        self.assertFalse(grouped['finite'])
        self.assertTrue(np.isinf(grouped['q']))
        with self.assertRaises(ValueError):
            radius(x, dict(m, q=grouped['q']))
        perfect = calibrate(x, np.zeros((40, 3)), m)
        self.assertEqual(perfect['q'], 0.)
        np.testing.assert_array_equal(radius(x, dict(m, q=perfect['q'])), np.zeros((40, 3)))

    def test_barrier_box_contains_interior_samples_and_collapses_at_points(self):
        from Aebs.vision_sls_parking.train_sbc_migration import Barrier, network_box
        torch.manual_seed(77)
        net = Barrier().double()
        low = torch.randn(6, 20, dtype=torch.float64)
        high = low+.2
        lo, hi = network_box(net, low, high)
        for _ in range(50):
            values = net(low+torch.rand_like(low)*(high-low))
            self.assertTrue(torch.all(values >= lo-1e-12))
            self.assertTrue(torch.all(values <= hi+1e-12))
        p_lo, p_hi = network_box(net, low, low)
        torch.testing.assert_close(p_lo, net(low), atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(p_hi, net(low), atol=1e-12, rtol=1e-12)

    def test_offline_planner_dynamics_match_environment(self):
        from Aebs.vision_sls_parking.warmstart import rollout_tensor
        from Aebs.vision_sls_parking.core import INITIAL_STATES, ACTION_HIGH, dynamics
        rng = np.random.default_rng(40)
        u = rng.uniform(-.8, .8, (30, 2))
        expected = [INITIAL_STATES[3]]
        for action in u:
            expected.append(dynamics(expected[-1], action*ACTION_HIGH))
        actual = rollout_tensor(INITIAL_STATES[3], torch.tensor(u)).numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)

    def test_surrogate_never_uses_calibration_or_audit_residuals(self):
        from Aebs.vision_sls_parking.residual_replay import ResidualReplaySensor
        from Aebs.vision_sls_parking.core import nominal_observation
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'samples.npz'
            states = np.tile([-1., 0., 0., 0.], (3, 1))
            np.savez(path, states=states, residuals=[[.1, .2, .3], [100, 100, 100], [-100, -100, -100]],
                     splits=['fit', 'calibration', 'audit'])
            sensor = ResidualReplaySensor(path, nominal_fraction=0)
            np.testing.assert_allclose(sensor(states[0])-nominal_observation(states[0]), [.1, .2, .3])
            nominal = ResidualReplaySensor(path, nominal_fraction=1)
            np.testing.assert_array_equal(nominal(states[0]), nominal_observation(states[0]))


if __name__ == '__main__':
    unittest.main()
