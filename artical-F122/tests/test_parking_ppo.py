import unittest
import numpy as np
from Aebs.vision_sls_parking.ppo_env import ParkingPPOEnv
from Aebs.vision_sls_parking.core import nominal_observation, ACTION_HIGH


class PPOInterfaceTests(unittest.TestCase):
    def test_delay_and_history(self):
        e = ParkingPPOEnv(scenario=1)
        obs, _ = e.reset(seed=7)
        y0 = nominal_observation(e.task.state)
        np.testing.assert_allclose(obs[:9], np.tile(y0/3, 3), atol=1e-7)
        obs1, _, _, _, _ = e.step([.2, .2])
        np.testing.assert_allclose(obs1[:9], obs[:9], atol=1e-7)
        y1 = nominal_observation(e.task.state)
        obs2, _, _, _, _ = e.step([.1, 0])
        np.testing.assert_allclose(obs2[6:9], y1/3, atol=1e-7)
        np.testing.assert_allclose(obs2[9:13], [.2, .2, .1, 0], atol=1e-7)

    def test_no_current_state_leak(self):
        e = ParkingPPOEnv(scenario=1)
        obs, _ = e.reset()
        e.task.state[:] = [200, -300, 100, 999]
        np.testing.assert_array_equal(e._observation(), obs)

    def test_goal_command_distinguishes_identical_start(self):
        e = ParkingPPOEnv()
        a, _ = e.reset(options={'scenario': 1})
        b, _ = e.reset(options={'scenario': 12})
        np.testing.assert_array_equal(a[:13], b[:13])
        self.assertFalse(np.array_equal(a[13:15], b[13:15]))

    def test_fixed_horizon_action_limits(self):
        e = ParkingPPOEnv(scenario=1)
        e.reset()
        for i in range(30):
            obs, r, terminated, truncated, info = e.step([4, -4])
            self.assertFalse(terminated)
            self.assertEqual(truncated, i == 29)
            self.assertTrue(np.isfinite(obs).all())
            self.assertTrue(np.isfinite(r))
            np.testing.assert_allclose(info['action_applied'], ACTION_HIGH*[1, -1])

    def test_gym_contract(self):
        from stable_baselines3.common.env_checker import check_env
        check_env(ParkingPPOEnv(), warn=True)

    def test_saved_rollout_audit_rejects_changed_state(self):
        from Aebs.vision_sls_parking.report_ppo import audit_rollouts
        rows, paths, actions = [], [], []
        e = ParkingPPOEnv()
        for i in range(15):
            e.reset(options={'scenario': i})
            ss, us = [e.task.state.copy()], []
            for _ in range(30):
                _, _, _, _, info = e.step([0, 0])
                ss.append(e.task.state.copy())
                us.append(info['action_applied'])
            rows.append(dict(success=info['episode_success'],
                             collision=info['collision_seen'],
                             out_of_bounds=info['out_of_bounds_seen']))
            paths.append(ss)
            actions.append(us)
        metric = dict(proxy=dict(per_scenario=rows, success_count=0,
                                 collision_count=0, out_of_bounds_count=0))
        data = dict(proxy_states=np.asarray(paths), proxy_actions=np.asarray(actions))
        self.assertEqual(audit_rollouts(metric, data)['status'],
                         'saved_rollout_replay_and_counts_pass')
        data['proxy_states'][0, 1, 0] += .01
        with self.assertRaises(ValueError):
            audit_rollouts(metric, data)


if __name__ == '__main__':
    unittest.main()
