import unittest
from pathlib import Path
import ast
from types import SimpleNamespace
import numpy as np

from Aebs.vision_sls_parking.core import (
    dynamics, nominal_observation, constraints, INITIAL_STATES,
    ACTION_HIGH, C, RENDER_OFFSET, DT, GOAL_LOW, GOAL_HIGH, OBSTACLES,
)
from Aebs.vision_sls_parking.env import ParkingEnv


class ParkingCoreTest(unittest.TestCase):
    def test_rest(self):
        x = [-2., -1., .4, 0.]
        np.testing.assert_array_equal(dynamics(x, [0, 0]), x)

    def test_straight_acceleration_exact(self):
        x = np.array([-2., -1., .3, .5])
        result = dynamics(x, [0, 1.2])
        length = x[3]*DT + .5*1.2*DT**2
        expected = x + [length*np.cos(.3), length*np.sin(.3), 0, 1.2*DT]
        np.testing.assert_allclose(result, expected, atol=1e-14)

    def test_turn_matches_exact_circle(self):
        theta, v, omega = .5, 1., .7
        result = dynamics([0, 0, theta, v], [omega, 0])
        expected = [v/omega*(np.sin(theta+omega*DT)-np.sin(theta)),
                    v/omega*(np.cos(theta)-np.cos(theta+omega*DT)), theta+omega*DT, v]
        np.testing.assert_allclose(result, expected, atol=1e-8)

    def test_noise_inside_ode_not_added_at_end(self):
        np.testing.assert_allclose(dynamics([0, 0, 0, 0], [0, 0], [1, 0, 0, 0]),
                                   [.01*DT, 0, 0, 0])

    def test_invalid_input(self):
        for state, action in [([0]*3, [0, 0]), ([0]*4, [0]), ([np.nan]*4, [0, 0])]:
            with self.assertRaises(ValueError): dynamics(state, action)
        with self.assertRaises(ValueError): dynamics([0]*4, [0, 0], [2, 0, 0, 0])
        with self.assertRaises(ValueError): dynamics([0]*4, [0, 0], dt=0)

    def test_coordinate_offset_once(self):
        x = INITIAL_STATES[0]
        np.testing.assert_allclose(nominal_observation(x), C @ (x+RENDER_OFFSET))
        self.assertEqual(C.shape, (3, 4))
        np.testing.assert_array_equal(C[:, 3], 0)

    def test_15_starts_and_goals(self):
        self.assertEqual(INITIAL_STATES.shape, (15, 4))
        self.assertEqual(GOAL_LOW.shape, (15, 4))
        for i, x in enumerate(INITIAL_STATES):
            f = constraints(x, i)
            self.assertTrue(f['in_bounds']); self.assertFalse(f['collision'])
            self.assertTrue(constraints((GOAL_LOW[i]+GOAL_HIGH[i])/2, i)['in_goal'])

    def test_obstacle_and_out_of_domain(self):
        self.assertTrue(constraints([*OBSTACLES[1, :2], 0, 0])['collision'])
        self.assertFalse(constraints([10, 0, 0, 0])['in_bounds'])

    def test_upstream_constants_from_actual_file(self):
        root = Path(__file__).resolve().parents[2] / 'VISION-SLS'
        if not root.is_dir(): self.skipTest('upstream checkout unavailable')
        tree = ast.parse((root/'expe/parking_lot/main_parking_lot.py').read_text())
        ns = {'np': np}
        wanted = {'x0_all', 'x_min_f0', 'x_min_f1', 'x_min_f2', 'x_min_f3',
                  'x_max_f0', 'x_max_f1', 'x_max_f2', 'x_max_f3',
                  'x_min_f_all', 'x_max_f_all', 'obstacles'}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name in wanted:
                    ns[name] = eval(compile(ast.Expression(node.value), '<upstream constant>', 'eval'), ns)
        np.testing.assert_allclose(INITIAL_STATES, ns['x0_all'])
        np.testing.assert_allclose(GOAL_LOW, ns['x_min_f_all'])
        np.testing.assert_allclose(GOAL_HIGH, ns['x_max_f_all'])
        np.testing.assert_allclose(OBSTACLES[:, :2], [o[0] for o in ns['obstacles']])
        np.testing.assert_allclose(OBSTACLES[:, 2], [o[1] for o in ns['obstacles']])

    def test_rk4_against_unmodified_upstream_function_bodies(self):
        root = Path(__file__).resolve().parents[2] / 'VISION-SLS'
        if not root.is_dir(): self.skipTest('upstream checkout unavailable')
        # Execute the actual two function bodies without importing the entire
        # CasADi/Drake/SLS dependency tree. Numeric vertsplit/vertcat suffice.
        ns = dict(np=np, ca=SimpleNamespace(
            vertsplit=lambda x: list(x), vertcat=lambda *x: np.asarray(x)))
        for path, name in [('dyn/unicycle.py', 'ode'), ('dyn/model.py', 'ddyn')]:
            tree = ast.parse((root/path).read_text())
            node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
            exec(compile(ast.Module(body=[node], type_ignores=[]), '<upstream '+name+'>', 'exec'), ns)
        model = SimpleNamespace(E=.01*np.eye(4), discretization_method='rk4')
        model.ode = lambda x, u, w: ns['ode'](model, x, u, w)
        rng = np.random.default_rng(7)
        for _ in range(100):
            x = rng.uniform([-3.5, -2, -6, -1], [.5, 1.9, 6, 2])
            u = rng.uniform([-np.pi, -4], [np.pi, 4])
            w = rng.uniform(-1, 1, 4)
            np.testing.assert_allclose(dynamics(x, u, w), ns['ddyn'](model, x, u, w, DT), atol=1e-14)


class FakeRenderer:
    def render(self, state):
        return np.broadcast_to(np.asarray(state), (2, 2, 4)).copy()
    def close(self): pass


class ParkingInterfaceTest(unittest.TestCase):
    def test_delayed_state_and_image_are_paired(self):
        env = ParkingEnv(FakeRenderer())
        first = env.reset()
        one, _, _ = env.step([0, .3])
        two, _, _ = env.step([0, .3])
        np.testing.assert_array_equal(one['observation_state'], first['current_state'])
        np.testing.assert_array_equal(two['observation_state'], one['current_state'])
        np.testing.assert_array_equal(two['rgb'][0, 0], two['observation_state'])
        self.assertEqual([first['observation_index'], one['observation_index'], two['observation_index']], [0, 0, 1])

    def test_current_observation_mode_explicit(self):
        env = ParkingEnv(observation_delay=0)
        env.reset()
        obs, _, _ = env.step([.1, .3])
        np.testing.assert_array_equal(obs['current_state'], obs['observation_state'])

    def test_clip_and_no_state_clip(self):
        env = ParkingEnv()
        env.reset(state=[0, 0, 0, 2.])
        obs, _, info = env.step([100, 100])
        np.testing.assert_array_equal(info['action_applied'], ACTION_HIGH)
        self.assertTrue(info['action_clipped'])
        self.assertGreater(obs['current_state'][3], 2.)

    def test_horizon_and_reset(self):
        env = ParkingEnv(horizon=2)
        with self.assertRaises(RuntimeError): env.step([0, 0])
        env.reset()
        self.assertFalse(env.step([0, 0])[1])
        self.assertTrue(env.step([0, 0])[1])
        with self.assertRaises(RuntimeError): env.step([0, 0])
        env.reset(1)
        self.assertFalse(env.step([0, 0])[1])

    def test_goal_not_early_stop_and_unsafe_history_preserved(self):
        env = ParkingEnv(horizon=2)
        env.reset(state=[.2, .5, 0, 0])
        self.assertFalse(env.step([0, 0])[1])
        _, done, info = env.step([0, 0])
        self.assertTrue(done); self.assertEqual(info['outcome'], 'success')
        env.reset(state=[10, 0, 0, 0])
        env.step([0, 0])
        _, _, info = env.step([0, 0])
        self.assertTrue(info['unsafe_seen'])


if __name__ == '__main__': unittest.main()
