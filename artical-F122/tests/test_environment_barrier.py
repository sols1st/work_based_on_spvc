"""Physics, paired masks, action hull coverage, and neural interval bounds."""
import unittest

import numpy as np
import torch

from Aebs.dino_latent.diagnose_environment_barrier import physical_next, summarize, quadrature
from Aebs.dino_latent.check_environment_action_envelope import group_actions, interval_barrier, noise_cells, evaluate_cells
from Aebs.system.env import Aebs
from Aebs.VT.utils import MLP


class EnvironmentBarrierTests(unittest.TestCase):
    def test_physical_successor_matches_existing_environment(self):
        env = Aebs(.05)
        x = torch.tensor([[15/env.std1, 2.5], [6.1/env.std1, .1], [8/env.std1, 3.]], dtype=torch.float32)
        a = torch.tensor([[2.], [-4.], [4.]], dtype=torch.float32)
        torch.testing.assert_close(physical_next(x, a, float(env.std1)), env.v_next(x, a), rtol=1e-6, atol=1e-6)

    def test_extra_violations_are_directional(self):
        report = summarize(np.ones(3), np.array([1.2, .5, .5]), np.array([.9, 1.1, 9.]), np.array([True, True, False]))
        self.assertEqual(report["checked_pairs"], 2)
        self.assertEqual(report["image_extra_violations"], 1)
        self.assertEqual(report["q_only_violations"], 1)

    def test_uniform_quadrature_on_linear_barrier(self):
        x = torch.tensor([[2., 3.], [4., 5.]])
        model = torch.nn.Linear(2, 1, bias=False)
        model.weight.data.fill_(1.)
        points = torch.tensor([[-.1, -.2], [.1, .2]])
        torch.testing.assert_close(quadrature(model, x, points), model(x).flatten())

    def test_interval_contains_random_interior_points(self):
        torch.manual_seed(7)
        for activation in ("tanh", "relu"):
            for output in (True, False):
                model = MLP([2, 16, 8, 1], activation=activation, square_output=output).double()
                low = torch.tensor([[-.2, -.5], [1., 2.]], dtype=torch.float64)
                high = low+torch.tensor([[.7, .8], [.1, .2]], dtype=torch.float64)
                lb, ub = interval_barrier(model, low, high)
                samples = low[:, None, :]+torch.rand((2, 1000, 2), dtype=torch.float64)*(high-low)[:, None, :]
                actual = model(samples.reshape(-1, 2)).reshape(2, 1000)
                self.assertTrue(torch.all(actual >= lb[:, None]-1e-12))
                self.assertTrue(torch.all(actual <= ub[:, None]+1e-12))

    def test_point_interval_matches_direct_output(self):
        model = MLP([2, 16, 8, 1], activation="tanh", square_output=True).double()
        point = torch.tensor([[2., 3.]], dtype=torch.float64)
        lb, ub = interval_barrier(model, point, point)
        torch.testing.assert_close(lb, model(point).flatten(), rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(ub, lb, rtol=1e-12, atol=1e-12)

    def test_group_hulls_include_all_appearances_and_nominal(self):
        rows = dict(distance_m=np.repeat([7., 8.], 12), speed_mps=np.ones(24),
                    appearance=np.tile([str(i) for i in range(12)], 2),
                    image_action=np.tile(np.linspace(-2., 2., 12), 2), q_action=np.full(24, 2.5),
                    current_barrier=np.ones(24), image_next_state=np.ones((24, 2)))
        unique, fields, appearances = group_actions(rows)
        self.assertEqual(len(unique), 2)
        np.testing.assert_allclose(fields["action_low"], [-2., -2.])
        np.testing.assert_allclose(fields["action_high"], [2.5, 2.5])
        _, nominal, _ = group_actions(rows, "nominal")
        np.testing.assert_allclose(nominal["action_low"], [2.5, 2.5])
        np.testing.assert_allclose(nominal["action_high"], [2.5, 2.5])
        _, full, _ = group_actions(rows, "full-actuator")
        np.testing.assert_allclose(full["action_low"], [-3., -3.])
        np.testing.assert_allclose(full["action_high"], [3., 3.])
        self.assertEqual(len(appearances), 12)
        repeated = {key: np.concatenate([value, value[:12]]) for key, value in rows.items()}
        _, wider, _ = group_actions(repeated)
        np.testing.assert_array_equal(wider["observations_per_state"], [24, 12])
        broken = {key: value.copy() for key, value in rows.items()}
        broken["appearance"][0] = broken["appearance"][1]
        with self.assertRaises(ValueError):
            group_actions(broken)

    def test_zero_noise_is_one_degenerate_cell(self):
        env = Aebs(.05)
        low, high = noise_cells(env, 20, torch.device("cpu"), "none")
        self.assertEqual(tuple(low.shape), (1, 2))
        self.assertTrue(torch.all(low == 0))
        torch.testing.assert_close(low, high)
        low, high = noise_cells(env, 20, torch.device("cpu"), "original-uniform")
        self.assertEqual(tuple(low.shape), (400, 2))
        self.assertTrue(torch.all(low < high))

    def test_zero_noise_action_hull_encloses_every_sampled_action(self):
        env = Aebs(.05)
        barrier = MLP([2,16,8,1], activation="tanh", square_output=True).double()
        states = np.array([[8., 1.]])
        fields = dict(action_low=np.array([-1.]), action_high=np.array([2.]),
                      next_distance_low=np.array([7.95/env.std1]), next_distance_high=np.array([7.95/env.std1]),
                      barrier_low=np.array([1.]))
        result = evaluate_cells(barrier, env, states, fields, np.array([0]), 20, torch.device("cpu"), 1, "none")
        a = torch.linspace(-1., 2., 101, dtype=torch.float64)
        next_states = torch.stack((torch.full_like(a, 7.95/env.std1), 1.-.05*a), dim=1)
        actual_margin = 1.-barrier(next_states).detach().numpy().flatten()
        self.assertTrue(np.all(actual_margin >= result["margin"][0]-1e-10))
        self.assertTrue(np.all(actual_margin <= result["optimistic_margin"][0]+1e-10))

    def test_action_subdivision_keeps_full_coverage_and_tightens_bound(self):
        torch.manual_seed(7)
        env = Aebs(.05)
        barrier = MLP([2,16,8,1], activation="tanh", square_output=True).double()
        states = np.array([[8., 2.9]])
        fields = dict(action_low=np.array([-3.]), action_high=np.array([3.]),
                      next_distance_low=np.array([(8.-.05*2.9)/env.std1]),
                      next_distance_high=np.array([(8.-.05*2.9)/env.std1]), barrier_low=np.array([1.]))
        coarse = evaluate_cells(barrier, env, states, fields, np.array([0]), 2, torch.device("cpu"), 1, "none", 1)
        fine = evaluate_cells(barrier, env, states, fields, np.array([0]), 2, torch.device("cpu"), 1, "none", 32)
        a = torch.linspace(-3., 3., 1001, dtype=torch.float64)
        successors = torch.stack((torch.full_like(a, (8.-.05*2.9)/env.std1), (2.9-.05*a).clamp(0.,3.)), dim=1)
        actual = 1.-barrier(successors).detach().numpy().flatten()
        self.assertTrue(np.all(actual >= fine["margin"][0]-1e-10))
        self.assertTrue(np.all(actual <= fine["optimistic_margin"][0]+1e-10))
        self.assertGreaterEqual(fine["margin"][0], coarse["margin"][0]-1e-10)
        self.assertAlmostEqual(float(fine["successor_speed_interval_width"][0]), .25)


if __name__ == "__main__":
    unittest.main()
