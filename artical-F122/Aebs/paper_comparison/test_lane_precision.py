"""Regression checks for failure inclusion, deadline semantics and plant parity."""
import unittest
import numpy as np
from .lane_precision import precision_metrics, calibrate_performance, performance_scores, summarize


class PrecisionTests(unittest.TestCase):
    def test_boolean_summary_works_with_numpy_quantile(self):
        result = summarize(dict(unsafe=np.array([False, True])))['unsafe']
        self.assertEqual(result['count'], 1)
        self.assertEqual(result['mean'], .5)
        self.assertAlmostEqual(result['p95'], .95)

    def test_reentry_does_not_hide_intermediate_failure(self):
        x = np.zeros((4, 81, 2))
        x[0, :20, 0] = .20  # Allowed transient before the 2-second deadline.
        x[1, 40, 0] = .06  # Last-frame-only metrics would hide this excursion.
        x[2, 1, 1] = .31  # Unsafe early; even perfect later tracking must fail.
        x[3, 20, 0] = .051  # Deadline is inclusive.
        m = precision_metrics(x, np.zeros((4, 80)))
        np.testing.assert_array_equal(m['strict_05cm'], [True, False, False, False])
        np.testing.assert_array_equal(m['unsafe'], [False, False, True, False])

    def test_missing_and_nonfinite_episodes_are_rejected(self):
        with self.assertRaises(ValueError):
            precision_metrics(np.zeros((1, 81, 2)), np.zeros((1, 79)))
        x = np.zeros((1, 81, 2)); x[0, 30, 0] = np.nan
        with self.assertRaises(ValueError):
            precision_metrics(x, np.zeros((1, 80)))

    def test_performance_cp_keeps_unsafe_at_infinity(self):
        metrics = dict(tail_peak_abs_e_m=np.arange(199, dtype=float)/1000,
                       unsafe=np.zeros(199, dtype=bool))
        result = calibrate_performance(metrics)
        self.assertEqual(result['rank'], 190)
        self.assertAlmostEqual(result['radius_m'], .189)
        metrics['unsafe'][:10] = True
        self.assertEqual(np.isinf(performance_scores(metrics)).sum(), 10)
        self.assertFalse(calibrate_performance(metrics)['finite_bound'])

    def test_training_dynamics_match_physical_plant(self):
        import torch
        from .lane_controller_comparison import successor
        from Aebs.lane_keeping.core import dynamics_batch
        rng = np.random.default_rng(938)
        x = rng.uniform(-1, 1, (50, 2))
        u = rng.uniform(-1, 1, 50)
        k = rng.choice([-.01, 0, .01], 50)
        actual = successor(torch.tensor(x), torch.tensor(u), torch.tensor(k)).numpy()
        expected = dynamics_batch(x*[.85, .30], u*.45, k)/[.85, .30]
        np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-14)

    def test_adversary_stays_inside_box_and_actor_has_gradient(self):
        import torch
        from .lane_precision import adversary, trajectory_cost
        torch.manual_seed(10)
        actor = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.Tanh(),
                                    torch.nn.Linear(8, 1), torch.nn.Tanh()).double()
        x = torch.tensor([[.2, .1], [-.2, -.1]], dtype=torch.float64)
        k = torch.tensor([.01, -.01], dtype=torch.float64)
        zero = torch.zeros((2, 8, 2), dtype=torch.float64)
        errors = adversary(actor, x, k, zero, .1, 4)
        self.assertLessEqual(float(errors.abs().max()), .1)
        self.assertGreater(float(errors.abs().sum()), 0.)
        trajectory_cost(actor, x, k, errors).mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in actor.parameters()))


if __name__ == '__main__':
    unittest.main()
