import unittest
import numpy as np

from Aebs.dino_latent.train_finite_horizon_sbc_lp import (
    bernstein1d, restriction, initial_control_matrix, barrier_features,
    expected_features, optimize_candidate, successors,
    audit_drift,
)


class FiniteHorizonSBCTest(unittest.TestCase):
    def test_positive_partition_of_unity(self):
        p = bernstein1d(np.linspace(0, 1, 51), 12)
        self.assertTrue((p >= 0).all())
        np.testing.assert_allclose(p.sum(1), 1)

    def test_de_casteljau_restriction_matches_original_polynomial(self):
        c = np.random.default_rng(1).uniform(0, 1, 13)
        r = restriction(12, .7, .95)
        t = np.linspace(0, 1, 101)
        np.testing.assert_allclose(bernstein1d(t, 12)@(r@c), bernstein1d(.7+.25*t, 12)@c, atol=1e-12)
        self.assertTrue((r >= 0).all())
        np.testing.assert_allclose(r.sum(1), 1)

    def test_initial_bound_is_valid_over_entire_box(self):
        w = np.random.default_rng(2).uniform(0, 1, 25)
        s = np.random.default_rng(3).uniform([15, 2.5], [16, 3], (1000, 2))
        p, o = barrier_features(s, 4)
        self.assertLessEqual((p@w+o).max(), (initial_control_matrix(4)@w).max()+1e-12)

    def test_global_region_values_are_constructed_not_fitted(self):
        s = np.array([[5.5, 1.], [4.9, .4], [5.5, .4], [8., 0.], [10., 2.]])
        p, o = barrier_features(s, 3)
        w = np.linspace(0, 1, 16)
        np.testing.assert_allclose((p@w+o)[:4], [1, 1, 0, 0])
        self.assertGreaterEqual((p@w+o)[4], 0)
        self.assertLessEqual((p@w+o)[4], 1)

    def test_expectation_is_not_worst_outcome(self):
        # At same current state, 10% unsafe and 90% successful successor.
        next_states = np.array([[[5.5, 1.], [5.5, .4]]])
        p, o = expected_features(next_states, [.1, .9], 3)
        np.testing.assert_allclose(p, 0)
        self.assertAlmostEqual(o[0], .1)
        self.assertLess(o[0], 1.)

    def test_LP_allows_increase_but_charges_horizon(self):
        width = 4
        current = np.zeros((1, width)); current[0, 0] = 1
        nxt = np.zeros_like(current)
        w, r = optimize_candidate(current, nxt, np.array([.1]), np.array([0.]), 1, 400)
        self.assertLess(r["maximum_LP_residual"], 1e-7)
        self.assertGreaterEqual(r["c_expected_drift"], 0)
        self.assertAlmostEqual(r["candidate_failure_budget"], r["gamma_initial_upper"]+400*r["c_expected_drift"])

    def test_successor_units_and_clipping(self):
        np.testing.assert_allclose(successors([[10., 1.], [6.1, .01]], [2., 10.]), [[9.95, .9], [6.0995, 0.]])

    def test_invalid_probability_distribution_rejected(self):
        with self.assertRaises(ValueError):
            expected_features(np.zeros((1, 2, 2)), [.5, .6], 3)

    def test_barrier_may_exceed_one_outside_unsafe(self):
        p, o = barrier_features(np.array([[8., 1.]]), 3)
        self.assertAlmostEqual(float(p@np.full(16, 2.)+o), 2.)

    def test_constant_baseline_never_has_budget_above_one(self):
        current = bernstein1d([.5], 1)
        current = (current[:, :, None]*current[:, None, :]).reshape(1, 4)
        _, r = optimize_candidate(current, np.zeros_like(current), np.ones(1), np.zeros(1), 1, 400, 1.)
        self.assertLessEqual(r["candidate_failure_budget"], 1.)

    def test_clipped_refinement_satisfies_actual_training_inequality(self):
        states = np.array([[6.1, 3.], [10., 2.], [15., 2.5]])
        nxt = successors(states, [3., 2., -1.])
        p, o = barrier_features(states, 3)
        pn, on = barrier_features(nxt, 3)
        w, _ = optimize_candidate(p, pn, on, o, 3, 400)
        w, r = optimize_candidate(p, pn, on, o, 3, 400, 20, w)
        audit = audit_drift(w, states, nxt, 3, r["c_expected_drift"], True)
        self.assertLess(audit["max_excess_over_c"], 2e-7)

    def test_value_one_state_is_safe_to_discharge_not_ignored_arbitrarily(self):
        # Current raw value2, successor raw value5; clipped values both1.
        current = np.array([[1., 0., 0., 0.]])
        nxt = np.array([[0., 1., 0., 0.]])
        w, r = optimize_candidate(current, nxt, np.zeros(1), np.zeros(1), 1, 400, 20, np.array([2., 5., 0., 0.]))
        self.assertEqual(r["current_value_one_states_discharged"], 1)
        self.assertGreaterEqual(float(current@w), 1-1e-7)
        self.assertLessEqual(min(float(nxt@w),1)-min(float(current@w),1), r["c_expected_drift"]+1e-7)


if __name__ == "__main__":
    unittest.main()
