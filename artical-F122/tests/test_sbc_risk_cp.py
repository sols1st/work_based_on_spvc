import unittest

import numpy as np

from Aebs.dino_latent.experiment_sbc_risk_cp import (
    path_scores, stopped_values, successors, calibrate_and_evaluate, SCORE_NAMES,
)
from Aebs.dino_latent.evaluate_risk_cp_library_shift import QuantizedProxyInterface
from tests.test_native_controllers import fixture


class SBCRiskCPTests(unittest.TestCase):
    def test_original_dynamics_and_action_saturation(self):
        actual = successors([[7., 1.], [7., .02], [7., 2.99]], [9., 3., -3.])
        np.testing.assert_allclose(actual, [[6.95, .85], [6.999, 0], [6.8505, 3]], atol=1e-6)

    def test_stopping_does_not_mislabel_unsafe_boundary(self):
        states = np.array([[6., .6], [6., .5], [8., 0.], [8., 1.]])
        result = stopped_values([0., 20., 20., 5.], states, 0., 10.)
        np.testing.assert_allclose(result, [1., 0., 0., .5])

    def test_unsafe_override_does_not_depend_on_unproved_neural_bound(self):
        result = stopped_values([-100., -100.], [[5.9, 1.], [6., .50001]], 0., 10.)
        np.testing.assert_array_equal(result, [1., 1.])

    def test_positive_variation_can_allow_some_increases(self):
        values = np.array([.1, .2, .15, .25, 0])
        increments = np.diff(values)
        score = path_scores(values, increments, np.zeros(4), 400)
        self.assertAlmostEqual(score["positive_variation"], -.7)
        self.assertGreater(score["horizon_max_increment"], 0)

    def test_decomposition_is_upper_bound_not_a_fake_tighter_score(self):
        values = np.array([.1, .15, .1])
        dq = np.array([-.1, .1]); de = np.diff(values)-dq
        score = path_scores(values, dq, de, 400)
        self.assertGreaterEqual(score["decomposed_budget"], score["positive_variation"])
        self.assertGreaterEqual(score["positive_variation"], score["max_barrier"])

    def test_every_unsafe_path_has_nonnegative_scores(self):
        rng = np.random.default_rng(7)
        for _ in range(100):
            values = np.r_[rng.uniform(0, 1, 20), 1.]
            dq = rng.uniform(-1, 1, 20); de = np.diff(values)-dq
            row = path_scores(values, dq, de, 400)
            self.assertTrue(all(v >= -1e-10 for v in row.values()))

    def test_off_policy_splice_rejected(self):
        with self.assertRaises(ValueError):
            path_scores([.1, .2, .3], [0., 0.], [0., 0.], 400)

    def test_negative_score_required_zero_does_not_certify(self):
        calibration = {name: np.zeros(459) for name in SCORE_NAMES}
        evaluation = {name: np.zeros(1000) for name in SCORE_NAMES}
        result = calibrate_and_evaluate(calibration, evaluation, .01, .01)
        self.assertFalse(result["positive_variation"]["q_strictly_negative"])

    def test_trajectory_sample_count_not_step_count(self):
        calibration = {name: np.full(458, -.5) for name in SCORE_NAMES}
        with self.assertRaises(ValueError):
            calibrate_and_evaluate(calibration, calibration, .01, .01)

    def test_same_requested_confidence_is_retained(self):
        calibration = {name: np.linspace(-.8, -.4, 459) for name in SCORE_NAMES}
        result = calibrate_and_evaluate(calibration, calibration, .01, .01)
        self.assertEqual(result["positive_variation"]["tolerance"]["q_hat"], -.4)
        self.assertGreaterEqual(result["positive_variation"]["tolerance"]["achieved_confidence"], .99)

    def test_lookup_only_counterfactual_keeps_source_and_replaces_latent(self):
        _, interface = fixture()
        counterfactual = QuantizedProxyInterface(interface)
        z, error, source = counterfactual.nearest([8., 13.], "a|b")
        np.testing.assert_array_equal(source, [1,4])
        np.testing.assert_allclose(error, [1.,2.])
        np.testing.assert_allclose(z, interface.proxy([7.,11.]),atol=1e-7)


if __name__ == "__main__": unittest.main()
