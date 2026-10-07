import unittest

import numpy as np

from Aebs.dino_latent.calibrate_transition_contract import (
    next_state_residual,
    trajectory_scores,
    validate_calibration_arrays,
)


class TransitionContractTests(unittest.TestCase):
    def test_next_state_residual_uses_same_physical_state(self):
        residual = next_state_residual(
            [10.0, 10.0], [2.0, 0.0], [3.0, -3.0], [1.0, 3.0]
        )
        np.testing.assert_allclose(residual[:, 0], 0.0)
        self.assertAlmostEqual(residual[0, 1], -0.1)
        # Nominal braking clips at zero; image-path acceleration reaches 0.15.
        self.assertAlmostEqual(residual[1, 1], 0.15)

    def test_trajectory_score_is_max_over_valid_steps(self):
        residual = np.asarray([
            [[0.0, 0.1], [0.2, 0.0], [99.0, 99.0]],
            [[0.0, 0.4], [0.0, 0.3], [0.0, 0.2]],
        ])
        valid = np.asarray([[1, 1, 0], [1, 1, 1]], dtype=bool)
        scores = trajectory_scores(residual, valid, [2.0, 0.5])
        np.testing.assert_allclose(scores, [0.2, 0.8])

    def test_rejects_non_independent_shape_or_invalid_state(self):
        features = np.zeros((2, 3, 384), dtype=np.float32)
        distance = np.full((2, 3), 10.0, dtype=np.float32)
        speed = np.full((2, 3), 1.0, dtype=np.float32)
        valid = np.ones((2, 3), dtype=bool)
        validate_calibration_arrays(features, distance, speed, valid, ["a", "b"], 3)
        with self.assertRaises(ValueError):
            validate_calibration_arrays(features, distance, speed, valid, ["a", "a"], 3)
        distance[0, 0] = 17.0
        with self.assertRaises(ValueError):
            validate_calibration_arrays(features, distance, speed, valid, ["a", "b"], 3)


if __name__ == "__main__":
    unittest.main()
