import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from Aebs.connect.collect_r2_calibration_trajectories import (
    advance_state,
    bgra_to_rgb_tensor,
    episode_plan,
    initialize_h5,
)
from Aebs.dino_latent.calibrate_transition_contract import SCHEMA


class R2CalibrationCollectorTests(unittest.TestCase):
    def test_plan_is_deterministic_independent_and_in_registered_domain(self):
        first = episode_plan(459, 1701)
        second = episode_plan(459, 1701)
        self.assertEqual(first, second)
        self.assertEqual(len({row["trajectory_id"] for row in first}), 459)
        self.assertTrue(all(15 <= row["initial_distance_m"] <= 16 for row in first))
        self.assertTrue(all(2.5 <= row["initial_speed_mps"] <= 3 for row in first))
        self.assertEqual(len({(row["weather"], row["color"]) for row in first}), 12)

    def test_dynamics_and_action_clipping(self):
        distance, speed = advance_state(10.0, 2.0, 99.0)
        self.assertAlmostEqual(distance, 9.9)
        self.assertAlmostEqual(speed, 1.85)
        _, speed = advance_state(10.0, 0.01, 3.0)
        self.assertEqual(speed, 0.0)

    def test_bgra_channel_conversion(self):
        # One BGRA pixel: blue=1, green=2, red=3, alpha ignored.
        tensor = bgra_to_rgb_tensor(bytes([1, 2, 3, 255]), 1, 1)
        np.testing.assert_allclose(tensor.numpy()[:, 0, 0], [3 / 255, 2 / 255, 1 / 255])

    def test_h5_schema_is_incomplete_until_collection_finishes(self):
        plan = episode_plan(2, 7)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.h5"
            provenance = {"dino_weights_sha256": "abc"}
            with h5py.File(path, "w") as stream:
                initialize_h5(stream, plan, 3, provenance)
            with h5py.File(path, "r") as stream:
                self.assertEqual(stream.attrs["schema"], SCHEMA)
                self.assertFalse(stream.attrs["complete"])
                self.assertTrue(stream.attrs["independent_from_training"])
                self.assertEqual(stream["dino_features"].shape, (2, 3, 384))
                self.assertFalse(stream["valid"][:].any())


if __name__ == "__main__":
    unittest.main()
