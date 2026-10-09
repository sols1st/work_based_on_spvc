import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from Aebs.connect.collect_r2_calibration_trajectories import episode_plan
from Aebs.connect.collect_r2_spvc_trajectories import (
    CAPTURE_PROTOCOL,
    DEVELOPMENT_SCHEMA,
    initialize_h5,
    screenshot_bgra_to_model_inputs,
)


class R2SpvcTrajectoryCollectorTests(unittest.TestCase):
    def test_screenshot_pipeline_returns_original_bgr_and_model_rgb(self):
        # BGRA pixel: blue=1, green=2, red=3.
        raw = np.asarray([[[1, 2, 3, 255]]], dtype=np.uint8)
        bgr, tensor = screenshot_bgra_to_model_inputs(raw, 1, 1)
        np.testing.assert_array_equal(bgr[0, 0], [1, 2, 3])
        np.testing.assert_allclose(tensor.numpy()[:, 0, 0], [3 / 255, 2 / 255, 1 / 255])

    def test_development_h5_cannot_be_mistaken_for_calibration(self):
        plan = episode_plan(2, 7)
        provenance = {"dino_weights_sha256": "abc"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "development.h5"
            with h5py.File(path, "w") as stream:
                initialize_h5(stream, plan, 3, provenance, "development")
            with h5py.File(path, "r") as stream:
                self.assertEqual(stream.attrs["schema"], DEVELOPMENT_SCHEMA)
                self.assertEqual(stream.attrs["capture_protocol"], CAPTURE_PROTOCOL)
                self.assertEqual(stream.attrs["split_role"], "development")
                self.assertFalse(stream.attrs["independent_from_training"])
                self.assertTrue(stream.attrs["used_for_model_selection"])
                self.assertFalse(stream.attrs["complete"])


if __name__ == "__main__":
    unittest.main()
