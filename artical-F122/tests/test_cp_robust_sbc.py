import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from Aebs.dino_latent.train_cp_robust_sbc import (
    load_contract,
    safe_terminal_mask,
    successor_scenarios,
    unsafe_mask,
)


class CpRobustSbcTests(unittest.TestCase):
    def test_successor_scenarios_apply_physical_speed_residual(self):
        states = np.asarray([[10.0, 2.0], [6.0, 0.1]], dtype=np.float32)
        actions = np.asarray([3.0, 3.0], dtype=np.float32)
        successors = successor_scenarios(states, actions, [-0.1, 0.0, 0.1])
        np.testing.assert_allclose(successors[0, :, 0], 9.9)
        np.testing.assert_allclose(successors[0, :, 1], [1.75, 1.85, 1.95])
        np.testing.assert_allclose(successors[1, :, 1], [0.0, 0.0, 0.1])

    def test_terminal_and_unsafe_partition(self):
        states = np.asarray([
            [5.5, 0.4],
            [5.5, 0.6],
            [10.0, 0.0],
            [10.0, 1.0],
        ])
        np.testing.assert_array_equal(safe_terminal_mask(states), [True, False, True, False])
        np.testing.assert_array_equal(unsafe_mask(states), [False, True, False, False])

    def test_contract_binds_models_and_calibration_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            representation = root / "representation.pt"
            controller = root / "controller.zip"
            calibration = root / "calibration.h5"
            for path, value in (
                (representation, b"r"), (controller, b"c"), (calibration, b"h")
            ):
                path.write_bytes(value)
            from Aebs.dino_latent.prepare_expanded_data import sha256
            contract = {
                "experiment": "R2_physical_next_state_trajectory_conformal_contract_v1",
                "registered_before_calibration": True,
                "representation_frozen": True,
                "controller_frozen": True,
                "test_used": False,
                "representation_sha256": sha256(representation),
                "controller_sha256": sha256(controller),
                "calibration_h5_sha256": sha256(calibration),
                "conformal": {"sample_count": 459, "q_hat": 0.15},
                "physical_residual_box": {
                    "distance_m": [0.0, 0.0], "speed_mps": [-0.15, 0.15]
                },
            }
            contract_path = root / "contract.json"
            contract_path.write_text(json.dumps(contract))
            loaded, low, high = load_contract(
                contract_path, representation, controller, calibration
            )
            self.assertEqual(loaded, contract)
            self.assertAlmostEqual(low, -0.15)
            self.assertAlmostEqual(high, 0.15)


if __name__ == "__main__":
    unittest.main()
