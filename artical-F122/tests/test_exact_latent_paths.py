import unittest

import numpy as np
import torch
from torch import nn

from Aebs.dino_latent.compare_exact_latent_paths import (
    to_raw, to_native, matched_errors, common_windows, validate_trajectories, actor_actions,
)
from Aebs.dino_latent.temporal_world_model import window_indices


class ExactLatentPathsTest(unittest.TestCase):
    def test_coordinate_roundtrip(self):
        z = torch.arange(96).float().reshape(3, 32) / 20
        mean, std = torch.linspace(-1, 1, 32), torch.linspace(.1, 2, 32)
        torch.testing.assert_close(to_raw(to_native(z, mean, std), mean, std), z)
        with self.assertRaises(ValueError):
            to_native(z, mean, torch.zeros(32))

    def test_positive_speed_residual_means_underbraking(self):
        da, dv = matched_errors([1.], [2.], [1.])
        np.testing.assert_allclose(da, [-1.])
        np.testing.assert_allclose(dv, [.05])

    def test_saturation_and_clipping(self):
        _, dv = matched_errors([.01, 3., 1.], [3., -3., 30.], [30., -30., 3.])
        np.testing.assert_allclose(dv, 0.)

    def test_common_windows_include_full_future10(self):
        mask = np.ones((2, 17), bool)
        mask[0, 15:] = False
        w = common_windows(mask)
        self.assertEqual(set(w[:, 0]), {0, 1})
        for ep, end in w:
            self.assertTrue(mask[ep, end-2:end+11].all())
        self.assertEqual(w[w[:, 0] == 0, 1].tolist(), [2, 3, 4])

    def test_missing_trajectory_rejected(self):
        mask = np.ones((2, 15), bool)
        mask[1, 5:] = False
        with self.assertRaises(ValueError):
            common_windows(mask)

    def test_one_step_includes_late_frames_omitted_by_common10(self):
        mask = np.ones((2, 20), bool)
        one = window_indices(mask, np.arange(2), 3, 1)
        shared = common_windows(mask)
        self.assertEqual(len(one) - len(shared), 18)
        self.assertEqual(one[:, 1].max(), 18)
        self.assertEqual(shared[:, 1].max(), 9)

    def test_synchronized_data_audit(self):
        v = np.array([[2., 1.95, 1.9]])
        d = np.array([[10., 9.9, 9.8025]])
        a = np.ones_like(v)
        mask = np.ones_like(v, bool)
        result = validate_trajectories(mask, d, v, a)
        self.assertLess(result["speed_transition_max_abs_mps"], 1e-10)
        d[0, 1] += .01
        with self.assertRaises(ValueError):
            validate_trajectories(mask, d, v, a)

    def test_no_holes_in_trajectory(self):
        mask = np.array([[True, False, True]])
        with self.assertRaises(ValueError):
            validate_trajectories(mask, np.ones((1, 3)), np.ones((1, 3)), np.ones((1, 3)))

    def test_actor_native_transform_and_output_clipping(self):
        actor = nn.Linear(33, 1, bias=False)
        with torch.no_grad():
            actor.weight.zero_(); actor.weight[0, 0] = 2.
        z = np.full((3, 32), 3., np.float32)
        native = dict(latent_mean=torch.ones(32), latent_std=torch.full((32,), 2.))
        np.testing.assert_allclose(actor_actions(actor, z, np.ones(3), native), 2.)
        np.testing.assert_allclose(actor_actions(actor, z, np.ones(3)), 3.)


if __name__ == "__main__":
    unittest.main()
