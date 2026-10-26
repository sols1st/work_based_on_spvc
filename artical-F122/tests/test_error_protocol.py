import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch

from Aebs.dino_latent.evaluate_error_protocol import (
    upper_tail_mean, physical_error_summary, clipping_audit, target_intersection,
    fixed_strata, group_table, safety_relevance,
    prediction_skill,
    native_feature_audit,
)


class ToyPredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(0.))

    def forward(self, z, v, a):
        return z[:, -1]+self.gain


class First32(torch.nn.Module):
    def forward(self, x):
        return x[:, :32]


class ErrorProtocolTest(unittest.TestCase):
    def test_tail_mean_has_exact_declared_count(self):
        self.assertEqual(upper_tail_mean(np.arange(20.)), 19.)
        self.assertEqual(upper_tail_mean(np.arange(21.)), 19.5)
        self.assertEqual(upper_tail_mean([2., 2., 2.]), 2.)
        with self.assertRaises(ValueError): upper_tail_mean([])
        with self.assertRaises(ValueError): upper_tail_mean([np.nan])

    def test_danger_direction_is_proxy_optimism_not_always_underbraking(self):
        r = physical_error_summary(np.array([-.2, .1]), np.array([0, 1]))
        self.assertAlmostEqual(r["image_faster_than_candidate"]["max_abs"], .2)
        self.assertAlmostEqual(r["candidate_faster_than_image"]["max_abs"], .1)

    def test_macro_does_not_overweight_long_trajectory(self):
        r = physical_error_summary(np.array([0., 0., 0., .2]), np.array([0, 0, 0, 1]))
        self.assertAlmostEqual(r["step_absolute"]["mae"], .05)
        self.assertAlmostEqual(r["trajectory_macro_mae_mps"], .1)

    def test_hidden_action_difference_at_speed_ceiling(self):
        r = clipping_audit(np.array([3.]), np.array([-1.]), np.array([-3.]))
        self.assertEqual(r["zero_physical_error_with_action_disagreement_count"], 1)
        self.assertAlmostEqual(r["unclipped_speed_equivalent_mae_mps"], .1)
        self.assertEqual(r["physical_speed_mae_mps"], 0)

    def test_no_hidden_difference_away_from_clipping(self):
        r = clipping_audit(np.array([1.]), np.array([-1.]), np.array([-3.]))
        self.assertEqual(r["zero_physical_error_with_action_disagreement_count"], 0)
        self.assertAlmostEqual(r["physical_speed_mae_mps"], .1)

    def test_target_intersection_is_not_context_intersection(self):
        a = dict(trajectory_index=np.array([0, 0, 1]), target_time=np.array([3, 12, 14]))
        b = dict(trajectory_index=np.array([1, 0]), target_time=np.array([14, 12]))
        targets, ix = target_intersection([a, b])
        np.testing.assert_array_equal(targets, [[0, 12], [1, 14]])
        np.testing.assert_array_equal(ix[0], [1, 2]); np.testing.assert_array_equal(ix[1], [1, 0])

    def test_duplicate_target_rejected(self):
        a = dict(trajectory_index=np.array([0, 0]), target_time=np.array([3, 3]))
        with self.assertRaises(ValueError): target_intersection([a])

    def test_group_balance_keeps_small_groups_visible(self):
        r = group_table(np.array([0., 0., 0., .2]), np.array(["a", "a", "a", "b"]))
        self.assertEqual(r["worst_group"], "b")
        self.assertAlmostEqual(r["balanced_group_mae_mps"], .1)
        self.assertEqual(r["groups"]["b"]["count"], 1)

    def test_physical_regions_do_not_define_new_safety_threshold(self):
        s = fixed_strata(np.array([6.01, 7., 15.]), np.array([.4, 1., 3.]))
        np.testing.assert_array_equal(s["next_position_reaches_unsafe_distance"], [True, False, False])
        np.testing.assert_array_equal(s["high_speed_v_ge_2p5"], [False, False, True])

    def test_proxy_safe_image_unsafe_is_distinguished(self):
        states = np.array([[6.01, 1.], [6.01, 1.]])
        s = safety_relevance(states, [.6, .4], [.4, .6])
        self.assertEqual(s["image_unsafe_candidate_safe"], 1)
        self.assertEqual(s["candidate_unsafe_image_safe"], 1)

    def test_prediction_skill_balances_trajectories(self):
        r = prediction_skill(np.array([0., 0., .5]), np.ones(3), np.array([0, 0, 1]))
        self.assertAlmostEqual(r["trajectory_balanced_skill_vs_persistence"], .75)
        self.assertEqual(r["trajectory_fraction_improved"], 1.)

    def test_zero_persistence_is_not_infinite_skill(self):
        r = prediction_skill(np.zeros(2), np.zeros(2), np.arange(2))
        self.assertIsNone(r["trajectory_balanced_skill_vs_persistence"])

    def test_native_audit_loads_nonzero_checkpoint_and_replays_saved_predictions(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for dim, native in ((32, "latent32"), (384, "dino384")):
                ck = dict(state_dict={"gain": torch.tensor(.2)}, kind="transformer", native=native,
                          normalization={native: (torch.zeros(dim), torch.ones(dim))})
                torch.save(ck, root/("transformer%d.pt" % dim))
                name = "pilot_transformer%d" % dim
                for h in (1, 3):
                    np.savez(root/("native_R2__%s__h%d.npz" % (name, h)),
                             trajectory_index=np.array([0]), target_time=np.array([2+h]))
                    np.savez(root/(name+"_h%d_predictions.npz" % h), raw_r2=np.full((1, 32), .2*h))
            kwargs = dict(pilot_dir=root, comparison_dir=root, z=torch.zeros(1, 8, 32),
                          x=torch.zeros(1, 8, 384), speeds=np.ones((1, 8), np.float32),
                          actions=np.zeros((1, 8), np.float32), projection=First32(), device=torch.device("cpu"))
            with patch("Aebs.dino_latent.evaluate_error_protocol.TemporalPredictor", side_effect=lambda *args, **kw: ToyPredictor()):
                result = native_feature_audit(**kwargs)
                self.assertAlmostEqual(result["pilot_transformer32/h1"]["normalized_coordinate_rmse"], .2, places=6)
                self.assertAlmostEqual(result["pilot_transformer384/h3"]["normalized_coordinate_rmse"], .6, places=6)
                np.savez(root/"pilot_transformer32_h1_predictions.npz", raw_r2=np.zeros((1, 32)))
                with self.assertRaises(ValueError):
                    native_feature_audit(**kwargs)


if __name__ == "__main__":
    unittest.main()
