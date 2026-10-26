import unittest
import numpy as np
from Aebs.dino_latent.error_tail_metrics import tail_summary,trajectory_tail_summary,align_rows
from Aebs.dino_latent.calibrate_transition_contract import next_state_residual
from Aebs.dino_latent.audit_cgan_cp_tails import paired_state_residual
from Aebs.dino_latent.calibrate_all_path_residuals import residual_calibration


class ErrorTailTest(unittest.TestCase):
    def test_absolute_mae_not_signed_mean_and_closed_threshold(self):
        r=tail_summary([-.15,.15,-.2,.2],[.15])
        self.assertEqual(r["bias"],0.)
        self.assertAlmostEqual(r["mae"],.175)
        t=r["thresholds"]["0.15"]
        self.assertEqual(t["count_above"],2)
        self.assertEqual(t["within_or_equal_count"],2)
        self.assertEqual((t["positive_above"],t["negative_below"]),(1,1))

    def test_trajectory_max_not_mean(self):
        r,s=trajectory_tail_summary([.01,.2,-.1],[0,0,1],3,[.15])
        self.assertEqual(r["trajectories_checked"],2)
        self.assertEqual(r["trajectories_absent"],1)
        self.assertEqual(r["thresholds"]["0.15"]["count_above"],1)
        np.testing.assert_allclose(s[:2],[.2,.1]); self.assertTrue(np.isnan(s[2]))

    def test_pair_alignment_not_position_guess(self):
        np.testing.assert_equal(align_rows([9,2],[2,4,9]),[2,0])
        with self.assertRaises(ValueError): align_rows([9],[2,4])
        with self.assertRaises(ValueError): align_rows([2],[2,2])

    def test_CP_sign_and_speed_clipping(self):
        r=next_state_residual([8,8],[1,.05],[-3,3],[3,-3])
        np.testing.assert_allclose(r,[[0,.3],[0,-.2]],atol=1e-12)
        self.assertGreater(r[0,1],0)  # image brakes less than proxy

    def test_invalid_residuals_rejected(self):
        with self.assertRaises(ValueError): tail_summary([np.nan])
        with self.assertRaises(ValueError): tail_summary([])
        with self.assertRaises(ValueError): trajectory_tail_summary([.1],[3],3)

    def test_monte_carlo_and_speed_broadcast_have_zero_distance_residual(self):
        r=paired_state_residual(np.full((2,1,1),8.),np.ones((1,1,4)),np.zeros((2,1,4)),np.ones((2,3,4)))
        self.assertEqual(r.shape,(2,3,4,2))
        np.testing.assert_array_equal(r[...,0],0.)
        np.testing.assert_allclose(r[...,1],.05)

    def test_CP_radius_is_max_at_registered459_and_not_safety_pass(self):
        r,s=residual_calibration(np.linspace(-.2,.1,459),np.arange(459),459)
        self.assertEqual(r["rank_from_largest"],1)
        self.assertEqual(r["cp_residual_radius_mps"],.2)
        self.assertFalse(r["safety_certificate"])
        with self.assertRaises(ValueError): residual_calibration([.1],[0],459)


if __name__=="__main__": unittest.main()
