import unittest
import numpy as np

from Aebs.dino_latent.fit_state_residual_envelope import (
    basis, trajectory_weights, split_trajectories, fit_envelope,
    predict_radius, trajectory_maxima, reference_quantile, summarize, physics_residual,
)


class StateEnvelopeTest(unittest.TestCase):
    def test_basis_nonnegative_partition_and_corners(self):
        x = np.random.default_rng(7).uniform([5,0],[16,3],(100,2))
        p = basis(x)
        self.assertEqual(p.shape,(100,9)); self.assertTrue((p>=0).all())
        np.testing.assert_allclose(p.sum(1),1)
        self.assertEqual(basis([[5,0]])[0,0],1)
        self.assertEqual(basis([[16,3]])[0,-1],1)

    def test_no_extrapolation(self):
        with self.assertRaises(ValueError): basis([[4.9,1.]])
        result = predict_radius([[4.9,1],[17,1],[10,1]],np.full(9,.01),multiplier=2)
        np.testing.assert_allclose(result,[.3,.3,.02])

    def test_split_whole_trajectory_and_repeatable(self):
        groups = np.repeat(np.arange(459),3)
        a, b = split_trajectories(groups), split_trajectories(groups)
        self.assertEqual([len(x) for x in a.values()],[183,183,93])
        for k in a: np.testing.assert_equal(a[k],b[k])
        self.assertFalse(set(a['fit']) & set(a['calibration_reference']))
        self.assertFalse(set(a['evaluation']) & set(a['fit']))
        np.testing.assert_equal(np.sort(np.concatenate(list(a.values()))),np.arange(459))

    def test_long_trajectory_does_not_dominate_fit(self):
        g = np.array([0,0,0,1]); w = trajectory_weights(g)
        self.assertAlmostEqual(w[g==0].sum(),.5); self.assertAlmostEqual(w[-1],.5)

    def test_fit_learns_state_dependence_and_exact_slack_constraints(self):
        d,v=np.meshgrid(np.linspace(5,16,10),np.linspace(0,3,8),indexing='ij')
        x=np.column_stack((d.ravel(),v.ravel())); y=.003+.03*(x[:,0]-5)/11
        beta, info=fit_envelope(x,y,np.arange(len(x)),penalty=40)
        p=predict_radius(x,beta)
        self.assertLess(abs(p-y).mean(),.002)
        self.assertGreater(predict_radius([[16,1]],beta)[0],predict_radius([[5,1]],beta)[0]+.02)
        self.assertLess(info['maximum_constraint_residual'],1e-8)
        self.assertEqual(info['ridge_gamma'],0.)

    def test_slack_can_leave_outlier_but_does_not_claim_coverage(self):
        x=np.tile([10.,1.],(100,1)); y=np.r_[np.full(99,.01),.2]
        beta,info=fit_envelope(x,y,np.arange(100),penalty=10)
        self.assertGreater(info['maximum_fitted_slack_mps'],.15)
        self.assertLess(predict_radius([[10,1]],beta)[0],.03)

    def test_CP_rank_has_finite_sample_plus_one(self):
        q,info=reference_quantile(np.arange(100,dtype=float),.01)
        self.assertEqual(q,99); self.assertEqual(info['ascending_order_statistic_rank'],100)
        self.assertFalse(info['actual_fresh_CP_guarantee'])
        self.assertFalse(info['iid_confidence_target_met'])
        with self.assertRaises(ValueError): reference_quantile(np.ones(50),.01)

    def test_calibration_uses_worst_normalized_frame_per_trajectory(self):
        ids,s=trajectory_maxima([.2,.1,3.,.4],[0,0,1,1])
        np.testing.assert_equal(ids,[0,1]); np.testing.assert_equal(s,[.2,3.])

    def test_zero_errors_and_positive_floor(self):
        x=np.tile([8.,1.],(20,1))
        beta,_=fit_envelope(x,np.zeros(20),np.arange(20))
        np.testing.assert_allclose(predict_radius(x,beta),.001,atol=1e-9)
        np.testing.assert_equal(predict_radius(x,beta,multiplier=0),0)

    def test_physical_sign_and_clipping(self):
        r=physics_residual([1,.05],[ -3,3],[3,-3])
        np.testing.assert_allclose(r,[.3,-.2])

    def test_coverage_is_whole_trajectory_not_only_step_average(self):
        s=summarize([.1,.01,.01],[.05,.05,.05],[0,0,1])
        self.assertAlmostEqual(s['step_coverage'],2/3)
        self.assertEqual(s['whole_recorded_trajectory_coverage'],.5)
        self.assertEqual(s['uncovered_trajectories'],1)

    def test_normalized_CP_envelope_changes_with_state(self):
        x=np.array([[5,0],[16,3]])
        beta=np.linspace(.001,.1,9)
        np.testing.assert_allclose(predict_radius(x,beta,multiplier=2),[.002,.2])
        np.testing.assert_allclose(predict_radius(x,beta,multiplier=20),[.02,.3])

    def test_bad_arrays_and_hyperparameters_fail_closed(self):
        for y in ([np.nan],[.31],[-.01]):
            with self.assertRaises(ValueError): fit_envelope([[8.,1]],y,[0])
        with self.assertRaises(ValueError): fit_envelope([[8.,1]],[.01],[0],penalty=0)
        with self.assertRaises(ValueError): summarize([np.nan],[.1],[0])
        with self.assertRaises(ValueError): predict_radius([[np.nan,1]],np.ones(9))


if __name__=='__main__': unittest.main()
