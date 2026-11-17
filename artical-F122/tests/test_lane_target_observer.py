"""Regression checks for the known-curvature correction at deployment."""
import unittest

import numpy as np
import torch

from Aebs.lane_keeping import core
from Aebs.lane_paired_synthesis.evaluate import draws, KAPPAS
from Aebs.lane_paired_synthesis.evaluate_target import CorrectedActor


class LaneTargetObserverTest(unittest.TestCase):
    def test_correction_goes_before_policy_and_uses_only_known_curvature(self):
        centers={str(k):[.3+i*.05,-.12-i*.03] for i,k in enumerate(KAPPAS)}
        # Identity readout exposes the actual three signals sent to the policy.
        actor=CorrectedActor(torch.nn.Identity(),centers)
        physical_curve=np.asarray(KAPPAS)
        raw=np.column_stack((np.linspace(-.25,.25,5),np.linspace(.11,.14,5),
                             physical_curve/core.KAPPA_SCALE))
        result=actor(torch.from_numpy(raw))
        shifted=raw.copy()
        shifted[:,:2]-=np.array([centers[str(k)] for k in physical_curve])
        np.testing.assert_allclose(result.numpy(),shifted,atol=1e-15,rtol=0)
        raw[0,2]=1.5
        with self.assertRaises(ValueError):
            actor(torch.from_numpy(raw))

    def test_paired_draws_and_heldout_independence(self):
        x,k=draws(202611178311,199)
        repeat,k_again=draws(202611178311,199)
        np.testing.assert_array_equal(x,repeat)
        np.testing.assert_array_equal(k,k_again)
        test,kt=draws(202611178312,100)
        self.assertEqual(set(KAPPAS),set(kt))
        self.assertFalse(set(map(tuple,x)) & set(map(tuple,test)))
        self.assertTrue(core.INITIAL.contains(x).all())


if __name__=='__main__':
    unittest.main()
