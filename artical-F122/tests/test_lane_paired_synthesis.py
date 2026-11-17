import unittest

import numpy as np
import torch

from Aebs.lane_keeping import core
from Aebs.lane_paired_synthesis.train import successor,within_cap
from Aebs.lane_paired_synthesis.audit import summarize


class PairedSynthesisTests(unittest.TestCase):
    def test_training_dynamics_match_full_physical_units_on_extended_curves(self):
        rng=np.random.default_rng(109)
        x=rng.uniform(-1,1,(50,2))
        k=rng.choice([-.02,0.,.02],50)
        u=rng.uniform(-1,1,50)
        prediction=successor(torch.tensor(x),torch.tensor(k),torch.tensor(u)).numpy()
        scales=[core.E_SCALE,core.THETA_SCALE]
        expected=core.dynamics_batch(x*scales,u*core.PHI_MAX,k)/scales
        np.testing.assert_allclose(expected,prediction,atol=5e-15,rtol=0)

    def test_joint_training_candidate_cap_has_correct_sign(self):
        nominal=torch.tensor([[.6],[-.4]])
        observed=torch.tensor([[-1.,.62,1.],[-.8,-.35,.2]])
        limited=within_cap(observed,nominal,.1)
        torch.testing.assert_close(limited,torch.tensor([[.5,.62,.7],[-.5,-.35,-.3]]))
        self.assertIs(within_cap(observed,nominal,None),observed)

    def test_extra_curve_unknown_or_witness_cannot_be_hidden_by_easy_curves(self):
        self.assertEqual(summarize([{'status':'pass'},{'status':'timeout'}]),'timeout')
        self.assertEqual(summarize([{'status':'unresolved'},{'status':'witness'}]),'witness')
        self.assertEqual(summarize([{'status':'pass'},{'status':'pass'}]),'pass')


if __name__=='__main__':
    unittest.main()
