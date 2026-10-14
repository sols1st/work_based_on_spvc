import unittest

import numpy as np
import torch

from Aebs.dino_latent.nbc_multistep import check_transition_chain,multistep_scores


class MultiStepTests(unittest.TestCase):
    def score(self,c,cn,ids,term,steps=5,trunc=None):
        if trunc is None: trunc=np.zeros(len(ids),bool)
        return multistep_scores(torch.tensor(c,dtype=torch.float32),torch.tensor(cn,dtype=torch.float32),
            ids,term,trunc,steps=steps)

    def test_one_step_is_original_hinge_even_at_terminal(self):
        c=torch.tensor([-.2,.1,-1.])
        cn=torch.tensor([.4,.1,-.5])
        result,valid,counts=multistep_scores(c,cn,[1,2,3],[True,True,True],[False]*3,steps=1)
        torch.testing.assert_close(result,torch.relu(cn-.9*c+.01))
        self.assertEqual(valid.tolist(),[1,1,1])
        self.assertEqual(counts,[3])

    def test_terminal_successor_is_included_but_next_episode_is_not(self):
        result,horizons,counts=self.score([-1.,-1.,100.,100.],[-1.,.5,100.,100.],
                                          [1,1,2,2],[False,True,False,False])
        expected=(0.+.99*(.5-.81*(-1.)+.019))/(1.+.99)
        self.assertAlmostEqual(float(result[0]),expected,places=6)
        self.assertEqual(horizons.tolist(),[2,1,2,1])
        self.assertEqual(counts,[4,2,0,0,0])

    def test_truncation_is_also_a_boundary(self):
        _,horizons,_=self.score([-1.]*3,[-1.]*3,[1,1,2],[False]*3,trunc=[False,True,False])
        self.assertEqual(horizons.tolist(),[2,1,1])

    def test_done_cannot_be_crossed_even_if_episode_id_is_reused(self):
        _,horizons,_=self.score([-1.]*3,[-1.]*3,[1,1,1],[False,True,False])
        self.assertEqual(horizons.tolist(),[2,1,1])

    def test_accumulated_margin_and_available_horizon_normalization(self):
        result,horizons,_=self.score([0.]*5,[0.]*5,[1]*5,[False]*5)
        expected=sum(.99**(k-1)*.01*(1-.9**k)/.1 for k in range(1,6))/sum(.99**(k-1) for k in range(1,6))
        self.assertAlmostEqual(float(result[0]),expected,places=6)
        self.assertAlmostEqual(float(result[-1]),.01,places=6)
        self.assertEqual(horizons.tolist(),[5,4,3,2,1])

    def test_certificate_gradient_preserved_for_training(self):
        c=torch.zeros(5,requires_grad=True);cn=torch.ones(5,requires_grad=True)
        result,_,_=multistep_scores(c,cn,[1]*5,[False]*5,[False]*5)
        result.mean().backward()
        self.assertGreater(float(cn.grad.abs().sum()),0.)
        self.assertGreater(float(c.grad.abs().sum()),0.)

    def test_chain_rejects_reset_as_successor(self):
        with self.assertRaisesRegex(ValueError,"successors"):
            check_transition_chain([[1.,1.],[9.,9.]],[[2.,2.],[10.,10.]],
                                    [1,1],[False,False],[False,False])
        # The same jump is legal after a true terminal, not treated as dynamics.
        check_transition_chain([[1.,1.],[9.,9.]],[[2.,2.],[10.,10.]],
                                [1,2],[True,False],[False,False])


if __name__=="__main__": unittest.main()
