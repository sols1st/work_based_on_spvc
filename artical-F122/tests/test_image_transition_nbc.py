import unittest
import numpy as np
import torch

from Aebs.dino_latent.train_image_transition_nbc import SignedNBC, barrier_losses, region_grids, evaluate_pairs


class ImageNBCTests(unittest.TestCase):
    def test_output_is_signed_not_softplus(self):
        model=SignedNBC(3.,8)
        with torch.no_grad():
            model.net[-1].weight.zero_()
            model.net[-1].bias.fill_(-1.)
        torch.testing.assert_close(model(torch.zeros(2,2)),torch.full((2,),-1.))

    def test_regions_use_task_bounds_not_recoverability_labels(self):
        init,unsafe=region_grids(3.,7,torch.device("cpu"))
        self.assertEqual(len(init),49)
        self.assertAlmostEqual(float(init[:,0].min()*3),15.)
        self.assertAlmostEqual(float(unsafe[:,0].max()*3),6.)
        self.assertTrue(torch.all(unsafe[:,1]>.5))

    def test_only_barrier_receives_gradients_from_recorded_pairs(self):
        model=SignedNBC(3.,8)
        x=torch.tensor([[3.,1.],[4.,2.]])
        y=torch.tensor([[2.9,1.1],[3.9,1.9]])
        initial,unsafe=region_grids(3.,4,torch.device("cpu"))
        loss,_=barrier_losses(model,x,y,initial,unsafe,.1,.01)
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters()))
        self.assertIsNone(x.grad)
        self.assertIsNone(y.grad)

    def test_nonpositive_region_escape_and_unsafe_are_not_identical(self):
        class Linear(torch.nn.Module):
            def forward(self,x): return x[:,1]-1.
        x=np.array([[2.5,.9],[2.5,2.]],dtype=np.float32)
        y=np.array([[2.4,1.1],[1.9,2.]],dtype=np.float32)
        result,_,_=evaluate_pairs(Linear(),x,y,np.zeros(2),np.zeros(2),3.,.1,.01,2,torch.device("cpu"))
        self.assertEqual(result["candidate_nonpositive_region"]["sign_escapes"],1)
        self.assertEqual(result["candidate_region_with_unsafe_successor"],0)
        self.assertEqual(result["all_pairs"]["unsafe_successors"],1)

    def test_negative_barrier_may_increase_without_raw_violation(self):
        class Identity(torch.nn.Module):
            def forward(self,x): return x[:,0]
        x=np.array([[-1.,0.]],dtype=np.float32)
        y=np.array([[-.95,0.]],dtype=np.float32)
        result,_,_=evaluate_pairs(Identity(),x,y,np.zeros(1),np.zeros(1),3.,.1,.01,1,torch.device("cpu"))
        self.assertEqual(result["all_pairs"]["raw_invariance_violations"],0)

    def test_hard_tail_emphasis_does_not_relax_mean_loss(self):
        torch.manual_seed(7)
        model=SignedNBC(3.,8)
        x=torch.rand(20,2)
        y=torch.rand(20,2)
        initial,unsafe=region_grids(3.,4,torch.device("cpu"))
        mean,_=barrier_losses(model,x,y,initial,unsafe,.1,.01,0.)
        focused,parts=barrier_losses(model,x,y,initial,unsafe,.1,.01,.1)
        self.assertGreaterEqual(float(focused),float(mean)-1e-8)
        self.assertGreaterEqual(float(parts["optimized_invariance"]),float(parts["invariance"])-1e-8)


if __name__=="__main__": unittest.main()
