import unittest
import numpy as np
import torch

from Aebs.dino_latent.diagnose_residual_sources import physical_action_bound, smoothness_penalty, appearance_floor


class ResidualSourcesTest(unittest.TestCase):
    def test_detached_input_gradient_penalty_cannot_regularize_weights(self):
        net=torch.nn.Linear(3,1,bias=False)
        with torch.no_grad(): net.weight.fill_(3.)
        x=torch.zeros(8,3)
        old=smoothness_penalty(net,x,False)
        fixed=smoothness_penalty(net,x,True)
        self.assertGreater(float(old),0)
        self.assertFalse(old.requires_grad)
        self.assertTrue(fixed.requires_grad)
        grad=torch.autograd.grad(fixed,net.weight)[0]
        self.assertGreater(float(grad.norm()),0)

    def test_interior_exact_action_residual_relationship(self):
        v=np.array([1.,2.]); ai=np.array([2.,-1.]); aq=np.array([-1.,1.])
        r,b=physical_action_bound(v,ai,aq)
        np.testing.assert_allclose(r,.05*(aq-ai),atol=1e-15)
        np.testing.assert_allclose(np.abs(r),b,atol=1e-15)

    def test_action_and_velocity_clipping_preserve_bound(self):
        v=np.array([0.,3.,.01,2.99]); ai=np.array([9.,-9.,3.,-3.]); aq=-ai
        r,b=physical_action_bound(v,ai,aq)
        self.assertTrue((np.abs(r)<=b+1e-12).all())
        self.assertTrue((np.abs(r)<b-1e-7).any())

    def test_identical_actions_have_zero_residual(self):
        r,b=physical_action_bound(np.array([0.,1.,3.]),np.array([3.,-3.,0.]),np.array([3.,-3.,0.]))
        np.testing.assert_array_equal(r,0.)
        np.testing.assert_array_equal(b,0.)

    def test_float32_actions_are_promoted_before_bound_arithmetic(self):
        rng=np.random.default_rng(7)
        v=rng.uniform(0,3,40000).astype(np.float32)
        ai=rng.uniform(-8,8,40000).astype(np.float32)
        aq=rng.uniform(-8,8,40000).astype(np.float32)
        r,b=physical_action_bound(v,ai,aq)
        self.assertTrue((np.abs(r)<=b+1e-12).all())

    def test_appearance_floor_is_half_observed_output_range(self):
        r,s,n=appearance_floor([7,7],[1],[[3],[-3]],[[True],[True]])
        np.testing.assert_allclose(r,[.15],atol=1e-14)
        np.testing.assert_array_equal(n,[2])
        np.testing.assert_array_equal(s,[[7,1]])

    def test_appearance_floor_accounts_for_speed_clipping(self):
        r,_,_=appearance_floor([7,7],[0],[[3],[-3]],[[True],[True]])
        np.testing.assert_allclose(r,[.075],atol=1e-14)

    def test_appearance_floor_requires_common_physical_state(self):
        with self.assertRaises(ValueError):
            appearance_floor([7,8],[1],[[3],[-3]],[[True],[True]])


if __name__=="__main__": unittest.main()
