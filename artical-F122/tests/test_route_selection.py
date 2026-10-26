import unittest
import numpy as np
import torch

from Aebs.dino_latent.train_matched_task_mlp import queried_speeds,task_auxiliary
from Aebs.dino_latent.latent_method_models import make_model
from Aebs.dino_latent.audit_route_selection import matched_indices,common_nontraining_ids,state_digest


class RouteSelectionTest(unittest.TestCase):
    def test_task_mlp_keeps_parameter_matched_initialization(self):
        torch.manual_seed(7);baseline=make_model("mlp_matched")
        torch.manual_seed(7);candidate=make_model("mlp_matched")
        self.assertEqual(sum(p.numel() for p in baseline.parameters()),sum(p.numel() for p in candidate.parameters()))
        for x,y in zip(baseline.state_dict().values(),candidate.state_dict().values()):
            self.assertTrue(torch.equal(x,y))

    def test_speed_query_uses_last_context_and_recorded_controls(self):
        v=torch.tensor([[2.,1.5,1.]])
        u=torch.tensor([[9.,9.,3.,-3.]])
        np.testing.assert_allclose(queried_speeds(v,u,3,2).numpy(),[[.85,1.]],atol=1e-7)
        with self.assertRaises(ValueError): queried_speeds(v,u[:,:3],3,2)

    def test_auxiliary_backpropagates_through_frozen_evaluators(self):
        torch.manual_seed(7)
        decoder=torch.nn.Linear(32,1);actor=torch.nn.Linear(33,1)
        for net in (decoder,actor):
            for p in net.parameters():p.requires_grad_(False)
        predicted=torch.randn(4,3,32,requires_grad=True)
        target=torch.randn(4,3,32)
        loss=task_auxiliary(predicted,target,torch.ones(4,3),torch.ones(4,3),decoder,actor,3.)
        loss.backward()
        self.assertGreater(float(predicted.grad.norm()),0.)
        self.assertTrue(all(p.grad is None for p in actor.parameters()))

    def test_same_predictions_and_labels_have_zero_auxiliary(self):
        decoder=torch.nn.Linear(32,1);actor=torch.nn.Linear(33,1)
        z=torch.randn(4,3,32);v=torch.ones(4,3)
        d=decoder(z).detach().squeeze(-1)*3
        loss=task_auxiliary(z,z,d,v,decoder,actor,3.)
        self.assertAlmostEqual(float(loss),0.,places=12)

    def test_match_target_identity_not_array_position(self):
        ix=matched_indices([2,1,1],[4,3,2],np.array([[1,1],[2,3]]),1)
        np.testing.assert_array_equal(ix,[2,0])
        with self.assertRaises(ValueError):matched_indices([1,1],[2,2],np.array([[1,1]]),1)
        with self.assertRaises(ValueError):matched_indices([1],[2],np.array([[1,2]]),1)

    def test_common_cohort_excludes_each_legacy_train_set(self):
        splits=[dict(train=[0,1],validation=[2,3,4]),dict(train=[4],validation=[1,2,3])]
        self.assertEqual(common_nontraining_ids([1,2,3,4],splits),{2,3})
        with self.assertRaises(ValueError):common_nontraining_ids([1,2],splits)
        with self.assertRaises(ValueError):common_nontraining_ids([1,2,3],[dict(train=[1],validation=[1,2,3])])

    def test_weight_digest_detects_duplicates_without_metadata(self):
        net=torch.nn.Linear(2,1)
        original=state_digest(net.state_dict())
        self.assertEqual(original,state_digest({k:v.clone() for k,v in net.state_dict().items()}))
        with torch.no_grad():net.weight.add_(.1)
        self.assertNotEqual(original,state_digest(net.state_dict()))


if __name__=="__main__":unittest.main()
