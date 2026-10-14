import copy
import unittest

import torch

from Aebs.dino_latent.nbc_retention import retention_scores, retention_failures, guarded_nbc_update


class RetentionTests(unittest.TestCase):
    def setUp(self):
        self.model=torch.nn.Sequential(torch.nn.Linear(2,1),torch.nn.Flatten(0))
        with torch.no_grad():
            self.model[0].weight.copy_(torch.tensor([[-1.,1.]]))
            self.model[0].bias.zero_()
        self.initial=torch.tensor([[2.,0.]])
        self.unsafe=torch.tensor([[0.,1.]])
        self.reference=retention_scores(self.model,self.initial,self.initial,self.initial,self.unsafe)

    def test_existing_constraints_pass_but_one_unsafe_error_rejects(self):
        self.assertEqual(retention_failures(self.reference,self.reference),[])
        bad=dict(self.reference,unsafe_min=-.001,unsafe_margin_violations=1,unsafe_sign_violations=1)
        self.assertIn("unsafe_margin",retention_failures(bad,self.reference))

    def test_old_tail_and_boundary_escapes_cannot_be_hidden_by_mean(self):
        bad=dict(self.reference,old_raw_violations=1,old_sign_escapes=1)
        failures=retention_failures(bad,self.reference)
        self.assertIn("old_raw_violations_increased",failures)
        self.assertIn("old_sign_escapes_increased",failures)

    def test_nonfinite_candidate_rejected(self):
        self.assertIn("nonfinite_candidate",retention_failures(dict(self.reference,old_hinge_mean=float("nan")),self.reference))

    def test_rejection_restores_weights_and_adam_moments(self):
        optimizer=torch.optim.Adam(self.model.parameters(),lr=.01)
        self.model(self.initial).square().mean().backward()
        optimizer.step()
        weights=copy.deepcopy(self.model.state_dict())
        state=copy.deepcopy(optimizer.state_dict())
        # An impossible reference forces rejection after a genuine update,
        # allowing an exact test of restoring nonempty optimizer moments.
        forced=dict(self.reference,old_hinge_mean=-1.)
        result=guarded_nbc_update(self.model,optimizer,self.initial,self.unsafe,
            self.initial,self.initial,self.initial,self.unsafe,
            self.initial,self.initial,self.initial,self.unsafe,forced,
            torch.Generator().manual_seed(7),steps=2)
        self.assertFalse(result["accepted"])
        for k,v in weights.items():
            torch.testing.assert_close(self.model.state_dict()[k],v,rtol=0,atol=0)
        restored=optimizer.state_dict()
        self.assertEqual(restored["param_groups"],state["param_groups"])
        for k,values in state["state"].items():
            for name,value in values.items():
                torch.testing.assert_close(restored["state"][k][name],value,rtol=0,atol=0)

    def test_memory_sampling_does_not_advance_policy_rng(self):
        before=torch.get_rng_state().clone()
        private=torch.Generator().manual_seed(7)
        torch.randperm(20,generator=private)
        torch.randint(20,(8,),generator=private)
        torch.testing.assert_close(before,torch.get_rng_state())

    def test_passing_candidate_keeps_a_real_parameter_update(self):
        optimizer=torch.optim.Adam(self.model.parameters(),lr=.01)
        before=copy.deepcopy(self.model.state_dict())
        result=guarded_nbc_update(self.model,optimizer,self.unsafe,self.unsafe,
            self.initial,self.initial,self.initial,self.unsafe,
            self.initial,self.initial,self.initial,self.unsafe,self.reference,
            torch.Generator().manual_seed(7),steps=1)
        self.assertTrue(result["accepted"])
        self.assertTrue(any(not torch.equal(before[k],v) for k,v in self.model.state_dict().items()))


if __name__=="__main__": unittest.main()
