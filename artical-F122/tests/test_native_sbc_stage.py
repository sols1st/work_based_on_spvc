import unittest
import numpy as np
import torch
from torch import nn
from Aebs.dino_latent.native_sbc_stage import FrozenNativeActor, roll_constant
from Aebs.dino_latent.train_native_sbc_legacy import NativeVTPolicy


class NativeSBCStageTests(unittest.TestCase):
    def test_vt_adapter_uses_new_q_and_not_a_distance_decoder(self):
        net = FrozenNativeActor(nn.Linear(1, 4), nn.Linear(5, 1), 10., 3.)
        vt = NativeVTPolicy(net, 3.18)
        physical = torch.tensor([[15., 2.5], [7., 1.]])
        norm = physical.clone(); norm[:, 0] /= 3.18
        torch.testing.assert_close(vt(None, norm), net(physical), atol=1e-6, rtol=1e-6)
        self.assertTrue(all(not p.requires_grad for p in vt.parameters()))

    def test_own_q_normalization_and_actor_clipping(self):
        q = nn.Linear(1, 1, bias=False)
        actor = nn.Linear(2, 1, bias=False)
        with torch.no_grad(): q.weight.fill_(2); actor.weight.copy_(torch.tensor([[1., 1.]]))
        net = FrozenNativeActor(q, actor, 10., 3.)
        torch.testing.assert_close(net(torch.tensor([[13., .5], [16., 1.]])), torch.tensor([[2.5], [3.]]))
        self.assertTrue(all(not p.requires_grad for p in net.parameters()))

    def test_no_error_is_not_an_automatic_robust_pass(self):
        class Brake(nn.Module):
            def forward(self, s): return torch.ones_like(s[:, :1]) * 3
        starts = np.array([[6.1, .6]], np.float32)
        zero, _, _ = roll_constant(Brake(), starts, 0., 20)
        bad, history, action = roll_constant(Brake(), starts, .15, 20)
        self.assertEqual(zero["outcomes"]["unsafe"], 0)
        self.assertEqual(bad["outcomes"]["unsafe"], 1)
        self.assertEqual(bad["first_counterexample"]["residual_each_step"], .15)
        self.assertEqual(history.shape[0], len(action) + 1)

    def test_goal_stopped_and_unsafe_are_distinct(self):
        class Brake(nn.Module):
            def forward(self, s): return torch.ones_like(s[:, :1]) * 3
        r, _, _ = roll_constant(Brake(), np.array([[5.9, .1], [10., .1], [5.9, 1.]]), 0., 5)
        self.assertEqual(r["outcomes"]["success"], 1)
        self.assertEqual(r["outcomes"]["stopped_safe_outside_goal"], 1)
        self.assertEqual(r["outcomes"]["unsafe"], 1)

    def test_state_gradient_available_without_parameter_updates(self):
        net = FrozenNativeActor(nn.Linear(1, 2), nn.Linear(3, 1), 10., 3.)
        s = torch.tensor([[11., 1.]], requires_grad=True)
        net(s).sum().backward()
        self.assertIsNotNone(s.grad)
        self.assertTrue(all(p.grad is None for p in net.parameters()))


if __name__ == "__main__": unittest.main()
