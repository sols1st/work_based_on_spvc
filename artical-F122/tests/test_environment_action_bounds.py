import unittest

import numpy as np
import torch

from Aebs.dino_latent.fit_environment_action_bounds import ActionErrorBounds, bounds_loss, envelope_targets, evaluate_summary


class ActionBoundsTests(unittest.TestCase):
    def test_order_and_nominal_inclusion(self):
        model = ActionErrorBounds(8)
        x = torch.tensor([[6., .5], [16., 3.]])
        bounds = model(x)
        self.assertTrue(torch.all(bounds[:, 0] <= 0))
        self.assertTrue(torch.all(bounds[:, 1] >= 0))
        self.assertTrue(torch.all(bounds[:, 0] >= -6))
        self.assertTrue(torch.all(bounds[:, 1] <= 6))
        bounds.sum().backward()
        self.assertIsNotNone(model.net[-1].bias.grad)

    def test_group_targets_include_every_appearance(self):
        stream = dict(distance_m=np.ones(12)*7., speed_mps=np.ones(12),
                      appearance=np.array([str(i) for i in range(12)]),
                      image_action=np.linspace(-2., 2., 12), q_action=np.ones(12)*.5,
                      current_barrier=np.ones(12), image_next_state=np.ones((12,2)))
        states, target, _ = envelope_targets(stream)
        np.testing.assert_allclose(target, [[-2.5,1.5]])
        np.testing.assert_allclose(states, [[7.,1.]])

    def test_miss_penalty_increases_for_undercoverage(self):
        target = torch.tensor([[-1.,1.]])
        self.assertGreater(float(bounds_loss(torch.tensor([[-.5,.5]]), target)),
                           float(bounds_loss(target, target)))

    def test_pair_coverage_and_state_coverage_are_distinct(self):
        states=np.array([[7.,1.], [7.,1.], [8.,1.]])
        result=evaluate_summary(states,np.array([0.,2.,0.]),np.tile([-1.,1.],(3,1)),
                                np.array(["a","b","a"]),np.ones(3,bool),np.zeros(3))
        self.assertAlmostEqual(result["all_ongoing"]["coverage"],2/3)
        self.assertAlmostEqual(result["all_ongoing"]["all_observed_appearance_state_coverage"],.5)
        self.assertEqual(result["worst_cases"][0]["outside_error"],1.)


if __name__ == "__main__":
    unittest.main()
