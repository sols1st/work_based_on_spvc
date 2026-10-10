"""Run directly: python Aebs/dino_latent/test_spvc_pair.py -v (no experiment)."""

import importlib.util
from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
spec = importlib.util.spec_from_file_location("pair_summary", Path(__file__).with_name("spvc_pair_summary.py"))
summary_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary_module)


class SummaryTests(unittest.TestCase):
    def test_regression_even_if_new_paths_agree(self):
        summary = summary_module.PairSummary()
        summary.add([3, 3, 0, 0], ["ongoing", "ongoing", "unsafe", "unsafe"])
        result = summary.result()
        self.assertEqual(result["counts"]["new_both_next_unsafe"], 1)
        self.assertEqual(result["counts"]["new_image_only_next_unsafe"], 0)
        self.assertEqual(result["counts"]["image_new_unsafe_old_not"], 1)
        self.assertEqual(result["counts"]["q_new_unsafe_old_not"], 1)
        self.assertEqual(result["action"]["new_dual_path"]["mae"], 0)

    def test_improvement_and_action_statistics(self):
        summary = summary_module.PairSummary()
        summary.add([0, 1, 2, 2], ["unsafe", "ongoing", "ongoing", "ongoing"])
        summary.add([0, 1, 1, 2], ["ongoing"] * 4)
        result = summary.result()
        self.assertEqual(result["counts"]["image_old_unsafe_new_not"], 1)
        self.assertEqual(result["action"]["old_dual_path"]["mae"], 1)
        self.assertEqual(result["action"]["new_dual_path"]["mae"], 0.5)
        self.assertEqual(result["action"]["image_policy_drift"]["mae"], 1.5)
        self.assertEqual(result["action"]["image_policy_drift"]["max_abs"], 2)

    def test_empty_and_nonfinite(self):
        summary = summary_module.PairSummary()
        self.assertIsNone(summary.result()["action"]["old_dual_path"]["mae"])
        with self.assertRaises(ValueError):
            summary.add([0, float("nan"), 0, 0], ["ongoing"]*4)


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "torch not installed locally")
class ActorTests(unittest.TestCase):
    def test_clipping_shapes_and_deterministic_only(self):
        import numpy as np
        import torch
        from Aebs.dino_latent.spvc_saved_actor import SavedSpvcActor

        class Toy(torch.nn.Module):
            latent_dimension = 2

            def __init__(self):
                super().__init__()
                self.controller_net = torch.nn.Linear(3, 1, bias=False)
                with torch.no_grad():
                    self.controller_net.weight.fill_(1)

        actor = SavedSpvcActor(Toy(), [-3], [3])
        np.testing.assert_array_equal(actor.predict([5, 0, 0])[0], [3])
        np.testing.assert_array_equal(actor.predict([[-5, 0, 0], [0.2, 0.3, 0.5]])[0], [[-3], [1]])
        with self.assertRaises(ValueError):
            actor.predict([0, 0, 0], deterministic=False)
        with self.assertRaises(ValueError):
            actor.predict([0, 0])

    def test_adapter_and_physical_scale_equivalence(self):
        import numpy as np
        import torch
        from Aebs.dino_latent.spvc_policy import DinoLatentSPVCPolicy
        from Aebs.dino_latent.spvc_saved_actor import SavedSpvcActor, check_adapter_equivalence

        q = torch.nn.Linear(1, 2, bias=False)
        controller = torch.nn.Linear(3, 1, bias=False)
        with torch.no_grad():
            q.weight.fill_(1)
            controller.weight.fill_(0.1)
        policy = DinoLatentSPVCPolicy(q, controller, 2.0, 2)
        policy.set_state_distance_scale(4.0)
        actor = SavedSpvcActor(policy, [-3], [3])
        distance = np.asarray([6, 8], dtype=np.float32)
        speed = np.asarray([1, 2], dtype=np.float32)
        q_obs = np.column_stack([distance/2, distance/2, speed]).astype(np.float32)
        result = check_adapter_equivalence(actor, actor, actor, q_obs, q_obs, distance, speed)
        self.assertLessEqual(max(result.values()), 1e-5)


if __name__ == "__main__":
    unittest.main()
