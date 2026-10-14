"""No torch/checkpoint needed: preprocessing, pairing and physical metrics."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
from PIL import Image

path = Path(__file__).resolve().parents[1]/"Aebs/dino_latent/proxy_comparison_math.py"
spec = importlib.util.spec_from_file_location("comparison_math",path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class ComparisonTests(unittest.TestCase):
    def test_downsample_matches_original_loops(self):
        image = Image.fromarray(np.random.default_rng(3).integers(0,256,(640,640,3),dtype=np.uint8))
        gray = np.array(image.convert("L").resize((160,160)))/255.0
        ref = np.zeros((32,32))
        for i in range(32):
            for j in range(32):
                ref[i,j] = np.mean(np.sort(gray[5*i:5*(i+1),5*j:5*(j+1)].reshape(-1))[-5:])
        ref -= ref.mean()
        ref += .5
        np.testing.assert_array_equal(m.legacy_pixels(image),np.clip(ref,0,1).astype(np.float32))

    def test_physics_sign_and_saturation(self):
        np.testing.assert_allclose(m.next_speed([1,3,.01,2],[3,-3,3,100]),[.85,3,0,1.85])
        self.assertAlmostEqual(float(m.next_speed(1,0)-m.next_speed(1,2)),.1)

    def test_paired_direction_and_unequal_groups(self):
        result = m.paired_summary([1,1,3],[2,2,2],[0,0,1])
        self.assertAlmostEqual(result["delta_mae_dino_minus_legacy"],-1/3)
        self.assertEqual(result["distance_block_macro_delta_mae"],0)
        self.assertFalse(result["generalization_or_safety_claim"])
        self.assertAlmostEqual(result["fraction_dino_abs_error_lower"],2/3)

    def test_mc_does_not_cancel_signed_errors(self):
        result = m.transition_summary(np.array([[.01,.02]]),np.array([[[.1,.2],[-.1,-.2]]]),
                                      np.array([[True,True]]),np.array([0]))
        self.assertAlmostEqual(result["legacy"]["mae"],.15)
        self.assertAlmostEqual(result["legacy"]["bias"],0)
        self.assertAlmostEqual(result["dino"]["mae"],.015)
        self.assertEqual(result["physical_state_pairs"],2)
        self.assertEqual(result["latent_draws"],2)

    def test_mask_preserves_image_speed_draw_pairing(self):
        result = m.transition_summary(np.array([[1,99],[99,2]]),np.array([[[3,99],[5,99]],[[99,4],[99,6]]]),
                                      np.eye(2,dtype=bool),[10,20])
        self.assertEqual(result["legacy"]["mae"],4.5)
        self.assertEqual(result["dino"]["mae"],1.5)
        self.assertEqual(result["by_distance_block_delta_mae"],{"10":-3.,"20":-3.})

    def test_validation_and_empty(self):
        with self.assertRaises(ValueError):
            m.errors([float("nan")])
        with self.assertRaises(ValueError):
            m.paired_summary([1,2],[1],[0])
        with self.assertRaises(ValueError):
            m.transition_summary(np.ones((1,2)),np.ones((1,1,3)),np.ones((1,2),bool),[0])
        result = m.transition_summary(np.zeros((1,2)),np.zeros((1,3,2)),np.zeros((1,2),bool),[0])
        self.assertEqual(result["status"],"empty")


if __name__ == "__main__":
    unittest.main()
