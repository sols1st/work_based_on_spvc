import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np

from Aebs.vision_sls_parking.core import C, RENDER_OFFSET
from Aebs.vision_sls_parking.perception import (
    fitted_bound, residual_summary, identifiability_witness, load_projection,
)
from Aebs.vision_sls_parking.renderer import visible_body_pixels
from Aebs.vision_sls_parking.test_perception import make_panel


class PerceptionMetricsTest(unittest.TestCase):
    def test_exact_target_and_algebraic_pose(self):
        s = np.array([[-2, .1, .5, 1], [-1, -.3, -.2, 0.]])
        r = residual_summary((s+RENDER_OFFSET)@C.T, s)
        self.assertEqual(r['latent_linf_max'], 0.)
        self.assertLess(r['pose_xy_euclidean_mae_m'], 1e-14)

    def test_absolute_error_not_signed_max(self):
        s = np.zeros((2, 4))
        y = (s+RENDER_OFFSET)@C.T - 1.
        r = residual_summary(y, s, bounds=np.full(2, .2))
        self.assertEqual(r['empirical_bound_coverage'], 0.)
        self.assertEqual(r['signed_max_instead_of_abs_coverage'], 1.)

    def test_speed_not_in_static_target(self):
        s = np.array([[-2, .1, .5, 1], [-2, .1, .5, -1]])
        y = (s+RENDER_OFFSET)@C.T
        np.testing.assert_array_equal(y[0], y[1])

    def test_polynomial_shift_and_order(self):
        s = np.array([[-2, .2, 0, 0], [-1, -.3, 0, 0]])
        coef = np.arange(10, dtype=float)
        x, y = s[:, 0]+3, s[:, 1]
        expected = (y+2*y*y+3*y**3+4*x+5*x*y+6*x*y*y+
                    7*x*x+8*x*x*y+9*x**3)
        np.testing.assert_allclose(fitted_bound(s, coef, 0), expected)
        np.testing.assert_equal(fitted_bound(s, np.full(10, -1.), .04), .04)

    def test_metrics_reject_bad_shapes_and_nonfinite(self):
        with self.assertRaises(ValueError): residual_summary(np.zeros((0, 3)), np.zeros((0, 4)))
        with self.assertRaises(ValueError): residual_summary(np.ones((1, 4)), np.ones((1, 4)))
        with self.assertRaises(ValueError): residual_summary(np.full((1, 3), np.nan), np.ones((1, 4)))
        with self.assertRaises(ValueError): fitted_bound([[0]*4], np.zeros(9), .01)

    def test_identical_image_witness(self):
        s = np.array([[-2, 0, 0, 0], [-1, 0, 0, 0], [-2, 0, 0, 2.]])
        target = (s+RENDER_OFFSET)@C.T
        images = np.zeros((3, 4, 4, 3), dtype=np.uint8)
        witness = identifiability_witness(s, images, target)
        self.assertEqual(len(witness), 1)
        self.assertAlmostEqual(witness[0]['any_single_image_estimator_minimum_worst_linf_error'],
                               np.max(abs(target[0]-target[1]))*.5)
        self.assertEqual(identifiability_witness(s[[0, 2]], images[:2], target[[0, 2]]), [])

    def test_visible_segmentation_handles_links_and_background(self):
        seg = np.array([-1, 3, 3+(4 << 24), 4])
        self.assertEqual(visible_body_pixels(seg, 3), 2)

    def test_panel_registered_fixed_and_same_state_targets(self):
        s, labels = make_panel(128)
        s2, _ = make_panel(128)
        self.assertEqual(s.shape, (147, 4))
        np.testing.assert_equal(s, s2)
        self.assertEqual((labels == 'occlusion_probe').sum(), 4)
        self.assertTrue((s[15:143, 3] == 0).all())


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Torch not installed in local light test runtime')
class FrozenProjectionTest(unittest.TestCase):
    def test_actual_checkpoint_and_upstream_architecture_equivalent(self):
        import torch
        from torch import nn
        root = Path(__file__).resolve().parents[2]/'VISION-SLS'
        ck = root/'expe/parking_lot/data/unicycle_checkpoint_observability_ft_075w_3pi4_097_300000_post.pt'
        if not ck.is_file() or not (root/'perception_utils/networks.py').is_file():
            self.skipTest('upstream assets unavailable')
        torch.set_num_threads(1)
        model, c, info = load_projection(ck)
        tree = ast.parse((root/'perception_utils/networks.py').read_text())
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                    and n.name == 'SupervisedDinoObservabilityLargeUnicycle')
        namespace = dict(torch=torch, nn=nn)
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<upstream projection>', 'exec'), namespace)
        upstream = namespace[node.name](SimpleNamespace(z_dim=1536, y_red_dim=3, x_dim=4))
        upstream.load_state_dict(torch.load(ck, map_location='cpu'))
        torch.manual_seed(7)
        data = torch.randn(20, 1536)
        with torch.no_grad():
            torch.testing.assert_close(model(data), upstream(data), rtol=0., atol=0.)
        self.assertFalse(any(p.requires_grad for p in model.parameters()))
        self.assertLess(info['normalized_C_max_gap'], 1e-6)


if __name__ == '__main__': unittest.main()
