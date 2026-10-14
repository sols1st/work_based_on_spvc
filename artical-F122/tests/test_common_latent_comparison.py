"""Offline metric/protocol tests. No torch, checkpoints, or experiment data."""
import ast
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Aebs/dino_latent"


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, SOURCE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


math = load("comparison_math", "proxy_comparison_math.py")
with patch.dict(sys.modules, {"Aebs.dino_latent.proxy_comparison_math": math}):
    m = load("common_latent_metrics", "common_latent_metrics.py")


class CommonLatentTests(unittest.TestCase):
    def test_fixed_generator_pixel_mapping(self):
        np.testing.assert_array_equal(m.generator_to_pixels([-1, 0, 1]), [0, .5, 1])
        # Constant and narrow-range outputs retain their absolute brightness.
        np.testing.assert_allclose(m.generator_to_pixels([.2, .2]), [.6, .6])
        np.testing.assert_allclose(m.generator_to_pixels([-.2, .2]), [.4, .6])
        for values in ([np.nan], [np.inf], [1.1], [-1.1]):
            with self.assertRaises(ValueError):
                m.generator_to_pixels(values)

    def test_rms_dimension_and_broadcasting(self):
        np.testing.assert_allclose(m.rms_gap(np.ones((2, 3, 32)), np.zeros((2, 1, 32))), 1)
        self.assertEqual(float(m.rms_gap(np.zeros(32), np.zeros(32))), 0)
        for left, right in ((np.ones(31), np.ones(32)), ([], []), ([np.nan], [0])):
            with self.assertRaises(ValueError):
                m.rms_gap(left, right)

    def test_all_draws_no_vector_cancellation(self):
        real = np.zeros((1, 1, 32))
        generated = np.asarray([[np.zeros(32), np.ones(32), -np.ones(32)]])
        gaps = m.rms_gap(generated, real)
        report = m.summarize_gaps([.5], gaps, [.2], [0], ["rain|red"], gaps)
        stats = report["all_selected"]["comparison"]["uniform_mc"]
        self.assertEqual(stats["legacy"]["mae"], 1)
        self.assertEqual(stats["draws_per_image"], 2)
        self.assertEqual(stats["unique_image_count"], 1)
        self.assertTrue(stats["sampled_mae_and_rmse_lower"])
        self.assertFalse(stats["generalization_or_safety_claim"])

    def test_scope_pairing_and_group_weighting(self):
        generated = np.array([[.1, 2, 4], [.1, 2, 4], [.1, 2, 4]])
        report = m.summarize_gaps([1, 1, 5], generated, [.2]*3,
                                  [0, 0, 1], ["a", "a", "b"], generated)
        stats = report["all_selected"]["comparison"]["uniform_mc"]
        self.assertAlmostEqual(stats["fraction_images_dino_below_cgan_mean"], 2/3)
        self.assertAlmostEqual(stats["distance_block_macro_delta_mae"], 0)
        self.assertAlmostEqual(stats["delta_mae_dino_minus_legacy"], -2/3)
        self.assertEqual(report["a"]["images"], 2)
        self.assertEqual(report["b"]["comparison"]["uniform_mc"]["dino"]["mae"], 5)

    def test_mean_better_does_not_hide_worse_maximum(self):
        dino = np.array([0.]*99 + [4.])
        generated = np.ones((100, 3))
        report = m.summarize_gaps(dino, generated, np.zeros(100),
                                  np.arange(100), np.repeat("a", 100), generated)
        stats = report["all_selected"]["comparison"]["uniform_mc"]
        self.assertTrue(stats["sampled_mae_and_rmse_lower"])
        self.assertGreater(stats["dino"]["max_abs"], stats["legacy"]["max_abs"])

    def test_invalid_inputs(self):
        valid = [[.1, .2], np.ones((2, 3)), [.3, .3], [0, 1], ["a", "b"], np.ones((2, 3))]
        for index, replacement in ((0, [np.nan, 1]), (1, np.ones((2, 1))),
                                    (2, [-1, 1]), (3, [0]), (5, np.ones((2, 2)))):
            args = valid.copy()
            args[index] = replacement
            with self.assertRaises(ValueError):
                m.summarize_gaps(*args)

    def test_python38_syntax_and_read_only_protocol(self):
        for name in ("compare_common_latent.py", "common_latent_metrics.py"):
            ast.parse((SOURCE/name).read_text(), feature_version=(3, 8))
        text = (SOURCE/"compare_common_latent.py").read_text()
        tree = ast.parse(text)
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        methods = [n.func.attr for n in calls if isinstance(n.func, ast.Attribute)]
        for forbidden in ("backward", "step", "learn", "train", "save"):
            self.assertNotIn(forbidden, methods)
        self.assertIn('r["split"] == "validation"', text)
        self.assertIn("identity = check_final_recheck(", text)
        self.assertIn('hashlib.sha256(content).hexdigest()', text)
        self.assertIn('args.output_dir.exists()', text)
        self.assertIn('generator_to_pixels(raw)', text)

    def test_plots_produce_three_readable_pngs(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib unavailable; pure metrics tests still run")
        names = [w+"|"+c for w in ("original", "overcast", "rain", "low_sun")
                 for c in ("220,220,220", "255,0,0", "40,80,160")]
        appearances = np.repeat(names, 5)
        rng = np.random.default_rng(7)
        dino = rng.uniform(.01, .5, len(appearances))
        generated = rng.uniform(.1, .7, (len(appearances), 4))
        reference = rng.uniform(.05, .6, len(appearances))
        summary = m.summarize_gaps(dino, generated, reference,
                                  np.tile(np.arange(5), 12), appearances, generated)
        with tempfile.TemporaryDirectory() as directory:
            paths = m.make_plots(Path(directory), dino, generated, reference, appearances, summary)
            self.assertEqual(len(paths), 3)
            for name in paths:
                with Image.open(Path(directory)/name) as image:
                    self.assertEqual(image.format, "PNG")
                    self.assertGreater(image.width, 800)
                    self.assertGreater(image.height, 500)


if __name__ == "__main__":
    unittest.main()
