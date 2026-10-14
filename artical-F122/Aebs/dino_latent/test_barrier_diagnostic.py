"""Lightweight logic tests; no model loading or experiments."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("logic", Path(__file__).with_name("barrier_diagnostic_logic.py"))
logic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(logic)


def row(b=1., safe=True, band=False, margin=-1., compensation=-2., distance=8.):
    return dict(barrier=b, all_safe=safe, legacy_band=band, margin=margin,
                compensated_margin=compensation, distance_m=distance, state=[distance/3, 1.])


class DiagnosticTests(unittest.TestCase):
    def test_target_threshold_includes_boundary_excludes_unsafe(self):
        rows = [row(2), row(4), row(4.1), row(1, safe=False)]
        result = logic.target_scope(rows, 2, 5, .5)
        self.assertEqual(result["raw_barrier_threshold"], 4)
        self.assertEqual([r["target_probability_scope"] for r in rows], [True, True, False, False])
        self.assertTrue(result["region_separation_sufficient"])

    def test_failed_separation_is_not_pass(self):
        result = logic.target_scope([row()], 2, 3, .5)
        self.assertFalse(result["region_separation_sufficient"])
        self.assertTrue(result["not_a_certified_probability"])

    def test_invalid_parameters(self):
        for a, p in [(0, .9), (-1, .9), (1, 0), (1, 1), (float("nan"), .9)]:
            with self.assertRaises(ValueError):
                logic.target_scope([], a, 20, p)
        with self.assertRaises(ValueError):
            logic.select_cases([], 2)

    def test_selection_prioritizes_target_and_preserves_band(self):
        rows = [row(1, margin=-2), row(1, margin=.1, compensation=-3),
                row(10, band=True, margin=.1, compensation=-1), row(10, margin=-100)]
        logic.target_scope(rows, 1, 20, .5)
        selected = logic.select_cases(rows, 3)
        self.assertEqual(set(i for i, _ in selected), {0, 1, 2})
        self.assertEqual(selected, logic.select_cases(rows, 3))

    def test_selection_deduplicates_and_handles_empty(self):
        rows = [row(band=True)]
        logic.target_scope(rows, 1, 20, .5)
        selected = logic.select_cases(rows, 12)
        self.assertEqual(len(selected), 1)
        self.assertEqual(len(selected[0][1]), 3)
        self.assertEqual(logic.select_cases([], 12), [])

    def test_interval_classification(self):
        self.assertEqual(logic.classify_point(.1, .2), "interval_supports_expected_increase_at_sample")
        self.assertEqual(logic.classify_point(-.2, -.1), "interval_supports_strict_decrease_at_sample")
        for bounds in [(-.1, .1), (-1e-8, 1e-8), (0, 0)]:
            self.assertEqual(logic.classify_point(*bounds), "unresolved_interval_or_near_zero")
        for bounds in [(2, 1), (float("nan"), 1), (-1, float("inf"))]:
            with self.assertRaises(ValueError):
                logic.classify_point(*bounds)


if __name__ == "__main__":
    unittest.main()
