"""Dependency-free report logic tests; no models or experiments are run."""

import unittest
import importlib.util
from pathlib import Path

# Load directly so local smoke tests do not import package-level DINO/torch.
_spec = importlib.util.spec_from_file_location(
    "scope_report", Path(__file__).with_name("compare_spvc_scopes.py")
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
summarize_scope = _module.summarize_scope


def row(margin, selected=True, compensated=None):
    return dict(legacy_band=selected, state=[2.0, 1.0], distance_m=6.0,
                barrier=1.0, expected=1.0-margin, margin=margin,
                compensated_margin=margin if compensated is None else compensated)


class ScopeTests(unittest.TestCase):
    def test_legacy_strict_endpoints(self):
        for value in (0.0, 9.5, 10.0, 11.0):
            self.assertFalse(_module.legacy_band_mask(value, 10.0))
        self.assertTrue(_module.legacy_band_mask(9.75, 10.0))

    def test_real_denominator_and_equality(self):
        report = summarize_scope([row(0.0), row(1.0), row(-1, False)], "legacy_band", 10000, 10000)
        self.assertEqual(report["checked_states"], 2)
        self.assertEqual(report["hard_violations"], 1)
        self.assertEqual(report["hard_violation_fraction_checked"], 0.5)
        self.assertEqual(report["hard_violation_fraction_total_grid"], 0.0001)
        self.assertTrue(report["legacy_ratio_rule_only"])
        self.assertFalse(report["zero_hard_violations_nonempty"])

    def test_empty_not_success(self):
        report = summarize_scope([row(1, False)], "legacy_band", 1, 1)
        self.assertEqual(report["status"], "empty_scope_not_evidence")
        self.assertIsNone(report["min_margin"])
        self.assertFalse(report["zero_hard_violations_nonempty"])

    def test_nonfinite_not_success(self):
        report = summarize_scope([row(float("nan"))], "legacy_band", 1, 1)
        self.assertEqual(report["status"], "nonfinite_not_evidence")
        self.assertIsNone(report["hard_violations"])

    def test_compensated_is_separate(self):
        report = summarize_scope([row(0.01, compensated=-0.01)], "legacy_band", 1, 1)
        self.assertEqual(report["hard_violations"], 0)
        self.assertEqual(report["compensated_violations"], 1)


if __name__ == "__main__":
    unittest.main()
