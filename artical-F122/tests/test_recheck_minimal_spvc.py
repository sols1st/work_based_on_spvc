import ast
import importlib.util
from pathlib import Path
import unittest

FILE = Path(__file__).resolve().parents[1]/"Aebs/dino_latent/recheck_minimal_spvc.py"
spec = importlib.util.spec_from_file_location("recheck",FILE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class RecheckTests(unittest.TestCase):
    def test_reported_training_example(self):
        r = m.summarize(1,10000,69,2,1.1836737394332886,20.141101837158203,.3146599233150482,True)
        self.assertEqual(r["status"],"saved_pair_passes_legacy_checks")
        self.assertAlmostEqual(r["legacy_probability_readout"],.9561689475149103)
        self.assertFalse(r["zero_hard_violations"])
        self.assertFalse(r["end_to_end_safety_certificate"])

    def test_threshold_inclusive_and_not_band_denominator(self):
        self.assertTrue(m.summarize(10,10000,20,11,1,20,0,True)["legacy_threshold_pass"])
        self.assertEqual(m.summarize(11,10000,20,12,1,20,0,False)["status"],"saved_pair_fails_legacy_decrease_threshold")

    def test_empty_band_separately_flagged(self):
        r = m.summarize(0,10000,0,0,1,20,0,True)
        self.assertTrue(r["legacy_threshold_pass"])
        self.assertEqual(r["status"],"empty_legacy_band_not_evidence")

    def test_invalid_regions(self):
        for bounds in [(1,1,1),(2,1,0),(1,2,0)]:
            r = m.summarize(0,10000,20,1,*bounds,True)
            self.assertEqual(r["status"],"saved_pair_does_not_reach_legacy_probability_target")
        self.assertEqual(m.summarize(0,10000,20,1,float("nan"),20,0,True)["status"],"invalid_nonfinite_region_bounds")

    def test_inconsistent_counts_rejected(self):
        for values in [(1,10000,0,0),(2,10000,20,1),(1,0,20,1)]:
            with self.assertRaises(ValueError):
                m.summarize(*values,1,20,0,True)
        with self.assertRaises(ValueError):
            m.summarize(11,10000,20,12,1,20,0,True)

    def test_no_training_and_only_original_verifier_import(self):
        tree = ast.parse(FILE.read_text())
        imports = [n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        self.assertIn("Aebs.VT.verify",imports)
        self.assertNotIn("Aebs.VT.train",imports)
        self.assertNotIn("Aebs.semantic_spvc.corrected_verify",imports)
        methods = [n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        self.assertNotIn("backward",methods)
        self.assertNotIn("step",methods)
        self.assertNotIn("train",methods)


if __name__ == "__main__":
    unittest.main()
