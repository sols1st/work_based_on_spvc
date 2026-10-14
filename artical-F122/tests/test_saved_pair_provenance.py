"""Final-pair attribution tests; no model loading or experiment execution."""
import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "provenance", ROOT / "Aebs/dino_latent/saved_pair_provenance.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rep = self.root / "representation.pt"
        self.base = self.root / "base.zip"
        self.recheck = self.root / "recheck.json"
        self.paths = dict(metrics=self.root/"metrics.json", policy=self.root/"latent_ppo_spvc.pt",
                          barrier=self.root/"sbc.pt", representation=self.rep, base_controller=self.base,
                          minimal_manifest=self.root/"minimal_run_manifest.json")
        for key, path in self.paths.items():
            path.write_text(key, encoding="utf-8")
        self.report = dict(experiment="minimal_R2_saved_pair_original_SPVC_recheck",
                           status="saved_pair_passes_legacy_checks", checkpoint_matches_verification=True,
                           checkpoint_files_unchanged=True, training_run=False, test_used=False,
                           source_hashes={key: m.digest(path) for key, path in self.paths.items()})
        self.save()

    def save(self):
        self.recheck.write_text(json.dumps(self.report), encoding="utf-8")

    def check(self):
        return m.check_final_recheck(self.rep, self.base, self.root, self.recheck)

    def test_final_pair_works_without_step22_scope_file(self):
        self.assertFalse((self.root/"scope_comparison.json").exists())
        result = self.check()
        self.assertTrue(result["checkpoint_matches_final_recheck"])
        self.assertEqual(result["final_recheck_source_hashes"], self.report["source_hashes"])

    def test_any_source_change_rejected(self):
        for key, path in self.paths.items():
            with self.subTest(source=key):
                original = path.read_bytes()
                path.write_bytes(original + b"changed")
                with self.assertRaisesRegex(ValueError, "different checkpoint"):
                    self.check()
                path.write_bytes(original)

    def test_invalid_flags_or_status_rejected(self):
        for key, value in dict(status="failed", checkpoint_matches_verification=False,
                               checkpoint_files_unchanged=False, training_run=True, test_used=True,
                               experiment="other_run").items():
            with self.subTest(field=key):
                previous = self.report[key]
                self.report[key] = value
                self.save()
                with self.assertRaises(ValueError):
                    self.check()
                self.report[key] = previous

    def test_missing_hash_rejected(self):
        del self.report["source_hashes"]["policy"]
        self.save()
        with self.assertRaises(ValueError):
            self.check()

    def test_removed_recorded_manifest_rejected(self):
        self.paths["minimal_manifest"].unlink()
        with self.assertRaises(ValueError):
            self.check()

    def test_preexisting_recheck_without_manifest_supported(self):
        self.paths["minimal_manifest"].unlink()
        del self.report["source_hashes"]["minimal_manifest"]
        self.save()
        self.assertTrue(self.check()["checkpoint_matches_final_recheck"])

    def test_evaluator_is_read_only_and_defaults_to_minimal(self):
        tree = ast.parse((ROOT/"Aebs/dino_latent/compare_legacy_dino.py").read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        methods = [n.func.attr for n in calls if isinstance(n.func, ast.Attribute)]
        for forbidden in ("backward", "step", "learn", "train"):
            self.assertNotIn(forbidden, methods)
        stage = next(n for n in calls if isinstance(n.func, ast.Attribute)
                     and n.func.attr == "add_argument" and n.args
                     and isinstance(n.args[0], ast.Constant) and n.args[0].value == "--dino-stage")
        self.assertEqual(next(k.value.value for k in stage.keywords if k.arg == "default"), "minimal")
        loader = next(n for n in calls if isinstance(n.func, ast.Name) and n.func.id == "load_spvc_pair")
        self.assertIn("final_recheck", [k.arg for k in loader.keywords])


if __name__ == "__main__":
    unittest.main()
