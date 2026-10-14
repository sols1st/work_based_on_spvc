"""Static integration checks; server --check-only tests actual tensor interfaces."""
import ast
import contextlib
import importlib.util
import io
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT/"Aebs/dino_latent/run_minimal_spvc.py"
spec = importlib.util.spec_from_file_location("minimal_launcher",FILE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def constant_settings(path, constructor):
    tree = ast.parse(path.read_text())
    call = next(n for n in ast.walk(tree) if isinstance(n,ast.Call)
                and isinstance(n.func,ast.Name) and n.func.id == constructor)
    values = {}
    for arg in call.keywords:
        try:
            values[arg.arg] = ast.literal_eval(arg.value)
        except (ValueError, TypeError):
            pass
    return values


class MinimalTests(unittest.TestCase):
    def test_default_original_compute_budget(self):
        args = m.make_parser().parse_args([])
        self.assertEqual(args.timeout_seconds,3600)
        self.assertEqual(args.max_iteration_index,100)
        self.assertFalse(args.check_only)

    def test_no_extra_experiment_forwarded(self):
        args = m.make_parser().parse_args(["--check-only"])
        forwarded = m.training_arguments(args)
        self.assertEqual(len(forwarded),12)
        forbidden = ("--check-only","--compare-scopes","--initial-sbc","--diagnose-only-from","--stop-on-zero-violation")
        self.assertFalse(any(f in forwarded for f in forbidden))

    def test_extra_flags_rejected(self):
        for flag in ("--compare-scopes","--initial-sbc","--stop-on-zero-violation"):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                m.make_parser().parse_args([flag])

    def test_learner_and_verifier_settings_match_existing_legacy_entry(self):
        original = ROOT/"Aebs/VT/loop.py"
        adapter = ROOT/"Aebs/dino_latent/run_spvc.py"
        for constructor in ("VTLearner","VTVerifier","Loop"):
            legacy = constant_settings(original,constructor)
            replacement = constant_settings(adapter,constructor)
            self.assertTrue(legacy)
            for key,value in legacy.items():
                self.assertEqual(replacement.get(key),value,(constructor,key))

    def test_legacy_verifier_only_and_lazy_optional_diagnostic(self):
        tree = ast.parse((ROOT/"Aebs/dino_latent/run_spvc.py").read_text())
        top_imports = [n.module for n in tree.body if isinstance(n,ast.ImportFrom)]
        self.assertIn("Aebs.VT.verify",top_imports)
        self.assertNotIn("Aebs.semantic_spvc.corrected_verify",top_imports)
        self.assertNotIn("Aebs.dino_latent.compare_spvc_scopes",top_imports)


if __name__ == "__main__":
    unittest.main()
