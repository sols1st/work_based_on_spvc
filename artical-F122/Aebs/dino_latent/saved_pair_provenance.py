"""Read-only identity checks for evaluating the independently rechecked minimal pair."""
import hashlib
import json


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_final_recheck(representation, base_controller, spvc_dir, recheck_path):
    paths = {
        "metrics": spvc_dir / "metrics.json",
        "policy": spvc_dir / "latent_ppo_spvc.pt",
        "barrier": spvc_dir / "sbc.pt",
        "representation": representation,
        "base_controller": base_controller,
    }
    manifest = spvc_dir / "minimal_run_manifest.json"
    if manifest.is_file():
        paths["minimal_manifest"] = manifest
    report = json.loads(recheck_path.read_text(encoding="utf-8"))
    if (report.get("experiment") != "minimal_R2_saved_pair_original_SPVC_recheck"
            or report.get("status") != "saved_pair_passes_legacy_checks"
            or report.get("checkpoint_matches_verification") is not True
            or report.get("checkpoint_files_unchanged") is not True
            or report.get("training_run") is not False
            or report.get("test_used") is not False):
        raise ValueError("expected a passing, read-only final saved-pair recheck")
    hashes = {name: digest(path) for name, path in paths.items()}
    # Exact keys also detect removal of a manifest that the recheck relied on.
    if report.get("source_hashes") != hashes:
        raise ValueError("final recheck belongs to different checkpoint/metadata inputs")
    return {
        "stage": "minimal_final_saved_pair",
        "final_recheck_sha256": digest(recheck_path),
        "final_recheck_status": report["status"],
        "final_recheck_source_hashes": hashes,
        "checkpoint_matches_final_recheck": True,
        "note": "identity/legacy-check evidence only; not an image-path safety certificate",
    }
