"""Export the frozen native R2 policy for the ORIGINAL fullscreen collector.

No learning, decoder, new camera, or new control law. Fold native latent
standardization into the projection's last affine layer and q's distance
standardization into its first affine layer. Verify both saved actor paths
against the original native checkpoints before declaring the export ready.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil

import h5py
import numpy as np
import torch
from stable_baselines3 import PPO

from Aebs.dino_latent.models import PhysicalToLatent, SafetyProjection
from Aebs.dino_latent.prepare_expanded_data import sha256


EXPORT_KIND = "frozen_native_R2_affine_collection_export_v1"


def fold_native_coordinates(representation, native):
    """Return a new checkpoint; neither source is mutated."""
    if native["result"]["family"] != "r2" or native["result"]["dimension"] != 32:
        raise ValueError("This compatibility export is only for native R2-32")
    mean = torch.as_tensor(native["latent_mean"]).reshape(-1)
    std = torch.as_tensor(native["latent_std"]).reshape(-1)
    dm, ds = float(native["distance_mean"]), float(native["distance_std"])
    scale = float(representation["distance_scale_m"])
    if (mean.shape != (32,) or std.shape != (32,)
            or not torch.isfinite(mean).all() or not torch.isfinite(std).all()
            or not bool((std > 0).all()) or not np.isfinite([dm, ds, scale]).all()
            or min(ds, scale) <= 0):
        raise ValueError("Invalid native coordinates")
    projection = copy.deepcopy(representation["projection_state_dict"])
    projection["network.2.weight"] = projection["network.2.weight"] / std[:, None]
    projection["network.2.bias"] = (projection["network.2.bias"] - mean) / std
    q = copy.deepcopy(native["q_state_dict"])
    first_weight = q["network.0.weight"].clone()
    q["network.0.weight"] = first_weight * (scale / ds)
    q["network.0.bias"] = q["network.0.bias"] - first_weight[:, 0] * (dm / ds)
    # Do not carry the old distance decoder: it uses different latent coordinates.
    return dict(projection_state_dict=projection, physical_to_latent_state_dict=q,
                feature_mean=representation["feature_mean"].clone(),
                feature_std=representation["feature_std"].clone(),
                distance_scale_m=scale, latent_dimension=32, dino_frozen=True,
                variant="R2_group_alignment", deployment_export=EXPORT_KIND,
                auxiliary_decoder_included=False,
                provenance={"source_provenance": copy.deepcopy(representation.get("provenance", {})),
                            "coordinate_change": "native standardization folded exactly into affine layers",
                            "training_performed": False})


def require_export_trajectory_identity(attrs, representation, representation_path, controller_path):
    """An old policy's trajectories must not become a new export's calibration."""
    if representation.get("deployment_export") != EXPORT_KIND:
        return
    provenance = json.loads(attrs.get("provenance_json", "{}"))
    if (provenance.get("representation_sha256") != sha256(representation_path)
            or provenance.get("controller_sha256") != sha256(controller_path)
            or provenance.get("model_updates") is not False):
        raise ValueError("Trajectories were not collected by this frozen native export")


@torch.no_grad()
def equivalence(representation, native, exported, ppo, features):
    old_projection, new_projection = SafetyProjection().eval(), SafetyProjection().eval()
    old_projection.load_state_dict(representation["projection_state_dict"])
    new_projection.load_state_dict(exported["projection_state_dict"])
    old_q, new_q = PhysicalToLatent().eval(), PhysicalToLatent().eval()
    old_q.load_state_dict(native["q_state_dict"])
    new_q.load_state_dict(exported["physical_to_latent_state_dict"])
    maxima = dict(image_latent_max_abs=0., image_action_max_abs=0.,
                  proxy_latent_max_abs=0., proxy_action_max_abs=0.)
    rng = np.random.default_rng(731)
    for start in range(0, len(features), 1024):
        x = torch.as_tensor(np.asarray(features[start:start+1024]), dtype=torch.float32)
        x = (x - representation["feature_mean"]) / representation["feature_std"]
        z = (old_projection(x) - native["latent_mean"]) / native["latent_std"]
        ze = new_projection(x)
        d = torch.from_numpy(rng.uniform(5, 16, size=(len(x), 1)).astype(np.float32))
        q = old_q((d - float(native["distance_mean"])) / float(native["distance_std"]))
        qe = new_q(d / float(exported["distance_scale_m"]))
        v = rng.uniform(0, 3, size=(len(x), 1)).astype(np.float32)
        for prefix, a, b in (("image", z, ze), ("proxy", q, qe)):
            maxima[prefix + "_latent_max_abs"] = max(maxima[prefix + "_latent_max_abs"], float((a-b).abs().max()))
            aa = ppo.predict(np.concatenate((a.numpy(), v), axis=1), deterministic=True)[0]
            ab = ppo.predict(np.concatenate((b.numpy(), v), axis=1), deterministic=True)[0]
            maxima[prefix + "_action_max_abs"] = max(maxima[prefix + "_action_max_abs"], float(abs(aa-ab).max()))
    if not len(features) or not np.isfinite(list(maxima.values())).all():
        raise ValueError("Empty/nonfinite export equivalence check")
    if max(maxima["image_action_max_abs"], maxima["proxy_action_max_abs"]) > 1e-4:
        raise ValueError("Export action equivalence failed: %s" % maxima)
    return dict(**maxima, checked_feature_rows=len(features), action_tolerance=1e-4,
                equivalence_pass=True, note="float32 numerical equivalence, not a safety proof")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--representation", type=Path, default=Path("results/dino_expanded_v1/08_R2_group_alignment/dino_safety_latent.pt"))
    p.add_argument("--native-checkpoint", type=Path, default=Path("results/native_representation_dimensions_20261019_v1/historical_R2.pt"))
    p.add_argument("--controller-dir", type=Path, default=Path("results/native_controllers_speed_residual_20261019_v1/historical_R2"))
    p.add_argument("--cache-dir", type=Path, default=Path("results/dino_expanded_v1/02_dino_h5"))
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    if a.output_dir.exists(): p.error("A new output directory is required")
    torch.set_num_threads(1)
    source = a.controller_dir / "native_ppo.zip"
    paths = [a.representation, a.native_checkpoint, source, a.controller_dir / "metrics.json",
             a.cache_dir / "cache_manifest.json", a.cache_dir / "features.h5"]
    hashes = {str(path): sha256(path) for path in paths}
    rep = torch.load(a.representation, map_location="cpu")
    native = torch.load(a.native_checkpoint, map_location="cpu")
    metrics = json.loads((a.controller_dir / "metrics.json").read_text())
    native_sources = native["config"]["source_sha256"]
    if (metrics["name"] != "historical_R2" or metrics["controller_sha256"] != hashes[str(source)]
            or metrics["representation_sha256"] != hashes[str(a.native_checkpoint)]
            or hashes[str(a.representation)] not in native_sources.values()
            or rep.get("dino_frozen") is not True):
        raise ValueError("Frozen native source identity mismatch")
    manifest = json.loads((a.cache_dir / "cache_manifest.json").read_text())
    if manifest.get("test_encoded") != 0 or manifest["feature_sha256"] != hashes[str(paths[-1])]:
        raise ValueError("Training-cache identity/test isolation mismatch")
    exported = fold_native_coordinates(rep, native)
    exported["export_source_sha256"] = hashes
    ppo = PPO.load(source, device="cpu")
    with h5py.File(paths[-1], "r") as stream:
        features = np.asarray(stream["features"], dtype=np.float32)
    check = equivalence(rep, native, exported, ppo, features)
    a.output_dir.mkdir(parents=True)
    rep_path = a.output_dir / "dino_safety_latent.pt"
    torch.save(exported, rep_path)
    shutil.copy2(source, a.output_dir / "latent_ppo.zip")
    # Verify the SERIALIZED checkpoint rather than just the pre-save dictionary.
    check = equivalence(rep, native, torch.load(rep_path, map_location="cpu"),
                        PPO.load(a.output_dir / "latent_ppo.zip", device="cpu"), features)
    result = dict(experiment=EXPORT_KIND, status="frozen_collection_bundle_ready", **check,
                  representation_sha256=sha256(rep_path), checkpoint_sha256=sha256(a.output_dir / "latent_ppo.zip"),
                  source_sha256=hashes, PPO_bytes_unchanged=sha256(a.output_dir / "latent_ppo.zip") == hashes[str(source)],
                  image_capture_changed=False, training_performed=False, test_used=False,
                  CP_calibrated=False, image_safety_certificate=False,
                  intended_collector="Aebs.connect.collect_r2_spvc_trajectories")
    for path, digest in hashes.items():
        if sha256(Path(path)) != digest: raise ValueError("Frozen export source changed")
    (a.output_dir / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print("[Frozen native R2 collection export]", result["status"], check, flush=True)
    print("PPO bytes unchanged:", result["PPO_bytes_unchanged"], "results:", a.output_dir, flush=True)


if __name__ == "__main__": main()
