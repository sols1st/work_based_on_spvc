"""Fixed-PPO one-step diagnostic at each image's own labeled distance.

Uses cached frozen-DINO projection outputs; never searches for a nearest image.
The surrogate is a comparison input, not a safe-action oracle.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from stable_baselines3 import PPO

from Aebs.dino_latent.holdout_experiment import sha256
from Aebs.dino_latent.models import PhysicalToLatent
from Aebs.system.env import AebsEnv
from Aebs.system.outcomes import classify_terminal_outcome


def paired_observations(latents, distances, indices, speeds, q_model, scale):
    image_ids = np.repeat(np.asarray(indices, dtype=int), len(speeds))
    velocities = np.tile(speeds, len(indices)).astype(np.float32)
    physical_distance = distances[image_ids].astype(np.float32)
    image_obs = np.column_stack([latents[image_ids], velocities]).astype(np.float32)
    with torch.no_grad():
        q = q_model(torch.from_numpy(physical_distance[:, None] / scale)).numpy()
    surrogate_obs = np.column_stack([q, velocities]).astype(np.float32)
    return image_ids, physical_distance, velocities, image_obs, surrogate_obs


def one_step(env, distance, speed, action):
    # Use exactly the existing environment's rounding, clipping and outcomes.
    env.state = np.asarray([distance / env.std1, speed], dtype=np.float32)
    env.elapsed_steps = 0
    state, reward, _, _, info = env.step(np.asarray([action], dtype=np.float32))
    return {"distance_m": float(state[0] * env.std1), "speed_mps": float(state[1]),
            "outcome": info.get("outcome", "ongoing"), "reward": float(reward)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", type=Path, default=Path("results/dino_improvement/02_train_only_ppo/latent_ppo.zip"))
    parser.add_argument("--data", type=Path, default=Path("Aebs/data/Downsampled.h5"))
    parser.add_argument("--representation-checkpoint", type=Path, default=Path("results/dino_safety_latent_stage1/dino_safety_latent.pt"))
    parser.add_argument("--latent-data", type=Path, default=Path("results/dino_safety_latent_stage1/latent_data.npz"))
    parser.add_argument("--manifest", type=Path, default=Path("results/dino_improvement/split.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/dino_improvement/04_matched_image_diagnostic"))
    parser.add_argument("--speed-count", type=int, default=31)
    args = parser.parse_args()
    if args.speed_count < 2:
        parser.error("speed-count must be at least 2")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output directory is nonempty; choose a new --output-dir")
    torch.set_num_threads(1)
    manifest = json.loads(args.manifest.read_text())
    hashes = {"data": sha256(args.data), "representation": sha256(args.representation_checkpoint),
              "latent_data": sha256(args.latent_data)}
    if hashes != manifest["hashes"]:
        parser.error("input hashes differ from registered split")
    with h5py.File(args.data, "r") as stream:
        distances = np.asarray(stream["y_train"], dtype=np.float32).reshape(-1)
    with np.load(args.latent_data) as stream:
        latents = np.asarray(stream["latent"], dtype=np.float32)
        if "available_indices" in stream:
            needed = manifest["indices"]["train"] + manifest["indices"]["validation"]
            if not np.isin(needed, stream["available_indices"]).all():
                parser.error("train/validation rows absent from latent cache")
    flat = [i for part in manifest["indices"].values() for i in part]
    if sorted(flat) != list(range(len(distances))):
        parser.error("manifest must partition all images without duplicates")
    if latents.shape != (len(distances), 32) or not np.isfinite(latents).all():
        parser.error("invalid cached latent shape or values")
    checkpoint = torch.load(args.representation_checkpoint, map_location="cpu")
    scale = float(checkpoint["distance_scale_m"])
    q_model = PhysicalToLatent()
    q_model.load_state_dict(checkpoint["physical_to_latent_state_dict"])
    q_model.eval()
    model = PPO.load(args.controller, device="cpu")
    env = AebsEnv(scale)
    speeds = np.linspace(0, 3, args.speed_count, dtype=np.float32)
    result = {"scope": "matched-image one-step diagnostic, not a certificate or rollout",
              "representation_is_newly_heldout": False,
              "surrogate_is_safe_oracle": False,
              "controller": str(args.controller), "controller_sha256": sha256(args.controller),
              "manifest_sha256": sha256(args.manifest), "hashes": hashes,
              "speed_grid": speeds.tolist(), "subsets": {}}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for subset in ("train", "validation"):
        ids, d, v, oi, oq = paired_observations(
            latents, distances, manifest["indices"][subset], speeds, q_model, scale)
        image_actions = model.predict(oi, deterministic=True)[0].reshape(-1)
        q_actions = model.predict(oq, deterministic=True)[0].reshape(-1)
        if not np.isfinite(image_actions).all() or not np.isfinite(q_actions).all():
            raise ValueError("nonfinite policy actions")
        rows = []
        excluded = {}
        for j in range(len(ids)):
            current = classify_terminal_outcome(float(d[j]), float(v[j]))
            if current is not None:
                excluded[current] = excluded.get(current, 0) + 1
                continue
            ai = float(np.clip(image_actions[j], -3, 3))
            aq = float(np.clip(q_actions[j], -3, 3))
            ni = one_step(env, float(d[j]), float(v[j]), ai)
            nq = one_step(env, float(d[j]), float(v[j]), aq)
            rows.append({"subset": subset, "image_id": int(ids[j]),
                         "distance_m": float(d[j]), "speed_mps": float(v[j]),
                         "image_action": ai, "surrogate_action": aq,
                         "action_difference": ai - aq,
                         "image_next": ni, "surrogate_next": nq})
        differences = np.asarray([r["action_difference"] for r in rows])
        summary = {"images": len(manifest["indices"][subset]), "candidate_pairs": len(ids),
                   "checked_nonterminal_pairs": len(rows), "excluded_already_terminal": excluded,
                   "lookup_distance_error_m": 0.0,
                   "action_mae": float(abs(differences).mean()) if len(rows) else None,
                   "action_max_abs": float(abs(differences).max()) if len(rows) else None,
                   "image_less_braking_fraction": float((differences < -1e-6).mean()) if len(rows) else None,
                   "image_only_next_unsafe": sum(r["image_next"]["outcome"] == "unsafe" and r["surrogate_next"]["outcome"] != "unsafe" for r in rows),
                   "surrogate_only_next_unsafe": sum(r["image_next"]["outcome"] != "unsafe" and r["surrogate_next"]["outcome"] == "unsafe" for r in rows),
                   "both_next_unsafe": sum(r["image_next"]["outcome"] == "unsafe" and r["surrogate_next"]["outcome"] == "unsafe" for r in rows),
                   "next_outcome_disagreement": sum(r["image_next"]["outcome"] != r["surrogate_next"]["outcome"] for r in rows),
                   "largest_action_differences": sorted(rows, key=lambda r: abs(r["action_difference"]), reverse=True)[:5]}
        result["subsets"][subset] = summary
        all_rows.extend(rows)
    (args.output_dir / "samples.json").write_text(json.dumps(all_rows, indent=2, allow_nan=False))
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print("[DINO matched-image one-step diagnostic]")
    print("representation_is_newly_heldout=False; no nearest-image lookup; no training")
    for subset, s in result["subsets"].items():
        print("%s: images=%d checked_pairs=%d excluded=%s" % (subset, s["images"], s["checked_nonterminal_pairs"], s["excluded_already_terminal"]))
        print("lookup_distance_error_m=0; action_MAE=%s max_abs=%s image_less_braking_fraction=%s" % (s["action_mae"], s["action_max_abs"], s["image_less_braking_fraction"]))
        print("image_only_next_unsafe=%d surrogate_only_next_unsafe=%d both_next_unsafe=%d outcome_disagreement=%d" % (s["image_only_next_unsafe"], s["surrogate_only_next_unsafe"], s["both_next_unsafe"], s["next_outcome_disagreement"]))
    print("No automatic pass: action agreement is not a safety guarantee.")
    print("metrics: %s" % (args.output_dir / "metrics.json"))
    print("per-pair data: %s" % (args.output_dir / "samples.json"))


if __name__ == "__main__":
    main()
