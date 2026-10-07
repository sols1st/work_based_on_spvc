"""Reproduce and record fixed-R1/fixed-PPO failures without training or test data.

The diagnostic has two parts:

* replay the original train-library evaluation grid and retain complete traces
  for every unsafe episode, including the exact nearest image and a same-state
  surrogate-policy comparison at every step;
* rescan exact-distance train/validation image pairs and retain every
  image-only next-step unsafe case, rather than only the largest action errors.

This is development evidence, not a safety certificate or an online CARLA
camera rollout.  It deliberately rejects caches containing test rows.
"""

import argparse
import hashlib
import heapq
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import torch
from stable_baselines3 import PPO

from Aebs.dino_latent.diagnose_matched_images import paired_observations, one_step
from Aebs.dino_latent.models import SafetyProjection
from Aebs.dino_latent.prepare_expanded_data import sha256
from Aebs.dino_latent.run_expanded_ppo import AppearanceEnv
from Aebs.system.outcomes import TIMEOUT, classify_terminal_outcome


def _json_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def nearest_image_record(env, rows, appearance, distance_m):
    """Return the immutable cache row selected by AppearanceEnv._latent."""
    pool = env.pools[appearance]
    local_index = int(pool[np.argmin(np.abs(env.image_distances_m[pool] - distance_m))])
    source_index = int(env.source_indices[local_index])
    row = rows[source_index]
    return {
        "cache_row": source_index,
        "filename": row["filename"],
        "weather": row["weather"],
        "color": row["color"],
        "label_distance_m": float(env.image_distances_m[local_index]),
        "lookup_error_m": float(abs(env.image_distances_m[local_index] - distance_m)),
    }


def surrogate_observation(env, distance_m, speed_mps):
    value = torch.tensor([[distance_m / env.distance_scale_m]], dtype=torch.float32)
    with torch.no_grad():
        latent = env.q_model(value).numpy()[0]
    return np.concatenate([latent, np.asarray([speed_mps], dtype=np.float32)]).astype(np.float32)


def trace_episode(
    model, env, rows, appearance, distance_m, speed_mps,
    action_divergence_threshold=0.1,
):
    """Replay one image-path episode and record a same-state q-path action."""
    observation = env.reset_to(distance_m, speed_mps)
    steps = []
    episode_return = 0.0
    outcome = None
    for step_number in range(1, env.base.max_episode_steps + 1):
        current_distance = float(env.base.state[0] * env.distance_scale_m)
        current_speed = float(env.base.state[1])
        selected = nearest_image_record(env, rows, appearance, current_distance)
        image_action = float(np.asarray(model.predict(observation, deterministic=True)[0]).reshape(-1)[0])
        q_observation = surrogate_observation(env, current_distance, current_speed)
        q_action = float(np.asarray(model.predict(q_observation, deterministic=True)[0]).reshape(-1)[0])
        saved_state = env.base.state.copy()
        saved_elapsed_steps = env.base.elapsed_steps
        q_next = one_step(env.base, current_distance, current_speed, q_action)
        env.base.state = saved_state
        env.base.elapsed_steps = saved_elapsed_steps
        observation, reward, terminated, truncated, info = env.step(
            np.asarray([image_action], dtype=np.float32)
        )
        next_distance = float(env.base.state[0] * env.distance_scale_m)
        next_speed = float(env.base.state[1])
        episode_return += float(reward)
        step_outcome = info.get("outcome", "ongoing")
        steps.append({
            "step": step_number,
            "state": {"distance_m": current_distance, "speed_mps": current_speed},
            "nearest_image": selected,
            "image_action": image_action,
            "surrogate_action": q_action,
            "action_difference": image_action - q_action,
            "next_state": {"distance_m": next_distance, "speed_mps": next_speed},
            "surrogate_next": q_next,
            "reward": float(reward),
            "outcome": step_outcome,
        })
        if terminated or truncated:
            outcome = info.get("outcome", TIMEOUT)
            break
    return {
        "appearance": appearance,
        "initial_distance_m": float(distance_m),
        "initial_speed_mps": float(speed_mps),
        "outcome": outcome or TIMEOUT,
        "step_count": len(steps),
        "return": episode_return,
        "first_material_action_divergence_step": next(
            (
                row["step"] for row in steps
                if abs(row["action_difference"]) >= action_divergence_threshold
            ),
            None,
        ),
        "max_abs_action_difference": max(
            (abs(row["action_difference"]) for row in steps), default=0.0
        ),
        "max_lookup_error_m": max(
            (row["nearest_image"]["lookup_error_m"] for row in steps), default=0.0
        ),
        "steps": steps,
    }


def scan_rollout_failures(
    model, make_env, rows, appearances, grid_size, action_divergence_threshold,
):
    distances = np.linspace(15.0, 16.0, grid_size, dtype=np.float32)
    speeds = np.linspace(2.5, 3.0, grid_size, dtype=np.float32)
    failures = []
    summaries = {}
    for appearance in appearances:
        env = make_env("image_nearest", appearance)
        counts = Counter()
        for distance_m in distances:
            for speed_mps in speeds:
                trace = trace_episode(
                    model, env, rows, appearance, float(distance_m), float(speed_mps),
                    action_divergence_threshold,
                )
                counts[trace["outcome"]] += 1
                if trace["outcome"] == "unsafe":
                    failures.append(trace)
        summaries[appearance] = {
            "episodes": grid_size * grid_size,
            "counts": dict(counts),
            "unsafe_count": counts["unsafe"],
        }
        print(
            "rollout %s: episodes=%d unsafe=%d" %
            (appearance, grid_size * grid_size, counts["unsafe"]), flush=True
        )
    return summaries, failures


def scan_matched_failures(model, env, latents, distances, ids, rows, speed_count):
    speeds = np.linspace(0, 3, speed_count, dtype=np.float32)
    counts = Counter()
    excluded = Counter()
    total = 0
    absolute = 0.0
    maximum = 0.0
    less_braking = 0
    image_only = []
    disagreements = []
    worst = []
    serial = 0
    for start in range(0, len(ids), 64):
        image_ids, d, v, image_obs, q_obs = paired_observations(
            latents, distances, ids[start:start + 64], speeds,
            env.q_model, env.distance_scale_m,
        )
        image_actions = model.predict(image_obs, deterministic=True)[0].reshape(-1)
        q_actions = model.predict(q_obs, deterministic=True)[0].reshape(-1)
        if not np.isfinite(image_actions).all() or not np.isfinite(q_actions).all():
            raise ValueError("Nonfinite policy actions")
        for j in range(len(d)):
            terminal = classify_terminal_outcome(float(d[j]), float(v[j]))
            if terminal:
                excluded[terminal] += 1
                continue
            image_action = float(np.clip(image_actions[j], -3, 3))
            q_action = float(np.clip(q_actions[j], -3, 3))
            image_next = one_step(env.base, float(d[j]), float(v[j]), image_action)
            q_next = one_step(env.base, float(d[j]), float(v[j]), q_action)
            image_unsafe = image_next["outcome"] == "unsafe"
            q_unsafe = q_next["outcome"] == "unsafe"
            counts["image_only_next_unsafe"] += int(image_unsafe and not q_unsafe)
            counts["surrogate_only_next_unsafe"] += int(q_unsafe and not image_unsafe)
            counts["both_next_unsafe"] += int(image_unsafe and q_unsafe)
            counts["outcome_disagreement"] += int(image_next["outcome"] != q_next["outcome"])
            difference = image_action - q_action
            total += 1
            absolute += abs(difference)
            maximum = max(maximum, abs(difference))
            less_braking += int(image_action < q_action - 1e-6)
            source_index = int(image_ids[j])
            source = rows[source_index]
            detail = {
                "cache_row": source_index,
                "filename": source["filename"],
                "weather": source["weather"],
                "color": source["color"],
                "distance_m": float(d[j]),
                "speed_mps": float(v[j]),
                "image_action": image_action,
                "surrogate_action": q_action,
                "action_difference": difference,
                "image_next": image_next,
                "surrogate_next": q_next,
            }
            if image_unsafe and not q_unsafe:
                image_only.append(detail)
            if image_next["outcome"] != q_next["outcome"]:
                disagreements.append(detail)
            serial += 1
            heapq.heappush(worst, (abs(difference), serial, detail))
            if len(worst) > 10:
                heapq.heappop(worst)
        if start % 1024 == 0:
            print("matched images %d/%d" % (min(start + 64, len(ids)), len(ids)), flush=True)
    summary = {
        "images": len(ids),
        "checked_nonterminal_pairs": total,
        "excluded": dict(excluded),
        "action_mae": absolute / total if total else None,
        "action_max_abs": maximum if total else None,
        "image_less_braking_fraction": less_braking / total if total else None,
        "lookup_distance_error_m": 0.0,
        **counts,
        "image_only_failure_count": len(image_only),
        "outcome_disagreement_count": len(disagreements),
        "worst_samples": [item[2] for item in sorted(worst, reverse=True)],
    }
    return summary, image_only, disagreements


def load_fixed_inputs(args, parser):
    cache_path = args.cache_dir / "cache_manifest.json"
    features_path = args.cache_dir / "features.h5"
    controller_path = args.ppo_dir / "latent_ppo.zip"
    data_path = args.ppo_dir / "physical_labels.h5"
    latent_path = args.ppo_dir / "latent_data.npz"
    config_path = args.ppo_dir / "config.json"
    metrics_path = args.ppo_dir / "metrics.json"
    required = [
        cache_path, features_path, args.representation, controller_path,
        data_path, latent_path, config_path, metrics_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error("Missing required inputs: " + ", ".join(missing))
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    representation = torch.load(args.representation, map_location="cpu")
    if (
        cache.get("schema") != "spvc_expanded_dino_h5_cache_v1"
        or cache.get("test_encoded") != 0
        or representation["provenance"]["cache_manifest_sha256"] != sha256(cache_path)
    ):
        parser.error("Representation/cache provenance mismatch or test data present")
    rows = cache["records"]
    if any(row["split"] not in ("train", "validation") for row in rows):
        parser.error("Test rows forbidden")
    if sha256(features_path) != cache["feature_sha256"]:
        parser.error("H5 feature hash changed")
    ppo_config = json.loads(config_path.read_text(encoding="utf-8"))
    source_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if (
        ppo_config.get("representation_sha256") != sha256(args.representation)
        or ppo_config.get("cache_manifest_sha256") != sha256(cache_path)
        or ppo_config.get("test_used") is not False
        or source_metrics.get("checkpoint_sha256") != sha256(controller_path)
        or source_metrics.get("test_used") is not False
    ):
        parser.error("PPO provenance mismatch, incomplete metrics, or test usage not explicitly false")
    with h5py.File(features_path, "r") as stream:
        if not stream.attrs["complete"]:
            parser.error("Incomplete H5")
        features = torch.from_numpy(stream["features"][:])
        distances = stream["distance_m"][:].astype(np.float32)
        if stream["filename"].asstr()[:].tolist() != [row["filename"] for row in rows]:
            parser.error("H5/cache row mismatch")
    projection = SafetyProjection().eval()
    projection.load_state_dict(representation["projection_state_dict"])
    with torch.no_grad():
        latents = projection(
            (features - representation["feature_mean"]) / representation["feature_std"]
        ).numpy()
    if latents.shape != (len(rows), 32) or not np.isfinite(latents).all():
        parser.error("Invalid projected latents")
    with h5py.File(data_path, "r") as stream:
        adapter_distances = np.asarray(stream["y_train"], dtype=np.float32)
    with np.load(latent_path) as stream:
        adapter_latents = np.asarray(stream["latent"], dtype=np.float32)
    if not np.array_equal(adapter_distances, distances) or not np.allclose(adapter_latents, latents):
        parser.error("PPO adapter data no longer matches the fixed cache/representation")
    return {
        "cache_path": cache_path,
        "features_path": features_path,
        "controller_path": controller_path,
        "data_path": data_path,
        "latent_path": latent_path,
        "metrics_path": metrics_path,
        "source_metrics": source_metrics,
        "rows": rows,
        "distances": distances,
        "latents": latents,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("results/dino_expanded_v1/02_dino_h5"))
    parser.add_argument(
        "--representation", type=Path,
        default=Path("results/dino_expanded_v1/03_alignment_r0_r1/R1_all_appearances/dino_safety_latent.pt"),
    )
    parser.add_argument("--ppo-dir", type=Path, default=Path("results/dino_expanded_v1/04_R1_ppo"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/dino_expanded_v1/05_failure_diagnostic"))
    parser.add_argument("--eval-grid-size", type=int, default=20)
    parser.add_argument("--speed-count", type=int, default=31)
    parser.add_argument("--action-divergence-threshold", type=float, default=0.1)
    parser.add_argument(
        "--appearance", action="append", default=None,
        help="weather|color key to replay; repeat for multiple. Default: all train appearances.",
    )
    args = parser.parse_args()
    if (
        args.output_dir.exists() or args.eval_grid_size < 2 or args.speed_count < 2
        or args.action_divergence_threshold <= 0
    ):
        parser.error("Choose a new output directory and sizes of at least 2")
    torch.set_num_threads(1)
    fixed = load_fixed_inputs(args, parser)
    model = PPO.load(fixed["controller_path"], device="cpu")

    def make_env(mode, appearance=None):
        return AppearanceEnv(
            args.representation, fixed["data_path"], fixed["latent_path"],
            fixed["rows"], mode, appearance,
        )

    available = sorted(make_env("image_nearest").pools)
    appearances = available if args.appearance is None else args.appearance
    unknown = sorted(set(appearances) - set(available))
    if unknown:
        parser.error("Unknown appearances %s; available=%s" % (unknown, available))

    args.output_dir.mkdir(parents=True)
    provenance = {
        "scope": "fixed-model development failure diagnostic; no training, SBC, QP, CP, or test",
        "test_used": False,
        "cache_manifest_sha256": sha256(fixed["cache_path"]),
        "features_sha256": sha256(fixed["features_path"]),
        "representation_sha256": sha256(args.representation),
        "controller_sha256": sha256(fixed["controller_path"]),
        "eval_grid_size": args.eval_grid_size,
        "speed_count": args.speed_count,
        "action_divergence_threshold": args.action_divergence_threshold,
        "appearances": appearances,
        "source_metrics_sha256": sha256(fixed["metrics_path"]),
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(provenance, indent=2, allow_nan=False), encoding="utf-8"
    )

    rollout_summary, rollout_failures = scan_rollout_failures(
        model, make_env, fixed["rows"], appearances, args.eval_grid_size,
        args.action_divergence_threshold,
    )
    (args.output_dir / "unsafe_rollout_traces.json").write_text(
        json.dumps(rollout_failures, indent=2, allow_nan=False), encoding="utf-8"
    )

    matched_summary = {}
    matched_image_only = {}
    matched_disagreements = {}
    for split in ("train", "validation"):
        ids = [i for i, row in enumerate(fixed["rows"]) if row["split"] == split]
        summary, image_only, disagreements = scan_matched_failures(
            model, make_env("surrogate"), fixed["latents"], fixed["distances"],
            ids, fixed["rows"], args.speed_count,
        )
        matched_summary[split] = summary
        matched_image_only[split] = image_only
        matched_disagreements[split] = disagreements
        print(
            "matched %s: image_only_unsafe=%d disagreements=%d" %
            (split, len(image_only), len(disagreements)), flush=True
        )
    (args.output_dir / "matched_image_only_unsafe.json").write_text(
        json.dumps(matched_image_only, indent=2, allow_nan=False), encoding="utf-8"
    )
    (args.output_dir / "matched_outcome_disagreements.json").write_text(
        json.dumps(matched_disagreements, indent=2, allow_nan=False), encoding="utf-8"
    )
    reproduction = {"rollouts": {}, "matched": {}}
    for appearance, actual in rollout_summary.items():
        expected = fixed["source_metrics"].get("evaluation", {}).get(appearance, {})
        expected_unsafe = (
            expected.get("counts", {}).get("unsafe", 0) if expected else None
        )
        reproduction["rollouts"][appearance] = {
            "expected_unsafe_count": expected_unsafe,
            "actual_unsafe_count": actual["unsafe_count"],
            "matches": expected_unsafe == actual["unsafe_count"],
        }
    for split, actual in matched_summary.items():
        expected = fixed["source_metrics"].get("matched", {}).get(split, {})
        reproduction["matched"][split] = {
            key: {"expected": expected.get(key), "actual": actual[key],
                  "matches": expected.get(key) == actual[key]}
            for key in (
                "image_only_next_unsafe", "surrogate_only_next_unsafe",
                "both_next_unsafe", "outcome_disagreement",
            )
        }
    reproduction["all_checked_counts_match"] = all(
        item["matches"]
        for item in reproduction["rollouts"].values()
    ) and all(
        item["matches"]
        for split in reproduction["matched"].values()
        for item in split.values()
    )
    metrics = {
        **provenance,
        "rollouts": rollout_summary,
        "unsafe_rollout_count": len(rollout_failures),
        "matched": matched_summary,
        "source_result_reproduction": reproduction,
        "evidence_digest": _json_hash({
            "rollout_failures": rollout_failures,
            "matched_image_only": matched_image_only,
            "matched_disagreements": matched_disagreements,
        }),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8"
    )
    print("[Expanded fixed-model failure diagnostic — no training/SBC/test]")
    print("unsafe_rollout_count=%d" % len(rollout_failures))
    print("source_result_counts_match=%s" % reproduction["all_checked_counts_match"])
    for split, summary in matched_summary.items():
        print(
            "%s image_only_next_unsafe=%d outcome_disagreement=%d action_mae=%.6f" %
            (split, summary["image_only_next_unsafe"], summary["outcome_disagreement"], summary["action_mae"])
        )
    print("metrics:", args.output_dir / "metrics.json")
    print("unsafe rollout traces:", args.output_dir / "unsafe_rollout_traces.json")
    print("matched image-only unsafe:", args.output_dir / "matched_image_only_unsafe.json")


if __name__ == "__main__":
    main()
