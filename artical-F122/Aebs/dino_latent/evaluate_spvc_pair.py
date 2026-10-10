"""Step23: exact-image old/new PPO paired one-step diagnostic. No training/test.

No nearest lookup, no online CARLA, no formal safety or trajectory-success claim.
"""

import argparse
from collections import Counter
import heapq
import json
from pathlib import Path
import time

import numpy as np
import torch

from Aebs.dino_latent.diagnose_expanded_failures import load_fixed_inputs
from Aebs.dino_latent.diagnose_matched_images import paired_observations, one_step
from Aebs.dino_latent.prepare_expanded_data import sha256
from Aebs.dino_latent.spvc_pair_summary import PairSummary
from Aebs.dino_latent.spvc_saved_actor import load_spvc_pair, check_adapter_equivalence
from Aebs.system.env import AebsEnv
from Aebs.system.outcomes import classify_terminal_outcome


def save_json(path, value):
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path("results/dino_expanded_v1")
    parser.add_argument("--cache-dir", type=Path, default=root / "02_dino_h5")
    parser.add_argument("--representation", type=Path, default=root / "08_R2_group_alignment/dino_safety_latent.pt")
    parser.add_argument("--ppo-dir", type=Path, default=root / "09_R2_ppo")
    parser.add_argument("--spvc-dir", type=Path, default=root / "22_R2_original_spvc_scope_compare")
    parser.add_argument("--output-dir", type=Path, default=root / "23_R2_spvc_matched_pair")
    parser.add_argument("--speed-count", type=int, default=31)
    parser.add_argument("--image-batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.output_dir.exists() or args.speed_count < 2 or args.image_batch_size < 1:
        parser.error("choose a NEW output directory, speed-count>=2, image-batch-size>=1")
    for name in ("metrics.json", "scope_comparison.json", "sbc.pt", "latent_ppo_spvc.pt"):
        if not (args.spvc_dir / name).is_file():
            parser.error("missing step22 artifact: %s" % (args.spvc_dir / name))
    torch.set_num_threads(1)
    started = time.monotonic()
    fixed = load_fixed_inputs(args, parser)
    old, new, old_adapter, identities = load_spvc_pair(args.representation, fixed["controller_path"], args.spvc_dir)
    speeds = np.linspace(0, 3, args.speed_count, dtype=np.float32)
    q_model, scale = new.policy.q_model, new.policy.distance_scale_m
    env = AebsEnv(scale)
    # Broad probe of both input branches, before evaluating or writing results.
    probe_ids = np.unique(np.linspace(0, len(fixed["rows"])-1, min(128, len(fixed["rows"])), dtype=int))
    _, d, v, oi, oq = paired_observations(
        fixed["latents"], fixed["distances"], probe_ids, speeds, q_model, scale
    )
    checks = check_adapter_equivalence(old, new, old_adapter, oi, oq, d, v)
    print("[Step23 loader checks]", checks, flush=True)
    args.output_dir.mkdir(parents=True)
    report = {
        "experiment": "R2_step09_vs_step22_exact_image_paired_one_step",
        "status": "running", "level": "development diagnostic, NOT a safety certificate",
        "training_run": False, "test_used": False, "online_carla": False,
        "nearest_image_lookup": False, "lookup_distance_error_m": 0.0,
        "representation_is_newly_heldout": False,
        "model_identities": identities, "loader_checks": checks,
        "representation_sha256": sha256(args.representation),
        "cache_manifest_sha256": sha256(fixed["cache_path"]),
        "feature_sha256": sha256(fixed["features_path"]),
        "speed_grid_mps": speeds.tolist(), "results": {},
        "outcome_codes": {"ongoing": 0, "success": 1, "unsafe": 2,
                          "stopped_safe_outside_goal": 3, "out_of_domain": 4, "timeout": 5},
        "array_columns": ["old_image", "old_q", "new_image", "new_q"],
        "next_state_units": ["distance_m", "speed_mps"],
    }
    save_json(args.output_dir / "metrics.json", report)
    for split in ("train", "validation"):
        ids = [i for i, row in enumerate(fixed["rows"]) if row["split"] == split]
        if not ids:
            raise ValueError("empty required split: " + split)
        summary, groups, excluded = PairSummary(), {}, Counter()
        capacity = len(ids) * len(speeds)
        records = {
            "image_row": np.empty(capacity, dtype=np.int64),
            "distance_m": np.empty(capacity, dtype=np.float32),
            "speed_mps": np.empty(capacity, dtype=np.float32),
            "actions": np.empty((capacity, 4), dtype=np.float32),
            "outcome_codes": np.empty((capacity, 4), dtype=np.uint8),
            "next_state": np.empty((capacity, 4, 2), dtype=np.float64),
        }
        worst, serial = [], 0
        for start in range(0, len(ids), args.image_batch_size):
            image_ids, d, v, oi, oq = paired_observations(
                fixed["latents"], fixed["distances"], ids[start:start+args.image_batch_size],
                speeds, q_model, scale,
            )
            actions = np.column_stack([
                old.predict(oi, deterministic=True)[0].reshape(-1),
                old.predict(oq, deterministic=True)[0].reshape(-1),
                new.predict(oi)[0].reshape(-1), new.predict(oq)[0].reshape(-1),
            ])
            if not np.isfinite(actions).all():
                raise ValueError("nonfinite paired actions")
            for i, source_id in enumerate(image_ids):
                terminal = classify_terminal_outcome(float(d[i]), float(v[i]))
                if terminal is not None:
                    excluded[terminal] += 1
                    continue
                next_states = [one_step(env, float(d[i]), float(v[i]), float(a)) for a in actions[i]]
                outcomes = [s["outcome"] for s in next_states]
                summary.add(actions[i], outcomes)
                row = fixed["rows"][int(source_id)]
                appearance = row["weather"] + "|" + row["color"]
                groups.setdefault(appearance, PairSummary()).add(actions[i], outcomes)
                position = summary.n - 1
                records["image_row"][position] = int(source_id)
                records["distance_m"][position] = d[i]
                records["speed_mps"][position] = v[i]
                records["actions"][position] = actions[i]
                records["outcome_codes"][position] = [report["outcome_codes"][o] for o in outcomes]
                records["next_state"][position] = [[s["distance_m"], s["speed_mps"]] for s in next_states]
                serial += 1
                regression = int(outcomes[2] == "unsafe" and outcomes[0] != "unsafe")
                drift = abs(float(actions[i, 2]-actions[i, 0]))
                detail = {"cache_row": int(source_id), "filename": row["filename"],
                          "appearance": appearance, "distance_m": float(d[i]), "speed_mps": float(v[i]),
                          "actions": actions[i].tolist(), "outcomes": outcomes,
                          "next_states": next_states, "new_image_unsafe_old_not": bool(regression)}
                heapq.heappush(worst, ((regression, drift), serial, detail))
                if len(worst) > 20:
                    heapq.heappop(worst)
            if start % (4 * args.image_batch_size) == 0 or start+args.image_batch_size >= len(ids):
                print("%s matched images=%d/%d pairs=%d" % (
                    split, min(start+args.image_batch_size, len(ids)), len(ids), summary.n
                ), flush=True)
        if not summary.n:
            raise ValueError("no nonterminal pairs in " + split)
        result = summary.result()
        result.update(images=len(ids), excluded_current_terminal=dict(excluded),
                      by_appearance={key: value.result() for key, value in sorted(groups.items())})
        report["results"][split] = result
        np.savez_compressed(args.output_dir / (split+"_pairs.npz"), **{
            key: value[:summary.n] for key, value in records.items()
        })
        save_json(args.output_dir / (split+"_worst_examples.json"), [item[2] for item in sorted(worst, reverse=True)])
        save_json(args.output_dir / "metrics.json", report)
    regressions = sum(s["counts"]["image_new_unsafe_old_not"] + s["counts"]["q_new_unsafe_old_not"]
                      for s in report["results"].values())
    report["status"] = "matched_unsafe_regressions_found" if regressions else "no_sampled_unsafe_regression_not_a_certificate"
    report["runtime_seconds"] = time.monotonic()-started
    save_json(args.output_dir / "metrics.json", report)
    print("\n[R2 old vs SPVC-updated PPO: exact-image matched diagnostic]")
    print("no training/test/nearest lookup/online CARLA; loader equivalence passed")
    print("changed actor tensors:", identities["changed_actor_tensors"])
    for split, result in report["results"].items():
        a, c = result["action"], result["counts"]
        print("%s: pairs=%d dual_path_MAE=%.6f -> %.6f image_policy_drift_MAE=%.6f max=%.6f" % (
            split, result["checked_nonterminal_pairs"], a["old_dual_path"]["mae"], a["new_dual_path"]["mae"],
            a["image_policy_drift"]["mae"], a["image_policy_drift"]["max_abs"]
        ))
        print("  image_only_unsafe=%d -> %d both_unsafe=%d -> %d" % (
            c["old_image_only_next_unsafe"], c["new_image_only_next_unsafe"],
            c["old_both_next_unsafe"], c["new_both_next_unsafe"]
        ))
        print("  image new_unsafe_old_not=%d old_unsafe_new_not=%d; q new_unsafe_old_not=%d old_unsafe_new_not=%d" % (
            c["image_new_unsafe_old_not"], c["image_old_unsafe_new_not"],
            c["q_new_unsafe_old_not"], c["q_old_unsafe_new_not"]
        ))
    print("status:", report["status"])
    print("metrics:", args.output_dir / "metrics.json")


if __name__ == "__main__":
    main()
