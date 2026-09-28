"""Comprehensive empirical evaluation of semantic-SPVC PPO with/without SBC-QP.

This experiment intentionally does not perform conformal prediction.  It uses
one fixed midpoint grid of initial states and paired observation conditions to
measure the empirical closed-loop behavior of the two frozen controllers.
"""

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from Aebs.conformal.calibrate_controller import (
    empirical_summary,
    file_sha256,
    rollout_batch,
)
from Aebs.semantic.robust_controller import load_contract
from Aebs.semantic_spvc.model import SemanticSPVCPolicy
from Aebs.semantic_spvc.qp_constraint import DiscreteSBCConstraint
from Aebs.semantic_spvc.qp_layer import ScalarSBCQP
from Aebs.system.env import Aebs
from Aebs.system.outcomes import SUCCESS, UNSAFE
from Aebs.VT.utils import MLP


SEMANTIC_MODES = (
    "exact",
    "dataset_nearest",
    "uniform_contract",
    "random_boundary_contract",
    "worst_endpoint_contract",
)


def midpoint_initial_grid(distance_points, speed_points):
    distances = 15.0 + (np.arange(distance_points) + 0.5) / distance_points
    speeds = 2.5 + 0.5 * (np.arange(speed_points) + 0.5) / speed_points
    distance_grid, speed_grid = np.meshgrid(distances, speeds, indexing="ij")
    return np.stack([distance_grid.reshape(-1), speed_grid.reshape(-1)], axis=1).astype(
        np.float32
    )


def load_semantic_lookup(policy, image_data, device):
    with h5py.File(image_data, "r") as data_file:
        images = np.asarray(data_file["X_train"], dtype=np.float32)
        distance_m = np.asarray(data_file["y_train"], dtype=np.float64).reshape(-1)
    predictions = []
    with torch.no_grad():
        for start in range(0, len(images), 256):
            image_tensor = torch.from_numpy(images[start:start + 256]).to(device)
            mean, _ = policy.encode_image(image_tensor)
            predictions.append(mean.cpu().numpy().reshape(-1))
    prediction_norm = np.concatenate(predictions).astype(np.float64)
    order = np.argsort(distance_m)
    return distance_m[order], prediction_norm[order]


def paired_comparison(baseline, filtered):
    difference = filtered["scores"] - baseline["scores"]
    return {
        "score_lower_is_safer": True,
        "mean_score_difference_qp_minus_ppo": float(np.mean(difference)),
        "minimum_score_difference_qp_minus_ppo": float(np.min(difference)),
        "maximum_score_difference_qp_minus_ppo": float(np.max(difference)),
        "qp_safer_trajectory_count": int(np.sum(difference < 0.0)),
        "qp_less_safe_trajectory_count": int(np.sum(difference > 0.0)),
        "score_tie_count": int(np.sum(difference == 0.0)),
        "different_outcome_count": int(np.sum(filtered["outcomes"] != baseline["outcomes"])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-dir", type=Path,
        default=Path("results/semantic_spvc_qp_original_init_lr1e4_full_20260926"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/semantic_spvc_no_cp_comprehensive_20260928"),
    )
    parser.add_argument("--semantic-checkpoint", type=Path,
                        default=Path("results/mvp/01_semantic/semantic_encoder.pt"))
    parser.add_argument("--controller", type=Path,
                        default=Path("Aebs/controller/best_model/best_model.zip"))
    parser.add_argument("--image-data", type=Path,
                        default=Path("Aebs/data/Downsampled.h5"))
    parser.add_argument("--distance-points", type=int, default=32)
    parser.add_argument("--speed-points", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--slack-weight", type=float, default=100.0)
    args = parser.parse_args()
    if min(args.distance_points, args.speed_points, args.horizon) < 1:
        parser.error("grid dimensions and horizon must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output directory is nonempty; choose a new directory")

    checkpoint_files = {
        "semantic_encoder": args.semantic_checkpoint,
        "semantic_ppo": args.checkpoint_dir / "semantic_ppo.pt",
        "sbc": args.checkpoint_dir / "sbc.pt",
        "image_dataset": args.image_data,
    }
    initial_states = midpoint_initial_grid(args.distance_points, args.speed_points)
    config = {
        "experiment": "semantic_spvc_no_cp_comprehensive_empirical_evaluation",
        "registered_before_rollout": True,
        "uses_conformal_prediction": False,
        "claim_level": "empirical_fixed_grid_evaluation_only_not_a_certificate",
        "checkpoint_dir": str(args.checkpoint_dir),
        "checkpoint_sha256": {
            name: file_sha256(path) for name, path in checkpoint_files.items()
        },
        "controllers": ["ppo", "ppo_plus_qp"],
        "semantic_modes": list(SEMANTIC_MODES),
        "initial_grid": {
            "distance_m": [15.0, 16.0],
            "speed_mps": [2.5, 3.0],
            "distance_points": args.distance_points,
            "speed_points": args.speed_points,
            "states": int(len(initial_states)),
            "construction": "cell midpoints",
        },
        "horizon": args.horizon,
        "fixed_seed": args.seed,
        "seed_policy": "one fixed seed; no multi-seed sweep",
        "paired_design": (
            "PPO and PPO+QP use identical initial states and identical pre-sampled "
            "observation-error sequences within each applicable mode"
        ),
        "semantic_mode_definitions": {
            "exact": "true simulator distance and speed",
            "dataset_nearest": (
                "semantic-encoder point estimate from the real dataset frame with "
                "nearest labeled distance"
            ),
            "uniform_contract": (
                "independent per-step Uniform(-radius,+radius) distance error"
            ),
            "random_boundary_contract": (
                "independent per-step random choice of -radius or +radius"
            ),
            "worst_endpoint_contract": (
                "at every step choose the contract endpoint yielding the smaller "
                "executed braking action for the evaluated controller"
            ),
        },
        "important_limitations": [
            "No CP calibration or confidence statement is used.",
            "The grid is finite and does not prove safety for all continuous initial states.",
            "Dataset-nearest is finite image replay, not new-camera distribution testing.",
            "Contract stress modes assume the saved semantic-error radii.",
            "The original deterministic AEBS dynamics are used without process disturbance.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "config.json", "w", encoding="utf-8") as output_file:
        json.dump(config, output_file, indent=2, ensure_ascii=False)

    started = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Aebs(0.05)
    policy = SemanticSPVCPolicy.from_checkpoints(
        args.semantic_checkpoint, args.controller, device
    )
    policy.load_state_dict(torch.load(checkpoint_files["semantic_ppo"], map_location=device))
    policy.eval()
    barrier = MLP([2, 16, 8, 1], activation="tanh", square_output=True).to(device)
    barrier.load_state_dict(torch.load(checkpoint_files["sbc"], map_location=device))
    barrier.eval()
    for model in (policy, barrier):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    constraint = DiscreteSBCConstraint(barrier, env, noise_bins=10).to(device)
    qp = ScalarSBCQP(
        float(env.action_space.low[0]), float(env.action_space.high[0]), args.slack_weight
    ).to(device)
    contract, checkpoint_scale = load_contract(args.semantic_checkpoint)
    if not np.isclose(checkpoint_scale, float(env.std1)):
        raise ValueError("semantic contract and environment distance scales differ")
    semantic_lookup = load_semantic_lookup(policy, args.image_data, device)

    rng = np.random.default_rng(args.seed)
    error_sequences = {
        "uniform_contract": rng.uniform(
            -1.0, 1.0, size=(len(initial_states), args.horizon)
        ).astype(np.float32),
        "random_boundary_contract": rng.choice(
            np.array([-1.0, 1.0], dtype=np.float32),
            size=(len(initial_states), args.horizon),
        ),
    }
    metrics = {"experiment": config["experiment"], "config": config, "modes": {}}
    arrays = {"initial_states": initial_states}
    for semantic_mode in SEMANTIC_MODES:
        print("evaluating %s: %d paired starts x H=%d" % (
            semantic_mode, len(initial_states), args.horizon
        ), flush=True)
        multipliers = error_sequences.get(semantic_mode)
        mode_results = {}
        for label, controller_mode in (("ppo", "baseline"), ("ppo_plus_qp", "qp")):
            result = rollout_batch(
                policy, constraint, qp, env, initial_states, controller_mode,
                args.horizon, device, semantic_mode, multipliers, contract,
                semantic_lookup,
            )
            mode_results[label] = result
            for key, value in result.items():
                arrays["%s_%s_%s" % (semantic_mode, label, key)] = value
        metrics["modes"][semantic_mode] = {
            "ppo": empirical_summary(mode_results["ppo"]),
            "ppo_plus_qp": empirical_summary(mode_results["ppo_plus_qp"]),
            "paired_qp_minus_ppo": paired_comparison(
                mode_results["ppo"], mode_results["ppo_plus_qp"]
            ),
        }
        for label in ("ppo", "ppo_plus_qp"):
            summary = metrics["modes"][semantic_mode][label]
            print("  %s: success=%d/%d, unsafe=%d/%d, timeout=%d/%d, steps=%.1f" % (
                label, summary["outcomes"][SUCCESS], len(initial_states),
                summary["outcomes"][UNSAFE], len(initial_states),
                summary["outcomes"]["timeout"], len(initial_states),
                summary["mean_steps"],
            ), flush=True)

    metrics["aggregate_descriptive_counts"] = {
        label: {
            "evaluated_trajectories": len(initial_states) * len(SEMANTIC_MODES),
            "successes": int(sum(
                metrics["modes"][mode][label]["outcomes"][SUCCESS]
                for mode in SEMANTIC_MODES
            )),
            "unsafe": int(sum(
                metrics["modes"][mode][label]["outcomes"][UNSAFE]
                for mode in SEMANTIC_MODES
            )),
        }
        for label in ("ppo", "ppo_plus_qp")
    }
    metrics["runtime_seconds"] = time.time() - started
    metrics["decision"] = (
        "comprehensive_empirical_evaluation_complete_no_cp_claim; "
        "inspect_each_mode_separately"
    )
    np.savez_compressed(args.output_dir / "trajectory_data.npz", **arrays)
    with open(args.output_dir / "metrics.json", "w", encoding="utf-8") as output_file:
        json.dump(metrics, output_file, indent=2, ensure_ascii=False, allow_nan=False)

    print("[Semantic-SPVC no-CP comprehensive empirical evaluation]")
    print("fixed starts=%d, modes=%d, paired trajectories/controller=%d" % (
        len(initial_states), len(SEMANTIC_MODES),
        len(initial_states) * len(SEMANTIC_MODES),
    ))
    for mode in SEMANTIC_MODES:
        ppo = metrics["modes"][mode]["ppo"]
        filtered = metrics["modes"][mode]["ppo_plus_qp"]
        print("%s: PPO success/unsafe=%.2f%%/%.2f%%; PPO+QP=%.2f%%/%.2f%%; "
              "QP intervention=%.2f%%, positive slack=%.2f%%" % (
                  mode, ppo["success_rate"] * 100, ppo["unsafe_rate"] * 100,
                  filtered["success_rate"] * 100, filtered["unsafe_rate"] * 100,
                  filtered["qp_intervention_rate"] * 100,
                  filtered["positive_slack_rate"] * 100,
              ))
    print("claim: empirical fixed-grid evaluation only; no CP confidence and no certificate")
    print("metrics: %s" % (args.output_dir / "metrics.json"))


if __name__ == "__main__":
    main()
