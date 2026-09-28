"""Trajectory-level conformal calibration for frozen AEBS controllers.

The calibration unit is one complete finite-horizon trajectory, not one time
step.  The observation model is explicitly registered as either exact semantic
state or synthetic uniform distance error inside the saved state-conditional
semantic contract.  Dynamics remain deterministic.  Each observation model is
a different deployment distribution and must be calibrated in a new output
directory.
"""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import h5py
from scipy.stats import beta as beta_distribution

from Aebs.semantic_spvc.model import SemanticSPVCPolicy
from Aebs.semantic_spvc.qp_constraint import DiscreteSBCConstraint
from Aebs.semantic_spvc.qp_layer import ScalarSBCQP
from Aebs.semantic.robust_controller import load_contract
from Aebs.system.env import Aebs
from Aebs.system.outcomes import (
    OUT_OF_DOMAIN,
    STOPPED_SAFE_OUTSIDE_GOAL,
    SUCCESS,
    TIMEOUT,
    UNSAFE,
)
from Aebs.VT.utils import MLP


OUTCOMES = (SUCCESS, UNSAFE, STOPPED_SAFE_OUTSIDE_GOAL, OUT_OF_DOMAIN, TIMEOUT)


def trajectory_safety_score(distance_m, speed_mps, distance_scale=1.0, speed_scale=1.0):
    """Positive exactly when the AEBS unsafe conjunction is true.

    Unsafe is distance <= 6 m AND speed > 0.5 m/s.  The minimum represents
    conjunction and the trajectory maximum later represents existence in time.
    """
    distance_term = (6.0 - np.asarray(distance_m)) / float(distance_scale)
    speed_term = (np.asarray(speed_mps) - 0.5) / float(speed_scale)
    raw = np.minimum(distance_term, speed_term)
    # The simulator treats d == 6 and v > 0.5 as unsafe. Preserve that closed
    # distance boundary instead of letting its continuous score equal zero.
    unsafe = (np.asarray(distance_m) <= 6.0) & (np.asarray(speed_mps) > 0.5)
    return np.where(unsafe, np.maximum(raw, 1e-9), raw)


def select_tolerance_rank(sample_count, epsilon, beta):
    """Largest paper-Theorem-2 rank satisfying the requested confidence.

    Scores are sorted from largest (worst) to smallest.  Rank ``l`` has
    calibration-conditional coverage distributed as Beta(N-l+1, l).
    """
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if not 0.0 < epsilon < 1.0 or not 0.0 < beta < 1.0:
        raise ValueError("epsilon and beta must lie in (0, 1)")
    feasible = []
    for rank in range(1, sample_count + 1):
        confidence_failure = float(beta_distribution.cdf(
            1.0 - epsilon, sample_count - rank + 1, rank
        ))
        if confidence_failure <= beta:
            feasible.append((rank, confidence_failure))
    if not feasible:
        required_zero_failure = math.ceil(math.log(beta) / math.log(1.0 - epsilon))
        raise ValueError(
            "No conformal rank meets the requested epsilon/beta with N=%d; "
            "the worst-score rank needs at least N=%d" % (
                sample_count, required_zero_failure
            )
        )
    return feasible[-1]


def conformal_summary(scores, epsilon, beta):
    scores = np.asarray(scores, dtype=np.float64)
    rank, confidence_failure = select_tolerance_rank(len(scores), epsilon, beta)
    ordered = np.sort(scores)[::-1]
    q_hat = float(ordered[rank - 1])
    # Choose alpha strictly inside the interval whose floor((N+1)*alpha)=rank.
    alpha = (rank + 0.5) / (len(scores) + 1.0)
    return {
        "sample_count": int(len(scores)),
        "target_violation_rate_epsilon": float(epsilon),
        "target_confidence_failure_beta": float(beta),
        "achieved_confidence": float(1.0 - confidence_failure),
        "rank_from_largest_l": int(rank),
        "alpha_representative": float(alpha),
        "q_hat": q_hat,
        "passes_q_hat_nonpositive": bool(q_hat <= 0.0),
        "claim_if_passed": (
            "With confidence at least %.6f, a new IID trajectory has probability "
            "at least %.6f of score <= q_hat over the configured finite horizon."
            % (1.0 - confidence_failure, 1.0 - epsilon)
        ),
    }


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classify_batch(distance_m, speed_mps, active, step, horizon):
    outcomes = np.full(len(distance_m), "", dtype=object)
    unsafe = active & (distance_m <= 6.0) & (speed_mps > 0.5)
    outcomes[unsafe] = UNSAFE
    success = active & ~unsafe & (distance_m >= 5.0) & (distance_m <= 6.0) & (speed_mps <= 0.5)
    outcomes[success] = SUCCESS
    out_of_domain = active & ~unsafe & ~success & ((distance_m < 5.0) | (distance_m > 16.0))
    outcomes[out_of_domain] = OUT_OF_DOMAIN
    stopped = active & ~unsafe & ~success & ~out_of_domain & (speed_mps <= 1e-6) & (distance_m > 6.0)
    outcomes[stopped] = STOPPED_SAFE_OUTSIDE_GOAL
    if step == horizon:
        timeout = active & (outcomes == "")
        outcomes[timeout] = TIMEOUT
    return outcomes


def nearest_semantic_prediction(distance_m, lookup_distance_m, lookup_prediction_norm):
    """Use the real dataset frame whose label is nearest to each true distance."""
    distance_m = np.asarray(distance_m)
    insertion = np.searchsorted(lookup_distance_m, distance_m)
    right = np.clip(insertion, 0, len(lookup_distance_m) - 1)
    left = np.clip(insertion - 1, 0, len(lookup_distance_m) - 1)
    choose_right = (
        np.abs(lookup_distance_m[right] - distance_m)
        < np.abs(lookup_distance_m[left] - distance_m)
    )
    selected = np.where(choose_right, right, left)
    return lookup_prediction_norm[selected]


def rollout_batch(
    policy,
    constraint,
    qp,
    env,
    initial_states,
    mode,
    horizon,
    device,
    semantic_mode="exact",
    semantic_error_multipliers=None,
    semantic_contract=None,
    semantic_lookup=None,
):
    count = len(initial_states)
    distance_m = initial_states[:, 0].astype(np.float64).copy()
    speed_mps = initial_states[:, 1].astype(np.float64).copy()
    active = np.ones(count, dtype=bool)
    outcomes = np.full(count, TIMEOUT, dtype=object)
    steps = np.zeros(count, dtype=np.int64)
    maximum_score = trajectory_safety_score(distance_m, speed_mps)
    intervention_steps = np.zeros(count, dtype=np.int64)
    positive_slack_steps = np.zeros(count, dtype=np.int64)
    maximum_slack = np.zeros(count, dtype=np.float64)
    semantic_observation_steps = np.zeros(count, dtype=np.int64)
    absolute_distance_error_sum_m = np.zeros(count, dtype=np.float64)
    maximum_absolute_distance_error_m = np.zeros(count, dtype=np.float64)

    if semantic_mode not in ("exact", "uniform_contract", "dataset_nearest"):
        raise ValueError("unknown semantic_mode: %s" % semantic_mode)
    if semantic_mode == "uniform_contract":
        if semantic_contract is None or semantic_error_multipliers is None:
            raise ValueError("uniform_contract requires a contract and error multipliers")
        if semantic_error_multipliers.shape != (count, horizon):
            raise ValueError("semantic_error_multipliers must have shape [episodes, horizon]")

    for step in range(1, horizon + 1):
        indices = np.flatnonzero(active)
        if not len(indices):
            break
        true_distance_norm = distance_m[indices] / float(env.std1)
        if semantic_mode == "uniform_contract":
            radius_norm = semantic_contract.radius(distance_m[indices]).astype(np.float64)
            error_norm = radius_norm * semantic_error_multipliers[indices, step - 1]
            observed_distance_norm = np.clip(
                true_distance_norm + error_norm,
                5.0 / float(env.std1),
                16.0 / float(env.std1),
            )
            realized_error_m = (observed_distance_norm - true_distance_norm) * float(env.std1)
        elif semantic_mode == "dataset_nearest":
            if semantic_lookup is None:
                raise ValueError("dataset_nearest requires a semantic lookup")
            observed_distance_norm = nearest_semantic_prediction(
                distance_m[indices], semantic_lookup[0], semantic_lookup[1]
            )
            observed_distance_norm = np.clip(
                observed_distance_norm,
                5.0 / float(env.std1),
                16.0 / float(env.std1),
            )
            realized_error_m = (
                observed_distance_norm - true_distance_norm
            ) * float(env.std1)
        else:
            observed_distance_norm = true_distance_norm
            realized_error_m = np.zeros(len(indices), dtype=np.float64)
        states = np.stack([observed_distance_norm, speed_mps[indices]], axis=1).astype(np.float32)
        semantic_observation_steps[indices] += 1
        absolute_distance_error_sum_m[indices] += np.abs(realized_error_m)
        maximum_absolute_distance_error_m[indices] = np.maximum(
            maximum_absolute_distance_error_m[indices], np.abs(realized_error_m)
        )
        state_tensor = torch.from_numpy(states).to(device)
        latent = torch.zeros((len(indices), 4), dtype=torch.float32, device=device)
        with torch.no_grad():
            nominal = policy(latent, state_tensor)
        clipped = nominal.clamp(
            min=float(env.action_space.low[0]), max=float(env.action_space.high[0])
        )
        if mode == "qp":
            affine = constraint.linearize(state_tensor, nominal)
            solution = qp(nominal, affine.coefficient, affine.right_hand_side)
            action = solution.action.detach()
            slack = solution.slack.detach().cpu().numpy().reshape(-1)
            changed = (action - clipped).abs().detach().cpu().numpy().reshape(-1) > 1e-6
            intervention_steps[indices] += changed.astype(np.int64)
            positive_slack_steps[indices] += (slack > 1e-6).astype(np.int64)
            maximum_slack[indices] = np.maximum(maximum_slack[indices], slack)
        else:
            action = clipped

        acceleration = action.cpu().numpy().reshape(-1).astype(np.float64)
        old_speed = speed_mps[indices].copy()
        distance_m[indices] -= old_speed * 0.05
        speed_mps[indices] = np.clip(old_speed - acceleration * 0.05, 0.0, 3.0)
        steps[indices] += 1
        maximum_score[indices] = np.maximum(
            maximum_score[indices],
            trajectory_safety_score(distance_m[indices], speed_mps[indices]),
        )
        step_outcomes = classify_batch(distance_m, speed_mps, active, step, horizon)
        finished = step_outcomes != ""
        outcomes[finished] = step_outcomes[finished]
        active[finished] = False

    return {
        "scores": maximum_score,
        "outcomes": outcomes,
        "steps": steps,
        "intervention_steps": intervention_steps,
        "positive_slack_steps": positive_slack_steps,
        "maximum_slack": maximum_slack,
        "semantic_observation_steps": semantic_observation_steps,
        "absolute_distance_error_sum_m": absolute_distance_error_sum_m,
        "maximum_absolute_distance_error_m": maximum_absolute_distance_error_m,
    }


def empirical_summary(result):
    total_steps = max(1, int(result["steps"].sum()))
    count = len(result["scores"])
    outcome_counts = {
        name: int(np.sum(result["outcomes"] == name)) for name in OUTCOMES
    }
    return {
        "episodes": count,
        "outcomes": outcome_counts,
        "success_rate": outcome_counts[SUCCESS] / count,
        "unsafe_rate": outcome_counts[UNSAFE] / count,
        "timeout_rate": outcome_counts[TIMEOUT] / count,
        "mean_steps": float(np.mean(result["steps"])),
        "maximum_trajectory_score": float(np.max(result["scores"])),
        "median_trajectory_score": float(np.median(result["scores"])),
        "nonpositive_score_rate": float(np.mean(result["scores"] <= 0.0)),
        "qp_intervention_rate": float(result["intervention_steps"].sum() / total_steps),
        "positive_slack_rate": float(result["positive_slack_steps"].sum() / total_steps),
        "maximum_slack": float(np.max(result["maximum_slack"])),
        "mean_absolute_distance_error_m": float(
            result["absolute_distance_error_sum_m"].sum()
            / max(1, result["semantic_observation_steps"].sum())
        ),
        "maximum_absolute_distance_error_m": float(
            np.max(result["maximum_absolute_distance_error_m"])
        ),
    }


def sample_initial_states(count, seed):
    rng = np.random.default_rng(seed)
    return np.stack([
        rng.uniform(15.0, 16.0, count),
        rng.uniform(2.5, 3.0, count),
    ], axis=1).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=Path("results/semantic_spvc_qp_original_init_lr1e4_full_20260926"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/conformal/trajectory_cp_mvp"))
    parser.add_argument("--semantic-checkpoint",
                        default="results/mvp/01_semantic/semantic_encoder.pt")
    parser.add_argument("--controller", default="Aebs/controller/best_model/best_model.zip")
    parser.add_argument("--calibration-episodes", type=int, default=459)
    parser.add_argument("--test-episodes", type=int, default=200)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--slack-weight", type=float, default=100.0)
    parser.add_argument(
        "--semantic-mode",
        choices=("exact", "uniform_contract", "dataset_nearest"),
        default="exact",
        help=("exact, or independent per-step uniform distance error inside the "
              "saved state-conditional semantic contract"),
    )
    parser.add_argument("--image-data", type=Path, default=Path("Aebs/data/Downsampled.h5"))
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output directory is nonempty; preserve results by choosing a new directory")
    if args.calibration_episodes < 1 or args.test_episodes < 1 or args.horizon < 1:
        parser.error("episode counts and horizon must be positive")
    # Fail before loading models if the sample size cannot support the request.
    select_tolerance_rank(args.calibration_episodes, args.epsilon, args.beta)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_files = {
        "semantic_encoder": Path(args.semantic_checkpoint),
        "semantic_ppo": args.checkpoint_dir / "semantic_ppo.pt",
        "sbc": args.checkpoint_dir / "sbc.pt",
    }
    if args.semantic_mode == "dataset_nearest":
        checkpoint_files["image_dataset"] = args.image_data
    config = {
        "experiment": "trajectory_level_conformal_mvp",
        "registered_before_rollout": True,
        "checkpoint_dir": str(args.checkpoint_dir),
        "checkpoint_sha256": {
            name: file_sha256(path) for name, path in checkpoint_files.items()
        },
        "controllers": ["ppo", "ppo_plus_qp"],
        "calibration_episodes": args.calibration_episodes,
        "test_episodes": args.test_episodes,
        "horizon": args.horizon,
        "target_epsilon": args.epsilon,
        "target_beta": args.beta,
        "calibration_seed": args.seed,
        "test_seed": args.seed + 1,
        "initial_distribution": {
            "distance_m": "Uniform(15,16)", "speed_mps": "Uniform(2.5,3.0)"
        },
        "semantics": (
            "exact simulator distance and speed"
            if args.semantic_mode == "exact"
            else (
                "exact speed plus semantic-encoder point estimate from the nearest "
                "labeled real dataset image"
                if args.semantic_mode == "dataset_nearest"
                else "exact speed plus independent per-step uniform distance error "
                     "inside the saved state-conditional semantic contract"
            )
        ),
        "semantic_mode": args.semantic_mode,
        "dynamics": "deterministic original AebsEnv dynamics",
        "trajectory_score": "max_t min((6-d_t)/1m, (v_t-0.5)/1mps)",
        "score_positive_means": "entered the configured unsafe set during the finite horizon",
        "not_covered": [
            "real image-distribution shift", "process disturbance",
            "distribution shift", "infinite horizon", "worst-endpoint stress mode",
        ],
    }
    if args.semantic_mode == "uniform_contract":
        config["semantic_error_distribution"] = {
            "multiplier": "IID Uniform(-1,1) at every active control step",
            "radius_source": str(args.semantic_checkpoint),
            "conditioning": "true physical distance bin",
            "important_limitation": (
                "synthetic distribution inside the calibrated interval; not the "
                "empirical image-error distribution and not a worst-case guarantee"
            ),
        }
    elif args.semantic_mode == "dataset_nearest":
        config["semantic_image_replay"] = {
            "dataset": str(args.image_data),
            "selection": "nearest labeled physical distance at each active step",
            "encoder_output": "point estimate",
            "important_limitation": (
                "finite 400-image nearest-neighbor replay; not a rendering engine, "
                "not new camera data, and not a guarantee under image distribution shift"
            ),
        }
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
        float(env.action_space.low[0]), float(env.action_space.high[0]),
        args.slack_weight,
    ).to(device)

    calibration_states = sample_initial_states(args.calibration_episodes, args.seed)
    test_states = sample_initial_states(args.test_episodes, args.seed + 1)
    semantic_contract = None
    semantic_lookup = None
    calibration_error_multipliers = None
    test_error_multipliers = None
    if args.semantic_mode == "uniform_contract":
        semantic_contract, checkpoint_scale = load_contract(args.semantic_checkpoint)
        if not np.isclose(checkpoint_scale, float(env.std1)):
            raise ValueError("semantic contract and environment distance scales differ")
        calibration_error_multipliers = np.random.default_rng(args.seed + 2).uniform(
            -1.0, 1.0, size=(args.calibration_episodes, args.horizon)
        ).astype(np.float32)
        test_error_multipliers = np.random.default_rng(args.seed + 3).uniform(
            -1.0, 1.0, size=(args.test_episodes, args.horizon)
        ).astype(np.float32)
    elif args.semantic_mode == "dataset_nearest":
        with h5py.File(args.image_data, "r") as data_file:
            lookup_images = np.asarray(data_file["X_train"], dtype=np.float32)
            lookup_distance_m = np.asarray(
                data_file["y_train"], dtype=np.float64
            ).reshape(-1)
        predicted = []
        with torch.no_grad():
            for start in range(0, len(lookup_images), 256):
                image_tensor = torch.from_numpy(lookup_images[start:start + 256]).to(device)
                mean, _ = policy.encode_image(image_tensor)
                predicted.append(mean.cpu().numpy().reshape(-1))
        prediction_norm = np.concatenate(predicted).astype(np.float64)
        order = np.argsort(lookup_distance_m)
        semantic_lookup = (lookup_distance_m[order], prediction_norm[order])
    metrics = {"experiment": config["experiment"], "config": config, "controllers": {}}
    arrays = {
        "calibration_initial_states": calibration_states,
        "test_initial_states": test_states,
    }
    for label, mode in (("ppo", "baseline"), ("ppo_plus_qp", "qp")):
        print("calibrating %s: N=%d, H=%d" % (
            label, args.calibration_episodes, args.horizon
        ), flush=True)
        calibration = rollout_batch(
            policy, constraint, qp, env, calibration_states, mode, args.horizon, device,
            args.semantic_mode, calibration_error_multipliers, semantic_contract,
            semantic_lookup,
        )
        test = rollout_batch(
            policy, constraint, qp, env, test_states, mode, args.horizon, device,
            args.semantic_mode, test_error_multipliers, semantic_contract,
            semantic_lookup,
        )
        cp = conformal_summary(calibration["scores"], args.epsilon, args.beta)
        metrics["controllers"][label] = {
            "conformal": cp,
            "calibration": empirical_summary(calibration),
            "independent_test": empirical_summary(test),
        }
        for split_name, result in (("calibration", calibration), ("test", test)):
            for key, value in result.items():
                arrays["%s_%s_%s" % (label, split_name, key)] = value
        print("%s: q_hat=%.9f, pass=%s, calibration unsafe=%d/%d, "
              "test unsafe=%d/%d" % (
                  label, cp["q_hat"], cp["passes_q_hat_nonpositive"],
                  metrics["controllers"][label]["calibration"]["outcomes"][UNSAFE],
                  args.calibration_episodes,
                  metrics["controllers"][label]["independent_test"]["outcomes"][UNSAFE],
                  args.test_episodes,
              ), flush=True)

    metrics["paired_comparison_qp_minus_ppo"] = {}
    for split_name in ("calibration", "test"):
        difference = (
            arrays["ppo_plus_qp_%s_scores" % split_name]
            - arrays["ppo_%s_scores" % split_name]
        )
        metrics["paired_comparison_qp_minus_ppo"][split_name] = {
            "score_lower_is_safer": True,
            "mean_score_difference": float(np.mean(difference)),
            "minimum_score_difference": float(np.min(difference)),
            "maximum_score_difference": float(np.max(difference)),
            "qp_safer_trajectory_count": int(np.sum(difference < 0.0)),
            "qp_less_safe_trajectory_count": int(np.sum(difference > 0.0)),
            "tie_count": int(np.sum(difference == 0.0)),
        }
    metrics["runtime_seconds"] = time.time() - started
    metrics["decision"] = {
        label: ("finite_horizon_cp_pass_under_registered_distribution"
                if item["conformal"]["passes_q_hat_nonpositive"]
                else "finite_horizon_cp_target_not_met")
        for label, item in metrics["controllers"].items()
    }
    np.savez_compressed(args.output_dir / "trajectory_data.npz", **arrays)
    with open(args.output_dir / "metrics.json", "w", encoding="utf-8") as output_file:
        json.dump(metrics, output_file, indent=2, ensure_ascii=False, allow_nan=False)
    print("[Trajectory-level conformal calibration: finite horizon]")
    print("target: safety>=%.2f%% with confidence>=%.2f%%" % (
        (1.0 - args.epsilon) * 100, (1.0 - args.beta) * 100
    ))
    for label, item in metrics["controllers"].items():
        print("%s: decision=%s, q_hat=%.9f, achieved confidence=%.4f%%, "
              "calibration success/unsafe=%d/%d, test success/unsafe=%d/%d" % (
                  label, metrics["decision"][label], item["conformal"]["q_hat"],
                  item["conformal"]["achieved_confidence"] * 100,
                  item["calibration"]["outcomes"][SUCCESS],
                  item["calibration"]["outcomes"][UNSAFE],
                  item["independent_test"]["outcomes"][SUCCESS],
                  item["independent_test"]["outcomes"][UNSAFE],
              ))
    paired = metrics["paired_comparison_qp_minus_ppo"]["calibration"]
    print("paired calibration score QP-PPO: mean=%.9f, QP safer/worse/tie=%d/%d/%d "
          "(lower score is safer)" % (
              paired["mean_score_difference"], paired["qp_safer_trajectory_count"],
              paired["qp_less_safe_trajectory_count"], paired["tie_count"],
          ))
    print("scope: semantic_mode=%s, deterministic dynamics, H=%d; not an infinite-horizon guarantee" % (
        args.semantic_mode, args.horizon
    ))
    print("metrics: %s" % (args.output_dir / "metrics.json"))


if __name__ == "__main__":
    main()
