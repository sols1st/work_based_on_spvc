"""Calibrate a frozen trajectory-scenario SBC on fresh independent rollouts.

For each complete trajectory, the score is

    B(s0) + sum_t max(B(s_{t+1}) - B(s_t), 0) - 1.

The stopped certificate is defined as B=0 on safe terminal states and B=1 on
unsafe states. Therefore every unsafe trajectory has score >= 0.  If the
trajectory-level conformal q_hat is negative, the conformal coverage statement
implies a finite-horizon safety statement for the registered IID deployment
distribution.  The barrier and collection protocol must be frozen before the
input trajectories are collected.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from Aebs.conformal.calibrate_controller import conformal_summary
from Aebs.dino_latent.calibrate_transition_contract import SCHEMA
from Aebs.dino_latent.prepare_expanded_data import sha256
from Aebs.dino_latent.train_cp_robust_sbc import (
    ScenarioBarrier,
    safe_terminal_mask,
    unsafe_mask,
)


DT = 0.05


def stopped_barrier_values(barrier, states, device, batch_size):
    states = np.asarray(states, dtype=np.float32)
    output = []
    with torch.no_grad():
        for start in range(0, len(states), batch_size):
            output.append(barrier(
                torch.from_numpy(states[start:start + batch_size]).to(device)
            ).cpu().numpy())
    values = np.concatenate(output).astype(np.float64)
    values[safe_terminal_mask(states)] = 0.0
    values[unsafe_mask(states)] = 1.0
    return values


def reconstruct_successors(distance, speed, action):
    next_distance = np.asarray(distance, dtype=np.float64) - np.asarray(speed) * DT
    next_speed = np.clip(
        np.asarray(speed, dtype=np.float64)
        - np.clip(np.asarray(action, dtype=np.float64), -3.0, 3.0) * DT,
        0.0, 3.0,
    )
    return np.stack((next_distance, next_speed), axis=1).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-h5", type=Path, required=True)
    parser.add_argument(
        "--registration", type=Path,
        default=Path(
            "results/dino_expanded_v1/20_trajectory_sbc_cp_registration_v1/registration.json"
        ),
    )
    parser.add_argument(
        "--barrier", type=Path,
        default=Path(
            "results/dino_expanded_v1/19_trajectory_scenario_sbc_boundary_v3/barrier.pt"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/dino_expanded_v1/21_trajectory_sbc_cp_v1"),
    )
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if (
        args.output_dir.exists() or not args.calibration_h5.is_file()
        or not args.registration.is_file()
        or not args.barrier.is_file() or args.horizon < 1 or args.batch_size < 1
        or not 0.0 < args.epsilon < 1.0 or not 0.0 < args.beta < 1.0
    ):
        parser.error("missing inputs, existing output, or invalid settings")

    registration = json.loads(args.registration.read_text(encoding="utf-8"))
    checkpoint = torch.load(args.barrier, map_location="cpu")
    if (
        checkpoint.get("experiment")
        != "R2_trajectory_scenario_SBC_boundary_focused_v3"
        or checkpoint.get("development_only") is not True
    ):
        parser.error("unexpected or non-development barrier checkpoint")
    registered_collection = registration.get("collection", {})
    if (
        registration.get("experiment")
        != "R2_frozen_trajectory_SBC_conformal_safety_v1"
        or registration.get("barrier_frozen") is not True
        or registration.get("models_frozen") is not True
        or registration.get("test_used") is not False
        or registration.get("barrier_sha256") != sha256(args.barrier)
        or registration.get("barrier_training_h5_sha256")
        != checkpoint.get("trajectory_h5_sha256")
        or registration.get("representation_sha256")
        != checkpoint.get("representation_sha256")
        or registration.get("controller_sha256")
        != checkpoint.get("controller_sha256")
        or float(registration.get("epsilon", -1.0)) != args.epsilon
        or float(registration.get("beta", -1.0)) != args.beta
        or registered_collection.get("protocol")
        != "original_spvc_primary_monitor_fullscreen_v1"
        or registered_collection.get("role") != "calibration"
        or int(registered_collection.get("horizon", -1)) != args.horizon
    ):
        parser.error("registration does not bind the requested frozen experiment")
    fresh_h5_hash = sha256(args.calibration_h5)
    if fresh_h5_hash == checkpoint.get("trajectory_h5_sha256"):
        parser.error("formal CP calibration must not reuse the SBC development trajectories")

    with h5py.File(args.calibration_h5, "r") as stream:
        attrs = dict(stream.attrs)
        try:
            provenance = json.loads(attrs["provenance_json"])
        except (KeyError, TypeError, json.JSONDecodeError):
            parser.error("missing trajectory collection provenance")
        if (
            attrs.get("schema") != SCHEMA
            or not bool(attrs.get("complete", False))
            or attrs.get("split_role") != "calibration"
            or not bool(attrs.get("independent_from_training", False))
            or bool(attrs.get("used_for_model_selection", True))
            or int(attrs.get("horizon", -1)) != args.horizon
            or provenance.get("registered_before_collection") is not True
            or provenance.get("model_updates") is not False
            or provenance.get("test_used") is not False
            or int(provenance.get("seed", -1))
            != int(registered_collection.get("seed", -2))
            or int(provenance.get("episodes", -1))
            != int(registered_collection.get("episodes", -2))
            or provenance.get("capture_protocol")
            != registered_collection.get("protocol")
            or provenance.get("representation_sha256")
            != checkpoint.get("representation_sha256")
            or provenance.get("controller_sha256")
            != checkpoint.get("controller_sha256")
        ):
            parser.error("fresh calibration independence/provenance check failed")
        distance = np.asarray(stream["distance_m"], dtype=np.float32)
        speed = np.asarray(stream["speed_mps"], dtype=np.float32)
        action = np.asarray(stream["image_action"], dtype=np.float32)
        valid = np.asarray(stream["valid"], dtype=bool)
        trajectory_ids = stream["trajectory_id"].asstr()[:]
        outcomes = stream["outcome"].asstr()[:]
    if (
        distance.shape != speed.shape or action.shape != distance.shape
        or valid.shape != distance.shape or distance.shape[1] != args.horizon
        or len(distance) != int(registered_collection.get("episodes", -1))
        or len(distance) < 459 or len(trajectory_ids) != len(distance)
        or not valid.any(axis=1).all()
    ):
        parser.error("invalid or insufficient calibration trajectory arrays")

    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    barrier = ScenarioBarrier().to(device)
    barrier.load_state_dict(checkpoint["state_dict"])
    barrier.eval()

    scores = np.empty(len(distance), dtype=np.float64)
    maximum_increments = np.empty(len(distance), dtype=np.float64)
    initial_values = np.empty(len(distance), dtype=np.float64)
    positive_variations = np.empty(len(distance), dtype=np.float64)
    max_increment_bound_scores = np.empty(len(distance), dtype=np.float64)
    derived_unsafe = np.zeros(len(distance), dtype=bool)
    for index in range(len(distance)):
        length = int(valid[index].sum())
        current = np.stack(
            (distance[index, :length], speed[index, :length]), axis=1
        ).astype(np.float32)
        successor = reconstruct_successors(
            distance[index, :length], speed[index, :length], action[index, :length]
        )
        current_value = stopped_barrier_values(
            barrier, current, device, args.batch_size
        )
        next_value = stopped_barrier_values(
            barrier, successor, device, args.batch_size
        )
        increments = next_value - current_value
        positive = np.maximum(increments, 0.0)
        initial_values[index] = current_value[0]
        maximum_increments[index] = max(0.0, float(positive.max()))
        positive_variations[index] = positive.sum()
        scores[index] = initial_values[index] + positive_variations[index] - 1.0
        max_increment_bound_scores[index] = (
            initial_values[index] + args.horizon * maximum_increments[index] - 1.0
        )
        derived_unsafe[index] = bool(unsafe_mask(successor).any())
    reported_unsafe = np.asarray([value == "unsafe" for value in outcomes])
    if not np.array_equal(derived_unsafe, reported_unsafe):
        parser.error("saved outcomes and reconstructed unsafe transitions disagree")
    if np.any(scores[reported_unsafe] < -1e-9):
        parser.error("unsafe trajectory produced a negative SBC score")

    conformal = conformal_summary(scores, args.epsilon, args.beta)
    max_increment_conformal = conformal_summary(
        max_increment_bound_scores, args.epsilon, args.beta
    )
    status = (
        "trajectory_sbc_cp_safety_pass"
        if conformal["q_hat"] < 0.0 else "trajectory_sbc_cp_safety_not_established"
    )
    metrics = {
        "experiment": "R2_frozen_trajectory_SBC_conformal_safety_v1",
        "status": status,
        "score": "B(s0) + sum positive stopped-barrier increments - 1",
        "score_logic": (
            "unsafe implies score >= 0 because stopped B is 1 on unsafe and 0 on safe "
            "terminal states; q_hat < 0 converts trajectory coverage into safety coverage"
        ),
        "trajectory_count": int(len(scores)),
        "reported_outcomes": {
            str(name): int(np.sum(outcomes == name)) for name in np.unique(outcomes)
        },
        "score_summary": {
            "min": float(scores.min()),
            "median": float(np.median(scores)),
            "p95": float(np.quantile(scores, 0.95)),
            "max": float(scores.max()),
        },
        "maximum_positive_increment": {
            "median": float(np.median(maximum_increments)),
            "p95": float(np.quantile(maximum_increments, 0.95)),
            "max": float(maximum_increments.max()),
        },
        "conformal": conformal,
        "conservative_horizon_times_max_increment_conformal": max_increment_conformal,
        "claim_boundary": (
            "finite-horizon IID trajectory safety for the preregistered original-SPVC "
            "capture, initial-state and appearance distribution; not continuous-domain or "
            "out-of-distribution safety"
        ),
        "barrier_sha256": sha256(args.barrier),
        "registration_sha256": sha256(args.registration),
        "calibration_h5_sha256": fresh_h5_hash,
        "barrier_training_h5_sha256": checkpoint["trajectory_h5_sha256"],
        "test_used": False,
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / "trajectory_scores.npz",
        trajectory_id=trajectory_ids,
        outcome=outcomes,
        initial_value=initial_values,
        maximum_positive_increment=maximum_increments,
        positive_variation=positive_variations,
        score=scores,
        max_increment_bound_score=max_increment_bound_scores,
    )
    print("[R2 frozen trajectory-SBC conformal safety calibration]")
    print("status:", status)
    print("trajectories=%d outcomes=%s" % (
        len(scores), metrics["reported_outcomes"]
    ))
    print("score: median=%.9f p95=%.9f max=%.9f" % (
        np.median(scores), np.quantile(scores, 0.95), scores.max()
    ))
    print("positive increment: median=%.9f p95=%.9f max=%.9f" % (
        np.median(maximum_increments), np.quantile(maximum_increments, 0.95),
        maximum_increments.max()
    ))
    print("CP: q_hat=%.9f confidence=%.6f target_coverage=%.6f" % (
        conformal["q_hat"], conformal["achieved_confidence"],
        1.0 - conformal["target_violation_rate_epsilon"],
    ))
    print("conservative H*max CP q_hat=%.9f" % max_increment_conformal["q_hat"])
    print("metrics:", args.output_dir / "metrics.json")


if __name__ == "__main__":
    main()
