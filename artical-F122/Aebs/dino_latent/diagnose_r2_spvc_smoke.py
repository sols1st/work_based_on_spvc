"""Diagnose original-SPVC full-screen R2 development trajectories.

This script is development-only.  It does not create a conformal contract and
refuses formal calibration H5 files.  It recomputes both frozen controller
paths at every saved state and reports task outcomes, action disagreement,
opposite saturation, and trajectory maximum next-speed residuals.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from stable_baselines3 import PPO

from Aebs.connect.collect_r2_spvc_trajectories import (
    CAPTURE_PROTOCOL,
    DEVELOPMENT_SCHEMA,
)
from Aebs.dino_latent.calibrate_transition_contract import (
    next_state_residual,
    trajectory_scores,
)
from Aebs.dino_latent.models import PhysicalToLatent, SafetyProjection
from Aebs.dino_latent.prepare_expanded_data import sha256


def percentiles(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": float(values.min()),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
    }


def outcome_table(weather, color, outcome):
    grouped = defaultdict(Counter)
    for w, c, o in zip(weather, color, outcome):
        grouped[str(w) + "|" + str(c)][str(o)] += 1
    return {key: dict(grouped[key]) for key in sorted(grouped)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory-h5", type=Path,
        default=Path("results/dino_expanded_v1/12_spvc_online_smoke_v1/trajectories.h5"),
    )
    parser.add_argument(
        "--representation", type=Path,
        default=Path("results/dino_expanded_v1/08_R2_group_alignment/dino_safety_latent.pt"),
    )
    parser.add_argument(
        "--ppo-dir", type=Path,
        default=Path("results/dino_expanded_v1/09_R2_ppo"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/dino_expanded_v1/12_spvc_online_smoke_v1/diagnostic"),
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()
    controller_path = args.ppo_dir / "latent_ppo.zip"
    metrics_path = args.ppo_dir / "metrics.json"
    missing = [str(path) for path in (
        args.trajectory_h5, args.representation, controller_path, metrics_path
    ) if not path.is_file()]
    if missing or args.output_dir.exists() or args.batch_size < 1:
        parser.error("missing inputs, existing output, or invalid batch size: " + ", ".join(missing))

    representation = torch.load(args.representation, map_location="cpu")
    ppo_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if (
        representation.get("variant") != "R2_group_alignment"
        or ppo_metrics.get("representation_sha256") != sha256(args.representation)
        or ppo_metrics.get("checkpoint_sha256") != sha256(controller_path)
        or ppo_metrics.get("test_used") is not False
    ):
        parser.error("R2/PPO provenance mismatch")

    with h5py.File(args.trajectory_h5, "r") as stream:
        attrs = dict(stream.attrs)
        if (
            attrs.get("schema") != DEVELOPMENT_SCHEMA
            or attrs.get("capture_protocol") != CAPTURE_PROTOCOL
            or not bool(attrs.get("complete", False))
            or attrs.get("split_role") != "development"
            or not bool(attrs.get("used_for_model_selection", False))
        ):
            parser.error("expected complete development-only original-SPVC H5")
        features = np.asarray(stream["dino_features"], dtype=np.float32)
        distance = np.asarray(stream["distance_m"], dtype=np.float32)
        speed = np.asarray(stream["speed_mps"], dtype=np.float32)
        valid = np.asarray(stream["valid"], dtype=bool)
        saved_action = np.asarray(stream["image_action"], dtype=np.float32)
        trajectory_id = stream["trajectory_id"].asstr()[:]
        weather = stream["weather"].asstr()[:]
        color = stream["color"].asstr()[:]
        outcome = stream["outcome"].asstr()[:]
    if (
        features.ndim != 3 or features.shape[2] != 384
        or distance.shape != features.shape[:2]
        or speed.shape != features.shape[:2]
        or valid.shape != features.shape[:2]
        or saved_action.shape != features.shape[:2]
        or not valid.any(axis=1).all()
    ):
        raise ValueError("invalid trajectory arrays")

    projection = SafetyProjection().eval()
    projection.load_state_dict(representation["projection_state_dict"])
    q_model = PhysicalToLatent().eval()
    q_model.load_state_dict(representation["physical_to_latent_state_dict"])
    controller = PPO.load(controller_path, device="cpu")
    flat_valid = np.flatnonzero(valid.reshape(-1))
    flat_features = features.reshape(-1, 384)
    flat_distance = distance.reshape(-1)
    flat_speed = speed.reshape(-1)
    flat_saved = saved_action.reshape(-1)
    image_actions = []
    q_actions = []
    torch.set_num_threads(1)
    for start in range(0, len(flat_valid), args.batch_size):
        indices = flat_valid[start:start + args.batch_size]
        with torch.no_grad():
            image_latent = projection(
                (torch.from_numpy(flat_features[indices]) - representation["feature_mean"])
                / representation["feature_std"]
            ).numpy()
            q_latent = q_model(
                torch.from_numpy(flat_distance[indices, None])
                / float(representation["distance_scale_m"])
            ).numpy()
        selected_speed = flat_speed[indices, None]
        image_actions.append(controller.predict(
            np.concatenate([image_latent, selected_speed], axis=1), deterministic=True
        )[0].reshape(-1))
        q_actions.append(controller.predict(
            np.concatenate([q_latent, selected_speed], axis=1), deterministic=True
        )[0].reshape(-1))
    image_actions = np.concatenate(image_actions).astype(np.float64)
    q_actions = np.concatenate(q_actions).astype(np.float64)
    saved_selected = flat_saved[flat_valid].astype(np.float64)
    replay_error = np.abs(image_actions - saved_selected)
    action_difference = image_actions - q_actions
    opposite = (image_actions >= 2.999) & (q_actions <= -2.999)

    residual_flat = np.full((valid.size, 2), np.nan, dtype=np.float64)
    residual_flat[flat_valid] = next_state_residual(
        flat_distance[flat_valid], flat_speed[flat_valid], image_actions, q_actions
    )
    residual = residual_flat.reshape(features.shape[0], features.shape[1], 2)
    scores = trajectory_scores(residual, valid, [1.0, 1.0])
    saturated = scores >= 0.3 - 1e-8
    counts = Counter(outcome.tolist())
    gate_checks = {
        "all_trajectories_success": counts.get("success", 0) == len(outcome),
        "zero_unsafe": counts.get("unsafe", 0) == 0,
        "zero_opposite_saturation_steps": int(opposite.sum()) == 0,
        "zero_saturated_trajectory_scores": int(saturated.sum()) == 0,
        "saved_action_replay_matches": float(replay_error.max()) <= 1e-4,
    }
    result = {
        "experiment": "R2_original_SPVC_fullscreen_smoke_diagnostic_v1",
        "scope": "development-only; not CP calibration, not SBC, test unused",
        "capture_protocol": CAPTURE_PROTOCOL,
        "trajectory_h5_sha256": sha256(args.trajectory_h5),
        "representation_sha256": sha256(args.representation),
        "controller_sha256": sha256(controller_path),
        "test_used": False,
        "trajectory_count": int(len(outcome)),
        "valid_step_count": int(valid.sum()),
        "outcomes": dict(counts),
        "outcomes_by_appearance": outcome_table(weather, color, outcome),
        "saved_action_replay_error": {
            "mean": float(replay_error.mean()), "max": float(replay_error.max()),
        },
        "actions": {
            "image": percentiles(image_actions),
            "q": percentiles(q_actions),
            "absolute_difference": percentiles(np.abs(action_difference)),
            "mae": float(np.mean(np.abs(action_difference))),
            "image_less_braking_fraction": float(np.mean(image_actions < q_actions - 1e-6)),
            "opposite_saturation_steps": int(opposite.sum()),
            "opposite_saturation_fraction": float(opposite.mean()),
        },
        "trajectory_max_next_state_residual": {
            **percentiles(scores),
            "physical_max": 0.3,
            "saturated_trajectory_count": int(saturated.sum()),
            "scores": [float(value) for value in scores],
        },
        "gate_checks": gate_checks,
        "development_gate_pass": bool(all(gate_checks.values())),
        "trajectory_ids": trajectory_id.tolist(),
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / "scores.npz",
        trajectory_id=trajectory_id,
        scores=scores,
        outcome=outcome,
    )
    print("[R2 original-SPVC full-screen smoke diagnostic — development only]")
    print("trajectories=%d steps=%d outcomes=%s" % (
        len(outcome), valid.sum(), dict(counts)
    ))
    print("action MAE=%.9f max=%.9f opposite_saturation=%d/%d replay_max=%.9g" % (
        np.mean(np.abs(action_difference)), np.max(np.abs(action_difference)),
        opposite.sum(), len(opposite), replay_error.max(),
    ))
    print("trajectory score median=%.9f p95=%.9f max=%.9f saturated=%d/%d" % (
        np.median(scores), np.quantile(scores, 0.95), scores.max(),
        saturated.sum(), len(scores),
    ))
    print("development_gate_pass=%s" % result["development_gate_pass"])
    print("metrics:", args.output_dir / "metrics.json")


if __name__ == "__main__":
    main()
