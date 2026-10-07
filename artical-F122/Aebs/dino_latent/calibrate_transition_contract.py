"""Calibrate a finite-horizon physical next-state residual contract for R2.

The calibration unit is one independent synchronized image/state trajectory.
This entry point never accepts train/validation rows or the old static image
cache.  It compares the frozen image-latent and q(distance) control paths at
the same physical state, maps their action difference through the registered
AEBS dynamics, and calibrates the maximum normalized residual over a complete
trajectory.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from stable_baselines3 import PPO

from Aebs.conformal.calibrate_controller import conformal_summary
from Aebs.dino_latent.models import PhysicalToLatent, SafetyProjection
from Aebs.dino_latent.prepare_expanded_data import sha256


SCHEMA = "spvc_r2_online_calibration_trajectories_v1"


def next_state_residual(distance_m, speed_mps, image_action, nominal_action, dt=0.05):
    """Return image-path minus q-path next state at the same current state."""
    distance_m = np.asarray(distance_m, dtype=np.float64)
    speed_mps = np.asarray(speed_mps, dtype=np.float64)
    image_action = np.clip(np.asarray(image_action, dtype=np.float64), -3.0, 3.0)
    nominal_action = np.clip(np.asarray(nominal_action, dtype=np.float64), -3.0, 3.0)
    image_next_d = distance_m - speed_mps * dt
    nominal_next_d = distance_m - speed_mps * dt
    image_next_v = np.clip(speed_mps - image_action * dt, 0.0, 3.0)
    nominal_next_v = np.clip(speed_mps - nominal_action * dt, 0.0, 3.0)
    return np.stack(
        [image_next_d - nominal_next_d, image_next_v - nominal_next_v], axis=-1
    )


def trajectory_scores(residual, valid, state_scales):
    residual = np.asarray(residual, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    scales = np.asarray(state_scales, dtype=np.float64)
    if (
        residual.ndim != 3 or residual.shape[-1] != 2
        or valid.shape != residual.shape[:2]
        or scales.shape != (2,) or not np.isfinite(scales).all()
        or np.any(scales <= 0) or not valid.any(axis=1).all()
    ):
        raise ValueError("invalid trajectory residuals, mask, or state scales")
    normalized = np.max(np.abs(residual) / scales, axis=2)
    normalized[~valid] = -np.inf
    return normalized.max(axis=1)


def validate_calibration_arrays(features, distance, speed, valid, trajectory_ids, horizon):
    if (
        features.ndim != 3 or features.shape[2] != 384
        or distance.shape != features.shape[:2]
        or speed.shape != features.shape[:2]
        or valid.shape != features.shape[:2]
        or len(trajectory_ids) != features.shape[0]
        or features.shape[1] != horizon
        or len(set(trajectory_ids)) != len(trajectory_ids)
        or not valid.any(axis=1).all()
    ):
        raise ValueError("calibration trajectory arrays have invalid shape or identity")
    selected = valid.astype(bool)
    if (
        not np.isfinite(features[selected]).all()
        or not np.isfinite(distance[selected]).all()
        or not np.isfinite(speed[selected]).all()
        or np.any((distance[selected] < 5.0) | (distance[selected] > 16.0))
        or np.any((speed[selected] < 0.0) | (speed[selected] > 3.0))
    ):
        raise ValueError("valid calibration states/features are nonfinite or out of domain")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration-h5", type=Path, required=True,
        help="Independent synchronized online trajectories; never train/validation cache.",
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
        "--training-cache-manifest", type=Path,
        default=Path("results/dino_expanded_v1/02_dino_h5/cache_manifest.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/dino_expanded_v1/10_transition_contract"),
    )
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--distance-residual-scale-m", type=float, default=1.0)
    parser.add_argument("--speed-residual-scale-mps", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()
    if (
        args.output_dir.exists() or args.horizon < 1 or args.batch_size < 1
        or not 0 < args.epsilon < 1 or not 0 < args.beta < 1
        or min(args.distance_residual_scale_m, args.speed_residual_scale_mps) <= 0
    ):
        parser.error("use a new output directory and valid preregistered settings")
    controller_path = args.ppo_dir / "latent_ppo.zip"
    metrics_path = args.ppo_dir / "metrics.json"
    required = [
        args.calibration_h5, args.representation, args.training_cache_manifest,
        controller_path, metrics_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error("missing inputs: " + ", ".join(missing))

    representation = torch.load(args.representation, map_location="cpu")
    ppo_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    training_cache = json.loads(args.training_cache_manifest.read_text(encoding="utf-8"))
    expected_dino_hash = training_cache.get("provenance", {}).get("dino_weights_sha256")
    if (
        representation.get("variant") != "R2_group_alignment"
        or representation.get("dino_frozen") is not True
        or ppo_metrics.get("representation_sha256") != sha256(args.representation)
        or ppo_metrics.get("checkpoint_sha256") != sha256(controller_path)
        or ppo_metrics.get("test_used") is not False
        or not expected_dino_hash
    ):
        parser.error("R2/PPO/training-cache provenance mismatch")

    with h5py.File(args.calibration_h5, "r") as stream:
        attrs = dict(stream.attrs)
        if (
            attrs.get("schema") != SCHEMA or not bool(attrs.get("complete", False))
            or attrs.get("split_role") != "calibration"
            or not bool(attrs.get("independent_from_training", False))
            or bool(attrs.get("used_for_model_selection", True))
            or attrs.get("dino_weights_sha256") != expected_dino_hash
            or int(attrs.get("horizon", -1)) != args.horizon
        ):
            parser.error("calibration H5 independence/schema/DINO provenance check failed")
        features = np.asarray(stream["dino_features"], dtype=np.float32)
        distance = np.asarray(stream["distance_m"], dtype=np.float32)
        speed = np.asarray(stream["speed_mps"], dtype=np.float32)
        valid = np.asarray(stream["valid"], dtype=bool)
        trajectory_ids = stream["trajectory_id"].asstr()[:].tolist()
    validate_calibration_arrays(
        features, distance, speed, valid, trajectory_ids, args.horizon
    )

    config = {
        "experiment": "R2_physical_next_state_trajectory_conformal_contract_v1",
        "registered_before_calibration": True,
        "calibration_unit": "one independent synchronized image/state trajectory",
        "score": "trajectory max of scaled L_inf(image-path minus q-path next-state residual)",
        "horizon": args.horizon,
        "epsilon": args.epsilon,
        "beta": args.beta,
        "state_scales": {
            "distance_m": args.distance_residual_scale_m,
            "speed_mps": args.speed_residual_scale_mps,
        },
        "dynamics": "d+=d-v*0.05; v+=clip(v-action*0.05,0,3)",
        "unsafe_set": "distance_m <= 6 and speed_mps > 0.5",
        "representation_frozen": True,
        "controller_frozen": True,
        "representation_sha256": sha256(args.representation),
        "controller_sha256": sha256(controller_path),
        "calibration_h5_sha256": sha256(args.calibration_h5),
        "training_cache_manifest_sha256": sha256(args.training_cache_manifest),
        "test_used": False,
        "claim_boundary": (
            "trajectory-level tolerance coverage for the registered calibration distribution; "
            "not yet a safety certificate until a robust SBC proves safety for every residual "
            "inside the saved physical-state contract"
        ),
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2, allow_nan=False), encoding="utf-8"
    )

    projection = SafetyProjection().eval()
    projection.load_state_dict(representation["projection_state_dict"])
    q_model = PhysicalToLatent().eval()
    q_model.load_state_dict(representation["physical_to_latent_state_dict"])
    controller = PPO.load(controller_path, device="cpu")
    flat_valid = np.flatnonzero(valid.reshape(-1))
    flat_features = features.reshape(-1, 384)
    flat_distance = distance.reshape(-1)
    flat_speed = speed.reshape(-1)
    residual_flat = np.full((features.shape[0] * args.horizon, 2), np.nan)
    torch.set_num_threads(1)
    for start in range(0, len(flat_valid), args.batch_size):
        indices = flat_valid[start:start + args.batch_size]
        feature_batch = torch.from_numpy(flat_features[indices])
        distance_batch = torch.from_numpy(flat_distance[indices, None])
        with torch.no_grad():
            image_latent = projection(
                (feature_batch - representation["feature_mean"])
                / representation["feature_std"]
            ).numpy()
            nominal_latent = q_model(
                distance_batch / float(representation["distance_scale_m"])
            ).numpy()
        speed_batch = flat_speed[indices, None]
        image_observation = np.concatenate([image_latent, speed_batch], axis=1)
        nominal_observation = np.concatenate([nominal_latent, speed_batch], axis=1)
        image_action = controller.predict(image_observation, deterministic=True)[0].reshape(-1)
        nominal_action = controller.predict(nominal_observation, deterministic=True)[0].reshape(-1)
        residual_flat[indices] = next_state_residual(
            flat_distance[indices], flat_speed[indices], image_action, nominal_action
        )
        print(
            "transition residuals %d/%d" %
            (min(start + args.batch_size, len(flat_valid)), len(flat_valid)),
            flush=True,
        )
    residual = residual_flat.reshape(features.shape[0], args.horizon, 2)
    scores = trajectory_scores(
        residual, valid,
        [args.distance_residual_scale_m, args.speed_residual_scale_mps],
    )
    conformal = conformal_summary(scores, args.epsilon, args.beta)
    q_hat = conformal["q_hat"]
    contract = {
        **config,
        "conformal": conformal,
        "physical_residual_box": {
            # With the registered AEBS transition, the current action changes
            # next speed only; both paths have exactly the same next distance.
            "distance_m": [0.0, 0.0],
            "speed_mps": [-q_hat * args.speed_residual_scale_mps,
                          q_hat * args.speed_residual_scale_mps],
        },
        "verifier_requirement": (
            "prove the SBC conditions for every state in the registered domain and every "
            "next-state residual in physical_residual_box"
        ),
    }
    metrics = {
        **config,
        "trajectory_count": int(len(scores)),
        "valid_step_count": int(valid.sum()),
        "score_summary": {
            "mean": float(scores.mean()),
            "median": float(np.median(scores)),
            "p95": float(np.quantile(scores, 0.95)),
            "max": float(scores.max()),
        },
        "conformal": conformal,
        "contract_path": str(args.output_dir / "contract.json"),
    }
    (args.output_dir / "contract.json").write_text(
        json.dumps(contract, indent=2, allow_nan=False), encoding="utf-8"
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / "calibration_scores.npz",
        trajectory_id=np.asarray(trajectory_ids), scores=scores,
    )
    print("[R2 trajectory-level physical next-state residual contract — no test/SBC]")
    print("trajectories=%d q_hat=%.9f confidence=%.6f" % (
        len(scores), q_hat, conformal["achieved_confidence"]
    ))
    print("contract:", args.output_dir / "contract.json")


if __name__ == "__main__":
    main()
