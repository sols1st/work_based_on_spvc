"""Develop an action-conditional residual envelope from used calibration data.

This is deliberately an exploratory diagnosis.  It uses the already inspected
459 trajectories to ask whether the global q_hat failed because residuals from
low-braking states were incorrectly transferred to maximum-braking states.
Its envelope is not a new CP contract.  If promising, the fixed binning and
score must be calibrated on newly collected independent trajectories.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from stable_baselines3 import PPO

from Aebs.dino_latent.calibrate_transition_contract import (
    SCHEMA,
    next_state_residual,
    validate_calibration_arrays,
)
from Aebs.dino_latent.diagnose_cp_box_feasibility import classify, initial_grid
from Aebs.dino_latent.models import PhysicalToLatent, SafetyProjection
from Aebs.dino_latent.prepare_expanded_data import sha256
from Aebs.dino_latent.train_cp_robust_sbc import load_contract, policy_actions


ACTION_LOW = -3.0
ACTION_HIGH = 3.0
DT = 0.05


def action_bin_indices(actions, edges):
    return np.clip(np.searchsorted(edges, actions, side="right") - 1, 0, len(edges) - 2)


def fit_empirical_envelope(actions, residual_speed, trajectory_index, edges, global_radius):
    indices = action_bin_indices(actions, edges)
    rows = []
    lower = np.empty(len(edges) - 1, dtype=np.float64)
    upper = np.empty(len(edges) - 1, dtype=np.float64)
    for index in range(len(edges) - 1):
        selected = indices == index
        values = residual_speed[selected]
        if len(values):
            lower[index] = min(0.0, float(values.min()))
            upper[index] = max(0.0, float(values.max()))
            trajectories = int(len(np.unique(trajectory_index[selected])))
            fallback = False
        else:
            lower[index] = -global_radius
            upper[index] = global_radius
            trajectories = 0
            fallback = True
        rows.append({
            "action_low": float(edges[index]),
            "action_high": float(edges[index + 1]),
            "steps": int(len(values)),
            "trajectories": trajectories,
            "residual_low_mps": float(lower[index]),
            "residual_high_mps": float(upper[index]),
            "used_global_fallback": fallback,
        })
    return lower, upper, rows


def conditional_rollout(controller, q_model, distance_scale_m, starts, edges, upper, horizon):
    states = starts.copy()
    labels = np.full(len(states), "running", dtype=object)
    first_counterexample = None
    residual_trace_max = 0.0
    for step in range(horizon):
        active = np.flatnonzero(labels == "running")
        if not len(active):
            break
        actions = policy_actions(controller, q_model, distance_scale_m, states[active])
        bins = action_bin_indices(actions, edges)
        residual = upper[bins]
        residual_trace_max = max(residual_trace_max, float(residual.max()))
        next_distance = states[active, 0] - states[active, 1] * DT
        nominal_speed = np.clip(states[active, 1] - actions * DT, 0.0, 3.0)
        next_speed = np.clip(nominal_speed + residual, 0.0, 3.0)
        next_states = np.stack((next_distance, next_speed), axis=1).astype(np.float32)
        states[active] = next_states
        next_labels = classify(next_states)
        done = next_labels != "running"
        labels[active[done]] = next_labels[done]
        if first_counterexample is None and np.any(next_labels == "unsafe"):
            local = int(np.flatnonzero(next_labels == "unsafe")[0])
            global_index = int(active[local])
            first_counterexample = {
                "initial_state": starts[global_index].astype(float).tolist(),
                "unsafe_state": next_states[local].astype(float).tolist(),
                "unsafe_at_step": int(step + 1),
                "nominal_action": float(actions[local]),
                "conditional_positive_residual": float(residual[local]),
                "action_bin": int(bins[local]),
            }
    labels[labels == "running"] = "timeout"
    return {
        "outcomes": {
            name: int(np.sum(labels == name))
            for name in ("safe_terminal", "unsafe", "timeout")
        },
        "first_counterexample": first_counterexample,
        "largest_residual_used_mps": residual_trace_max,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration-h5", type=Path,
        default=Path("results/dino_expanded_v1/13_spvc_calibration_trajectories_v1/trajectories.h5"),
    )
    parser.add_argument(
        "--contract", type=Path,
        default=Path("results/dino_expanded_v1/14_transition_contract_spvc_v1/contract.json"),
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
        default=Path("results/dino_expanded_v1/16_action_conditional_diagnostic"),
    )
    parser.add_argument("--action-bins", type=int, default=12)
    parser.add_argument("--initial-grid-size", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()
    if (
        args.output_dir.exists()
        or min(args.action_bins, args.initial_grid_size, args.horizon, args.batch_size) < 1
    ):
        parser.error("use a new output directory and positive settings")
    controller_path = args.ppo_dir / "latent_ppo.zip"
    required = [args.calibration_h5, args.contract, args.representation, controller_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error("missing inputs: " + ", ".join(missing))
    contract, _, global_high = load_contract(
        args.contract, args.representation, controller_path, args.calibration_h5
    )
    representation = torch.load(args.representation, map_location="cpu")
    with h5py.File(args.calibration_h5, "r") as stream:
        if stream.attrs.get("schema") != SCHEMA:
            parser.error("unexpected calibration schema")
        features = np.asarray(stream["dino_features"], dtype=np.float32)
        distance = np.asarray(stream["distance_m"], dtype=np.float32)
        speed = np.asarray(stream["speed_mps"], dtype=np.float32)
        valid = np.asarray(stream["valid"], dtype=bool)
        trajectory_ids = stream["trajectory_id"].asstr()[:].tolist()
    validate_calibration_arrays(
        features, distance, speed, valid, trajectory_ids, features.shape[1]
    )

    projection = SafetyProjection().eval()
    projection.load_state_dict(representation["projection_state_dict"])
    q_model = PhysicalToLatent().eval()
    q_model.load_state_dict(representation["physical_to_latent_state_dict"])
    controller = PPO.load(controller_path, device="cpu")
    flat_valid = np.flatnonzero(valid.reshape(-1))
    flat_features = features.reshape(-1, features.shape[-1])
    flat_distance = distance.reshape(-1)
    flat_speed = speed.reshape(-1)
    flat_trajectory = np.repeat(np.arange(len(features)), features.shape[1])
    nominal_actions = []
    residual_speed = []
    selected_trajectory = []
    for start in range(0, len(flat_valid), args.batch_size):
        indices = flat_valid[start:start + args.batch_size]
        with torch.no_grad():
            image_latent = projection(
                (torch.from_numpy(flat_features[indices]) - representation["feature_mean"])
                / representation["feature_std"]
            ).numpy()
            nominal_latent = q_model(
                torch.from_numpy(flat_distance[indices, None])
                / float(representation["distance_scale_m"])
            ).numpy()
        speeds = flat_speed[indices, None]
        image_actions = controller.predict(
            np.concatenate((image_latent, speeds), axis=1), deterministic=True
        )[0].reshape(-1)
        nominal = controller.predict(
            np.concatenate((nominal_latent, speeds), axis=1), deterministic=True
        )[0].reshape(-1)
        residual = next_state_residual(
            flat_distance[indices], flat_speed[indices], image_actions, nominal
        )[:, 1]
        nominal_actions.append(nominal)
        residual_speed.append(residual)
        selected_trajectory.append(flat_trajectory[indices])
        print("residual analysis %d/%d" % (
            min(start + args.batch_size, len(flat_valid)), len(flat_valid)
        ), flush=True)
    nominal_actions = np.concatenate(nominal_actions)
    residual_speed = np.concatenate(residual_speed)
    selected_trajectory = np.concatenate(selected_trajectory)
    edges = np.linspace(ACTION_LOW, ACTION_HIGH, args.action_bins + 1, dtype=np.float64)
    edges[-1] += 1e-6
    lower, upper, rows = fit_empirical_envelope(
        nominal_actions, residual_speed, selected_trajectory, edges, global_high
    )
    rollout = conditional_rollout(
        controller, q_model, float(representation["distance_scale_m"]),
        initial_grid(args.initial_grid_size), edges, upper, args.horizon,
    )
    promising = rollout["outcomes"]["unsafe"] == 0 and rollout["outcomes"]["timeout"] == 0
    metrics = {
        "experiment": "R2_action_conditional_residual_development_diagnostic_v1",
        "status": (
            "promising_freeze_design_then_collect_fresh_calibration"
            if promising else "action_only_conditioning_still_infeasible"
        ),
        "claim_level": (
            "exploratory only; the inspected calibration data were reused to design the "
            "envelope, so this is not a valid new conformal contract"
        ),
        "global_q_hat": float(contract["conformal"]["q_hat"]),
        "valid_steps": int(len(residual_speed)),
        "trajectory_count": int(len(features)),
        "action_edges": edges.astype(float).tolist(),
        "empirical_bins": rows,
        "conditional_positive_rollout": rollout,
        "next_required_step": (
            "Freeze these bins and the normalized score, then collect new independent "
            "trajectories for formal calibration."
            if promising else
            "Add preregistered distance/speed conditioning before collecting fresh calibration."
        ),
        "provenance": {
            "calibration_h5_sha256": sha256(args.calibration_h5),
            "representation_sha256": sha256(args.representation),
            "controller_sha256": sha256(controller_path),
        },
        "test_used": False,
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8"
    )
    print("\n[R2 action-conditional residual development diagnostic]")
    print("status:", metrics["status"])
    print("steps=%d trajectories=%d global_q_hat=%.9f" % (
        len(residual_speed), len(features), float(contract["conformal"]["q_hat"])
    ))
    for index, row in enumerate(rows):
        print("bin %02d action=[%.3f,%.3f): steps=%d trajectories=%d residual=[%.9f,%.9f]%s" % (
            index, row["action_low"], row["action_high"], row["steps"],
            row["trajectories"], row["residual_low_mps"], row["residual_high_mps"],
            " fallback" if row["used_global_fallback"] else "",
        ))
    print("conditional positive rollout:", rollout)
    print("metrics:", args.output_dir / "metrics.json")


if __name__ == "__main__":
    main()
