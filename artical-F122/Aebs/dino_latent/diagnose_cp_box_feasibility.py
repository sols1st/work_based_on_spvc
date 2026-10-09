"""Find a decisive counterexample to the global CP residual-box SBC.

This is a diagnosis, not training and not a certificate. The robust SBC
specification permits every residual in [-q_hat, q_hat] at every step. A
single unsafe rollout from the registered initial set using one such allowed
sequence proves that the requested initial/unsafe/nonincrease conditions
cannot all hold for that global box.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from Aebs.dino_latent.models import PhysicalToLatent
from Aebs.dino_latent.prepare_expanded_data import sha256
from Aebs.dino_latent.train_cp_robust_sbc import (
    INITIAL_DISTANCE_M,
    INITIAL_SPEED_MPS,
    load_contract,
    policy_actions,
    safe_terminal_mask,
    successor_scenarios,
    unsafe_mask,
)


def classify(states):
    labels = np.full(len(states), "running", dtype=object)
    labels[safe_terminal_mask(states)] = "safe_terminal"
    labels[unsafe_mask(states)] = "unsafe"
    return labels


def initial_grid(size):
    distance = np.linspace(*INITIAL_DISTANCE_M, size, dtype=np.float32)
    speed = np.linspace(*INITIAL_SPEED_MPS, size, dtype=np.float32)
    mesh_d, mesh_v = np.meshgrid(distance, speed)
    return np.stack((mesh_d.reshape(-1), mesh_v.reshape(-1)), axis=1)


def rollout_mode(controller, q_model, distance_scale_m, starts, residual, horizon):
    states = starts.copy()
    labels = np.full(len(states), "running", dtype=object)
    finished_step = np.full(len(states), -1, dtype=np.int32)
    first_counterexample = None
    for step in range(horizon):
        active = np.flatnonzero(labels == "running")
        if not len(active):
            break
        actions = policy_actions(controller, q_model, distance_scale_m, states[active])
        next_states = successor_scenarios(states[active], actions, [residual])[:, 0]
        states[active] = next_states
        next_labels = classify(next_states)
        done = next_labels != "running"
        labels[active[done]] = next_labels[done]
        finished_step[active[done]] = step + 1
        if first_counterexample is None and np.any(next_labels == "unsafe"):
            local = int(np.flatnonzero(next_labels == "unsafe")[0])
            global_index = int(active[local])
            first_counterexample = {
                "initial_state": starts[global_index].astype(float).tolist(),
                "unsafe_state": next_states[local].astype(float).tolist(),
                "unsafe_at_step": int(step + 1),
                "action_before_unsafe": float(actions[local]),
                "residual_each_step": float(residual),
            }
    labels[labels == "running"] = "timeout"
    return {
        "outcomes": {
            name: int(np.sum(labels == name))
            for name in ("safe_terminal", "unsafe", "timeout")
        },
        "mean_finished_step": (
            float(finished_step[finished_step >= 0].mean())
            if np.any(finished_step >= 0) else None
        ),
        "first_counterexample": first_counterexample,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract", type=Path,
        default=Path("results/dino_expanded_v1/14_transition_contract_spvc_v1/contract.json"),
    )
    parser.add_argument(
        "--calibration-h5", type=Path,
        default=Path("results/dino_expanded_v1/13_spvc_calibration_trajectories_v1/trajectories.h5"),
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
        default=Path("results/dino_expanded_v1/15b_cp_box_feasibility"),
    )
    parser.add_argument("--initial-grid-size", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=400)
    args = parser.parse_args()
    if args.output_dir.exists() or min(args.initial_grid_size, args.horizon) < 1:
        parser.error("use a new output directory and positive settings")

    controller_path = args.ppo_dir / "latent_ppo.zip"
    required = [args.contract, args.calibration_h5, args.representation, controller_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error("missing inputs: " + ", ".join(missing))
    contract, speed_low, speed_high = load_contract(
        args.contract, args.representation, controller_path, args.calibration_h5
    )
    representation = torch.load(args.representation, map_location="cpu")
    q_model = PhysicalToLatent().eval()
    q_model.load_state_dict(representation["physical_to_latent_state_dict"])
    controller = PPO.load(controller_path, device="cpu")
    starts = initial_grid(args.initial_grid_size)
    modes = {
        "negative_q_hat_each_step": speed_low,
        "zero_residual": 0.0,
        "positive_q_hat_each_step": speed_high,
    }
    results = {
        name: rollout_mode(
            controller, q_model, float(representation["distance_scale_m"]),
            starts, residual, args.horizon,
        )
        for name, residual in modes.items()
    }
    decisive = results["positive_q_hat_each_step"]["outcomes"]["unsafe"] > 0
    status = (
        "global_box_certificate_infeasible_counterexample"
        if decisive else "no_simple_counterexample_not_a_feasibility_proof"
    )
    metrics = {
        "experiment": "R2_global_CP_box_feasibility_diagnostic_v1",
        "level": "diagnostic; one counterexample is decisive, no counterexample is not proof",
        "status": status,
        "reason": (
            "An allowed residual sequence takes a registered initial state to unsafe; "
            "therefore initial-low, unsafe-high and global nonincrease SBC conditions "
            "cannot all hold for this residual box."
            if decisive else
            "The three simple constant residual sequences did not disprove feasibility."
        ),
        "initial_states": int(len(starts)),
        "horizon": args.horizon,
        "q_hat": float(contract["conformal"]["q_hat"]),
        "residual_box_speed_mps": [speed_low, speed_high],
        "modes": results,
        "provenance": {
            "contract_sha256": sha256(args.contract),
            "representation_sha256": sha256(args.representation),
            "controller_sha256": sha256(controller_path),
        },
        "test_used": False,
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8"
    )
    print("[R2 global CP residual-box feasibility diagnostic]")
    print("status:", status)
    print("initial states=%d horizon=%d q_hat=%.9f" % (
        len(starts), args.horizon, float(contract["conformal"]["q_hat"])
    ))
    for name, result in results.items():
        print("%s: %s" % (name, result["outcomes"]))
    print("counterexample:", results["positive_q_hat_each_step"]["first_counterexample"])
    print("metrics:", args.output_dir / "metrics.json")


if __name__ == "__main__":
    main()
