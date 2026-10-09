"""Train an SBC on observed closed-loop trajectory transition scenarios.

The already inspected 459 trajectories are development data here.  They are
split by complete trajectory into training and validation subsets.  This
script does not claim a conformal guarantee: after the barrier is frozen, a
new independent trajectory set is required to calibrate its violation score.
"""

import argparse
import json
import random
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from Aebs.dino_latent.calibrate_transition_contract import SCHEMA
from Aebs.dino_latent.prepare_expanded_data import sha256
from Aebs.dino_latent.train_cp_robust_sbc import (
    INITIAL_DISTANCE_M,
    INITIAL_SPEED_MPS,
    ScenarioBarrier,
    grid_states,
    safe_terminal_mask,
    unsafe_mask,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_transitions(distance, speed, valid, trajectory_indices):
    current_parts = []
    next_parts = []
    trajectory_parts = []
    unsafe_next_count = 0
    for trajectory_index in trajectory_indices:
        length = int(valid[trajectory_index].sum())
        if length < 2:
            continue
        states = np.stack(
            (distance[trajectory_index, :length], speed[trajectory_index, :length]), axis=1
        ).astype(np.float32)
        current = states[:-1]
        successor = states[1:]
        keep = ~(safe_terminal_mask(current) | unsafe_mask(current))
        current = current[keep]
        successor = successor[keep]
        unsafe_next_count += int(unsafe_mask(successor).sum())
        current_parts.append(current)
        next_parts.append(successor)
        trajectory_parts.append(
            np.full(len(current), trajectory_index, dtype=np.int32)
        )
    if not current_parts:
        raise ValueError("no usable trajectory transitions")
    return (
        np.concatenate(current_parts),
        np.concatenate(next_parts),
        np.concatenate(trajectory_parts),
        unsafe_next_count,
    )


def sample_regions(rng, count):
    initial = np.stack((
        rng.uniform(*INITIAL_DISTANCE_M, count),
        rng.uniform(*INITIAL_SPEED_MPS, count),
    ), axis=1).astype(np.float32)
    # Uniform unsafe sampling almost never touches the open boundary v>0.5,
    # although that is where a continuous neural classifier is hardest to
    # separate. Reserve half of the samples for fixed near-boundary speeds.
    boundary_count = count // 2
    uniform_count = count - boundary_count
    boundary_speeds = np.resize(
        np.asarray([0.5001, 0.501, 0.505, 0.51], dtype=np.float32),
        boundary_count,
    )
    unsafe = np.concatenate((
        np.stack((
            np.linspace(5.0, 6.0, boundary_count, dtype=np.float32),
            boundary_speeds,
        ), axis=1),
        np.stack((
            rng.uniform(5.0, 6.0, uniform_count),
            rng.uniform(0.5001, 3.0, uniform_count),
        ), axis=1),
    ), axis=0).astype(np.float32)
    rng.shuffle(unsafe)
    return initial, unsafe


def values(barrier, states, device, batch_size):
    result = []
    with torch.no_grad():
        for start in range(0, len(states), batch_size):
            result.append(barrier(
                torch.from_numpy(states[start:start + batch_size]).to(device)
            ).cpu().numpy())
    return np.concatenate(result)


def evaluate(barrier, current, successor, device, batch_size, init_target, unsafe_target):
    current_value = values(barrier, current, device, batch_size)
    next_value = values(barrier, successor, device, batch_size)
    next_value[safe_terminal_mask(successor)] = 0.0
    decrease_margin = current_value - next_value
    initial = grid_states(INITIAL_DISTANCE_M, INITIAL_SPEED_MPS, 100)
    unsafe = grid_states((5.0, 6.0), (0.5001, 3.0), 100)
    initial_value = values(barrier, initial, device, batch_size)
    unsafe_value = values(barrier, unsafe, device, batch_size)
    decrease_violations = decrease_margin < -1e-6
    initial_violations = initial_value > init_target + 1e-6
    unsafe_violations = unsafe_value < unsafe_target - 1e-6
    worst = int(np.argmin(decrease_margin))
    return {
        "transition_count": int(len(current)),
        "safe_occupancy_max": float(current_value.max()),
        "safe_occupancy_p95": float(np.quantile(current_value, 0.95)),
        "decrease_violation_count": int(decrease_violations.sum()),
        "decrease_violation_rate": float(decrease_violations.mean()),
        "min_decrease_margin": float(decrease_margin.min()),
        "mean_decrease_margin": float(decrease_margin.mean()),
        "initial_violation_count": int(initial_violations.sum()),
        "initial_max": float(initial_value.max()),
        "initial_target_max": float(init_target),
        "unsafe_violation_count": int(unsafe_violations.sum()),
        "unsafe_min": float(unsafe_value.min()),
        "unsafe_target_min": float(unsafe_target),
        "worst_transition": {
            "state": current[worst].astype(float).tolist(),
            "next_state": successor[worst].astype(float).tolist(),
            "margin": float(decrease_margin[worst]),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory-h5", type=Path,
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
        default=Path("results/dino_expanded_v1/17_trajectory_scenario_sbc_v1"),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--region-samples", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--init-target", type=float, default=0.01)
    parser.add_argument("--unsafe-target", type=float, default=1.0)
    parser.add_argument("--decrease-weight", type=float, default=500.0)
    parser.add_argument("--safe-occupancy-weight", type=float, default=50.0)
    parser.add_argument("--region-weight", type=float, default=200.0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if (
        args.output_dir.exists()
        or not 0.0 < args.train_fraction < 1.0
        or min(args.epochs, args.batch_size, args.region_samples, args.log_every) < 1
        or args.learning_rate <= 0.0
        or min(args.decrease_weight, args.safe_occupancy_weight, args.region_weight) <= 0.0
        or args.init_target < 0.0
        or args.unsafe_target <= args.init_target
    ):
        parser.error("use a new output directory and valid settings")
    controller_path = args.ppo_dir / "latent_ppo.zip"
    required = [args.trajectory_h5, args.representation, controller_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error("missing inputs: " + ", ".join(missing))

    with h5py.File(args.trajectory_h5, "r") as stream:
        attrs = dict(stream.attrs)
        if (
            attrs.get("schema") != SCHEMA
            or not bool(attrs.get("complete", False))
            or attrs.get("split_role") != "calibration"
        ):
            parser.error("trajectory H5 schema/completion check failed")
        distance = np.asarray(stream["distance_m"], dtype=np.float32)
        speed = np.asarray(stream["speed_mps"], dtype=np.float32)
        valid = np.asarray(stream["valid"], dtype=bool)
        trajectory_ids = stream["trajectory_id"].asstr()[:].tolist()
    if distance.shape != speed.shape or valid.shape != distance.shape:
        parser.error("invalid trajectory state arrays")

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(distance))
    train_count = int(np.floor(len(indices) * args.train_fraction))
    train_indices = np.sort(indices[:train_count])
    validation_indices = np.sort(indices[train_count:])
    train_current, train_next, _, train_unsafe_next = extract_transitions(
        distance, speed, valid, train_indices
    )
    validation_current, validation_next, _, validation_unsafe_next = extract_transitions(
        distance, speed, valid, validation_indices
    )
    if train_unsafe_next or validation_unsafe_next:
        parser.error(
            "observed trajectory data contain unsafe successors; the frozen controller "
            "itself must be fixed before fitting an SBC"
        )
    initial_samples, unsafe_samples = sample_regions(rng, args.region_samples)
    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    barrier = ScenarioBarrier().to(device)
    optimizer = torch.optim.Adam(barrier.parameters(), lr=args.learning_rate)
    order = np.arange(len(train_current))
    history = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(order)
        totals = []
        decreases = []
        initial_losses = []
        unsafe_losses = []
        safe_occupancy_losses = []
        violation_rates = []
        for start in range(0, len(order), args.batch_size):
            selected = order[start:start + args.batch_size]
            current = torch.from_numpy(train_current[selected]).to(device)
            successor_np = train_next[selected]
            successor = torch.from_numpy(successor_np).to(device)
            initial = torch.from_numpy(initial_samples[
                rng.integers(0, len(initial_samples), len(selected))
            ]).to(device)
            unsafe = torch.from_numpy(unsafe_samples[
                rng.integers(0, len(unsafe_samples), len(selected))
            ]).to(device)
            current_value = barrier(current)
            next_value = barrier(successor)
            terminal = torch.from_numpy(safe_terminal_mask(successor_np)).to(device)
            next_value = torch.where(terminal, torch.zeros_like(next_value), next_value)
            violation = F.relu(next_value - current_value)
            decrease_loss = violation.square().mean() + violation.max().square()
            # Every recorded transition belongs to an actually safe development
            # trajectory.  Keeping its occupancy basin close to zero prevents
            # the smooth classifier from rising a little at almost every step
            # merely to reach the adjacent synthetic unsafe region.
            nonterminal_next = barrier(successor[~terminal])
            safe_occupancy_loss = current_value.square().mean()
            if len(nonterminal_next):
                safe_occupancy_loss = (
                    safe_occupancy_loss + nonterminal_next.square().mean()
                )
            # Train with a small interior margin; evaluation still uses the
            # registered targets. This avoids declaring success from values
            # that sit exactly on a floating-point threshold.
            initial_train_target = args.init_target * 0.5
            unsafe_train_target = args.unsafe_target + 0.20
            initial_loss = F.relu(
                barrier(initial) - initial_train_target
            ).square().mean()
            unsafe_loss = F.relu(
                unsafe_train_target - barrier(unsafe)
            ).square().mean()
            loss = (
                args.decrease_weight * decrease_loss
                + args.safe_occupancy_weight * safe_occupancy_loss
                + args.region_weight * (initial_loss + unsafe_loss)
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(barrier.parameters(), 5.0)
            optimizer.step()
            totals.append(float(loss.detach().cpu()))
            decreases.append(float(decrease_loss.detach().cpu()))
            initial_losses.append(float(initial_loss.detach().cpu()))
            unsafe_losses.append(float(unsafe_loss.detach().cpu()))
            safe_occupancy_losses.append(float(safe_occupancy_loss.detach().cpu()))
            violation_rates.append(float((violation > 1e-6).float().mean().cpu()))
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            record = {
                "epoch": epoch,
                "loss": float(np.mean(totals)),
                "decrease_loss": float(np.mean(decreases)),
                "initial_loss": float(np.mean(initial_losses)),
                "unsafe_loss": float(np.mean(unsafe_losses)),
                "safe_occupancy_loss": float(np.mean(safe_occupancy_losses)),
                "train_decrease_violation_rate": float(np.mean(violation_rates)),
            }
            history.append(record)
            print(json.dumps(record), flush=True)

    barrier.eval()
    train_evaluation = evaluate(
        barrier, train_current, train_next, device, args.batch_size,
        args.init_target, args.unsafe_target,
    )
    validation_evaluation = evaluate(
        barrier, validation_current, validation_next, device, args.batch_size,
        args.init_target, args.unsafe_target,
    )
    validation_pass = (
        validation_evaluation["decrease_violation_count"] == 0
        and validation_evaluation["initial_violation_count"] == 0
        and validation_evaluation["unsafe_violation_count"] == 0
    )
    status = (
        "development_pass_ready_for_fresh_cp_collection"
        if validation_pass else "development_constraints_violated"
    )
    args.output_dir.mkdir(parents=True)
    checkpoint = {
        "experiment": "R2_trajectory_scenario_SBC_boundary_focused_v3",
        "state_dict": barrier.state_dict(),
        "input_units": ["distance_m", "speed_mps"],
        "trajectory_h5_sha256": sha256(args.trajectory_h5),
        "representation_sha256": sha256(args.representation),
        "controller_sha256": sha256(controller_path),
        "development_only": True,
    }
    torch.save(checkpoint, args.output_dir / "barrier.pt")
    metrics = {
        "experiment": checkpoint["experiment"],
        "status": status,
        "claim_level": (
            "development scenario train/validation only; a new independent trajectory "
            "set is required for conformal calibration"
        ),
        "trajectory_count": int(len(distance)),
        "train_trajectory_count": int(len(train_indices)),
        "validation_trajectory_count": int(len(validation_indices)),
        "train_trajectory_ids": [trajectory_ids[i] for i in train_indices],
        "validation_trajectory_ids": [trajectory_ids[i] for i in validation_indices],
        "training": {
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "region_samples": args.region_samples,
            "learning_rate": args.learning_rate,
            "init_target": args.init_target,
            "internal_init_train_target": args.init_target * 0.5,
            "unsafe_target": args.unsafe_target,
            "internal_unsafe_train_target": args.unsafe_target + 0.20,
            "decrease_weight": args.decrease_weight,
            "safe_occupancy_weight": args.safe_occupancy_weight,
            "region_weight": args.region_weight,
        },
        "train": train_evaluation,
        "validation": validation_evaluation,
        "history": history,
        "runtime_seconds": float(time.time() - started),
        "provenance": {
            "trajectory_h5_sha256": sha256(args.trajectory_h5),
            "representation_sha256": sha256(args.representation),
            "controller_sha256": sha256(controller_path),
        },
        "test_used": False,
        "checkpoint": str(args.output_dir / "barrier.pt"),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8"
    )
    print("\n[R2 trajectory-scenario SBC development]")
    print("status:", status)
    print("trajectories: train=%d validation=%d" % (
        len(train_indices), len(validation_indices)
    ))
    print("train:", train_evaluation)
    print("validation:", validation_evaluation)
    print("checkpoint:", args.output_dir / "barrier.pt")
    print("metrics:", args.output_dir / "metrics.json")


if __name__ == "__main__":
    main()
