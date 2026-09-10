"""Minimal robust SBC trainer and fixed-grid verifier for AEBS MVP."""

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO

from Aebs.VT.utils import MLP
from Aebs.semantic.robust_controller import load_contract
from Aebs.semantic.safety_filter import load_controller
from Aebs.uncertainty.model import StateDependentUncertainty


DISTURBANCE_LABELS = (
    "mean",
    "support_low",
    "support_high",
    "low_distance_high_speed",
    "high_distance_low_speed",
)
SEMANTIC_MULTIPLIERS = (-1.0, 0.0, 1.0)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def controller_actions(controller: PPO, states: np.ndarray) -> np.ndarray:
    actions, _ = controller.predict(states.astype(np.float32), deterministic=True)
    return np.asarray(actions, dtype=np.float32).reshape(-1)


def next_state(states: np.ndarray, actions: np.ndarray, distance_scale: float) -> np.ndarray:
    distance_m = states[:, 0] * distance_scale
    speed = states[:, 1]
    next_distance_m = distance_m - speed * 0.05
    next_speed = np.clip(speed - actions * 0.05, 0.0, 3.0)
    return np.stack((next_distance_m / distance_scale, next_speed), axis=1)


def sample_box(low: Tuple[float, float], high: Tuple[float, float], count: int, rng: np.random.Generator) -> np.ndarray:
    low_array = np.asarray(low, dtype=np.float64)
    high_array = np.asarray(high, dtype=np.float64)
    return rng.uniform(low_array, high_array, size=(count, 2)).astype(np.float32)


def sample_hard_case(
    distance_scale: float,
    distance_m: float,
    speed: float,
    distance_radius_m: float,
    speed_radius: float,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if count <= 0:
        return np.empty((0, 2), dtype=np.float32)
    distance = rng.uniform(distance_m - distance_radius_m, distance_m + distance_radius_m, size=count)
    velocity = rng.uniform(speed - speed_radius, speed + speed_radius, size=count)
    distance = np.clip(distance, 5.0, 16.0)
    velocity = np.clip(velocity, 0.0, 3.0)
    velocity[0] = np.clip(speed, 0.0, 3.0)
    distance[0] = np.clip(distance_m, 5.0, 16.0)
    return np.stack((distance / distance_scale, velocity), axis=1).astype(np.float32)


def verifier_grid_states(distance_scale: float, grid_size: int) -> np.ndarray:
    distances = np.linspace(5.0 / distance_scale, 16.0 / distance_scale, grid_size)
    speeds = np.linspace(0.0, 3.0, grid_size)
    mesh_d, mesh_v = np.meshgrid(distances, speeds)
    return np.stack((mesh_d.reshape(-1), mesh_v.reshape(-1)), axis=1).astype(np.float32)


def region_grid_states(
    distance_scale: float,
    distance_low_m: float,
    distance_high_m: float,
    speed_low: float,
    speed_high: float,
    grid_size: int,
) -> np.ndarray:
    distances = np.linspace(distance_low_m / distance_scale, distance_high_m / distance_scale, grid_size)
    speeds = np.linspace(speed_low, speed_high, grid_size)
    mesh_d, mesh_v = np.meshgrid(distances, speeds)
    return np.stack((mesh_d.reshape(-1), mesh_v.reshape(-1)), axis=1).astype(np.float32)


def certificate_masks(
    states: np.ndarray,
    distance_scale: float,
    terminal_speed_threshold: float,
    goal_distance_m: float,
    goal_speed: float,
    unsafe_distance_low_m: float,
    unsafe_distance_high_m: float,
    unsafe_speed: float,
) -> Dict[str, np.ndarray]:
    distance_m = states[:, 0] * distance_scale
    speed = states[:, 1]
    stopped = speed <= terminal_speed_threshold
    goal = (distance_m <= goal_distance_m) & (speed <= goal_speed)
    unsafe = (
        (distance_m >= unsafe_distance_low_m)
        & (distance_m <= unsafe_distance_high_m)
        & (speed >= unsafe_speed)
    )
    terminal = stopped | goal
    return {
        "terminal": terminal,
        "goal": goal,
        "unsafe": unsafe,
        "decrease": ~(terminal | unsafe),
    }


def top_fraction_mean(values: torch.Tensor, fraction: float) -> torch.Tensor:
    count = max(1, int(math.ceil(len(values) * fraction)))
    return torch.topk(values, k=count).values.mean()


def robust_successors(
    states: np.ndarray,
    controller: PPO,
    uncertainty: StateDependentUncertainty,
    distance_scale: float,
    semantic_radii: np.ndarray,
) -> np.ndarray:
    candidates = []
    for semantic_multiplier in SEMANTIC_MULTIPLIERS:
        semantic_states = states.copy()
        semantic_states[:, 0] = semantic_states[:, 0] + semantic_multiplier * semantic_radii
        semantic_states[:, 0] = np.clip(semantic_states[:, 0], 5.0 / distance_scale, 16.0 / distance_scale)
        actions = controller_actions(controller, semantic_states)
        base_next = next_state(states, actions, distance_scale)
        box = uncertainty.lookup(states, distance_scale)
        disturbance_points = [
            box["mean"],
            box["support_low"],
            box["support_high"],
            np.stack((box["support_low"][:, 0], box["support_high"][:, 1]), axis=1),
            np.stack((box["support_high"][:, 0], box["support_low"][:, 1]), axis=1),
        ]
        for disturbance in disturbance_points:
            perturbed = base_next + disturbance
            perturbed[:, 0] = np.clip(perturbed[:, 0], 5.0 / distance_scale, 16.0 / distance_scale)
            perturbed[:, 1] = np.clip(perturbed[:, 1], 0.0, 3.0)
            candidates.append(perturbed)
    return np.stack(candidates, axis=1).astype(np.float32)


def candidate_label(index: int) -> Dict[str, object]:
    semantic_index = index // len(DISTURBANCE_LABELS)
    disturbance_index = index % len(DISTURBANCE_LABELS)
    return {
        "semantic_multiplier": SEMANTIC_MULTIPLIERS[semantic_index],
        "disturbance": DISTURBANCE_LABELS[disturbance_index],
    }


def evaluate_grid(
    barrier: MLP,
    controller: PPO,
    uncertainty: StateDependentUncertainty,
    semantic_checkpoint: str,
    data_path: str,
    grid_size: int,
    batch_size: int,
    epsilon: float,
    terminal_speed_threshold: float,
    init_target: float,
    unsafe_target: float,
    goal_target: float,
    goal_distance_m: float,
    goal_speed: float,
    unsafe_distance_low_m: float,
    unsafe_distance_high_m: float,
    unsafe_speed: float,
    device: torch.device,
) -> Dict:
    contract, checkpoint_scale = load_contract(semantic_checkpoint)
    with h5py.File(data_path, "r") as stream:
        distance_scale = float(np.std(np.asarray(stream["y_train"], dtype=np.float32)))
    if not np.isclose(checkpoint_scale, distance_scale):
        raise ValueError("semantic checkpoint and dataset use different distance scaling")
    all_states = verifier_grid_states(distance_scale, grid_size)
    masks = certificate_masks(
        all_states,
        distance_scale,
        terminal_speed_threshold,
        goal_distance_m,
        goal_speed,
        unsafe_distance_low_m,
        unsafe_distance_high_m,
        unsafe_speed,
    )
    states = all_states[masks["decrease"]]
    margins = []
    worst = {
        "margin": float("inf"),
        "state": None,
    }
    barrier.eval()
    with torch.no_grad():
        for start in range(0, len(states), batch_size):
            batch = states[start : start + batch_size]
            distance_m = batch[:, 0] * distance_scale
            radii = contract.radius(distance_m).astype(np.float32)
            successors = robust_successors(batch, controller, uncertainty, distance_scale, radii)
            current = barrier(torch.from_numpy(batch).to(device)).view(-1)
            next_values = barrier(torch.from_numpy(successors.reshape(-1, 2)).to(device)).view(len(batch), -1)
            robust_next, robust_index = next_values.max(dim=1)
            margin = (current - robust_next - epsilon).detach().cpu().numpy()
            margins.append(margin)
            local_index = int(np.argmin(margin))
            if float(margin[local_index]) < worst["margin"]:
                label = candidate_label(int(robust_index[local_index].detach().cpu()))
                worst = {
                    "margin": float(margin[local_index]),
                    "state": batch[local_index].astype(float).tolist(),
                    "state_units": ["normalized_distance", "m_per_s"],
                    "distance_m": float(batch[local_index, 0] * distance_scale),
                    "speed": float(batch[local_index, 1]),
                    "current_barrier": float(current[local_index].detach().cpu()),
                    "robust_next_barrier": float(robust_next[local_index].detach().cpu()),
                    "semantic_multiplier": label["semantic_multiplier"],
                    "disturbance": label["disturbance"],
                    "successor": successors[local_index, int(robust_index[local_index].detach().cpu())].astype(float).tolist(),
                }
    margins = np.concatenate(margins)
    violation_count = int(np.sum(margins < 0.0))
    speed_edges = (terminal_speed_threshold, 0.5, 1.0, 2.0, 3.0)
    speed_bins = []
    for index in range(len(speed_edges) - 1):
        low = speed_edges[index]
        high = speed_edges[index + 1]
        if index == len(speed_edges) - 2:
            mask = (states[:, 1] > low) & (states[:, 1] <= high)
        else:
            mask = (states[:, 1] > low) & (states[:, 1] <= high)
        bin_margins = margins[mask]
        if len(bin_margins) == 0:
            continue
        speed_bins.append(
            {
                "speed_low": float(low),
                "speed_high": float(high),
                "count": int(len(bin_margins)),
                "violation_count": int(np.sum(bin_margins < 0.0)),
                "min_margin": float(np.min(bin_margins)),
                "mean_margin": float(np.mean(bin_margins)),
            }
        )
    init_states = region_grid_states(
        distance_scale, 15.0, 16.0, 2.5, 3.0, grid_size
    )
    unsafe_states = region_grid_states(
        distance_scale,
        unsafe_distance_low_m,
        unsafe_distance_high_m,
        unsafe_speed,
        3.0,
        grid_size,
    )
    goal_states = region_grid_states(
        distance_scale, unsafe_distance_low_m, goal_distance_m, 0.0, goal_speed, grid_size
    )
    with torch.no_grad():
        all_values = barrier(torch.from_numpy(all_states).to(device)).view(-1).cpu().numpy()
        init_values = barrier(torch.from_numpy(init_states).to(device)).view(-1).cpu().numpy()
        unsafe_values = barrier(torch.from_numpy(unsafe_states).to(device)).view(-1).cpu().numpy()
        goal_values = barrier(torch.from_numpy(goal_states).to(device)).view(-1).cpu().numpy()
    nonnegative_violation_count = int(np.sum(all_values < 0.0))
    init_violation_count = int(np.sum(init_values > init_target))
    unsafe_violation_count = int(np.sum(unsafe_values < unsafe_target))
    goal_violation_count = int(np.sum(goal_values > goal_target))
    region_verified = init_violation_count == 0 and unsafe_violation_count == 0 and goal_violation_count == 0
    status = "verified" if violation_count == 0 and region_verified and nonnegative_violation_count == 0 else "violated"
    return {
        "status": status,
        "grid_size": int(grid_size),
        "checked_states": int(len(states)),
        "excluded_terminal_states": int(np.sum(masks["terminal"])),
        "excluded_goal_states": int(np.sum(masks["goal"])),
        "excluded_unsafe_states": int(np.sum(masks["unsafe"])),
        "terminal_speed_threshold": float(terminal_speed_threshold),
        "violation_count": violation_count,
        "min_margin": float(np.min(margins)),
        "mean_margin": float(np.mean(margins)),
        "speed_bins": speed_bins,
        "worst_case": worst,
        "nonnegative": {
            "violation_count": nonnegative_violation_count,
            "minimum": float(np.min(all_values)),
            "target_min": 0.0,
        },
        "regions": {
            "sample_count": int(grid_size * grid_size),
            "init_violation_count": init_violation_count,
            "init_max": float(np.max(init_values)),
            "init_target_max": float(init_target),
            "unsafe_violation_count": unsafe_violation_count,
            "unsafe_min": float(np.min(unsafe_values)),
            "unsafe_target_min": float(unsafe_target),
            "goal_violation_count": goal_violation_count,
            "goal_max": float(np.max(goal_values)),
            "goal_target_max": float(goal_target),
        },
    }


def train_barrier(
    semantic_checkpoint: str,
    uncertainty_path: str,
    controller_path: str,
    data_path: str,
    output_dir: Path,
    seed: int,
    epochs: int,
    train_states: int,
    batch_size: int,
    learning_rate: float,
    epsilon: float,
    grid_size: int,
    decrease_weight: float,
    init_weight: float,
    unsafe_weight: float,
    square_output: bool,
    region_warmup_epochs: int,
    warmup_decrease_weight: float,
    init_target: float,
    unsafe_target: float,
    include_verifier_grid_train: bool,
    max_decrease_weight: float,
    topk_decrease_weight: float,
    topk_decrease_fraction: float,
    initial_barrier: str,
    terminal_speed_threshold: float,
    hard_case_count: int,
    hard_case_distance_m: float,
    hard_case_speed: float,
    hard_case_distance_radius_m: float,
    hard_case_speed_radius: float,
    region_topk_fraction: float,
    goal_distance_m: float,
    goal_speed: float,
    unsafe_distance_low_m: float,
    unsafe_distance_high_m: float,
    unsafe_speed: float,
    goal_weight: float,
    goal_target: float,
) -> Dict:
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    contract, checkpoint_scale = load_contract(semantic_checkpoint)
    with h5py.File(data_path, "r") as stream:
        distance_scale = float(np.std(np.asarray(stream["y_train"], dtype=np.float32)))
    if not np.isclose(checkpoint_scale, distance_scale):
        raise ValueError("semantic checkpoint and dataset use different distance scaling")
    controller = load_controller(controller_path)
    uncertainty = StateDependentUncertainty.from_npz(uncertainty_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    barrier = MLP([2, 16, 8, 1], activation="tanh", square_output=square_output).to(device)
    if initial_barrier:
        if not Path(initial_barrier).exists():
            raise FileNotFoundError(f"initial barrier not found: {initial_barrier}")
        state_dict = torch.load(initial_barrier, map_location=device)
        barrier.load_state_dict(state_dict)
    if not 0.0 < topk_decrease_fraction <= 1.0:
        raise ValueError("topk_decrease_fraction must be in (0, 1]")
    if not 0.0 < region_topk_fraction <= 1.0:
        raise ValueError("region_topk_fraction must be in (0, 1]")
    optimizer = torch.optim.Adam(barrier.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed)
    random_states = sample_box((5.0 / distance_scale, 0.0), (16.0 / distance_scale, 3.0), train_states, rng)
    if include_verifier_grid_train:
        states = np.concatenate((random_states, verifier_grid_states(distance_scale, grid_size)), axis=0)
    else:
        states = random_states
    hard_case_states = sample_hard_case(
        distance_scale,
        hard_case_distance_m,
        hard_case_speed,
        hard_case_distance_radius_m,
        hard_case_speed_radius,
        hard_case_count,
        rng,
    )
    if len(hard_case_states) > 0:
        states = np.concatenate((states, hard_case_states), axis=0)
    train_masks = certificate_masks(
        states,
        distance_scale,
        terminal_speed_threshold,
        goal_distance_m,
        goal_speed,
        unsafe_distance_low_m,
        unsafe_distance_high_m,
        unsafe_speed,
    )
    excluded_train_terminal_states = int(np.sum(train_masks["terminal"]))
    excluded_train_unsafe_states = int(np.sum(train_masks["unsafe"]))
    states = states[train_masks["decrease"]]
    init_states = region_grid_states(distance_scale, 15.0, 16.0, 2.5, 3.0, grid_size)
    unsafe_states = region_grid_states(
        distance_scale,
        unsafe_distance_low_m,
        unsafe_distance_high_m,
        unsafe_speed,
        3.0,
        grid_size,
    )
    goal_states = region_grid_states(
        distance_scale, unsafe_distance_low_m, goal_distance_m, 0.0, goal_speed, grid_size
    )
    history = []
    started = time.time()
    for epoch in range(1, epochs + 1):
        permutation = rng.permutation(len(states))
        epoch_losses = []
        epoch_decrease_losses = []
        epoch_init_losses = []
        epoch_unsafe_losses = []
        epoch_goal_losses = []
        epoch_max_decrease_losses = []
        epoch_topk_decrease_losses = []
        epoch_violations = []
        active_decrease_weight = warmup_decrease_weight if epoch <= region_warmup_epochs else decrease_weight
        for start in range(0, len(states), batch_size):
            batch = states[permutation[start : start + batch_size]]
            distance_m = batch[:, 0] * distance_scale
            radii = contract.radius(distance_m).astype(np.float32)
            successors = robust_successors(batch, controller, uncertainty, distance_scale, radii)
            batch_tensor = torch.from_numpy(batch).to(device)
            successor_tensor = torch.from_numpy(successors.reshape(-1, 2)).to(device)
            current = barrier(batch_tensor).view(-1)
            next_values = barrier(successor_tensor).view(len(batch), -1)
            robust_next = next_values.max(dim=1).values
            per_state_decrease = F.relu(robust_next - current + epsilon)
            decrease_loss = per_state_decrease.mean()
            max_decrease_loss = per_state_decrease.max()
            topk_decrease_loss = top_fraction_mean(per_state_decrease, topk_decrease_fraction)
            init_value = barrier(torch.from_numpy(init_states).to(device)).view(-1)
            unsafe_value = barrier(torch.from_numpy(unsafe_states).to(device)).view(-1)
            goal_value = barrier(torch.from_numpy(goal_states).to(device)).view(-1)
            init_loss = top_fraction_mean(F.relu(init_value - init_target), region_topk_fraction)
            unsafe_loss = top_fraction_mean(F.relu(unsafe_target - unsafe_value), region_topk_fraction)
            goal_loss = top_fraction_mean(F.relu(goal_value - goal_target), region_topk_fraction)
            loss = (
                active_decrease_weight * decrease_loss
                + max_decrease_weight * max_decrease_loss
                + topk_decrease_weight * topk_decrease_loss
                + init_weight * init_loss
                + unsafe_weight * unsafe_loss
                + goal_weight * goal_loss
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(barrier.parameters(), 5.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
            epoch_decrease_losses.append(float(decrease_loss.detach().cpu()))
            epoch_init_losses.append(float(init_loss.detach().cpu()))
            epoch_unsafe_losses.append(float(unsafe_loss.detach().cpu()))
            epoch_goal_losses.append(float(goal_loss.detach().cpu()))
            epoch_max_decrease_losses.append(float(max_decrease_loss.detach().cpu()))
            epoch_topk_decrease_losses.append(float(topk_decrease_loss.detach().cpu()))
            epoch_violations.append(float(torch.mean((per_state_decrease > 0.0).float()).detach().cpu()))
        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            record = {
                "epoch": int(epoch),
                "loss": float(np.mean(epoch_losses)),
                "decrease_loss": float(np.mean(epoch_decrease_losses)),
                "max_decrease_loss": float(np.mean(epoch_max_decrease_losses)),
                "topk_decrease_loss": float(np.mean(epoch_topk_decrease_losses)),
                "init_loss": float(np.mean(epoch_init_losses)),
                "unsafe_loss": float(np.mean(epoch_unsafe_losses)),
                "goal_loss": float(np.mean(epoch_goal_losses)),
                "active_decrease_weight": float(active_decrease_weight),
                "train_robust_decrease_violation_rate": float(np.mean(epoch_violations)),
            }
            history.append(record)
            print(json.dumps(record), flush=True)
    torch.save(barrier.state_dict(), output_dir / "barrier.pt")
    verification = evaluate_grid(
        barrier,
        controller,
        uncertainty,
        semantic_checkpoint,
        data_path,
        grid_size,
        batch_size,
        epsilon,
        terminal_speed_threshold,
        init_target,
        unsafe_target,
        goal_target,
        goal_distance_m,
        goal_speed,
        unsafe_distance_low_m,
        unsafe_distance_high_m,
        unsafe_speed,
        device,
    )
    metrics = {
        "experiment": "robust_sbc_mvp",
        "seed": seed,
        "controller": controller_path,
        "epochs": epochs,
        "train_states": train_states,
        "effective_train_states": int(len(states)),
        "excluded_train_terminal_states": excluded_train_terminal_states,
        "excluded_train_unsafe_states": excluded_train_unsafe_states,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epsilon": epsilon,
        "decrease_weight": decrease_weight,
        "init_weight": init_weight,
        "unsafe_weight": unsafe_weight,
        "square_output": bool(square_output),
        "region_warmup_epochs": int(region_warmup_epochs),
        "warmup_decrease_weight": float(warmup_decrease_weight),
        "init_target": float(init_target),
        "unsafe_target": float(unsafe_target),
        "goal_weight": float(goal_weight),
        "goal_target": float(goal_target),
        "include_verifier_grid_train": bool(include_verifier_grid_train),
        "max_decrease_weight": float(max_decrease_weight),
        "topk_decrease_weight": float(topk_decrease_weight),
        "topk_decrease_fraction": float(topk_decrease_fraction),
        "initial_barrier": initial_barrier or None,
        "terminal_speed_threshold": float(terminal_speed_threshold),
        "region_topk_fraction": float(region_topk_fraction),
        "goal_distance_m": float(goal_distance_m),
        "goal_speed": float(goal_speed),
        "unsafe_distance_low_m": float(unsafe_distance_low_m),
        "unsafe_distance_high_m": float(unsafe_distance_high_m),
        "unsafe_speed": float(unsafe_speed),
        "hard_case_count": int(hard_case_count),
        "hard_case_distance_m": float(hard_case_distance_m),
        "hard_case_speed": float(hard_case_speed),
        "hard_case_distance_radius_m": float(hard_case_distance_radius_m),
        "hard_case_speed_radius": float(hard_case_speed_radius),
        "runtime_seconds": float(time.time() - started),
        "history": history,
        "verification": verification,
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2, ensure_ascii=False)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-checkpoint", required=True)
    parser.add_argument("--uncertainty", required=True)
    parser.add_argument("--controller", required=True)
    parser.add_argument("--data", default="Aebs/data/Downsampled.h5")
    parser.add_argument("--output-dir", default="results/mvp/04_robust_sbc")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--train-states", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--grid-size", type=int, default=80)
    parser.add_argument("--decrease-weight", type=float, default=100.0)
    parser.add_argument("--init-weight", type=float, default=1.0)
    parser.add_argument("--unsafe-weight", type=float, default=1.0)
    parser.add_argument("--square-output", action="store_true")
    parser.add_argument("--region-warmup-epochs", type=int, default=25)
    parser.add_argument("--warmup-decrease-weight", type=float, default=0.0)
    parser.add_argument("--init-target", type=float, default=1.0)
    parser.add_argument("--unsafe-target", type=float, default=10.0)
    parser.add_argument("--include-verifier-grid-train", action="store_true")
    parser.add_argument("--max-decrease-weight", type=float, default=25.0)
    parser.add_argument("--topk-decrease-weight", type=float, default=0.0)
    parser.add_argument("--topk-decrease-fraction", type=float, default=0.1)
    parser.add_argument("--initial-barrier", default="")
    parser.add_argument("--terminal-speed-threshold", type=float, default=0.0)
    parser.add_argument("--hard-case-count", type=int, default=2048)
    parser.add_argument("--hard-case-distance-m", type=float, default=13.35)
    parser.add_argument("--hard-case-speed", type=float, default=3.0)
    parser.add_argument("--hard-case-distance-radius-m", type=float, default=0.8)
    parser.add_argument("--hard-case-speed-radius", type=float, default=0.2)
    parser.add_argument("--region-topk-fraction", type=float, default=0.1)
    parser.add_argument("--goal-distance-m", type=float, default=6.0)
    parser.add_argument("--goal-speed", type=float, default=0.5)
    parser.add_argument("--unsafe-distance-low-m", type=float, default=5.0)
    parser.add_argument("--unsafe-distance-high-m", type=float, default=6.0)
    parser.add_argument("--unsafe-speed", type=float, default=0.5)
    parser.add_argument("--goal-weight", type=float, default=10.0)
    parser.add_argument("--goal-target", type=float, default=1.0)
    args = parser.parse_args()
    metrics = train_barrier(
        args.semantic_checkpoint,
        args.uncertainty,
        args.controller,
        args.data,
        Path(args.output_dir),
        args.seed,
        args.epochs,
        args.train_states,
        args.batch_size,
        args.learning_rate,
        args.epsilon,
        args.grid_size,
        args.decrease_weight,
        args.init_weight,
        args.unsafe_weight,
        args.square_output,
        args.region_warmup_epochs,
        args.warmup_decrease_weight,
        args.init_target,
        args.unsafe_target,
        args.include_verifier_grid_train,
        args.max_decrease_weight,
        args.topk_decrease_weight,
        args.topk_decrease_fraction,
        args.initial_barrier,
        args.terminal_speed_threshold,
        args.hard_case_count,
        args.hard_case_distance_m,
        args.hard_case_speed,
        args.hard_case_distance_radius_m,
        args.hard_case_speed_radius,
        args.region_topk_fraction,
        args.goal_distance_m,
        args.goal_speed,
        args.unsafe_distance_low_m,
        args.unsafe_distance_high_m,
        args.unsafe_speed,
        args.goal_weight,
        args.goal_target,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
