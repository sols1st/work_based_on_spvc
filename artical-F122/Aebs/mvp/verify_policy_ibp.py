"""Adaptive interval-bound check for the repaired standalone PPO.

For each true-state cell, this verifier constructs the complete semantic input
box induced by the state-conditional perception contract, computes a sound IBP
lower bound on the PPO actor output, and compares it with a conservative upper
bound on the braking action required to keep the next state recoverable.

This verifies a local robust action condition.  It is not yet a complete
end-to-end probability or global inductive certificate: unresolved cells,
successor closure, and conformal trajectory-risk accounting remain explicit.
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch.nn as nn
from stable_baselines3 import PPO

from Aebs.mvp.check_certificate_transfer import minimum_safe_action
from Aebs.mvp.robust_sbc import discrete_stopping_distance
from Aebs.semantic.robust_controller import load_contract


@dataclass(frozen=True)
class Cell:
    d_low: float
    d_high: float
    v_low: float
    v_high: float
    depth: int

    @property
    def area(self) -> float:
        return (self.d_high - self.d_low) * (self.v_high - self.v_low)


def linear_interval(
    low: np.ndarray, high: np.ndarray, layer: nn.Linear
) -> Tuple[np.ndarray, np.ndarray]:
    weight = layer.weight.detach().cpu().numpy().astype(np.float64)
    bias = layer.bias.detach().cpu().numpy().astype(np.float64)
    positive = np.maximum(weight, 0.0)
    negative = np.minimum(weight, 0.0)
    output_low = low @ positive.T + high @ negative.T + bias
    output_high = high @ positive.T + low @ negative.T + bias
    return output_low, output_high


def actor_ibp_bounds(
    policy, input_low: np.ndarray, input_high: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sound interval bounds for the SB3 MLP actor used in this project."""

    low = np.asarray(input_low, dtype=np.float64)
    high = np.asarray(input_high, dtype=np.float64)
    for layer in policy.mlp_extractor.policy_net:
        if isinstance(layer, nn.Linear):
            low, high = linear_interval(low, high, layer)
        elif isinstance(layer, nn.Tanh):
            low, high = np.tanh(low), np.tanh(high)
        elif isinstance(layer, nn.ReLU):
            low, high = np.maximum(low, 0.0), np.maximum(high, 0.0)
        elif isinstance(layer, nn.Identity):
            continue
        else:
            raise TypeError(f"unsupported actor layer for IBP: {type(layer).__name__}")
    raw_low, raw_high = linear_interval(low, high, policy.action_net)
    clipped_low = np.clip(raw_low, -3.0, 3.0)
    clipped_high = np.clip(raw_high, -3.0, 3.0)
    return (
        raw_low.reshape(-1),
        raw_high.reshape(-1),
        clipped_low.reshape(-1),
        clipped_high.reshape(-1),
    )


def maximum_contract_radius(
    d_low: np.ndarray,
    d_high: np.ndarray,
    bin_edges: np.ndarray,
    radii: np.ndarray,
) -> np.ndarray:
    result = np.zeros_like(d_low, dtype=np.float64)
    for index, radius in enumerate(radii):
        intersects = (d_high >= bin_edges[index]) & (
            d_low <= bin_edges[index + 1]
        )
        result = np.where(intersects, np.maximum(result, radius), result)
    return result


def split_cell(cell: Cell) -> List[Cell]:
    d_mid = 0.5 * (cell.d_low + cell.d_high)
    v_mid = 0.5 * (cell.v_low + cell.v_high)
    depth = cell.depth + 1
    return [
        Cell(cell.d_low, d_mid, cell.v_low, v_mid, depth),
        Cell(d_mid, cell.d_high, cell.v_low, v_mid, depth),
        Cell(cell.d_low, d_mid, v_mid, cell.v_high, depth),
        Cell(d_mid, cell.d_high, v_mid, cell.v_high, depth),
    ]


def initial_cells(
    distance_low_m: float,
    distance_high_m: float,
    speed_low: float,
    speed_high: float,
    distance_cells: int,
    speed_cells: int,
    contract_edges: np.ndarray,
) -> List[Cell]:
    distance_edges = np.linspace(distance_low_m, distance_high_m, distance_cells + 1)
    internal_contract_edges = contract_edges[
        (contract_edges > distance_low_m) & (contract_edges < distance_high_m)
    ]
    distance_edges = np.unique(np.concatenate((distance_edges, internal_contract_edges)))
    speed_edges = np.linspace(speed_low, speed_high, speed_cells + 1)
    cells = []
    for v_index in range(len(speed_edges) - 1):
        for d_index in range(len(distance_edges) - 1):
            cells.append(
                Cell(
                    float(distance_edges[d_index]),
                    float(distance_edges[d_index + 1]),
                    float(speed_edges[v_index]),
                    float(speed_edges[v_index + 1]),
                    0,
                )
            )
    return cells


def evaluate_cells(
    cells: List[Cell],
    policy,
    distance_scale: float,
    contract_edges: np.ndarray,
    contract_radii: np.ndarray,
    tolerance: float,
    semantic_splits: int,
) -> Dict[str, object]:
    d_low = np.asarray([cell.d_low for cell in cells], dtype=np.float64)
    d_high = np.asarray([cell.d_high for cell in cells], dtype=np.float64)
    v_low = np.asarray([cell.v_low for cell in cells], dtype=np.float64)
    v_high = np.asarray([cell.v_high for cell in cells], dtype=np.float64)

    maximum_current_margin = (
        d_high - 6.0 - discrete_stopping_distance(v_low, 0.5, 3.0, 0.05)
    )
    minimum_current_margin = (
        d_low - 6.0 - discrete_stopping_distance(v_high, 0.5, 3.0, 0.05)
    )
    fully_unrecoverable = maximum_current_margin < 0.0
    boundary_cell = (~fully_unrecoverable) & (minimum_current_margin < 0.0)

    required_upper = minimum_safe_action(d_low, v_high, 6.0, 0.5, 3.0)
    # On the viability boundary, the required current action is the braking
    # needed to reduce v_high by one maximum-braking time step.  This equals
    # max braking for v >= target + a_max*dt, but is smaller in the final
    # low-speed band (0.5, 0.65).  Using 3.0 for every boundary cell was sound
    # but unnecessarily conservative near the terminal-speed threshold.
    boundary_required = np.clip((v_high - 0.5) / 0.05, 0.0, 3.0)
    required_upper = np.where(boundary_cell, boundary_required, required_upper)

    radius = maximum_contract_radius(
        d_low, d_high, contract_edges, contract_radii
    )
    input_low = np.stack(
        (
            np.clip(d_low / distance_scale - radius, 5.0 / distance_scale, 16.0 / distance_scale),
            v_low,
        ),
        axis=1,
    )
    input_high = np.stack(
        (
            np.clip(d_high / distance_scale + radius, 5.0 / distance_scale, 16.0 / distance_scale),
            v_high,
        ),
        axis=1,
    )
    if semantic_splits < 1:
        raise ValueError("semantic_splits must be at least 1")
    # Partition only the semantic-distance input.  The union of these boxes is
    # exactly the original perception interval, so taking the minimum lower
    # bound remains sound while avoiding a single unnecessarily wide IBP box.
    fractions_low = np.arange(semantic_splits, dtype=np.float64) / semantic_splits
    fractions_high = (
        np.arange(1, semantic_splits + 1, dtype=np.float64) / semantic_splits
    )
    semantic_width = input_high[:, 0] - input_low[:, 0]
    split_low = np.repeat(input_low, semantic_splits, axis=0)
    split_high = np.repeat(input_high, semantic_splits, axis=0)
    split_low[:, 0] = (
        input_low[:, 0, None]
        + semantic_width[:, None] * fractions_low[None, :]
    ).reshape(-1)
    split_high[:, 0] = (
        input_low[:, 0, None]
        + semantic_width[:, None] * fractions_high[None, :]
    ).reshape(-1)
    split_raw_low, split_raw_high, split_action_low, split_action_high = (
        actor_ibp_bounds(policy, split_low, split_high)
    )
    raw_low = split_raw_low.reshape(len(cells), semantic_splits).min(axis=1)
    raw_high = split_raw_high.reshape(len(cells), semantic_splits).max(axis=1)
    action_low = split_action_low.reshape(len(cells), semantic_splits).min(axis=1)
    action_high = split_action_high.reshape(len(cells), semantic_splits).max(axis=1)
    bound_margin = action_low - required_upper
    certified = (~fully_unrecoverable) & (bound_margin >= -tolerance)
    unresolved = (~fully_unrecoverable) & ~certified
    return {
        "fully_unrecoverable": fully_unrecoverable,
        "boundary_cell": boundary_cell,
        "required_action_upper": required_upper,
        "semantic_radius_norm": radius,
        "input_low": input_low,
        "input_high": input_high,
        "raw_action_low": raw_low,
        "raw_action_high": raw_high,
        "action_low": action_low,
        "action_high": action_high,
        "bound_margin": bound_margin,
        "certified": certified,
        "unresolved": unresolved,
    }


def cell_record(cell: Cell, result: Dict[str, object], index: int) -> Dict[str, float]:
    return {
        "distance_low_m": cell.d_low,
        "distance_high_m": cell.d_high,
        "speed_low_m_per_s": cell.v_low,
        "speed_high_m_per_s": cell.v_high,
        "depth": int(cell.depth),
        "boundary_cell": bool(result["boundary_cell"][index]),
        "required_action_upper": float(result["required_action_upper"][index]),
        "raw_action_lower": float(result["raw_action_low"][index]),
        "clipped_action_lower": float(result["action_low"][index]),
        "bound_margin": float(result["bound_margin"][index]),
    }


def verify(
    student_path: str,
    semantic_checkpoint: str,
    output_dir: Path,
    distance_low_m: float,
    distance_high_m: float,
    speed_low: float,
    speed_high: float,
    distance_cells: int,
    speed_cells: int,
    max_depth: int,
    max_evaluated_cells: int,
    tolerance: float,
    semantic_splits: int,
) -> Dict:
    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    contract, distance_scale = load_contract(semantic_checkpoint)
    student = PPO.load(student_path, device="cpu")
    student.policy.set_training_mode(False)
    contract_edges = np.asarray(contract.bin_edges, dtype=np.float64)
    contract_radii = np.asarray(contract.radii, dtype=np.float64)

    pending = initial_cells(
        distance_low_m,
        distance_high_m,
        speed_low,
        speed_high,
        distance_cells,
        speed_cells,
        contract_edges,
    )
    certified_records = []
    unresolved_records = []
    unrecoverable_area = 0.0
    certified_area = 0.0
    evaluated_cells = 0
    per_depth = {}

    while pending:
        if evaluated_cells + len(pending) > max_evaluated_cells:
            unresolved_records.extend(
                {
                    "distance_low_m": cell.d_low,
                    "distance_high_m": cell.d_high,
                    "speed_low_m_per_s": cell.v_low,
                    "speed_high_m_per_s": cell.v_high,
                    "depth": int(cell.depth),
                    "reason": "cell_budget_exhausted",
                }
                for cell in pending
            )
            break
        current = pending
        pending = []
        result = evaluate_cells(
            current,
            student.policy,
            distance_scale,
            contract_edges,
            contract_radii,
            tolerance,
            semantic_splits,
        )
        evaluated_cells += len(current)
        for index, cell in enumerate(current):
            depth_key = str(cell.depth)
            per_depth.setdefault(
                depth_key,
                {"evaluated": 0, "certified": 0, "unrecoverable": 0, "split": 0},
            )
            per_depth[depth_key]["evaluated"] += 1
            if result["fully_unrecoverable"][index]:
                unrecoverable_area += cell.area
                per_depth[depth_key]["unrecoverable"] += 1
            elif result["certified"][index]:
                certified_area += cell.area
                per_depth[depth_key]["certified"] += 1
                certified_records.append(cell_record(cell, result, index))
            elif cell.depth < max_depth:
                pending.extend(split_cell(cell))
                per_depth[depth_key]["split"] += 1
            else:
                record = cell_record(cell, result, index)
                record["reason"] = "ibp_bound_insufficient_at_max_depth"
                unresolved_records.append(record)

    region_area = (distance_high_m - distance_low_m) * (speed_high - speed_low)
    unresolved_area = max(region_area - certified_area - unrecoverable_area, 0.0)
    unresolved_records.sort(key=lambda row: row.get("bound_margin", -np.inf))
    status = (
        "local_action_condition_verified"
        if len(unresolved_records) == 0
        else "partial_local_verification"
    )
    all_certified_margins = [row["bound_margin"] for row in certified_records]
    metrics = {
        "experiment": "adaptive_ibp_policy_action_verification",
        "status": status,
        "student": student_path,
        "semantic_checkpoint": semantic_checkpoint,
        "verified_property": (
            "For each certified true-state cell and every semantic observation in "
            "its contract-induced input box, the clipped PPO action is no smaller "
            "than the conservative action threshold for a recoverable successor."
        ),
        "boundary_action_rule": (
            "clip((v_high - target_speed) / dt, 0, max_braking); exact for the "
            "one-step terminal band and equal to max braking above 0.65 m/s"
        ),
        "scope_limit": (
            "This is a local robust action-condition proof, not yet a complete global "
            "inductive or trajectory-level probabilistic safety certificate."
        ),
        "region": {
            "distance_low_m": float(distance_low_m),
            "distance_high_m": float(distance_high_m),
            "speed_low_m_per_s": float(speed_low),
            "speed_high_m_per_s": float(speed_high),
            "area": float(region_area),
        },
        "partition": {
            "initial_distance_cells": int(distance_cells),
            "initial_speed_cells": int(speed_cells),
            "max_depth": int(max_depth),
            "max_evaluated_cells": int(max_evaluated_cells),
            "semantic_splits": int(semantic_splits),
            "evaluated_cells": int(evaluated_cells),
            "certified_cells": int(len(certified_records)),
            "unresolved_cells": int(len(unresolved_records)),
            "per_depth": per_depth,
        },
        "area": {
            "certified_box_area": float(certified_area),
            "unrecoverable_box_area": float(unrecoverable_area),
            "unresolved_box_area": float(unresolved_area),
            "certified_fraction_of_region": float(certified_area / region_area),
            "unresolved_fraction_of_region": float(unresolved_area / region_area),
        },
        "certified_bound_margin": {
            "minimum": (
                float(np.min(all_certified_margins))
                if all_certified_margins
                else None
            ),
            "maximum": (
                float(np.max(all_certified_margins))
                if all_certified_margins
                else None
            ),
        },
        "worst_unresolved_cells": unresolved_records[:100],
        "runtime_seconds": float(time.time() - started),
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2, ensure_ascii=False)

    print("[Adaptive IBP policy verification]")
    print(f"status: {status}")
    print(
        f"region: d=[{distance_low_m:.3f}, {distance_high_m:.3f}] m, "
        f"v=[{speed_low:.3f}, {speed_high:.3f}] m/s"
    )
    print(
        f"cells: evaluated={evaluated_cells}, certified={len(certified_records)}, "
        f"unresolved={len(unresolved_records)}"
    )
    print(
        f"area: certified={100.0 * certified_area / region_area:.2f}%, "
        f"unrecoverable={100.0 * unrecoverable_area / region_area:.2f}%, "
        f"unresolved={100.0 * unresolved_area / region_area:.2f}%"
    )
    if unresolved_records and "bound_margin" in unresolved_records[0]:
        worst = unresolved_records[0]
        print(
            "worst unresolved: "
            f"d=[{worst['distance_low_m']:.6f}, {worst['distance_high_m']:.6f}], "
            f"v=[{worst['speed_low_m_per_s']:.6f}, {worst['speed_high_m_per_s']:.6f}], "
            f"required<={worst['required_action_upper']:.6f}, "
            f"action>={worst['clipped_action_lower']:.6f}, "
            f"margin={worst['bound_margin']:.6f}"
        )
    print(f"metrics: {output_dir / 'metrics.json'}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--student",
        default=(
            "results/mvp/08_standalone_ppo_saturation_repair/"
            "standalone_ppo_repaired.zip"
        ),
    )
    parser.add_argument(
        "--semantic-checkpoint",
        default="results/mvp/01_semantic/semantic_encoder.pt",
    )
    parser.add_argument(
        "--output-dir",
        default="results/mvp/09_policy_ibp_local",
    )
    parser.add_argument("--distance-low-m", type=float, default=6.0)
    parser.add_argument("--distance-high-m", type=float, default=8.0)
    parser.add_argument("--speed-low", type=float, default=2.0)
    parser.add_argument("--speed-high", type=float, default=3.0)
    parser.add_argument("--distance-cells", type=int, default=16)
    parser.add_argument("--speed-cells", type=int, default=8)
    parser.add_argument("--max-depth", type=int, default=7)
    parser.add_argument("--max-evaluated-cells", type=int, default=200000)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--semantic-splits", type=int, default=1)
    args = parser.parse_args()
    verify(
        args.student,
        args.semantic_checkpoint,
        Path(args.output_dir),
        args.distance_low_m,
        args.distance_high_m,
        args.speed_low,
        args.speed_high,
        args.distance_cells,
        args.speed_cells,
        args.max_depth,
        args.max_evaluated_cells,
        args.tolerance,
        args.semantic_splits,
    )


if __name__ == "__main__":
    main()
