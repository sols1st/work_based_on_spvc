"""Direct fixed-grid barrier feasibility check for the 2-D AEBS MVP.

This module does not train a neural network.  It assigns one barrier value to
each verifier-grid point and solves the strengthened robust-decrease conditions
as a sparse linear program.  A successful result is a grid certificate, not a
continuous-state formal certificate.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from Aebs.mvp.robust_sbc import (
    certificate_masks,
    robust_successors,
    verifier_grid_states,
)
from Aebs.semantic.robust_controller import load_contract
from Aebs.semantic.safety_filter import load_controller
from Aebs.uncertainty.model import StateDependentUncertainty


def bilinear_indices_weights(
    points: np.ndarray,
    distance_scale: float,
    grid_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return four flattened grid indices and bilinear weights per point."""

    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    d_low, d_high = 5.0 / distance_scale, 16.0 / distance_scale
    v_low, v_high = 0.0, 3.0
    d_position = np.clip(
        (points[:, 0] - d_low) / (d_high - d_low) * (grid_size - 1),
        0.0,
        grid_size - 1,
    )
    v_position = np.clip(
        (points[:, 1] - v_low) / (v_high - v_low) * (grid_size - 1),
        0.0,
        grid_size - 1,
    )
    d0 = np.minimum(np.floor(d_position).astype(np.int64), grid_size - 2)
    v0 = np.minimum(np.floor(v_position).astype(np.int64), grid_size - 2)
    d1, v1 = d0 + 1, v0 + 1
    wd = d_position - d0
    wv = v_position - v0

    indices = np.stack(
        (
            v0 * grid_size + d0,
            v0 * grid_size + d1,
            v1 * grid_size + d0,
            v1 * grid_size + d1,
        ),
        axis=1,
    )
    weights = np.stack(
        (
            (1.0 - wd) * (1.0 - wv),
            wd * (1.0 - wv),
            (1.0 - wd) * wv,
            wd * wv,
        ),
        axis=1,
    )
    return indices, weights


def mask_arguments() -> Dict[str, object]:
    return {
        "terminal_speed_threshold": 0.5,
        "goal_distance_m": 6.0,
        "goal_speed": 0.5,
        "unsafe_distance_low_m": 5.0,
        "unsafe_distance_high_m": 6.0,
        "unsafe_speed": 0.5,
        "use_recoverable_domain": True,
        "max_braking": 3.0,
    }


def solve_grid_certificate(
    controller_path: str,
    semantic_checkpoint: str,
    uncertainty_path: str,
    output_dir: Path,
    grid_size: int,
    solver_time_limit_seconds: float,
) -> Dict:
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except ImportError as error:
        raise SystemExit(
            "SciPy is required for the grid LP. Install once with: "
            "/opt/miniconda3/bin/conda run -n vt pip install scipy==1.10.1"
        ) from error

    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    contract, distance_scale = load_contract(semantic_checkpoint)
    controller = load_controller(controller_path)
    uncertainty = StateDependentUncertainty.from_npz(uncertainty_path)
    states = verifier_grid_states(distance_scale, grid_size)
    masks = certificate_masks(states, distance_scale, **mask_arguments())
    decrease_indices = np.flatnonzero(masks["decrease"])
    decrease_states = states[decrease_indices]
    semantic_radii = contract.radius(
        decrease_states[:, 0] * distance_scale
    ).astype(np.float32)
    successors = robust_successors(
        decrease_states,
        controller,
        uncertainty,
        distance_scale,
        semantic_radii,
    )

    state_variable_count = len(states)
    epsilon_index = state_variable_count
    variable_count = state_variable_count + 1
    row_indices, column_indices, coefficients, upper_bounds = [], [], [], []
    row_count = 0

    def add_row(columns, values, upper_bound) -> None:
        nonlocal row_count
        for column, value in zip(columns, values):
            if abs(float(value)) > 1e-14:
                row_indices.append(row_count)
                column_indices.append(int(column))
                coefficients.append(float(value))
        upper_bounds.append(float(upper_bound))
        row_count += 1

    flat_successors = successors.reshape(-1, 2)
    successor_indices, successor_weights = bilinear_indices_weights(
        flat_successors, distance_scale, grid_size
    )
    successor_terminal = certificate_masks(
        flat_successors, distance_scale, **mask_arguments()
    )["terminal"]
    successor_unsafe = certificate_masks(
        flat_successors, distance_scale, **mask_arguments()
    )["unsafe"]
    candidate_count = successors.shape[1]
    unsafe_successor_matrix = successor_unsafe.reshape(len(decrease_states), candidate_count)
    unsafe_successor_count = int(np.sum(unsafe_successor_matrix))
    states_with_unsafe_successor_count = int(
        np.sum(np.any(unsafe_successor_matrix, axis=1))
    )
    for local_state_index, grid_index in enumerate(decrease_indices):
        for candidate_index in range(candidate_count):
            flat_index = local_state_index * candidate_count + candidate_index
            columns = [grid_index, epsilon_index]
            values = [-1.0, 1.0]
            if not successor_terminal[flat_index]:
                columns.extend(successor_indices[flat_index].tolist())
                values.extend(successor_weights[flat_index].tolist())
            add_row(columns, values, 0.0)

    # This is a fixed-grid certificate, so region constraints must be imposed
    # on the same grid nodes.  Interpolating a point just above v=0.5 onto the
    # main grid mixes it with a terminal node just below v=0.5; demanding that
    # interpolation be >=10 conflicts with the terminal value B=0 and makes
    # the LP infeasible by construction.
    distance_m = states[:, 0] * distance_scale
    speed = states[:, 1]
    init_mask = (
        (distance_m >= 15.0)
        & (distance_m <= 16.0)
        & (speed >= 2.5)
        & (speed <= 3.0)
    )
    init_indices = np.flatnonzero(init_mask)
    for grid_index in init_indices:
        add_row([grid_index], [1.0], 1.0)

    # Unsafe grid nodes are already fixed to B=10 by variable_bounds below.
    # No extra off-grid unsafe interpolation constraints are added.

    matrix = coo_matrix(
        (coefficients, (row_indices, column_indices)),
        shape=(row_count, variable_count),
    ).tocsr()
    objective = np.zeros(variable_count, dtype=np.float64)
    objective[epsilon_index] = -1.0

    variable_bounds = []
    for terminal, unsafe in zip(masks["terminal"], masks["unsafe"]):
        if terminal:
            variable_bounds.append((0.0, 0.0))
        elif unsafe:
            variable_bounds.append((10.0, 10.0))
        else:
            # This B<=10 restriction makes the LP non-degenerate and stronger
            # than the threshold-gated decrease condition in Theorem 2.2.
            variable_bounds.append((0.0, 10.0))
    variable_bounds.append((0.0, 1.0))

    result = linprog(
        objective,
        A_ub=matrix,
        b_ub=np.asarray(upper_bounds),
        bounds=variable_bounds,
        method="highs",
        options={"time_limit": float(solver_time_limit_seconds)},
    )
    status_names = {
        0: "feasible",
        1: "limit_reached",
        2: "infeasible",
        3: "unbounded",
        4: "solver_error",
    }
    status = status_names.get(int(result.status), "unknown")
    metrics = {
        "experiment": "strengthened_grid_certificate_lp",
        "certificate_level": "fixed_grid_bilinear_diagnostic",
        "status": status,
        "solver_status_code": int(result.status),
        "solver_message": str(result.message),
        "controller": controller_path,
        "semantic_checkpoint": semantic_checkpoint,
        "uncertainty": uncertainty_path,
        "grid_size": int(grid_size),
        "state_variables": int(state_variable_count),
        "decrease_states": int(len(decrease_states)),
        "successor_candidates_per_state": int(candidate_count),
        "unsafe_successor_count": unsafe_successor_count,
        "states_with_unsafe_successor_count": states_with_unsafe_successor_count,
        "linear_constraints": int(row_count),
        "excluded_terminal_states": int(np.sum(masks["terminal"])),
        "excluded_unsafe_states": int(np.sum(masks["unsafe"])),
        "excluded_inevitable_states": int(np.sum(masks["inevitable"])),
        "initial_grid_nodes": int(len(init_indices)),
        "unsafe_grid_nodes": int(np.sum(masks["unsafe"])),
        "region_constraint_mode": "main_grid_nodes",
        "runtime_seconds": float(time.time() - started),
        "interpretation": (
            "A feasible result proves only the stated strengthened conditions on the "
            "fixed grid and bilinear table. It is not a continuous-state formal proof."
        ),
    }
    if result.success:
        barrier_values = np.asarray(result.x[:state_variable_count])
        epsilon = float(result.x[epsilon_index])
        residual = matrix @ result.x - np.asarray(upper_bounds)
        metrics.update(
            {
                "epsilon_max": epsilon,
                "maximum_constraint_residual": float(np.max(residual)),
                "minimum_barrier": float(np.min(barrier_values)),
                "maximum_barrier": float(np.max(barrier_values)),
            }
        )
        np.save(output_dir / "barrier_grid.npy", barrier_values.reshape(grid_size, grid_size))

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2, ensure_ascii=False)
    print("[Grid certificate LP]")
    print(f"status: {status}")
    print(f"states: {state_variable_count}, decrease states: {len(decrease_states)}")
    print(f"constraints: {row_count}")
    print(
        "states with one-step unsafe robust successor: "
        f"{states_with_unsafe_successor_count}"
    )
    if result.success:
        print(f"epsilon_max: {metrics['epsilon_max']:.9f}")
        print(f"maximum constraint residual: {metrics['maximum_constraint_residual']:.3e}")
    else:
        print(f"solver: {result.message}")
    print(f"metrics: {output_dir / 'metrics.json'}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--controller",
        default="results/mvp/02_standalone_ppo_distilled_v2/standalone_ppo.zip",
    )
    parser.add_argument(
        "--semantic-checkpoint",
        default="results/mvp/01_semantic/semantic_encoder.pt",
    )
    parser.add_argument(
        "--uncertainty",
        default=(
            "results/mvp/03_process_uncertainty/"
            "state_dependent_uncertainty.npz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="results/mvp/04_grid_certificate_lp_standalone_ppo_v2",
    )
    parser.add_argument("--grid-size", type=int, default=80)
    parser.add_argument("--solver-time-limit-seconds", type=float, default=600.0)
    args = parser.parse_args()
    if args.grid_size < 2:
        raise ValueError("grid-size must be at least 2")
    solve_grid_certificate(
        args.controller,
        args.semantic_checkpoint,
        args.uncertainty,
        Path(args.output_dir),
        args.grid_size,
        args.solver_time_limit_seconds,
    )


if __name__ == "__main__":
    main()
