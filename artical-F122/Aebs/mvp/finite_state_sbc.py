"""Finite-state supermartingale barrier backed by the full-domain IBP proof.

The free neural SBC failed because it tried to learn an arbitrary continuous
function with strict decrease across terminal and viability boundaries.  This
module uses a different certificate class and solver:

* the sound result-16 proof supplies the robust closure R -> {R, T};
* R is the recoverable operational set, T the safe absorbing terminal set;
* U is unsafe or physically unrecoverable and is assigned barrier value one;
* a small HiGHS LP solves the non-increasing supermartingale conditions.

The resulting certificate is piecewise constant and may be discontinuous at
the viability boundary.  It is a conditional robust/degenerate-stochastic SBC,
not an unconditional trajectory probability claim: it assumes the semantic
observation stays inside its calibrated contract and process residual is zero.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from Aebs.mvp.robust_sbc import discrete_stopping_distance


ABSTRACT_STATES = ("recoverable", "safe_terminal", "unsafe_or_unrecoverable")


def load_closure_evidence(metrics_path: Path) -> Tuple[Dict, str]:
    raw = metrics_path.read_bytes()
    metrics = json.loads(raw.decode("utf-8"))
    errors = []
    if metrics.get("status") != "local_action_condition_verified":
        errors.append("status is not local_action_condition_verified")
    partition = metrics.get("partition", {})
    if int(partition.get("unresolved_cells", -1)) != 0:
        errors.append("unresolved_cells is not zero")
    region = metrics.get("region", {})
    expected_region = {
        "distance_low_m": 5.0,
        "distance_high_m": 16.0,
        "speed_low_m_per_s": 0.5,
        "speed_high_m_per_s": 3.0,
    }
    for key, expected in expected_region.items():
        if not np.isclose(float(region.get(key, np.nan)), expected):
            errors.append(f"unexpected region field {key}")
    expected_student = (
        "results/mvp/14_standalone_ppo_dual_boundary_repair/"
        "standalone_ppo_repaired.zip"
    )
    if metrics.get("student") != expected_student:
        errors.append("closure evidence is not bound to the result-14 policy")
    if errors:
        raise ValueError("invalid closure evidence: " + "; ".join(errors))
    return metrics, hashlib.sha256(raw).hexdigest()


def solve_abstract_sbc() -> Tuple[object, np.ndarray, np.ndarray, np.ndarray]:
    """Solve the three-mode supermartingale barrier as a linear program.

    Variables are [B_R, B_T, B_U, beta].  The robust successor relation is
    R -> {R,T}, T -> {T}, U -> {U}.  For every possible successor j we impose
    B_j <= B_i.  beta upper-bounds the initial-set barrier value B_R.
    """

    try:
        from scipy.optimize import linprog
    except ImportError as error:
        raise SystemExit(
            "SciPy is required. Install it in vt with: "
            "pip install scipy==1.10.1"
        ) from error

    # A_ub @ x <= b_ub.
    rows = [
        # R -> T: B_T - B_R <= 0.  R -> R and T -> T are identities.
        [-1.0, 1.0, 0.0, 0.0],
        # Initial set lies in R: B_R - beta <= 0.
        [1.0, 0.0, 0.0, -1.0],
        # Unsafe normalization: 1 - B_U <= 0, written -B_U <= -1.
        [0.0, 0.0, -1.0, 0.0],
    ]
    upper = [0.0, 0.0, -1.0]
    objective = np.asarray([1e-9, 0.0, 0.0, 1.0], dtype=np.float64)
    bounds = [
        (0.0, 1.0),  # recoverable
        (0.0, 0.0),  # safe terminal
        (1.0, 1.0),  # unsafe or physically unrecoverable
        (0.0, 1.0),  # initial failure-probability upper bound beta
    ]
    matrix = np.asarray(rows, dtype=np.float64)
    rhs = np.asarray(upper, dtype=np.float64)
    result = linprog(
        objective,
        A_ub=matrix,
        b_ub=rhs,
        bounds=bounds,
        method="highs",
    )
    return result, matrix, rhs, objective


def run(metrics_path: Path, output_dir: Path) -> Dict:
    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    closure, closure_sha256 = load_closure_evidence(metrics_path)
    result, matrix, rhs, objective = solve_abstract_sbc()

    status = "verified" if result.success else "solver_failed"
    values = None
    maximum_residual = None
    if result.success:
        values = np.asarray(result.x, dtype=np.float64)
        maximum_residual = float(np.max(matrix @ values - rhs))
        if maximum_residual > 1e-9:
            status = "constraint_residual_failed"

    initial_min_margin_m = float(
        15.0 - 6.0 - discrete_stopping_distance(
            np.asarray([3.0]), 0.5, 3.0, 0.05
        )[0]
    )
    certificate = {
        "experiment": "finite_state_supermartingale_barrier",
        "algorithm": "HiGHS linear programming on a three-mode robust abstraction",
        "status": status,
        "certificate_class": "piecewise_constant_supermartingale_barrier",
        "stochastic_interpretation": (
            "Degenerate stochastic/robust certificate under zero process residual; "
            "semantic inputs are treated adversarially inside the calibrated contract."
        ),
        "closure_evidence": {
            "path": str(metrics_path),
            "sha256": closure_sha256,
            "status": closure["status"],
            "controller": closure["student"],
            "evaluated_cells": closure["partition"]["evaluated_cells"],
            "certified_cells": closure["partition"]["certified_cells"],
            "unresolved_cells": closure["partition"]["unresolved_cells"],
            "certified_fraction_of_region": closure["area"][
                "certified_fraction_of_region"
            ],
            "unrecoverable_fraction_of_region": (
                closure["area"]["unrecoverable_box_area"]
                / closure["region"]["area"]
            ),
        },
        "abstract_states": list(ABSTRACT_STATES),
        "robust_successor_relation": {
            "recoverable": ["recoverable", "safe_terminal"],
            "safe_terminal": ["safe_terminal"],
            "unsafe_or_unrecoverable": ["unsafe_or_unrecoverable"],
        },
        "barrier_definition": {
            "recoverable": 0.0,
            "safe_terminal": 0.0,
            "unsafe_or_unrecoverable": 1.0,
        },
        "initial_region": {
            "distance_m": [15.0, 16.0],
            "speed_m_per_s": [2.5, 3.0],
            "minimum_recoverability_margin_m": initial_min_margin_m,
            "contained_in_recoverable_set": bool(initial_min_margin_m >= 0.0),
        },
        "lp": {
            "success": bool(result.success),
            "solver_status_code": int(result.status),
            "solver_message": str(result.message),
            "variables": ["B_recoverable", "B_safe_terminal", "B_unsafe", "beta"],
            "objective": objective.tolist(),
            "constraint_count": int(len(rhs)),
            "maximum_constraint_residual": maximum_residual,
        },
        "conditional_guarantee": (
            "For every initial state in the stated initial region, if every semantic "
            "observation remains inside the calibrated state-conditional contract and "
            "the process residual is zero, the supermartingale bound on reaching the "
            "unsafe_or_unrecoverable abstract state is beta."
        ),
        "scope_limit": (
            "beta is not an unconditional trajectory-risk bound. A trajectory-level "
            "probability for remaining inside the semantic contract must be calibrated "
            "separately before combining it with this conditional certificate."
        ),
        "runtime_seconds": float(time.time() - started),
    }
    if values is not None:
        certificate["lp"].update(
            {
                "barrier_values": {
                    "recoverable": float(values[0]),
                    "safe_terminal": float(values[1]),
                    "unsafe_or_unrecoverable": float(values[2]),
                },
                "initial_failure_probability_upper_bound_beta": float(values[3]),
                "objective_value": float(result.fun),
            }
        )

    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as stream:
        json.dump(certificate, stream, indent=2, ensure_ascii=False)
    np.savez_compressed(
        output_dir / "lp_system.npz",
        A_ub=matrix,
        b_ub=rhs,
        objective=objective,
        solution=(values if values is not None else np.asarray([])),
    )

    print("[Finite-state supermartingale barrier]")
    print(f"status: {status}")
    print(
        "closure evidence: "
        f"evaluated={closure['partition']['evaluated_cells']}, "
        f"certified={closure['partition']['certified_cells']}, "
        f"unresolved={closure['partition']['unresolved_cells']}"
    )
    print(f"initial recoverability margin: {initial_min_margin_m:.6f} m")
    if values is not None:
        print(
            "barrier: "
            f"B(R)={values[0]:.6f}, B(T)={values[1]:.6f}, "
            f"B(U)={values[2]:.6f}"
        )
        print(f"conditional failure bound beta: {values[3]:.6f}")
        print(f"maximum LP residual: {maximum_residual:.3e}")
    else:
        print(f"solver: {result.message}")
    print("note: beta is conditional on contract membership and zero process residual")
    print(f"metrics: {metrics_file}")
    return certificate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--closure-metrics",
        default="results/mvp/16_policy_ibp_full_final/metrics.json",
    )
    parser.add_argument(
        "--output-dir",
        default="results/mvp/17_finite_state_sbc",
    )
    args = parser.parse_args()
    run(Path(args.closure_metrics), Path(args.output_dir))


if __name__ == "__main__":
    main()
