"""Minimal repair for the remaining low-speed viability-boundary cells.

The full-domain IBP run leaves a very small unresolved strip around the first
discrete stopping-step transition (roughly v=0.65 m/s).  This script samples
that physical boundary directly, repairs only states whose deployed PPO action
is below the one-step recoverability threshold, retains the source policy on a
broad grid, and then runs both rollout regression tests and the full-domain
sound IBP verifier.  No runtime safety filter is added.
"""

import json
from pathlib import Path

import numpy as np

from Aebs.mvp.check_certificate_transfer import minimum_safe_action, predict_actions
from Aebs.mvp.repair_standalone_ppo import train_repair
from Aebs.mvp.robust_sbc import discrete_stopping_distance
from Aebs.mvp.verify_policy_ibp import verify
from Aebs.semantic.robust_controller import load_contract
from Aebs.semantic.safety_filter import load_controller


SOURCE_STUDENT = (
    "results/mvp/08_standalone_ppo_saturation_repair/"
    "standalone_ppo_repaired.zip"
)
TEACHER = (
    "results/mvp/02_safety_filter_anti_stall_compare/"
    "adaptive_anti_stall_filter.json"
)
SEMANTIC_CHECKPOINT = "results/mvp/01_semantic/semantic_encoder.pt"
OUTPUT_DIR = Path("results/mvp/13_standalone_ppo_low_speed_repair")


def make_boundary_diagnostics(output_path: Path) -> dict:
    """Build a small dataset aligned with the discontinuous viability curve."""

    contract, distance_scale = load_contract(SEMANTIC_CHECKPOINT)
    student = load_controller(SOURCE_STUDENT)

    # The remaining formal cells occupy v about 0.64--0.72 m/s.  Sampling a
    # slightly wider band avoids fitting only the reported worst rectangle.
    speeds = np.linspace(0.62, 0.75, 131, dtype=np.float64)
    boundary_distance = 6.0 + discrete_stopping_distance(
        speeds, 0.5, 3.0, 0.05
    )
    distance_offsets = np.asarray(
        [0.0, 0.0005, 0.001, 0.0025, 0.005, 0.01], dtype=np.float64
    )
    distance_m = np.repeat(boundary_distance, len(distance_offsets))
    distance_m += np.tile(distance_offsets, len(speeds))
    speed = np.repeat(speeds, len(distance_offsets))
    states = np.stack((distance_m / distance_scale, speed), axis=1).astype(
        np.float32
    )
    required_action = minimum_safe_action(
        distance_m, speed, 6.0, 0.5, 3.0
    )

    semantic_samples = 65
    multipliers = np.linspace(-1.0, 1.0, semantic_samples, dtype=np.float64)
    radius_norm = contract.radius(distance_m).astype(np.float64)
    observed_distance = (
        states[:, 0, None].astype(np.float64)
        + radius_norm[:, None] * multipliers[None, :]
    )
    observed_distance = np.clip(
        observed_distance, 5.0 / distance_scale, 16.0 / distance_scale
    )
    observations = np.stack(
        (
            observed_distance,
            np.broadcast_to(speed[:, None], observed_distance.shape),
        ),
        axis=2,
    ).reshape(-1, 2)
    action = predict_actions(student, observations, 65536).reshape(
        len(states), semantic_samples
    )
    action_margin = np.min(action - required_action[:, None], axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        states=states,
        distance_m=distance_m,
        speed=speed,
        required_action=required_action,
        student_action_margin=action_margin,
    )
    summary = {
        "states": int(len(states)),
        "semantic_samples_per_state": int(semantic_samples),
        "violating_states": int(np.sum(action_margin < 0.0)),
        "minimum_action_margin": float(np.min(action_margin)),
        "speed_range_m_per_s": [float(np.min(speed)), float(np.max(speed))],
        "distance_range_m": [float(np.min(distance_m)), float(np.max(distance_m))],
    }
    print("[Low-speed boundary diagnostic]")
    print(
        f"states={summary['states']}, violations={summary['violating_states']}, "
        f"min action margin={summary['minimum_action_margin']:.9f}"
    )
    return summary


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    diagnostics_path = OUTPUT_DIR / "source_boundary_diagnostics.npz"
    diagnostic_summary = make_boundary_diagnostics(diagnostics_path)

    metrics = train_repair(
        student_path=SOURCE_STUDENT,
        teacher_path=TEACHER,
        semantic_checkpoint=SEMANTIC_CHECKPOINT,
        diagnostics_path=str(diagnostics_path),
        output_dir=OUTPUT_DIR,
        seed=7,
        epochs=15,
        batch_size=512,
        learning_rate=5e-6,
        selection_margin=0.1,
        target_margin=0.02,
        raw_target_headroom=0.05,
        semantic_samples=65,
        local_weight=10.0,
        underbraking_weight=10.0,
        eval_episodes=200,
        device_name="auto",
    )

    verification = verify(
        student_path=str(OUTPUT_DIR / "standalone_ppo_repaired.zip"),
        semantic_checkpoint=SEMANTIC_CHECKPOINT,
        output_dir=OUTPUT_DIR / "full_domain_verification",
        distance_low_m=5.0,
        distance_high_m=16.0,
        speed_low=0.5,
        speed_high=3.0,
        distance_cells=44,
        speed_cells=10,
        max_depth=7,
        max_evaluated_cells=400000,
        tolerance=1e-9,
        semantic_splits=16,
    )

    metrics["source_boundary_diagnostic"] = diagnostic_summary
    metrics["full_domain_verification"] = {
        "status": verification["status"],
        "evaluated_cells": verification["partition"]["evaluated_cells"],
        "certified_cells": verification["partition"]["certified_cells"],
        "unresolved_cells": verification["partition"]["unresolved_cells"],
        "certified_fraction_of_region": verification["area"][
            "certified_fraction_of_region"
        ],
        "unresolved_fraction_of_region": verification["area"][
            "unresolved_fraction_of_region"
        ],
    }
    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2, ensure_ascii=False)

    final = metrics["full_domain_verification"]
    print("\n[Low-speed repair final decision]")
    print(
        f"status={final['status']}, unresolved={final['unresolved_cells']}, "
        f"unresolved_area={100.0 * final['unresolved_fraction_of_region']:.6f}%"
    )
    print(f"metrics: {OUTPUT_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
