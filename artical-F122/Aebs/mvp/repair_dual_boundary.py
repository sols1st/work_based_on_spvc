"""Jointly replay the low-speed and previously repaired high-speed boundaries.

Result 13 learned the new v~=0.65 m/s boundary but forgot 17 high-speed
critical states.  This single follow-up starts again from the successful result
08 checkpoint and trains on both boundary sets at once.  It uses the same
short repair settings, rollout regression, dense high-speed diagnostic, and
full-domain IBP verification as result 13.
"""

import json
from pathlib import Path

import numpy as np

from Aebs.mvp.repair_low_speed_boundary import (
    SEMANTIC_CHECKPOINT,
    SOURCE_STUDENT,
    TEACHER,
    make_boundary_diagnostics,
)
from Aebs.mvp.repair_standalone_ppo import train_repair
from Aebs.mvp.verify_policy_ibp import verify


OUTPUT_DIR = Path("results/mvp/14_standalone_ppo_dual_boundary_repair")
HIGH_SPEED_DIAGNOSTICS = Path(
    "results/mvp/08_standalone_ppo_saturation_repair/"
    "local_refinement/state_diagnostics.npz"
)


def make_joint_diagnostics(low_path: Path, output_path: Path) -> dict:
    """Merge new low-speed states with result 08's high-speed replay states."""

    low = np.load(low_path)
    high = np.load(HIGH_SPEED_DIAGNOSTICS)
    states = np.concatenate((low["states"], high["states"]), axis=0)
    required_action = np.concatenate(
        (low["required_action"], high["required_action"]), axis=0
    )
    action_margin = np.concatenate(
        (low["student_action_margin"], high["student_action_margin"]), axis=0
    )
    np.savez_compressed(
        output_path,
        states=states,
        required_action=required_action,
        student_action_margin=action_margin,
    )
    selected_low = int(np.sum(low["student_action_margin"] < 0.1))
    selected_high = int(np.sum(high["student_action_margin"] < 0.1))
    summary = {
        "low_speed_states": int(len(low["states"])),
        "high_speed_replay_states": int(len(high["states"])),
        "selected_low_speed_states": selected_low,
        "selected_high_speed_states": selected_high,
        "selected_total": selected_low + selected_high,
    }
    print("[Joint boundary replay]")
    print(
        f"selected low-speed={selected_low}, high-speed={selected_high}, "
        f"total={summary['selected_total']}"
    )
    return summary


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    low_path = OUTPUT_DIR / "low_speed_boundary_diagnostics.npz"
    low_summary = make_boundary_diagnostics(low_path)
    joint_path = OUTPUT_DIR / "joint_boundary_diagnostics.npz"
    replay_summary = make_joint_diagnostics(low_path, joint_path)

    metrics = train_repair(
        student_path=SOURCE_STUDENT,
        teacher_path=TEACHER,
        semantic_checkpoint=SEMANTIC_CHECKPOINT,
        diagnostics_path=str(joint_path),
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
        device_name="cpu",
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

    metrics["low_speed_source_diagnostic"] = low_summary
    metrics["joint_boundary_replay"] = replay_summary
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
    print("\n[Dual-boundary repair final decision]")
    print(
        f"status={final['status']}, unresolved={final['unresolved_cells']}, "
        f"unresolved_area={100.0 * final['unresolved_fraction_of_region']:.6f}%"
    )
    print(f"metrics: {OUTPUT_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
