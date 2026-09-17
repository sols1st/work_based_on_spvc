"""Final sound numerical refinement for the result 14 full-domain proof.

This keeps the controller, semantic contract, physical viability condition,
and verification tolerance fixed.  It only partitions the same continuous
input set more finely: 256 semantic subintervals and two additional adaptive
state-subdivision levels relative to result 15.
"""

from pathlib import Path

from Aebs.mvp.verify_policy_ibp import verify


def main() -> None:
    verify(
        student_path=(
            "results/mvp/14_standalone_ppo_dual_boundary_repair/"
            "standalone_ppo_repaired.zip"
        ),
        semantic_checkpoint="results/mvp/01_semantic/semantic_encoder.pt",
        output_dir=Path("results/mvp/16_policy_ibp_full_final"),
        distance_low_m=5.0,
        distance_high_m=16.0,
        speed_low=0.5,
        speed_high=3.0,
        distance_cells=44,
        speed_cells=10,
        max_depth=9,
        max_evaluated_cells=400000,
        tolerance=1e-9,
        semantic_splits=256,
    )


if __name__ == "__main__":
    main()
