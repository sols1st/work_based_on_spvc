"""Sound full-domain IBP for result 14 with finer semantic partitioning.

The semantic contract is unchanged.  Each continuous perception interval is
partitioned into 64 boxes instead of 16, and the minimum sound action lower
bound across their union is used.  This only tightens interval propagation; it
does not omit observations or relax the safety threshold.
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
        output_dir=Path("results/mvp/15_policy_ibp_full_semantic64"),
        distance_low_m=5.0,
        distance_high_m=16.0,
        speed_low=0.5,
        speed_high=3.0,
        distance_cells=44,
        speed_cells=10,
        max_depth=7,
        max_evaluated_cells=400000,
        tolerance=1e-9,
        semantic_splits=64,
    )


if __name__ == "__main__":
    main()
