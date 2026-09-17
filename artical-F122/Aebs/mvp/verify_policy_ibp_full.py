"""Verify the repaired PPO action condition on the full AEBS domain."""

from pathlib import Path

from Aebs.mvp.verify_policy_ibp import verify


def main() -> None:
    verify(
        student_path=(
            "results/mvp/08_standalone_ppo_saturation_repair/"
            "standalone_ppo_repaired.zip"
        ),
        semantic_checkpoint="results/mvp/01_semantic/semantic_encoder.pt",
        output_dir=Path("results/mvp/11_policy_ibp_full_domain"),
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


if __name__ == "__main__":
    main()
