"""Run the local IBP verifier with a sound 16-way semantic partition."""

from pathlib import Path

from Aebs.mvp.verify_policy_ibp import verify


def main() -> None:
    verify(
        student_path=(
            "results/mvp/08_standalone_ppo_saturation_repair/"
            "standalone_ppo_repaired.zip"
        ),
        semantic_checkpoint="results/mvp/01_semantic/semantic_encoder.pt",
        output_dir=Path("results/mvp/10_policy_ibp_semantic_split"),
        distance_low_m=6.0,
        distance_high_m=8.0,
        speed_low=2.0,
        speed_high=3.0,
        distance_cells=16,
        speed_cells=8,
        max_depth=7,
        max_evaluated_cells=200000,
        tolerance=1e-9,
        semantic_splits=16,
    )


if __name__ == "__main__":
    main()
