"""Second minimal repair: make critical deployed actions clip to exactly 3.0."""

from pathlib import Path

from Aebs.mvp.repair_standalone_ppo import train_repair


def main() -> None:
    train_repair(
        student_path=(
            "results/mvp/07_standalone_ppo_local_repair/"
            "standalone_ppo_repaired.zip"
        ),
        teacher_path=(
            "results/mvp/02_safety_filter_anti_stall_compare/"
            "adaptive_anti_stall_filter.json"
        ),
        semantic_checkpoint="results/mvp/01_semantic/semantic_encoder.pt",
        diagnostics_path=(
            "results/mvp/07_standalone_ppo_local_repair/"
            "local_refinement/state_diagnostics.npz"
        ),
        output_dir=Path("results/mvp/08_standalone_ppo_saturation_repair"),
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


if __name__ == "__main__":
    main()
