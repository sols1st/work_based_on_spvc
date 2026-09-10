"""Re-verify an existing MVP barrier without retraining it."""

import argparse
import json
from pathlib import Path

import torch

from Aebs.mvp.robust_sbc import build_barrier, evaluate_grid
from Aebs.semantic.safety_filter import load_controller
from Aebs.uncertainty.model import StateDependentUncertainty


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="Aebs/mvp/config.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epsilons", default="0,0.001,0.005,0.01")
    parser.add_argument("--terminal-speed-threshold", type=float)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as stream:
        config = json.load(stream)
    settings = config["robust_sbc"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    barrier = build_barrier(
        str(settings.get("barrier_output_transform", "")),
        bool(settings["square_output"]),
        device,
    )
    barrier.load_state_dict(torch.load(args.checkpoint, map_location=device))
    controller = load_controller(config["safety_filter_controller"])
    uncertainty_path = (
        Path(config["output_dir"])
        / config["uncertainty"].get("output_name", "03_uncertainty")
        / "state_dependent_uncertainty.npz"
    )
    uncertainty = StateDependentUncertainty.from_npz(str(uncertainty_path))
    semantic_checkpoint = str(Path(config["output_dir"]) / "01_semantic" / "semantic_encoder.pt")

    results = {}
    terminal_speed_threshold = (
        float(args.terminal_speed_threshold)
        if args.terminal_speed_threshold is not None
        else float(settings.get("terminal_speed_threshold", 0.0))
    )
    for epsilon in (float(value) for value in args.epsilons.split(",")):
        verification = evaluate_grid(
            barrier,
            controller,
            uncertainty,
            semantic_checkpoint,
            config["data"],
            int(settings["grid_size"]),
            int(settings["batch_size"]),
            epsilon,
            terminal_speed_threshold,
            float(settings.get("init_target", 1.0)),
            float(settings.get("unsafe_target", 10.0)),
            float(settings.get("goal_target", 1.0)),
            float(settings.get("goal_distance_m", 6.0)),
            float(settings.get("goal_speed", 0.5)),
            float(settings.get("unsafe_distance_low_m", 5.0)),
            float(settings.get("unsafe_distance_high_m", 6.0)),
            float(settings.get("unsafe_speed", 0.5)),
            bool(settings.get("use_recoverable_domain", False)),
            float(settings.get("max_braking", 3.0)),
            bool(settings.get("enforce_goal_constraint", False)),
            float(settings.get("terminal_value", 0.0)),
            device,
        )
        results[f"{epsilon:g}"] = verification
        print(
            f"epsilon={epsilon:g}: violations={verification['violation_count']}/"
            f"{verification['checked_states']}, min_margin={verification['min_margin']:.6f}, "
            f"mean_margin={verification['mean_margin']:.6f}"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as stream:
        json.dump(
            {
                "checkpoint": args.checkpoint,
                "terminal_speed_threshold": terminal_speed_threshold,
                "results": results,
            },
            stream,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()
