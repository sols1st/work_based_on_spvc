"""Build and compare global-radius and adaptive-radius AEBS safety filters.

This is an evaluation-only fast path.  It does not train PPO, the semantic
encoder, or a barrier network.
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
from stable_baselines3 import PPO

from Aebs.semantic.robust_controller import evaluate, load_contract
from Aebs.semantic.safety_filter import load_controller


MODES = ("exact", "uniform", "random_boundary", "worst_endpoint")


def write_spec(path: Path, spec: Dict) -> None:
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(spec, stream, indent=2, ensure_ascii=False)


def base_spec(
    baseline_controller: str,
    distance_scale: float,
    global_radius_m: float,
    target_speed: float,
    distance_buffer_m: float,
) -> Dict:
    return {
        "type": "analytic_aebs_safety_filter",
        "baseline_controller": baseline_controller,
        "distance_scale": float(distance_scale),
        "perception_radius_m": float(global_radius_m),
        "safety_distance_m": 6.0,
        "target_speed": float(target_speed),
        "distance_buffer_m": float(distance_buffer_m),
        "max_braking": 3.0,
    }


def compact_row(metrics: Dict) -> str:
    row = (
        f"success={100.0 * metrics['success_rate']:.1f}% "
        f"unsafe={100.0 * metrics['unsafe_rate']:.1f}% "
        f"stopped_safe={100.0 * metrics['stopped_safe_outside_goal_rate']:.1f}% "
        f"out_of_domain={100.0 * metrics['out_of_domain_rate']:.1f}% "
        f"timeout={100.0 * metrics['timeout_rate']:.1f}% "
        f"steps={metrics['mean_steps']:.1f}"
    )
    if metrics.get("intervention_rate") is not None:
        row += (
            f" intervention={100.0 * metrics['intervention_rate']:.1f}%"
            f" extra_braking={metrics['mean_extra_braking']:.3f}"
        )
    if metrics.get("overshoot_cap_rate"):
        row += f" overshoot_cap={100.0 * metrics['overshoot_cap_rate']:.1f}%"
    return row


def compare(
    baseline_controller: str,
    semantic_checkpoint: str,
    output_dir: Path,
    episodes: int,
    seed: int,
    target_speed: float,
    distance_buffer_m: float,
) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    contract, distance_scale = load_contract(semantic_checkpoint)
    radii_m = np.asarray(contract.radii, dtype=np.float64) * distance_scale
    global_radius_m = float(np.max(radii_m))

    global_spec = base_spec(
        baseline_controller,
        distance_scale,
        global_radius_m,
        target_speed,
        distance_buffer_m,
    )
    global_spec["radius_mode"] = "global_max"

    adaptive_spec = base_spec(
        baseline_controller,
        distance_scale,
        global_radius_m,
        target_speed,
        distance_buffer_m,
    )
    adaptive_spec.update(
        {
            "radius_mode": "observed_distance_bin_diagnostic",
            "radius_bin_edges_m": [float(value) for value in contract.bin_edges],
            "radius_values_m": [float(value) for value in radii_m],
            "validity_note": (
                "Diagnostic only: the source contract was calibrated by true-state bins; "
                "using observed-distance bins is not yet a formal coverage guarantee."
            ),
        }
    )
    adaptive_anti_stall_spec = dict(adaptive_spec)
    adaptive_anti_stall_spec.update(
        {
            "radius_mode": "observed_distance_bin_anti_stall_diagnostic",
            "prevent_filter_overshoot": True,
            "dt": 0.05,
            "validity_note": (
                "Diagnostic only: observed-distance bins are not yet a formal coverage "
                "guarantee. The filter contribution is capped so one discrete step "
                "does not reduce speed below target_speed."
            ),
        }
    )

    global_path = output_dir / "global_max_filter.json"
    adaptive_path = output_dir / "adaptive_observed_bin_filter.json"
    adaptive_anti_stall_path = output_dir / "adaptive_anti_stall_filter.json"
    write_spec(global_path, global_spec)
    write_spec(adaptive_path, adaptive_spec)
    write_spec(adaptive_anti_stall_path, adaptive_anti_stall_spec)

    controllers = {
        "baseline": PPO.load(baseline_controller, device="cpu"),
        "global_max_filter": load_controller(str(global_path)),
        "adaptive_observed_bin_filter": load_controller(str(adaptive_path)),
        "adaptive_anti_stall_filter": load_controller(str(adaptive_anti_stall_path)),
    }
    metrics = {
        "experiment": "safety_filter_fast_comparison",
        "evaluation_semantics": "mutually_exclusive_terminal_outcomes_v2",
        "seed": int(seed),
        "episodes": int(episodes),
        "semantic_contract": {
            "alpha": float(contract.alpha),
            "bin_edges_m": [float(value) for value in contract.bin_edges],
            "radii_m": [float(value) for value in radii_m],
            "global_max_radius_m": global_radius_m,
        },
        "specs": {
            "global_max_filter": str(global_path),
            "adaptive_observed_bin_filter": str(adaptive_path),
            "adaptive_anti_stall_filter": str(adaptive_anti_stall_path),
        },
        "controllers": {},
    }
    for controller_name, controller in controllers.items():
        metrics["controllers"][controller_name] = {
            mode: evaluate(
                controller,
                contract,
                distance_scale,
                mode,
                episodes,
                seed,
            )
            for mode in MODES
        }

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2, ensure_ascii=False)

    print("[Safety filter fast comparison]")
    print("No model training was performed.")
    for mode in MODES:
        print(f"\n{mode}")
        for controller_name in controllers:
            print(
                f"  {controller_name}: "
                f"{compact_row(metrics['controllers'][controller_name][mode])}"
            )
    print(f"\nmetrics: {metrics_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-controller",
        default="Aebs/controller/best_model/best_model.zip",
    )
    parser.add_argument(
        "--semantic-checkpoint",
        default="results/mvp/01_semantic/semantic_encoder.pt",
    )
    parser.add_argument(
        "--output-dir",
        default="results/mvp/02_safety_filter_anti_stall_compare",
    )
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--target-speed", type=float, default=0.45)
    parser.add_argument("--distance-buffer-m", type=float, default=0.25)
    args = parser.parse_args()
    compare(
        args.baseline_controller,
        args.semantic_checkpoint,
        Path(args.output_dir),
        args.episodes,
        args.seed,
        args.target_speed,
        args.distance_buffer_m,
    )


if __name__ == "__main__":
    main()
