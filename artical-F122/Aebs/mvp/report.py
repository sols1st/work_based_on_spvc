"""Summarize MVP result files without running any experiment."""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable


def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def print_controller(metrics: Dict) -> None:
    print("\n[Controller]")
    baseline = metrics.get("baseline", {})
    robust = metrics.get("robust", {})
    for mode in ("exact", "uniform", "random_boundary", "worst_endpoint"):
        if mode not in baseline or mode not in robust:
            continue
        b = baseline[mode]
        r = robust[mode]
        print(
            f"{mode}: unsafe {percent(b['unsafe_rate'])} -> {percent(r['unsafe_rate'])}, "
            f"return {b['mean_return']:.3f} -> {r['mean_return']:.3f}"
        )


def print_uncertainty(metrics: Dict) -> None:
    print("\n[Uncertainty]")
    print(f"cells: {metrics['cells']}")
    print(f"count per cell: {metrics['min_count_per_cell']}..{metrics['max_count_per_cell']}")
    print(f"support low min: {metrics['support_low_min']}")
    print(f"support high max: {metrics['support_high_max']}")
    print(f"mean radius max: {metrics['mean_radius_max']}")


def print_sbc(name: str, metrics: Dict) -> None:
    verification = metrics["verification"]
    worst = verification["worst_case"]
    history = metrics.get("history", [])
    final = history[-1] if history else {}
    print(f"\n[SBC: {name}]")
    print(f"status: {verification['status']}")
    print(f"violations: {verification['violation_count']} / {verification['checked_states']}")
    print(f"min margin: {verification['min_margin']:.6f}")
    print(f"mean margin: {verification['mean_margin']:.6f}")
    print(f"worst state: d={worst['distance_m']:.3f} m, v={worst['speed']:.3f} m/s")
    if "semantic_multiplier" in worst:
        print(f"worst semantic multiplier: {worst['semantic_multiplier']}")
        print(f"worst disturbance: {worst['disturbance']}")
        print(f"B(s)={worst['current_barrier']:.6f}, max B(next)={worst['robust_next_barrier']:.6f}")
    if final:
        print(
            f"final train: loss={final['loss']:.6f}, "
            f"decrease_loss={final.get('decrease_loss', float('nan')):.6f}, "
            f"violation={percent(final['train_robust_decrease_violation_rate'])}"
        )


def metric_files(root: Path, pattern: str) -> Iterable[Path]:
    return sorted(root.glob(pattern), key=lambda item: str(item))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/mvp")
    args = parser.parse_args()
    root = Path(args.results)
    summary = root / "summary.json"
    if summary.exists():
        print(f"[Summary] {summary}")
    semantic = root / "01_semantic" / "metrics.json"
    if semantic.exists():
        metrics = load_json(semantic)
        point = metrics["semantic_point"]
        contract = metrics["state_conditional_contract"]
        print("\n[Semantic]")
        print(f"MAE: {point['mae_m']:.4f} m")
        print(f"RMSE: {point['rmse_m']:.4f} m")
        print(f"contract coverage: {percent(contract['test_coverage'])}")
        print(f"contract radii m: {[round(value, 4) for value in contract['radii_m']]}")
    controller = root / "02_controller" / "metrics.json"
    if controller.exists():
        print_controller(load_json(controller))
    uncertainty = root / "03_uncertainty" / "metrics.json"
    if uncertainty.exists():
        print_uncertainty(load_json(uncertainty))
    for path in metric_files(root, "04_robust_sbc*/metrics.json"):
        print_sbc(path.parent.name, load_json(path))


if __name__ == "__main__":
    main()
