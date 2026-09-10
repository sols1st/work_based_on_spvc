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


def print_safety_filter(metrics: Dict) -> None:
    print("\n[Safety filter]")
    baseline = metrics.get("baseline", {})
    filtered = metrics.get("filtered", {})
    for mode in ("exact", "uniform", "random_boundary", "worst_endpoint"):
        if mode not in baseline or mode not in filtered:
            continue
        before = baseline[mode]
        after = filtered[mode]
        print(
            f"{mode}: unsafe {percent(before['unsafe_rate'])} -> {percent(after['unsafe_rate'])}, "
            f"return {before['mean_return']:.3f} -> {after['mean_return']:.3f}"
        )


def print_uncertainty(name: str, metrics: Dict) -> None:
    print(f"\n[Uncertainty: {name}]")
    if "source" in metrics:
        print(f"source: {metrics['source']}")
    else:
        print("source: legacy_semantic_action_delta (historical; do not use for SBC)")
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
    if "excluded_terminal_states" in verification:
        print(f"excluded terminal states: {verification['excluded_terminal_states']}")
    if "excluded_goal_states" in verification:
        print(f"excluded goal states: {verification['excluded_goal_states']}")
    if "excluded_unsafe_states" in verification:
        print(f"excluded unsafe states: {verification['excluded_unsafe_states']}")
    print(f"min margin: {verification['min_margin']:.6f}")
    print(f"mean margin: {verification['mean_margin']:.6f}")
    print(f"worst state: d={worst['distance_m']:.3f} m, v={worst['speed']:.3f} m/s")
    if "semantic_multiplier" in worst:
        print(f"worst semantic multiplier: {worst['semantic_multiplier']}")
        print(f"worst disturbance: {worst['disturbance']}")
        print(f"B(s)={worst['current_barrier']:.6f}, max B(next)={worst['robust_next_barrier']:.6f}")
    regions = verification.get("regions")
    if regions:
        print(
            "regions: "
            f"init violations={regions['init_violation_count']}/{regions['sample_count']}, "
            f"max={regions['init_max']:.6f} (target <= {regions['init_target_max']:.6f}); "
            f"unsafe violations={regions['unsafe_violation_count']}/{regions['sample_count']}, "
            f"min={regions['unsafe_min']:.6f} (target >= {regions['unsafe_target_min']:.6f})"
        )
    nonnegative = verification.get("nonnegative")
    if nonnegative:
        print(
            f"nonnegative: violations={nonnegative['violation_count']}, "
            f"min={nonnegative['minimum']:.6f} (target >= {nonnegative['target_min']:.6f})"
        )
    if final:
        print(
            f"final train: loss={final['loss']:.6f}, "
            f"decrease_loss={final.get('decrease_loss', float('nan')):.6f}, "
            f"max_decrease_loss={final.get('max_decrease_loss', float('nan')):.6f}, "
            f"topk_decrease_loss={final.get('topk_decrease_loss', float('nan')):.6f}, "
            f"init_loss={final.get('init_loss', float('nan')):.6f}, "
            f"unsafe_loss={final.get('unsafe_loss', float('nan')):.6f}, "
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
    safety_filter = root / "02_safety_filter" / "metrics.json"
    if safety_filter.exists():
        print_safety_filter(load_json(safety_filter))
    for uncertainty in metric_files(root, "03*uncertainty/metrics.json"):
        print_uncertainty(uncertainty.parent.name, load_json(uncertainty))
    for path in metric_files(root, "04_robust_sbc*/metrics.json"):
        print_sbc(path.parent.name, load_json(path))


if __name__ == "__main__":
    main()
