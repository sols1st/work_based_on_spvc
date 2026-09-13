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


def print_filter_fast_comparison(name: str, metrics: Dict) -> None:
    print(f"\n[Safety filter comparison: {name}]")
    controllers = metrics.get("controllers", {})
    for mode in ("exact", "uniform", "random_boundary", "worst_endpoint"):
        if not any(mode in values for values in controllers.values()):
            continue
        print(f"{mode}:")
        for name in (
            "baseline",
            "global_max_filter",
            "adaptive_observed_bin_filter",
            "adaptive_anti_stall_filter",
        ):
            if name not in controllers or mode not in controllers[name]:
                continue
            values = controllers[name][mode]
            line = (
                f"  {name}: success={percent(values['success_rate'])}, "
                f"unsafe={percent(values['unsafe_rate'])}, "
                f"stopped_safe={percent(values['stopped_safe_outside_goal_rate'])}, "
                f"out_of_domain={percent(values['out_of_domain_rate'])}, "
                f"timeout={percent(values['timeout_rate'])}, "
                f"steps={values['mean_steps']:.1f}"
            )
            if values.get("intervention_rate") is not None:
                line += (
                    f", intervention={percent(values['intervention_rate'])}, "
                    f"extra_braking={values['mean_extra_braking']:.3f}"
                )
            if values.get("overshoot_cap_rate"):
                line += f", overshoot_cap={percent(values['overshoot_cap_rate'])}"
            print(line)


def print_grid_certificate_lp(metrics: Dict) -> None:
    print("\n[Grid certificate LP]")
    print(f"status: {metrics['status']}")
    print(
        f"grid: {metrics['grid_size']}x{metrics['grid_size']}, "
        f"decrease states: {metrics['decrease_states']}, "
        f"constraints: {metrics['linear_constraints']}"
    )
    if "epsilon_max" in metrics:
        print(f"epsilon_max: {metrics['epsilon_max']:.9f}")
        print(
            "maximum constraint residual: "
            f"{metrics['maximum_constraint_residual']:.3e}"
        )
    else:
        print(f"solver: {metrics['solver_message']}")
    print(f"certificate level: {metrics['certificate_level']}")


def print_standalone_ppo(metrics: Dict) -> None:
    print("\n[Standalone PPO: no runtime safety filter]")
    match = metrics["action_match"]
    print(
        f"teacher action MAE={match['action_mae']:.6f}, "
        f"max_abs={match['action_max_abs_error']:.6f}, "
        f"underbraking={percent(match['underbraking_rate'])}"
    )
    for mode in ("exact", "uniform", "random_boundary", "worst_endpoint"):
        values = metrics["student_evaluation"][mode]
        print(
            f"{mode}: success={percent(values['success_rate'])}, "
            f"unsafe={percent(values['unsafe_rate'])}, "
            f"timeout={percent(values['timeout_rate'])}, "
            f"steps={values['mean_steps']:.1f}"
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
    if "excluded_inevitable_states" in verification:
        print(f"excluded inevitable states: {verification['excluded_inevitable_states']}")
    print(f"min margin: {verification['min_margin']:.6f}")
    print(f"mean margin: {verification['mean_margin']:.6f}")
    zero_verification = metrics.get("zero_epsilon_verification")
    if zero_verification is not None and metrics.get("epsilon", 0.0) > 0.0:
        print(
            f"epsilon=0 diagnostic: violations={zero_verification['violation_count']}/"
            f"{zero_verification['checked_states']}, min_margin={zero_verification['min_margin']:.6f}"
        )
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
            f"unsafe violations={regions['unsafe_violation_count']}/{regions.get('unsafe_sample_count', regions['sample_count'])}, "
            f"min={regions['unsafe_min']:.6f} (target >= {regions['unsafe_target_min']:.6f})"
        )
        if "goal_violation_count" in regions:
            print(
                f"goal region: enforced={regions.get('goal_enforced', True)}, "
                f"violations={regions['goal_violation_count']}/{regions['sample_count']}, "
                f"max={regions['goal_max']:.6f} (target <= {regions['goal_target_max']:.6f})"
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
            f"epsilon={final.get('active_epsilon', metrics.get('epsilon', float('nan'))):.6f}, "
            f"decrease_loss={final.get('decrease_loss', float('nan')):.6f}, "
            f"max_decrease_loss={final.get('max_decrease_loss', float('nan')):.6f}, "
            f"topk_decrease_loss={final.get('topk_decrease_loss', float('nan')):.6f}, "
            f"init_loss={final.get('init_loss', float('nan')):.6f}, "
            f"unsafe_loss={final.get('unsafe_loss', float('nan')):.6f}, "
            f"goal_loss={final.get('goal_loss', float('nan')):.6f}, "
            f"violation={percent(final['train_robust_decrease_violation_rate'])}"
        )


def metric_files(root: Path, pattern: str) -> Iterable[Path]:
    return sorted(root.glob(pattern), key=lambda item: str(item))


def print_domain_analysis(metrics: Dict) -> None:
    physical = metrics["physical_recoverability"]
    discrete = metrics.get("discrete_recoverability")
    rollout = metrics["worst_endpoint_rollout"]
    comparison = metrics["comparison"]
    print("\n[Certificate domain analysis]")
    print(f"continuous recoverable/unrecoverable: {physical['recoverable_count']}/{physical['unrecoverable_count']}")
    print(f"initial-region minimum recoverability margin: {physical['initial_region_minimum_margin_m']:.6f} m")
    if discrete:
        print(f"discrete recoverable/unrecoverable: {discrete['recoverable_count']}/{discrete['unrecoverable_count']}")
        print(f"discrete initial-region minimum margin: {discrete['initial_region_minimum_margin_m']:.6f} m")
    print(
        f"worst-endpoint rollout: goal={rollout['goal_count']}, "
        f"stopped_safe={rollout['stopped_safe_count']}, unsafe={rollout['unsafe_count']}, "
        f"unresolved={rollout['unresolved_count']}"
    )
    if "discrete_recoverable_but_rollout_unsafe" in comparison:
        print(
            f"continuous recoverable but unsafe={comparison['continuous_recoverable_but_rollout_unsafe']}, "
            f"discrete recoverable but unsafe={comparison['discrete_recoverable_but_rollout_unsafe']}"
        )
        print(
            f"continuous/discrete conservative exclusions="
            f"{comparison['continuous_unrecoverable_but_rollout_not_unsafe']}/"
            f"{comparison['discrete_unrecoverable_but_rollout_not_unsafe']}"
        )
        print(
            "max discrete margin among rollout-unsafe states: "
            f"{comparison['max_discrete_margin_among_rollout_unsafe_m']}"
        )
    else:
        print(
            f"recoverable but unsafe={comparison['recoverable_but_rollout_unsafe']}, "
            f"unrecoverable but not unsafe={comparison['unrecoverable_but_rollout_not_unsafe']}"
        )


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
    for comparison_name in (
        "02_safety_filter_fast_compare",
        "02_safety_filter_anti_stall_compare",
    ):
        comparison_path = root / comparison_name / "metrics.json"
        if comparison_path.exists():
            print_filter_fast_comparison(comparison_name, load_json(comparison_path))
    standalone_ppo = root / "02_standalone_ppo_distilled" / "metrics.json"
    if standalone_ppo.exists():
        print_standalone_ppo(load_json(standalone_ppo))
    grid_lp = root / "04_grid_certificate_lp" / "metrics.json"
    if grid_lp.exists():
        print_grid_certificate_lp(load_json(grid_lp))
    for uncertainty in metric_files(root, "03*uncertainty/metrics.json"):
        print_uncertainty(uncertainty.parent.name, load_json(uncertainty))
    domain_analysis = root / "03_domain_analysis" / "metrics.json"
    if domain_analysis.exists():
        print_domain_analysis(load_json(domain_analysis))
    for path in metric_files(root, "04_robust_sbc*/metrics.json"):
        print_sbc(path.parent.name, load_json(path))


if __name__ == "__main__":
    main()
