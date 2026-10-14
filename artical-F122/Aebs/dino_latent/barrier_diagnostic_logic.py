"""Dependency-free selection and conservative labels for point diagnostics."""

import math


def target_scope(rows, initial_upper, unsafe_lower, probability):
    """Unshifted B/a normalization; only a point mask, not a certificate."""
    if (not all(math.isfinite(x) for x in (initial_upper, unsafe_lower, probability))
            or initial_upper <= 0 or not 0 < probability < 1):
        raise ValueError("positive initial bound and probability in (0,1) required")
    threshold = initial_upper / (1-probability)
    for row in rows:
        row["target_probability_scope"] = bool(row["all_safe"] and row["barrier"] <= threshold)
    return {"probability_target": probability, "normalization": "B / initial_upper; no domain-minimum shift",
            "initial_upper": initial_upper, "unsafe_lower": unsafe_lower,
            "raw_barrier_threshold": threshold,
            "region_separation_sufficient": unsafe_lower >= threshold,
            "checked_grid_points": sum(r["target_probability_scope"] for r in rows),
            "not_a_certified_probability": True}


def select_cases(rows, max_cases):
    if max_cases < 3:
        raise ValueError("at least three case slots required")
    selected = []
    groups = [
        ("band_compensated", [i for i, r in enumerate(rows) if r["legacy_band"] and r["compensated_margin"] <= 0], "compensated_margin"),
        ("target_scope_hard", [i for i, r in enumerate(rows) if r["target_probability_scope"] and r["margin"] <= 0], "margin"),
        ("target_scope_compensated", [i for i, r in enumerate(rows) if r["target_probability_scope"] and r["compensated_margin"] <= 0], "compensated_margin"),
    ]
    reasons = {}
    for name, indices, key in groups:
        if indices:
            index = min(indices, key=lambda i: (rows[i][key], i))
            if index not in selected:
                selected.append(index)
            reasons.setdefault(index, []).append(name+"_worst")
    candidates = sorted(set(i for _, indices, _ in groups for i in indices))
    # Cover spatially separated failures rather than choosing only adjacent minima.
    while len(selected) < max_cases:
        remaining = [i for i in candidates if i not in selected]
        if not remaining:
            break
        def separation(i):
            return min(((rows[i]["distance_m"]-rows[j]["distance_m"])/11.0)**2
                       + ((rows[i]["state"][1]-rows[j]["state"][1])/3.0)**2 for j in selected)
        index = max(remaining, key=lambda i: (separation(i), -i))
        selected.append(index)
        reasons[index] = ["spatially_separated_violation"]
    return [(i, reasons[i]) for i in selected]


def classify_point(lower_residual, upper_residual, tolerance=1e-5):
    if not all(math.isfinite(x) for x in (lower_residual, upper_residual, tolerance)) or tolerance < 0:
        raise ValueError("finite residual bounds and nonnegative tolerance required")
    if lower_residual > upper_residual + tolerance:
        raise ValueError("inverted expectation bounds")
    if lower_residual > tolerance:
        return "interval_supports_expected_increase_at_sample"
    if upper_residual < -tolerance:
        return "interval_supports_strict_decrease_at_sample"
    return "unresolved_interval_or_near_zero"
