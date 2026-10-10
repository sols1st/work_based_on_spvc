"""Post-training scope comparison; preserve legacy noise, bounds and inequalities.

This is NOT a replacement verifier or a continuous-domain certificate.
The full grid is evaluated once with no early exit, using one fixed model pair.
"""

import math


def legacy_band_mask(normalized, threshold):
    return (normalized > 0.95 * threshold) & (normalized < threshold)


def summarize_scope(rows, scope, grid_total, legacy_denominator):
    selected = [row for row in rows if row[scope]]
    count = len(selected)
    finite = all(
        math.isfinite(row[key])
        for row in selected for key in ("barrier", "expected", "margin", "compensated_margin")
    )
    hard = sum(row["margin"] <= 0 for row in selected)
    compensated = sum(row["compensated_margin"] <= 0 for row in selected)
    numeric_ok = count > 0 and finite
    worst = min(selected, key=lambda row: row["margin"]) if numeric_ok else None
    return {
        "checked_states": count,
        "grid_total": grid_total,
        "finite": finite,
        "hard_violations": hard if finite else None,
        "hard_violation_fraction_checked": hard / count if numeric_ok else None,
        "hard_violation_fraction_total_grid": hard / grid_total if finite else None,
        "compensated_violations": compensated if finite else None,
        "min_margin": worst["margin"] if worst else None,
        "mean_margin": sum(row["margin"] for row in selected) / count if numeric_ok else None,
        "min_compensated_margin": min(row["compensated_margin"] for row in selected) if numeric_ok else None,
        "zero_hard_violations_nonempty": numeric_ok and hard == 0,
        # Literal legacy arithmetic is recorded even if the selected set is empty.
        "legacy_ratio_rule_only": (hard / legacy_denominator <= 0.001)
        if finite and legacy_denominator > 0 else None,
        "legacy_denominator": legacy_denominator,
        "status": "empty_scope_not_evidence" if count == 0 else (
            "nonfinite_not_evidence" if not finite else (
                "sampled_zero_hard_violations" if hard == 0 else "sampled_violated"
            )
        ),
        "worst_state": {key: worst[key] for key in (
            "state", "distance_m", "barrier", "expected", "margin"
        )} if worst else None,
    }


def compare_scopes(verifier, output_dir, legacy_denominator, k_except_l=1.2):
    import json
    import numpy as np
    import torch

    learner, env = verifier.learner, verifier.env
    _, init_upper = verifier.compute_bound_init(verifier.grid_size)
    unsafe_lower, _ = verifier.compute_bound_unsafe(verifier.grid_size)
    domain_lower, _ = verifier.compute_bound_domain(verifier.grid_size)
    if not all(math.isfinite(v) for v in (init_upper, unsafe_lower, domain_lower)):
        raise ValueError("nonfinite region bounds; comparison is invalid")
    threshold = verifier.normalize(unsafe_lower, init_upper, domain_lower)
    grid_total = int(np.prod(verifier.grid_size))
    delta = float(np.linalg.norm(
        (env.observation_space.high - env.observation_space.low) / verifier.grid_size / 2
    ))
    pmass, noise_low, noise_high = verifier.get_pmass_grid()
    rows = []
    for start in range(0, grid_total, verifier.batch_size):
        end = min(start + verifier.batch_size, grid_total)
        states = verifier.v_get_grid_item(torch.arange(start, end)).to(verifier.device)
        with torch.no_grad():
            action = learner.p_net(torch.zeros((len(states), 4), device=verifier.device), states)
            value = learner.l_model(states).flatten()
            expected = verifier.compute_expected_l(states, action, pmass, noise_low, noise_high)
            normalized = verifier.normalize(value, init_upper, domain_lower)
            unsafe = torch.zeros(len(states), dtype=torch.bool, device=verifier.device)
            for box in env.unsafe_spaces:
                low = torch.as_tensor(box.low, device=verifier.device, dtype=states.dtype)
                high = torch.as_tensor(box.high, device=verifier.device, dtype=states.dtype)
                unsafe |= ((states >= low) & (states <= high)).all(dim=1)
            # Exact active legacy predicate, including strict endpoints.
            narrow = legacy_band_mask(normalized, threshold)
            # Controlled ablation: remove only the 95% lower cutoff; exclude unsafe.
            broad = (~unsafe) & (normalized < threshold)
            safe = ~unsafe
        lips = verifier.compute_local_lipschitz_batch(states)
        margin = value - expected
        compensated = margin - lips * k_except_l * delta
        arrays = [v.detach().cpu().numpy() for v in (
            states, value, expected, margin, compensated, narrow, broad, safe
        )]
        for i in range(len(states)):
            state = arrays[0][i].tolist()
            rows.append({
                "state": state, "distance_m": float(state[0] * env.std1),
                "barrier": float(arrays[1][i]), "expected": float(arrays[2][i]),
                "margin": float(arrays[3][i]), "compensated_margin": float(arrays[4][i]),
                "legacy_band": bool(arrays[5][i]), "below_unsafe_safe": bool(arrays[6][i]),
                "all_safe": bool(arrays[7][i]),
            })
        print("scope comparison grid %d/%d" % (end, grid_total), flush=True)
    all_finite = all(math.isfinite(row[key]) for row in rows for key in (
        "barrier", "expected", "margin", "compensated_margin"
    ))
    bounds_ordered = domain_lower < init_upper < unsafe_lower
    probability = 1 - (init_upper - domain_lower) / (unsafe_lower - domain_lower) if bounds_ordered else None
    scopes = {name: summarize_scope(rows, name, grid_total, legacy_denominator) for name in (
        "legacy_band", "below_unsafe_safe", "all_safe"
    )}
    result = {
        "experiment": "R2_original_SPVC_fixed_checkpoint_scope_comparison",
        "level": "legacy sampled diagnostic, NOT a formal safety certificate",
        "all_grid_values_finite": all_finite,
        "status": "comparison_complete" if all_finite else "invalid_nonfinite",
        "grid_total": grid_total,
        "initial_upper": init_upper, "unsafe_lower": unsafe_lower, "domain_lower": domain_lower,
        "bounds_ordered": bounds_ordered, "normalized_unsafe_threshold": threshold,
        "region_ratio_only_not_certified_probability": probability,
        "noise_cells": int(pmass.numel()), "k_except_l": k_except_l, "delta": delta,
        "hard_rule": "legacy: expected - B >= 0 is violation (equality violates)",
        "scopes": scopes,
        "scope_definitions": {
            "legacy_band": "0.95 * normalized_unsafe_lower < normalized_B < normalized_unsafe_lower; no extra exclusions",
            "below_unsafe_safe": "not unsafe AND normalized_B < normalized_unsafe_lower",
            "all_safe": "all grid states outside original unsafe boxes; no terminal exclusions",
        },
        "limitations": [
            "legacy interval implementation and local gradient compensation unchanged",
            "broad scopes are diagnostics, not a claim of satisfying the paper theorem",
            "same proxy PPO/SBC in all scopes; no CP box or image-path coverage",
            "empty scopes and nonfinite values cannot establish safety",
        ],
    }
    with (output_dir / "scope_comparison.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
    np.savez_compressed(output_dir / "scope_samples.npz", **{
        key: np.asarray([row[key] for row in rows]) for key in rows[0]
    })
    print("\n[R2 original-SPVC same-checkpoint scope comparison]")
    print("grid=%d finite=%s init_upper=%.6f unsafe_lower=%.6f domain_lower=%.6f" % (
        grid_total, all_finite, init_upper, unsafe_lower, domain_lower
    ))
    for name, data in scopes.items():
        print("%s: checked=%d/%d hard=%s/%d compensated=%s min_margin=%s status=%s" % (
            name, data["checked_states"], grid_total, data["hard_violations"],
            data["checked_states"], data["compensated_violations"], data["min_margin"], data["status"]
        ))
    print("region ratio only (NOT certified probability):", probability)
    print("metrics:", output_dir / "scope_comparison.json")
    return result
