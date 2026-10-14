"""Mainline step24: frozen step22 SBC violation/noise-integration diagnostic.

No training, no policy change, no data collection and no replacement of legacy
results. Corrected interval inputs are a separate sensitivity check at samples.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from auto_LiRPA import BoundedModule

from Aebs.dino_latent.barrier_diagnostic_logic import select_cases, classify_point, target_scope
from Aebs.dino_latent.prepare_expanded_data import sha256
from Aebs.dino_latent.spvc_saved_actor import load_spvc_pair
from Aebs.semantic_spvc.corrected_verify import CorrectedIntervalVTVerifier
from Aebs.system.env import Aebs
from Aebs.VT.utils import MLP
from Aebs.VT.verify import VTVerifier


def write_json(path, data):
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)


def load_samples(path, scope):
    keys = ("state", "distance_m", "barrier", "expected", "margin", "compensated_margin",
            "legacy_band", "below_unsafe_safe", "all_safe")
    with np.load(path, allow_pickle=False) as stream:
        arrays = {key: stream[key].copy() for key in keys}
    n = int(scope["grid_total"])
    if arrays["state"].shape != (n, 2):
        raise ValueError("saved state grid shape mismatch")
    for key in keys[1:]:
        if arrays[key].shape != (n,):
            raise ValueError("bad sample array: " + key)
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("nonfinite saved grid values")
    if not np.allclose(arrays["barrier"]-arrays["expected"], arrays["margin"], rtol=1e-5, atol=1e-5):
        raise ValueError("saved residual arrays disagree")
    for name in ("legacy_band", "below_unsafe_safe", "all_safe"):
        mask = arrays[name]
        if mask.dtype != np.bool_:
            raise ValueError("scope masks must be boolean")
        source = scope["scopes"][name]
        if (int(mask.sum()) != source["checked_states"]
                or int(np.sum(mask & (arrays["margin"] <= 0))) != source["hard_violations"]
                or int(np.sum(mask & (arrays["compensated_margin"] <= 0))) != source["compensated_violations"]):
            raise ValueError("saved scope counts disagree: " + name)
    return [{key: arrays[key][i].tolist() for key in keys} for i in range(n)]


def summarize_locations(rows, scope, key):
    selected = [r for r in rows if r[scope] and r[key] <= 0]
    if not selected:
        return {"count": 0}
    distances = np.asarray([r["distance_m"] for r in selected])
    speeds = np.asarray([r["state"][1] for r in selected])
    hist, _, _ = np.histogram2d(distances, speeds, bins=([5, 6, 8, 12, 16.00001], [0, 0.5, 1, 2, 3.00001]))
    return {
        "count": len(selected), "distance_range_m": [float(distances.min()), float(distances.max())],
        "speed_range_mps": [float(speeds.min()), float(speeds.max())],
        "distance_quantiles_m": np.quantile(distances, [0, .25, .5, .75, 1]).tolist(),
        "low_speed_at_most_0_5_count": int(np.sum(speeds <= .5)),
        "distance_edges_m": [5, 6, 8, 12, 16.00001], "speed_edges_mps": [0, .5, 1, 2, 3.00001],
        "histogram_distance_by_speed": hist.astype(int).tolist(),
        "outside_histogram_count": len(selected)-int(hist.sum()),
        "worst": min(selected, key=lambda r: r[key]),
    }


def bound_region_boxes(verifier, boxes, bins=20):
    """Partition the region boxes themselves, avoiding partial-overlap selection."""
    bounds = []
    for box in boxes:
        axes = [np.linspace(float(lo), float(hi), bins+1, dtype=np.float32)
                for lo, hi in zip(box.low, box.high)]
        low = np.stack(np.meshgrid(*(a[:-1] for a in axes), indexing="ij"), axis=-1).reshape(-1, 2)
        high = np.stack(np.meshgrid(*(a[1:] for a in axes), indexing="ij"), axis=-1).reshape(-1, 2)
        lo, hi = verifier.compute_bounds_on_set(low, high)
        if not np.isfinite([lo, hi]).all() or lo > hi:
            raise ValueError("invalid region interval bounds")
        bounds.append({"lower": lo, "upper": hi, "cells": len(low)})
    if not bounds:
        raise ValueError("empty region collection")
    return {"lower": min(b["lower"] for b in bounds), "upper": max(b["upper"] for b in bounds),
            "bins_per_dimension": bins, "boxes": bounds}


def integrate_point(barrier, env, corrected, legacy, state, action, bins, chunk_size):
    legacy.pmass_n = np.full(2, bins, dtype=int)
    legacy._cached_pmass_grid = None
    mass, low, high = legacy.get_pmass_grid()
    mass_sum = float(mass.double().sum())
    if abs(mass_sum-1) > 1e-5 or not torch.all(mass > 0).item():
        raise ValueError("invalid noise probability masses")
    sums = {key: 0.0 for key in ("lower", "upper", "midpoint", "lower_corner")}
    outside = 0.0
    with torch.no_grad():
        nominal = env.v_next(state, action)
        current = float(barrier(state).item())
        legacy_expected = float(legacy.compute_expected_l(state, action, mass, low, high).item())
        if not np.isfinite([current, legacy_expected]).all():
            raise ValueError("nonfinite barrier/legacy expectation")
        domain_low = torch.as_tensor(env.observation_space.low, device=state.device, dtype=state.dtype)
        domain_high = torch.as_tensor(env.observation_space.high, device=state.device, dtype=state.dtype)
        for start in range(0, len(mass), chunk_size):
            lower, upper = nominal+low[start:start+chunk_size], nominal+high[start:start+chunk_size]
            midpoint = (lower+upper)/2
            lb, ub = corrected._interval_bounds(lower, upper)
            mid = barrier(midpoint).flatten()
            corner = barrier(lower).flatten()
            lb, ub = lb.flatten(), ub.flatten()
            values = (lb, ub, mid, corner)
            if any(not torch.isfinite(value).all().item() for value in values):
                raise ValueError("nonfinite interval/integration output")
            slack = 1e-5 * (1 + torch.maximum(lb.abs(), ub.abs()))
            if not ((lb <= mid+slack) & (mid <= ub+slack) & (lb <= corner+slack) & (corner <= ub+slack)).all().item():
                raise ValueError("interval does not enclose sampled values; do not interpret as bounds")
            weights = mass[start:start+chunk_size].double()
            for key, value in zip(sums, values):
                sums[key] += float((weights*value.double()).sum())
            out = ((midpoint < domain_low) | (midpoint > domain_high)).any(dim=1)
            outside += float(weights[out].sum())
    return {
        "bins_per_dimension": bins, "noise_cells": bins*bins, "mass_sum": mass_sum,
        "barrier": current, "legacy_residual": legacy_expected-current,
        "interval_lower_residual": sums["lower"]-current,
        "interval_upper_residual": sums["upper"]-current,
        "midpoint_residual_estimate": sums["midpoint"]-current,
        "lower_corner_residual_estimate": sums["lower_corner"]-current,
        "legacy_minus_lower_corner": legacy_expected-sums["lower_corner"],
        "noise_midpoint_mass_outside_state_domain": outside,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path("results/dino_expanded_v1")
    parser.add_argument("--spvc-dir", type=Path, default=root / "22_R2_original_spvc_scope_compare")
    parser.add_argument("--representation", type=Path, default=root / "08_R2_group_alignment/dino_safety_latent.pt")
    parser.add_argument("--controller", type=Path, default=root / "09_R2_ppo/latent_ppo.zip")
    parser.add_argument("--output-dir", type=Path, default=root / "24_R2_sbc_mainline_diagnostic")
    parser.add_argument("--max-cases", type=int, default=12)
    parser.add_argument("--noise-bins", default="10,20,40,80")
    parser.add_argument("--interval-batch-size", type=int, default=2048)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--target-probability", type=float, default=.90)
    args = parser.parse_args()
    try:
        bins_list = sorted(set(int(x) for x in args.noise_bins.split(",")))
    except ValueError:
        parser.error("noise-bins must contain integer resolutions")
    if (args.output_dir.exists() or not 3 <= args.max_cases <= 64 or args.interval_batch_size < 1
            or not bins_list or bins_list[0] != 10 or bins_list[-1] > 100
            or not 0 < args.target_probability < 1):
        parser.error("new output required; max-cases 3..64; resolutions start at 10 and end <=100")
    torch.set_num_threads(1)
    scope_path = args.spvc_dir / "scope_comparison.json"
    samples_path = args.spvc_dir / "scope_samples.npz"
    for path in (scope_path, samples_path, args.representation, args.controller, args.spvc_dir / "sbc.pt"):
        if not path.is_file():
            parser.error("missing input: " + str(path))
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; use --device cpu")
    scope = json.loads(scope_path.read_text(encoding="utf-8"))
    rows = load_samples(samples_path, scope)
    _, actor, _, identities = load_spvc_pair(args.representation, args.controller, args.spvc_dir)
    env = Aebs(0.05)
    if not np.isclose(env.std1, actor.policy.state_distance_scale_m, rtol=1e-6, atol=1e-7):
        raise ValueError("current VT distance scale differs from saved run; investigate data provenance")
    if not np.allclose([r["distance_m"] for r in rows],
                       [r["state"][0]*env.std1 for r in rows], rtol=1e-6, atol=1e-6):
        raise ValueError("saved metric distances do not match current VT coordinates")
    device = torch.device(args.device)
    policy = actor.policy.to(device).eval()
    barrier = MLP([2, 16, 8, 1], activation="tanh", square_output=True).to(device)
    barrier.load_state_dict(torch.load(args.spvc_dir / "sbc.pt", map_location=device), strict=True)
    barrier.eval()
    for model in (policy, barrier):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    learner = SimpleNamespace(p_net=policy, l_model=barrier)
    legacy = VTVerifier(learner, env, BoundedModule(barrier, torch.zeros(1, 2, device=device), device=device), 1, .9, False)
    corrected = CorrectedIntervalVTVerifier(
        learner, env, BoundedModule(barrier, torch.zeros(1, 2, device=device), device=device), 1, .9, False
    )
    legacy.device = device
    corrected.device = device
    corrected.batch_size = args.interval_batch_size
    region_bounds = {"initial": bound_region_boxes(corrected, env.init_spaces),
                     "unsafe": bound_region_boxes(corrected, env.unsafe_spaces)}
    target = target_scope(rows, region_bounds["initial"]["upper"], region_bounds["unsafe"]["lower"],
                          args.target_probability)
    print("target=%.2f corrected init upper=%.6f unsafe lower=%.6f threshold=%.6f points=%d separation=%s" % (
        args.target_probability, target["initial_upper"], target["unsafe_lower"],
        target["raw_barrier_threshold"], target["checked_grid_points"], target["region_separation_sufficient"]), flush=True)
    selected = select_cases(rows, args.max_cases)
    report = {
        "experiment": "R2_frozen_step22_SBC_mainline_violation_diagnostic",
        "status": "running", "level": "selected-point numerical diagnostic, NOT a certificate",
        "training_run": False, "models_frozen": True, "test_used": False,
        "model_identities": identities, "scope_samples_sha256": sha256(samples_path),
        "vt_data_sha256": sha256(Path("Aebs/data/Downsampled.h5")),
        "vt_utils_sha256": sha256(Path(__file__).resolve().parents[1] / "VT/utils.py"),
        "noise_support": [np.asarray(x).tolist() for x in env.noise_bounds],
        "scope_masks": "three saved step22 masks unchanged; separate target-probability point mask added",
        "target_scope": target, "corrected_region_bounds": region_bounds,
        "barrier_output": "original MLP square_output=True flag actually applies softplus",
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "sign_tolerance_diagnostic_only": 1e-5,
        "locations": {name: {"hard": summarize_locations(rows, name, "margin"),
                              "compensated": summarize_locations(rows, name, "compensated_margin")}
                      for name in ("legacy_band", "below_unsafe_safe", "all_safe", "target_probability_scope")},
        "selected_cases": [],
        "limitations": ["midpoint quadrature is not a rigorous expectation bound",
                        "interval arithmetic uses standard floats, not outward-rounded proof arithmetic",
                        "selected states do not establish whole-domain validity",
                        "selection uses legacy failures; corrected-only violations at other points may be missed",
                        "raw expectation signs are not a check of every original training epsilon constraint",
                        "target probability defines a candidate proxy scope, not image-path error coverage",
                        "noise is added after nominal dynamics exactly as legacy; no new clipping or terminal overrides",
                        "local-gradient state-grid compensation is unchanged, not a certified global Lipschitz bound"],
    }
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "metrics.json", report)
    for position, (index, reasons) in enumerate(selected):
        saved = rows[index]
        state = torch.tensor([saved["state"]], dtype=torch.float32, device=device)
        with torch.no_grad():
            action = policy(torch.zeros(1, 4, device=device), state)
        case = {"grid_index": index, "selection_reason": reasons, "saved": saved,
                "raw_action": float(action.item()), "clipped_action": float(action.clamp(-3, 3).item()), "refinement": []}
        for bins in bins_list:
            row = integrate_point(barrier, env, corrected, legacy, state, action, bins, args.interval_batch_size)
            if bins == 10:
                # Confirm snapshot identity and legacy computation before interpreting a refined result.
                if not np.allclose([row["barrier"], row["legacy_residual"]],
                                   [saved["barrier"], -saved["margin"]], rtol=1e-5, atol=1e-4):
                    raise ValueError("cannot reproduce saved point %d: %s" % (index, row))
            penalty = saved["margin"]-saved["compensated_margin"]
            row["interval_upper_plus_saved_compensation_residual"] = row["interval_upper_residual"]+penalty
            row["interval_upper_plus_training_epsilon_0_1"] = row["interval_upper_residual"]+.1
            row["interpretation"] = classify_point(row["interval_lower_residual"], row["interval_upper_residual"])
            case["refinement"].append(row)
            print("case %d/%d grid=%d d=%.4f v=%.4f bins=%d legacy=%+.6f midpoint=%+.6f interval=[%+.6f,%+.6f]" % (
                position+1, len(selected), index, saved["distance_m"], saved["state"][1], bins,
                row["legacy_residual"], row["midpoint_residual_estimate"], row["interval_lower_residual"], row["interval_upper_residual"]
            ), flush=True)
        case["final_interpretation"] = case["refinement"][-1]["interpretation"]
        case["coarse_upper_positive_fine_upper_negative"] = (
            case["refinement"][0]["interval_upper_residual"] > 1e-5
            and case["refinement"][-1]["interval_upper_residual"] < -1e-5
        )
        report["selected_cases"].append(case)
        write_json(args.output_dir / "metrics.json", report)
    report["status"] = "diagnostic_complete" if selected else "no_saved_violations_to_refine"
    report["selected_count"] = len(selected)
    report["final_interpretation_counts"] = dict(Counter(c["final_interpretation"] for c in report["selected_cases"]))
    write_json(args.output_dir / "metrics.json", report)
    print("\n[R2 frozen SBC mainline diagnostic — no training]")
    for name, item in report["locations"].items():
        print("%s: hard=%d compensated=%d hard_distance_range=%s hard_speed_range=%s" % (
            name, item["hard"]["count"], item["compensated"]["count"],
            item["hard"].get("distance_range_m"), item["hard"].get("speed_range_mps")
        ))
    print("selected:", len(selected), "interpretations:", report["final_interpretation_counts"])
    print("metrics:", args.output_dir / "metrics.json")


if __name__ == "__main__":
    main()
