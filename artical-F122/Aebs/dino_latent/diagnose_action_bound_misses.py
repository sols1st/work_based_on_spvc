"""Read-only development diagnosis of action-bound fitting/generalization.

Neighbor labels are diagnostic evidence, never substituted for validation
actions. No model updates, CP, test, rollout or SBC verification.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from Aebs.dino_latent.check_environment_action_envelope import group_actions
from Aebs.dino_latent.diagnose_environment_barrier import write_json
from Aebs.dino_latent.prepare_expanded_data import sha256


def label_grid(stream):
    states, _, appearances = group_actions(stream)
    distances, speeds = np.unique(states[:, 0]), np.unique(states[:, 1])
    if len(states) != len(distances)*len(speeds):
        raise ValueError("expected a complete train distance-speed rectangle")
    di = np.searchsorted(distances, stream["distance_m"])
    vi = np.searchsorted(speeds, stream["speed_mps"])
    errors = stream["image_action"]-stream["q_action"]
    shape = (len(distances), len(speeds))
    def reduce(mask):
        low, high = np.full(shape, np.inf), np.full(shape, -np.inf)
        np.minimum.at(low, (di[mask], vi[mask]), errors[mask])
        np.maximum.at(high, (di[mask], vi[mask]), errors[mask])
        if not np.isfinite(low).all() or not np.isfinite(high).all():
            raise ValueError("incomplete appearance grid")
        return low, high
    all_bounds = reduce(np.ones(len(errors), bool))
    by_appearance = {name: reduce(stream["appearance"] == name) for name in appearances}
    return distances, speeds, all_bounds, by_appearance


def bracket_bounds(train_distances, low, high, query_distances, query_speed_indices):
    right = np.searchsorted(train_distances, query_distances, side="left")
    valid = (right > 0) & (right < len(train_distances))
    left = np.clip(right-1, 0, len(train_distances)-1)
    right = np.clip(right, 0, len(train_distances)-1)
    lo = np.minimum(low[left, query_speed_indices], low[right, query_speed_indices])
    hi = np.maximum(high[left, query_speed_indices], high[right, query_speed_indices])
    return lo, hi, valid, left, right


def miss_partition(errors, predicted, neighbor_low, neighbor_high, has_evidence, scope, tolerance=1e-6):
    miss = (errors < predicted[:, 0]-tolerance) | (errors > predicted[:, 1]+tolerance)
    inside = (errors >= neighbor_low-tolerance) & (errors <= neighbor_high+tolerance)
    chosen = scope & miss
    return dict(pairs=int(scope.sum()), predicted_interval_misses=int(chosen.sum()),
                missed_but_inside_observed_neighbor_hull=int((chosen & has_evidence & inside).sum()),
                missed_and_outside_observed_neighbor_hull=int((chosen & has_evidence & ~inside).sum()),
                missed_without_required_neighbor_evidence=int((chosen & ~has_evidence).sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path("results/environment_nbc_20261013_001")
    parser.add_argument("--diagnostic-dir", type=Path, default=root/"01_frozen_diagnostic")
    parser.add_argument("--fit-dir", type=Path, default=root/"06_action_bounds_train_v1")
    parser.add_argument("--spvc-dir", type=Path, default=Path("results/dino_spvc_minimal_v1"))
    parser.add_argument("--representation", type=Path, default=Path("results/dino_expanded_v1/08_R2_group_alignment/dino_safety_latent.pt"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("new output directory required")
    started = time.monotonic()
    fit_path = args.fit_dir/"metrics.json"
    fit = json.loads(fit_path.read_text())
    if fit.get("status") != "development_only_action_bounds_fitted" or fit.get("test_used") is not False or fit.get("frozen_models_unchanged") is not True or fit.get("pair_data_unchanged") is not True:
        raise ValueError("fit not complete or not frozen development-only")
    files = dict(fit_metrics=fit_path, diagnostic_metrics=args.diagnostic_dir/"metrics.json",
                 bounds_checkpoint=args.fit_dir/"action_error_bounds.pt",
                 actor=args.spvc_dir/"latent_ppo_spvc.pt", barrier=args.spvc_dir/"sbc.pt", representation=args.representation)
    files.update({split+"_pairs": args.diagnostic_dir/(split+"_pairs.npz") for split in ("train", "validation")})
    files.update({split+"_predictions": args.fit_dir/(split+"_predictions.npz") for split in ("train", "validation")})
    hashes = {k: sha256(p) for k, p in files.items()}
    for k in ("actor", "barrier", "representation"):
        if hashes[k] != fit["source_hashes"][k]:
            raise ValueError("frozen model identity changed")
    if hashes["bounds_checkpoint"] != fit["checkpoint_sha256"] or hashes["diagnostic_metrics"] != fit["diagnostic_sha256"]:
        raise ValueError("checkpoint/diagnostic changed")
    for split in ("train", "validation"):
        if hashes[split+"_pairs"] != fit["pair_hashes"][split]:
            raise ValueError("paired data identity changed")
    report = dict(experiment="frozen_action_bounds_neighbor_label_diagnosis_v1", status="running",
                  training_run=False, model_updates=False, test_used=False, formal_calibration=False,
                  neighbor_scope="same speed; bracket distances or fixed physical distance windows; all 12 appearances and same appearance reported separately",
                  interpretation="outside an observed neighbor hull means absent from those recorded labels, not physically impossible or a proof of causal failure",
                  source_hashes=hashes, script_sha256=sha256(Path(__file__)), results={})
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir/"metrics.json", report)
    try:
        with np.load(files["train_pairs"], allow_pickle=False) as train:
            distances, speeds, (train_low, train_high), appearance_bounds = label_grid(train)
            train_errors = train["image_action"]-train["q_action"]
            global_low, global_high = float(train_errors.min()), float(train_errors.max())
            with np.load(files["train_predictions"], allow_pickle=False) as saved:
                pred = saved["predicted_error_bounds"]
            if pred.shape != (len(train_errors), 2):
                raise ValueError("train prediction shape mismatch")
            miss = (train_errors < pred[:, 0]-1e-6) | (train_errors > pred[:, 1]+1e-6)
            report["train_fit_failure"] = dict(pairs=len(train_errors), misses=int(miss.sum()),
                                              distance_count=len(distances), speed_count=len(speeds),
                                              meaning="these are recorded training labels missed by the fitted interval, not heldout generalization failures")
        with np.load(files["validation_pairs"], allow_pickle=False) as val:
            d, v = val["distance_m"], val["speed_mps"]
            errors = val["image_action"]-val["q_action"]
            names, band = val["appearance"], val["fixed_legacy_band"]
            vi = np.searchsorted(speeds, v)
            if np.any(vi >= len(speeds)) or not np.array_equal(speeds[vi], v):
                raise ValueError("validation speed not on train speed grid")
            with np.load(files["validation_predictions"], allow_pickle=False) as saved:
                pred = saved["predicted_error_bounds"]
            if pred.shape != (len(errors), 2) or not np.isfinite(pred).all():
                raise ValueError("invalid saved validation predictions")
            scopes = dict(all_ongoing=np.ones(len(errors), bool), legacy_band=band)
            report["global_train_observed_error_range"] = [global_low, global_high]
            report["validation_observed_error_range"] = [float(errors.min()), float(errors.max())]
            report["results"]["global_training_hull"] = {scope: miss_partition(errors, pred,
                np.full(len(errors), global_low), np.full(len(errors), global_high),
                np.ones(len(errors), bool), mask) for scope, mask in scopes.items()}
            lo, hi, has, left, right = bracket_bounds(distances, train_low, train_high, d, vi)
            report["results"]["two_sided_neighbor_distances_all_appearances"] = {
                scope: miss_partition(errors, pred, lo, hi, has, mask) for scope, mask in scopes.items()}
            same_low, same_high = np.empty(len(errors)), np.empty(len(errors))
            for name, (al, ah) in appearance_bounds.items():
                mask = names == name
                same_low[mask] = np.minimum(al[left[mask], vi[mask]], al[right[mask], vi[mask]])
                same_high[mask] = np.maximum(ah[left[mask], vi[mask]], ah[right[mask], vi[mask]])
            report["results"]["two_sided_neighbors_same_appearance"] = {
                scope: miss_partition(errors, pred, same_low, same_high, has, mask) for scope, mask in scopes.items()}
            nearest_gap = np.minimum(abs(d-distances[left]), abs(d-distances[right]))
            report["nearest_train_distance_gap_m_pair_weighted"] = dict(median=float(np.median(nearest_gap)),
                p95=float(np.quantile(nearest_gap, .95)), max=float(nearest_gap.max()),
                validation_pairs_without_two_sided_bracket=int((~has).sum()))
            unique_d, inverse = np.unique(d, return_inverse=True)
            report["results"]["distance_windows"] = {}
            for radius in (.05, .15, .30):
                lows = np.full((len(unique_d), len(speeds)), np.nan)
                highs = lows.copy()
                evidence = np.zeros(len(unique_d), bool)
                for i, distance in enumerate(unique_d):
                    start = np.searchsorted(distances, float(distance)-radius, side="left")
                    end = np.searchsorted(distances, float(distance)+radius, side="right")
                    if end > start:
                        lows[i], highs[i] = train_low[start:end].min(axis=0), train_high[start:end].max(axis=0)
                        evidence[i] = True
                row = {scope: miss_partition(errors, pred, lows[inverse, vi], highs[inverse, vi],
                                             evidence[inverse], mask) for scope, mask in scopes.items()}
                row["all_pairs_without_window_evidence"] = int((~evidence[inverse]).sum())
                report["results"]["distance_windows"]["plus_minus_%.2f_m" % radius] = row
            outside = np.maximum(np.maximum(pred[:, 0]-errors, errors-pred[:, 1]), 0.)
            cases = []
            for scope, mask in scopes.items():
                ids = np.flatnonzero(mask & (outside > 1e-6))
                for i in ids[np.argsort(-outside[ids], kind="stable")[:5]]:
                    cases.append(dict(scope=scope, distance_m=float(d[i]), speed_mps=float(v[i]), appearance=str(names[i]),
                        image_action=float(val["image_action"][i]), q_action=float(val["q_action"][i]), error=float(errors[i]),
                        prediction=pred[i].tolist(), outside_error=float(outside[i]),
                        left_train_distance_m=float(distances[left[i]]), right_train_distance_m=float(distances[right[i]]),
                        two_sided_bracket=bool(has[i]), neighbor_hull_all_appearances=[float(lo[i]), float(hi[i])],
                        neighbor_hull_same_appearance=[float(same_low[i]), float(same_high[i])]))
            report["worst_cases_with_neighbor_labels"] = cases
        report["source_files_unchanged"] = {k: sha256(p) for k, p in files.items()} == hashes
        if not report["source_files_unchanged"]:
            raise ValueError("source changed during diagnostic")
        report["status"] = "development_miss_diagnosis_complete"
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        report["runtime_seconds"] = time.monotonic()-started
        write_json(args.output_dir/"metrics.json", report)
    print("[Frozen action-bound miss diagnosis: no training]")
    print("train fit:", report["train_fit_failure"])
    for key in ("global_training_hull", "two_sided_neighbor_distances_all_appearances"):
        print(key, report["results"][key], flush=True)
    print("distance gap:", report["nearest_train_distance_gap_m_pair_weighted"])
    print("metrics:", args.output_dir/"metrics.json", flush=True)


if __name__ == "__main__":
    main()
