"""Safety-oriented reanalysis of frozen synchronized-frame predictions.

No new models, nearest-image replay, calibration data or safety certificate.
Separate static consistency from forecasting; assess signed physical errors,
trajectory-balanced accuracy, clipping, group tails and a frozen SBC diagnostic.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from Aebs.dino_latent.compare_exact_latent_paths import actor_actions, matched_errors
from Aebs.dino_latent.error_tail_metrics import tail_summary
from Aebs.dino_latent.models import SafetyProjection
from Aebs.dino_latent.native_sbc_stage import load_frozen
from Aebs.dino_latent.prepare_expanded_data import sha256
from Aebs.dino_latent.spvc_saved_actor import load_spvc_pair
from Aebs.dino_latent.temporal_world_model import TemporalPredictor, rollout
from Aebs.system.env import Aebs
from Aebs.VT.utils import MLP


def upper_tail_mean(values, fraction=.05):
    """Mean of largest ceil(fraction*N) entries, including ties by count."""
    x = np.asarray(values, np.float64).reshape(-1)
    if not len(x) or not np.isfinite(x).all() or not 0 < fraction <= 1:
        raise ValueError("Nonempty finite values and valid tail fraction required")
    k = max(1, int(np.ceil(len(x)*fraction)))
    return float(np.partition(x, len(x)-k)[-k:].mean())


def scalar_summary(x, unit="m/s"):
    result = tail_summary(x, unit=unit)
    result["worst_5pct_mean"] = upper_tail_mean(np.abs(x))
    return result


def physical_error_summary(residual, groups):
    r, g = np.asarray(residual, np.float64), np.asarray(groups)
    if r.ndim != 1 or g.shape != r.shape or not len(r) or not np.isfinite(r).all():
        raise ValueError("Need paired finite residual and trajectory ID arrays")
    labels = np.unique(g)
    absolute = np.abs(r)
    optimistic, extra_speed = np.maximum(-r, 0), np.maximum(r, 0)
    macro = np.array([absolute[g == label].mean() for label in labels])
    abs_max = np.array([absolute[g == label].max() for label in labels])
    optimistic_max = np.array([optimistic[g == label].max() for label in labels])
    extra_max = np.array([extra_speed[g == label].max() for label in labels])
    return dict(step_absolute=scalar_summary(absolute),
                image_faster_than_candidate=scalar_summary(optimistic),
                candidate_faster_than_image=scalar_summary(extra_speed),
                trajectory_macro_mae_mps=float(macro.mean()),
                trajectory_mae_distribution=scalar_summary(macro),
                trajectory_absolute_max=scalar_summary(abs_max),
                trajectory_proxy_optimism_max=scalar_summary(optimistic_max),
                trajectory_candidate_extra_speed_max=scalar_summary(extra_max),
                trajectory_count=len(labels),
                empirical_only=True, independent_CP_coverage=False)


def clipping_audit(speed, image_action, candidate_action):
    v, ai, ac = (np.asarray(x, np.float64) for x in (speed, image_action, candidate_action))
    if v.shape != ai.shape or ai.shape != ac.shape or not np.isfinite(v+ai+ac).all():
        raise ValueError("Mismatched physical arrays")
    ai, ac = np.clip(ai, -3, 3), np.clip(ac, -3, 3)
    raw_i, raw_c = v-.05*ai, v-.05*ac
    vi, vc = np.clip(raw_i, 0, 3), np.clip(raw_c, 0, 3)
    action_disagreement = np.abs(ac-ai) > 1e-3
    hidden = action_disagreement & (np.abs(vc-vi) <= 1e-7)
    saturation = (raw_i < 0) | (raw_i > 3) | (raw_c < 0) | (raw_c > 3)
    return dict(count=len(v), action_mae=float(np.abs(ac-ai).mean()),
                action_disagreement_count=int(action_disagreement.sum()),
                unclipped_speed_equivalent_mae_mps=float(np.abs(raw_c-raw_i).mean()),
                physical_speed_mae_mps=float(np.abs(vc-vi).mean()),
                speed_saturation_count=int(saturation.sum()),
                zero_physical_error_with_action_disagreement_count=int(hidden.sum()),
                zero_physical_error_with_action_disagreement_fraction=float(hidden.mean()),
                scope="pre-speed-clipping equivalent is explanatory, not a valid physical successor outside [0,3]")


def target_intersection(arrays):
    """Align horizon-specific results by TARGET state, not context end."""
    if not arrays:
        raise ValueError("Need at least one target set")
    maps = []
    for a in arrays:
        pairs = list(zip(a["trajectory_index"].tolist(), a["target_time"].tolist()))
        if len(set(pairs)) != len(pairs):
            raise ValueError("Duplicate target frame")
        maps.append({pair: i for i, pair in enumerate(pairs)})
    shared = sorted(set(maps[0]).intersection(*(set(m) for m in maps[1:])))
    if not shared:
        raise ValueError("No common targets")
    return np.asarray(shared, np.int64), [np.array([m[p] for p in shared]) for m in maps]


def fixed_strata(distance, speed):
    d, v = np.asarray(distance), np.asarray(speed)
    return dict(near_obstacle_d_le_7=d <= 7,
                low_speed_near_obstacle=(d <= 7) & (v > .25) & (v <= 1),
                high_speed_v_ge_2p5=v >= 2.5,
                next_position_reaches_unsafe_distance=d-.05*v <= 6)


def group_table(residual, labels):
    r, labels = np.asarray(residual), np.asarray(labels)
    table = {}
    for name in np.unique(labels):
        mask = labels == name
        s = physical_error_summary(r[mask], np.zeros(mask.sum(), np.int64))
        table[str(name)] = dict(count=int(mask.sum()), mae_mps=s["step_absolute"]["mae"],
                                p99_mps=s["step_absolute"]["p99_abs"],
                                max_abs_mps=s["step_absolute"]["max_abs"],
                                proxy_optimism_max_mps=s["image_faster_than_candidate"]["max_abs"])
    worst = max(table, key=lambda key: table[key]["mae_mps"])
    return dict(groups=table, balanced_group_mae_mps=float(np.mean([x["mae_mps"] for x in table.values()])),
                worst_group=worst, worst_group_mae_mps=table[worst]["mae_mps"])


def safety_relevance(states, image_next_speed, candidate_next_speed):
    d, v = np.asarray(states).T
    dn = d-.05*v
    image_unsafe = (dn <= 6) & (np.asarray(image_next_speed) > .5)
    candidate_unsafe = (dn <= 6) & (np.asarray(candidate_next_speed) > .5)
    return dict(image_unsafe_candidate_safe=int((image_unsafe & ~candidate_unsafe).sum()),
                candidate_unsafe_image_safe=int((candidate_unsafe & ~image_unsafe).sum()),
                both_unsafe=int((candidate_unsafe & image_unsafe).sum()),
                scope="same-state one-step classification only, not an autonomous rollout rate")


def prediction_skill(predicted_mse, persistence_mse, groups):
    predicted_mse, persistence_mse, groups = map(np.asarray, (predicted_mse, persistence_mse, groups))
    if (predicted_mse.ndim != 1 or predicted_mse.shape != persistence_mse.shape or groups.shape != predicted_mse.shape
            or not len(groups) or not np.isfinite(predicted_mse+persistence_mse).all()
            or (predicted_mse < 0).any() or (persistence_mse < 0).any()):
        raise ValueError("Need paired finite nonnegative squared errors")
    labels = np.unique(groups)
    pm = np.array([predicted_mse[groups == g].mean() for g in labels])
    bm = np.array([persistence_mse[groups == g].mean() for g in labels])
    return dict(normalized_coordinate_rmse=float(np.sqrt(predicted_mse.mean())),
                persistence_normalized_coordinate_rmse=float(np.sqrt(persistence_mse.mean())),
                trajectory_balanced_skill_vs_persistence=float(1-pm.mean()/bm.mean()) if bm.mean() > 0 else None,
                trajectory_fraction_improved=float((pm < bm).mean()),
                trajectories=len(labels), count=len(groups),
                scope="within its OWN feature space; improvement over persistence, not an absolute cross-space accuracy ranking")


@torch.no_grad()
def native_feature_audit(pilot_dir, comparison_dir, z, x, speeds, actions, projection, device):
    """Do not judge 384D feature accuracy only after a lossy 32D projection."""
    results = {}
    for dim in (32, 384):
        name = "pilot_transformer%d" % dim
        ck = torch.load(pilot_dir / ("transformer%d.pt" % dim), map_location="cpu")
        arm = ck.get("arm", {})
        model = TemporalPredictor(dim, ck["kind"], speed_mode=arm.get("speed_mode", "concat"),
                                  use_action=arm.get("use_action", True)).to(device).eval().requires_grad_(False)
        model.load_state_dict(ck["state_dict"], strict=True)
        mean, std = [t.to(device) for t in ck["normalization"][ck["native"]]]
        if not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("Invalid training-only forecast normalization")
        source = (z if dim == 32 else x).to(device)
        vv, aa = torch.from_numpy(np.nan_to_num(speeds)).to(device), torch.from_numpy(np.nan_to_num(actions)).to(device)
        for h in (1, 3):
            rows = np.load(comparison_dir / ("native_R2__%s__h%d.npz" % (name, h)))
            ep, tt = rows["trajectory_index"], rows["target_time"]
            saved_r2 = np.load(comparison_dir / (name+"_h%d_predictions.npz" % h))["raw_r2"]
            pm, bm = [], []
            replay_max = 0.
            for start in range(0, len(ep), 1024):
                ei = torch.as_tensor(ep[start:start+1024], device=device)[:, None]
                ends = torch.as_tensor(tt[start:start+1024]-h, device=device)[:, None]
                ci = ends + torch.arange(-2, 1, device=device)
                ai = ends + torch.arange(-2, h, device=device)
                context = (source[ei, ci]-mean)/std
                forecast = rollout(model, context, vv[ei, ci], aa[ei, ai], h)[:, -1]
                raw_prediction = forecast*std+mean
                r2_prediction = raw_prediction if dim == 32 else projection.to(device)(raw_prediction)
                replay_max = max(replay_max, float(np.max(abs(r2_prediction.cpu().numpy()-saved_r2[start:start+len(ei)]))))
                target = (source[ei[:, 0], ends[:, 0]+h]-mean)/std
                pm.append((forecast-target).square().mean(1).cpu().numpy())
                bm.append((context[:, -1]-target).square().mean(1).cpu().numpy())
            if replay_max > 1e-5:
                raise ValueError("Native feature audit does not replay saved frozen predictions")
            result = prediction_skill(np.concatenate(pm), np.concatenate(bm), ep)
            result["saved_R2_prediction_replay_max_abs"] = replay_max
            results[name+"/h%d" % h] = result
        del model, source
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--comparison-dir", type=Path, default=Path("results/exact_latent_path_comparison_20261020_v2"))
    p.add_argument("--native-barrier-dir", type=Path, default=Path("results/native_R2_sbc_stage2_legacy_20261019_v1"))
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    if a.output_dir.exists():
        p.error("New output directory required")
    torch.set_num_threads(1)
    previous_path = a.comparison_dir / "metrics.json"
    previous = json.loads(previous_path.read_text())
    cfg = previous["config"]
    if (previous["status"] != "synchronized_frozen_comparison_complete_not_a_certificate"
            or cfg["window_scope"] != "per_horizon" or cfg["nearest_lookup"] or not cfg["reference_reused"]):
        raise ValueError("Need the exact per-horizon frozen comparison")
    paths = {k: Path(v) for k, v in cfg["args"].items() if isinstance(v, str) and k not in ("device", "window_scope")}
    hashes = dict(cfg["source_sha256"])
    hashes[str(previous_path)] = sha256(previous_path)
    hashes[str(paths["trajectories"])] = cfg["trajectory_sha256"]
    # Include the loaded metric-defining source, not only the command entry point.
    for file in (Path(__file__), Path("Aebs/dino_latent/error_tail_metrics.py"),
                 Path("Aebs/dino_latent/temporal_world_model.py"),
                 Path("Aebs/VT/utils.py"), a.native_barrier_dir / "sbc.pt",
                 a.native_barrier_dir / "metrics.json"):
        hashes[str(file)] = sha256(file)
    for path, digest in hashes.items():
        if sha256(Path(path)) != digest:
            raise ValueError("Changed registered source: " + path)
    native_policy, _, native_identity = load_frozen(paths["native_dir"], paths["controller_dir"])
    _, saved, _, old_identity = load_spvc_pair(paths["representation"], paths["base_ppo"], paths["spvc_dir"], paths["final_recheck"])
    native = torch.load(paths["native_dir"] / "historical_R2.pt", map_location="cpu")
    actors = {"minimal_SPVC": (saved.policy.controller_net.eval().requires_grad_(False), None),
              "native_R2": (native_policy.actor, native)}
    rep = torch.load(paths["representation"], map_location="cpu")
    projection = SafetyProjection().eval().requires_grad_(False)
    projection.load_state_dict(rep["projection_state_dict"])
    with h5py.File(paths["trajectories"], "r") as f:
        features = f["dino_features"][:]
        distance, speed, recorded_actions = (f[k][:] for k in ("distance_m", "speed_mps", "image_action"))
        weather, color = f["weather"].asstr()[:], f["color"].asstr()[:]
    with torch.no_grad():
        x = (torch.from_numpy(np.nan_to_num(features)) - rep["feature_mean"]) / rep["feature_std"]
        z = projection(x.flatten(0, 1)).reshape(*distance.shape, 32)
    barrier_meta = json.loads((a.native_barrier_dir / "metrics.json").read_text())
    if (not barrier_meta["checkpoint_matches_verification"] or
            barrier_meta["saved_barrier_sha256"] != hashes[str(a.native_barrier_dir / "sbc.pt")] or
            barrier_meta["config"]["provenance"]["source_sha256"] != native_identity["source_sha256"]):
        raise ValueError("Native barrier/controller provenance mismatch")
    old_meta = json.loads(paths["final_recheck"].read_text())
    scale = float(Aebs(.05).std1)
    if abs(scale-old_identity["state_distance_scale_m"]) > 1e-7:
        raise ValueError("Barrier distance coordinates changed")
    barriers = {}
    for actor, path, bounds in (("minimal_SPVC", paths["spvc_dir"] / "sbc.pt", old_meta),
                                 ("native_R2", a.native_barrier_dir / "sbc.pt", barrier_meta["verification"])):
        model = MLP([2, 16, 8, 1], activation="tanh", square_output=True).eval().requires_grad_(False)
        model.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
        span = float(bounds["unsafe_lower"] - bounds["domain_lower"])
        if not np.isfinite(span) or span <= 0:
            raise ValueError("Invalid frozen barrier normalization")
        barriers[actor] = (model, span, float(bounds["domain_lower"]))

    @torch.no_grad()
    def barrier_values(actor, states):
        model, _, _ = barriers[actor]
        s = torch.from_numpy(np.asarray(states, np.float32).copy())
        s[:, 0] /= scale
        return model(s).reshape(-1).numpy().astype(np.float64)

    a.output_dir.mkdir(parents=True)
    protocol = dict(source_sha256=hashes, source_comparison=str(previous_path),
                    model_training=False, nearest_lookup=False, fresh_CP=False,
                    data_scope="reused PPO09 reference trajectories, not new-policy autonomous rollouts",
                    absolute_residual="candidate next speed minus image next speed, at same target state",
                    proxy_optimism="max(image next speed minus candidate next speed,0); threatens image-from-proxy inference",
                    candidate_extra_speed="max(candidate next speed minus image next speed,0); replacement-path underbraking",
                    main_aggregation="equal trajectory MAE plus tail metrics; frame MAE retained for comparison",
                    worst_tail="mean of largest ceil(0.05*N) absolute errors; empirical, not a population risk bound",
                    clipping="report action differences and pre-speed-clipping equivalents alongside true physical error",
                    fixed_strata=dict(distance_edges_m=[5, 6, 7, 10, 13, 16], speed_edges_mps=[0, .5, 1, 2, 3]),
                    horizon_comparison="intersection of identical TARGET (trajectory,time), not identical context ends",
                    comparison_cohorts=dict(static_consistency=["old_q32", "native_q32"],
                        pilot_forecasts=["pilot_transformer32", "pilot_transformer384", "persistence32"],
                        latest_forecasts=["matched_mlp32", "transformer32", "task_transformer32", "no_speed_transformer32", "persistence32"]),
                    excluded_claims=["overall architecture ranking", "independent CP", "global barrier proof", "real vehicle dynamics accuracy"],
                    frozen_barrier_metric="image B(next)-candidate B(next), divided by unsafe-lower minus domain-lower; no clamping or terminal overrides",
                    barrier_test="noise-free pointwise diagnostic only, NOT original expected/interval decrease verification")
    (a.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2, allow_nan=False))
    out = dict(status="running", protocol=protocol, paths={}, same_target_horizons={})
    arrays = {}
    for key, old in previous["paths"].items():
        actor_name, name, h = key.split("/")
        source = a.comparison_dir / (key.replace("/", "__") + ".npz")
        prediction_path = a.comparison_dir / (name + "_" + h + "_predictions.npz")
        hashes[str(source)] = sha256(source); hashes[str(prediction_path)] = sha256(prediction_path)
        record = dict(np.load(source))
        ep, tt = record["trajectory_index"], record["target_time"]
        dd, vv = distance[ep, tt], speed[ep, tt]
        states = np.column_stack((dd, vv))
        actor, norm = actors[actor_name]
        ai = actor_actions(actor, z[ep, tt], vv, norm)
        # All targets in the registered run were nonterminal, hence no row filtering.
        if old["excluded_terminal_targets"] or len(ep) != old["count"]:
            raise ValueError("Unexpected target filtering; explicitly realign predictions first")
        candidate_z = np.load(prediction_path)["raw_r2"]
        ac = actor_actions(actor, candidate_z, vv, norm)
        delta_a, residual = matched_errors(vv, ai, ac)
        if np.max(abs(residual-record["speed_delta"])) > 1e-6 or np.max(abs(delta_a-record["action_delta"])) > 1e-5:
            raise ValueError("Saved predictions/actions/residuals disagree")
        summary = physical_error_summary(residual, ep)
        summary["family"] = old["family"]
        summary["clipping"] = clipping_audit(vv, ai, ac)
        appearance = np.array([w + "|" + c for w, c in zip(weather[ep], color[ep])])
        summary["appearance"] = group_table(residual, appearance)
        summary["physical_regions"] = {}
        for region, mask in fixed_strata(dd, vv).items():
            summary["physical_regions"][region] = (physical_error_summary(residual[mask], ep[mask]) if mask.any()
                                                    else dict(count=0, unavailable="no reference samples"))
        db = np.clip(np.searchsorted([5, 6, 7, 10, 13, 16], dd, side="right")-1, 0, 4)
        vb = np.clip(np.searchsorted([0, .5, 1, 2, 3], vv, side="right")-1, 0, 3)
        labels = np.array(["d%d_v%d" % (i, j) for i, j in zip(db, vb)])
        summary["physical_cell"] = group_table(residual, labels)
        vi, vc = np.clip(vv.astype(np.float64)-.05*ai, 0, 3), np.clip(vv.astype(np.float64)-.05*ac, 0, 3)
        summary["one_step_outcomes"] = safety_relevance(states, vi, vc)
        ni, nc = np.column_stack((dd-.05*vv, vi)), np.column_stack((dd-.05*vv, vc))
        current, bi, bc = barrier_values(actor_name, states), barrier_values(actor_name, ni), barrier_values(actor_name, nc)
        _, span, lower = barriers[actor_name]
        gap = (bi-bc)/span
        narrow = (current > lower+.95*span) & (current < lower+span)
        summary["SBC_relevance"] = dict(normalized_image_minus_candidate_gap=scalar_summary(gap, "normalized frozen B"),
            positive_gap_p99=float(np.quantile(np.maximum(gap, 0), .99)),
            positive_gap_max=float(np.maximum(gap, 0).max()),
            candidate_nonincrease_but_image_increase=int(((bc-current <= 1e-7) & (bi-current > 1e-7)).sum()),
            legacy_band_samples=int(narrow.sum()),
            legacy_band_positive_gap_max=float(np.maximum(gap[narrow], 0).max()) if narrow.any() else None,
            scope="deterministic same-state NN discrepancy; not a stochastic or interval certificate")
        worst = int(np.argmax(np.maximum(-residual, 0)))
        summary["largest_proxy_optimism_example"] = dict(trajectory_index=int(ep[worst]), target_time=int(tt[worst]),
            distance_m=float(dd[worst]), speed_mps=float(vv[worst]), appearance=str(appearance[worst]),
            image_action=float(ai[worst]), candidate_action=float(ac[worst]),
            image_faster_mps=float(max(-residual[worst], 0)))
        out["paths"][key] = summary
        arrays[key] = dict(trajectory_index=ep, target_time=tt, speed_delta=residual)
        np.savez_compressed(a.output_dir / (key.replace("/", "__") + ".npz"),
                            **arrays[key], action_delta=delta_a, normalized_B_image_minus_candidate=gap,
                            speed_saturated=np.array((vv-.05*ai < 0) | (vv-.05*ai > 3) | (vv-.05*ac < 0) | (vv-.05*ac > 3)))
        if actor_name == "native_R2" and h == "h1":
            print(name, "macro-MAE=%.9f P99=%.9f optimistic-max=%.9f" % (
                summary["trajectory_macro_mae_mps"], summary["step_absolute"]["p99_abs"],
                summary["image_faster_than_candidate"]["max_abs"]), flush=True)
    for actor_name in actors:
        for name in ("old_q32", "native_q32", "persistence32", "matched_mlp32", "transformer32",
                     "task_transformer32", "no_speed_transformer32", "pilot_transformer32", "pilot_transformer384"):
            horizons = [h for h in (1, 3, 10) if "%s/%s/h%d" % (actor_name, name, h) in arrays]
            keys = ["%s/%s/h%d" % (actor_name, name, h) for h in horizons]
            targets, indices = target_intersection([arrays[k] for k in keys])
            entry = dict(targets=len(targets), trajectories=len(np.unique(targets[:, 0])),
                         horizons={})
            for h, key, ix in zip(horizons, keys, indices):
                entry["horizons"][h] = physical_error_summary(arrays[key]["speed_delta"][ix], targets[:, 0])
            if name in ("old_q32", "native_q32"):
                difference = max(np.max(abs(arrays[key]["speed_delta"][ix]-arrays[keys[0]]["speed_delta"][indices[0]]))
                                 for key, ix in zip(keys, indices))
                if difference > 1e-6:
                    raise ValueError("Static q changed across identical target states")
                entry["static_horizon_invariance_max_abs"] = float(difference)
            out["same_target_horizons"][actor_name+"/"+name] = entry
    out["native_forecast_accuracy"] = native_feature_audit(paths["pilot_dir"], a.comparison_dir,
        z, x, speed, recorded_actions, projection, torch.device(a.device))
    for file, digest in hashes.items():
        if sha256(Path(file)) != digest:
            raise ValueError("Source changed during reanalysis: " + file)
    out.update(status="task_separated_safety_relevant_error_diagnostic_complete", all_sources_unchanged=True,
               no_global_ranking=True, safety_certificate=False)
    (a.output_dir / "metrics.json").write_text(json.dumps(out, indent=2, allow_nan=False))
    (a.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2, allow_nan=False))
    print("[Revised error protocol]", out["status"], flush=True)
    print("metrics:", a.output_dir / "metrics.json", flush=True)


if __name__ == "__main__":
    main()
