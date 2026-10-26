"""Frozen, synchronized-frame comparison; no nearest-image or policy rollout.

Static q receives the target distance; temporal predictors receive past images
and recorded action queries. These are different tasks, explicitly separated.
The retained reference trajectories have already been inspected: no fresh CP.
"""
import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from Aebs.conformal.calibrate_controller import classify_batch
from Aebs.dino_latent.error_tail_metrics import tail_summary, trajectory_tail_summary
from Aebs.dino_latent.latent_method_models import make_model, predict, paired_cluster_bootstrap
from Aebs.dino_latent.models import SafetyProjection, PhysicalToLatent
from Aebs.dino_latent.native_sbc_stage import load_frozen
from Aebs.dino_latent.prepare_expanded_data import sha256
from Aebs.dino_latent.spvc_saved_actor import load_spvc_pair
from Aebs.dino_latent.temporal_world_model import TemporalPredictor, window_indices, rollout


def to_raw(z, mean, std):
    if not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Invalid latent standardization")
    return z * std + mean


def to_native(z, mean, std):
    if not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Invalid latent standardization")
    return (z - mean) / std


def matched_errors(speed, image_action, candidate_action):
    """Positive residual means candidate brakes less and leaves higher speed."""
    speed, image_action, candidate_action = np.broadcast_arrays(
        np.asarray(speed, np.float64), np.asarray(image_action, np.float64),
        np.asarray(candidate_action, np.float64))
    image_action = np.clip(image_action, -3, 3)
    candidate_action = np.clip(candidate_action, -3, 3)
    return (candidate_action - image_action,
            np.clip(speed - .05 * candidate_action, 0, 3)
            - np.clip(speed - .05 * image_action, 0, 3))


def common_windows(valid):
    """Require full three-frame context AND ten future frames for every arm."""
    windows = window_indices(valid, np.arange(len(valid)), 3, 10)
    if not len(windows) or len(np.unique(windows[:, 0])) != len(valid):
        raise ValueError("Each trajectory needs at least one common window")
    return windows


def validate_trajectories(valid, distance, speed, actions):
    if valid.ndim != 2 or not valid.any(1).all():
        raise ValueError("Empty or malformed trajectories")
    lengths = valid.sum(1)
    if not np.array_equal(valid, np.arange(valid.shape[1])[None, :] < lengths[:, None]):
        raise ValueError("Valid frames must be a contiguous prefix")
    for value in (distance, speed, actions):
        if value.shape != valid.shape or not np.isfinite(value[valid]).all():
            raise ValueError("Nonfinite/misaligned states/actions")
    adjacent = valid[:, 1:] & valid[:, :-1]
    de = np.abs(distance[:, 1:] - (distance[:, :-1] - .05 * speed[:, :-1]))[adjacent]
    ve = np.abs(speed[:, 1:] - np.clip(speed[:, :-1] - .05 * np.clip(actions[:, :-1], -3, 3), 0, 3))[adjacent]
    if not len(de) or max(de.max(), ve.max()) > 5e-4:
        raise ValueError("Saved frames are not synchronized physical transitions")
    return dict(distance_transition_max_abs_m=float(de.max()),
                speed_transition_max_abs_mps=float(ve.max()))


@torch.no_grad()
def actor_actions(actor, raw_z, speed, native=None):
    z = torch.as_tensor(raw_z, dtype=torch.float32)
    if native is not None:
        z = to_native(z, native["latent_mean"], native["latent_std"])
    v = torch.as_tensor(np.array(speed, copy=True), dtype=torch.float32).reshape(-1, 1)
    obs = torch.cat((z, v), 1)
    return torch.cat([actor(obs[i:i+8192]).clamp(-3, 3).reshape(-1)
                      for i in range(0, len(obs), 8192)]).numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trajectories", type=Path, default=Path("results/dino_expanded_v1/20_trajectory_sbc_cp_calibration_v1/trajectories.h5"))
    p.add_argument("--representation", type=Path, default=Path("results/dino_expanded_v1/08_R2_group_alignment/dino_safety_latent.pt"))
    p.add_argument("--base-ppo", type=Path, default=Path("results/dino_expanded_v1/09_R2_ppo/latent_ppo.zip"))
    p.add_argument("--spvc-dir", type=Path, default=Path("results/dino_spvc_minimal_v1"))
    p.add_argument("--final-recheck", type=Path, default=Path("results/dino_spvc_minimal_v1_final_recheck/metrics.json"))
    p.add_argument("--native-dir", type=Path, default=Path("results/native_representation_dimensions_20261019_v1"))
    p.add_argument("--controller-dir", type=Path, default=Path("results/native_controllers_speed_residual_20261019_v1/historical_R2"))
    p.add_argument("--temporal-dir", type=Path, default=Path("results/latent_methods_benchmark_20261018_run1"))
    p.add_argument("--pilot-dir", type=Path, default=Path("results/transformer_pilot_20261018"))
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--window-scope", choices=("per_horizon", "common10"), default="per_horizon")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists() or args.batch_size < 1:
        p.error("New output directory and positive batch size required")
    torch.set_num_threads(1)
    device = torch.device(args.device)
    started = time.perf_counter()
    specs = [("matched_mlp32", "mlp32_parameter_matched", "mlp_matched"),
             ("transformer32", "transformer32", "transformer"),
             ("task_transformer32", "transformer32_task_aware", "task_aware"),
             ("no_speed_transformer32", "transformer32_no_speed", "no_speed")]
    files = [args.representation, args.base_ppo, args.final_recheck,
             args.spvc_dir / "latent_ppo_spvc.pt", args.spvc_dir / "sbc.pt",
             args.spvc_dir / "metrics.json", args.native_dir / "historical_R2.pt",
             args.controller_dir / "native_ppo.zip", args.controller_dir / "metrics.json",
             args.temporal_dir / "coordinates.pt", args.temporal_dir / "config.json",
             args.pilot_dir / "config.json", Path(__file__)]
    files += [args.temporal_dir / (stem + ".pt") for _, stem, _ in specs]
    files += [args.pilot_dir / ("transformer%d.pt" % dim) for dim in (32, 384)]
    hashes = {str(f): sha256(f) for f in files}
    for folder, key in ((args.temporal_dir, "fresh_CP_used"), (args.pilot_dir, "fresh_CP_data_used")):
        config = json.loads((folder / "config.json").read_text())
        if config.get(key) is not False or config.get("test_used", False):
            raise ValueError("Forecast training provenance does not exclude reference/test use")
    native_policy, _, native_identity = load_frozen(args.native_dir, args.controller_dir)
    native = torch.load(args.native_dir / "historical_R2.pt", map_location="cpu")
    base, saved, _, old_identity = load_spvc_pair(args.representation, args.base_ppo, args.spvc_dir, args.final_recheck)
    actors = {"minimal_SPVC": (saved.policy.controller_net.eval().requires_grad_(False), None),
              "native_R2": (native_policy.actor, native)}
    rep = torch.load(args.representation, map_location="cpu")
    projection, old_q = SafetyProjection().eval(), PhysicalToLatent().eval()
    projection.load_state_dict(rep["projection_state_dict"])
    old_q.load_state_dict(rep["physical_to_latent_state_dict"])
    projection.requires_grad_(False); old_q.requires_grad_(False)
    with h5py.File(args.trajectories, "r") as f:
        provenance = json.loads(f.attrs["provenance_json"])
        if (not f.attrs.get("complete") or provenance["representation_sha256"] != sha256(args.representation)
                or provenance["controller_sha256"] != sha256(args.base_ppo)):
            raise ValueError("Wrong/incomplete synchronized reference trajectories")
        valid = f["valid"][:].astype(bool)
        d, v, a = (f[k][:] for k in ("distance_m", "speed_mps", "image_action"))
        features = f["dino_features"][:]
        ids = f["trajectory_id"].asstr()[:].tolist()
    if len(ids) != 459 or len(set(ids)) != len(ids) or features.shape != (*valid.shape, 384):
        raise ValueError("Need the fixed459-trajectory synchronized panel")
    if not np.isfinite(features[valid]).all():
        raise ValueError("Nonfinite image features")
    audit = validate_trajectories(valid, d, v, a)
    windows = common_windows(valid)
    window_sets = {h: windows if args.window_scope == "common10" else
                   window_indices(valid, np.arange(len(valid)), 3, h) for h in (1, 3, 10)}
    with torch.no_grad():
        x = (torch.from_numpy(np.nan_to_num(features)) - rep["feature_mean"]) / rep["feature_std"]
        z = projection(x.flatten(0, 1)).reshape(*valid.shape, 32)
    ve, vt = np.where(valid)
    collector_replay = base.predict(np.column_stack((z[ve, vt].numpy(), v[ve, vt])), deterministic=True)[0].reshape(-1)
    audit["collector_recorded_action_replay_max_abs"] = float(np.max(np.abs(collector_replay - a[ve, vt])))
    if audit["collector_recorded_action_replay_max_abs"] > 1e-4:
        raise ValueError("Image encodings do not replay actual collector actions")
    args.output_dir.mkdir(parents=True)
    config = dict(args={k: str(value) if isinstance(value, Path) else value for k, value in vars(args).items()},
                  source_sha256=hashes, trajectory_sha256=sha256(args.trajectories),
                  trajectory_count=len(ids), common_contexts=len(windows), context_length=3,
                  windows_per_horizon={h: len(w) for h, w in window_sets.items()},
                  eligibility="ALL methods share EXACT same window set within each horizon; nonterminal targets only",
                  window_scope=args.window_scope,
                  near_terminal_policy="per_horizon includes all eligible late frames; common10 restricts to ten-step-capable contexts",
                  static_task="q gets exact target distance; consistency, not forecasting",
                  temporal_task="past3 synchronized features + recorded open-loop actions; no future image/speed inputs",
                  actor_task="same target TRUE speed supplied to all actors; isolates latent effects, not full speed prediction",
                  residual_sign="candidate_next_speed minus matched_real_image_next_speed; positive=underbraking",
                  collector="old PPO09, not the candidate models or native_R2",
                  nearest_lookup=False, training=False, autonomous_rollout=False, fresh_CP=False,
                  final_test_used=False, safety_certificate=False,
                  reference_reused=True, trajectory_audit=audit,
                  minimal_actor_identity=old_identity, native_actor_identity=native_identity)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, allow_nan=False))
    np.savez_compressed(args.output_dir / "common_windows.npz", windows=windows, trajectory_id=np.array(ids))
    np.savez_compressed(args.output_dir / "windows_by_horizon.npz",
                        **{"h%d" % h: w for h, w in window_sets.items()})
    targets, references = {}, {}
    output = dict(status="running", config=config, paths={})
    residuals = {}
    for horizon in (1, 3, 10):
        ep, end = window_sets[horizon].T
        tt = end + horizon
        target_d, target_v, target_z = d[ep, tt], v[ep, tt], z[ep, tt]
        outcomes = classify_batch(target_d, target_v, np.ones(len(ep), bool), 0, 400)
        keep = outcomes == ""
        if len(np.unique(ep[keep])) != len(ids):
            raise ValueError("Some trajectories lack nonterminal common targets")
        targets[horizon] = (target_d, target_v, target_z, keep)
        for actor_name, (actor, norm) in actors.items():
            references[(actor_name, horizon)] = actor_actions(actor, target_z, target_v, norm)

    def add(name, horizon, prediction, family):
        ep, end = window_sets[horizon].T
        dd, vv, truth, keep = targets[horizon]
        pred = np.asarray(prediction, np.float32)
        if pred.shape != (len(ep), 32) or not np.isfinite(pred).all():
            raise ValueError("Prediction coordinates must be finite raw R2, not another native space")
        latent_rms = np.sqrt(np.mean((pred.astype(np.float64) - truth.numpy()) ** 2, axis=1))
        latent_result = tail_summary(latent_rms[keep], unit="raw R2 per-window coordinate RMS")
        for actor_name, (actor, norm) in actors.items():
            candidate = actor_actions(actor, pred, vv, norm)
            reference = references[(actor_name, horizon)]
            action_delta, speed_delta = matched_errors(vv, reference, candidate)
            # These are local outcomes at the recorded target state, not new rollouts.
            next_d = dd - .05 * vv
            ref_v = np.clip(vv.astype(np.float64) - .05 * reference, 0, 3)
            cand_v = np.clip(vv.astype(np.float64) - .05 * candidate, 0, 3)
            ref_unsafe = (next_d <= 6) & (ref_v > .5)
            cand_unsafe = (next_d <= 6) & (cand_v > .5)
            r = speed_delta[keep]
            trajectory_summary, maxima = trajectory_tail_summary(r, ep[keep], len(ids), (.05, .1, .15))
            key = "%s/%s/h%d" % (actor_name, name, horizon)
            residuals[key] = r
            result = dict(family=family, count=int(keep.sum()), excluded_terminal_targets=int((~keep).sum()),
                          horizon=horizon, latent=latent_result, action=tail_summary(action_delta[keep], unit="action"),
                          next_speed=tail_summary(r, (.05, .1, .15)), trajectory_maxima=trajectory_summary,
                          image_only_next_unsafe=int((ref_unsafe & ~cand_unsafe & keep).sum()),
                          candidate_only_next_unsafe=int((cand_unsafe & ~ref_unsafe & keep).sum()),
                          both_next_unsafe=int((cand_unsafe & ref_unsafe & keep).sum()))
            output["paths"][key] = result
            np.savez_compressed(args.output_dir / (key.replace("/", "__") + ".npz"),
                                trajectory_index=ep[keep], target_time=(end+horizon)[keep],
                                action_delta=action_delta[keep], speed_delta=r,
                                latent_rms=latent_rms[keep], trajectory_maxima=maxima)
            print("%s n=%d MAE=%.9f max=%.9f" % (key, len(r), result["next_speed"]["mae"], result["next_speed"]["max_abs"]), flush=True)
        # Predictions are shared by both actors and retained for reproducibility.
        np.savez_compressed(args.output_dir / (name + "_h%d_predictions.npz" % horizon), raw_r2=pred)

    with torch.no_grad():
        for h in (1, 3, 10):
            ep, end = window_sets[h].T
            dd = torch.from_numpy(np.array(targets[h][0], copy=True)).float()[:, None]
            old_prediction = old_q(dd / float(rep["distance_scale_m"]))
            new_prediction = to_raw(native_policy.q((dd - native_policy.distance_mean) / native_policy.distance_std),
                                    native["latent_mean"], native["latent_std"])
            add("old_q32", h, old_prediction.numpy(), "static_exact_target_distance")
            add("native_q32", h, new_prediction.numpy(), "static_exact_target_distance")
            add("persistence32", h, z[ep, end].numpy(), "temporal_nonlearned")
    sources = {32: z.to(device), 384: x.to(device)}
    speeds, controls = (torch.from_numpy(np.nan_to_num(t)).to(device) for t in (v, a))
    coords = torch.load(args.temporal_dir / "coordinates.pt", map_location="cpu")
    forecast_specs = [(name, args.temporal_dir / (stem + ".pt"), kind, 32, 10, False)
                      for name, stem, kind in specs]
    forecast_specs += [("pilot_transformer%d" % dim, args.pilot_dir / ("transformer%d.pt" % dim),
                        "transformer", dim, 3, True) for dim in (32, 384)]
    gpu_projection = projection.to(device)
    for name, path, kind, dim, horizon, pilot in forecast_specs:
        ck = torch.load(path, map_location="cpu")
        if pilot:
            arm = ck.get("arm", {})
            model = TemporalPredictor(dim, ck["kind"], speed_mode=arm.get("speed_mode", "concat"),
                                      use_action=arm.get("use_action", True)).to(device)
            mean, std = ck["normalization"][ck["native"]]
        else:
            model = make_model(kind).to(device)
            mean, std = coords["normalization"]["r2"]
        model.load_state_dict(ck["state_dict"], strict=True)
        model.eval().requires_grad_(False)
        mean, std = mean.to(device), std.to(device)
        for h in (h for h in (1, 3, 10) if h <= horizon):
            eligible = window_sets[h]
            batches = []
            for start in range(0, len(eligible), args.batch_size):
                w = torch.as_tensor(eligible[start:start+args.batch_size], device=device)
                rows, ends = w[:, 0, None], w[:, 1, None]
                ci = ends + torch.arange(-2, 1, device=device)
                ai = ends + torch.arange(-2, h, device=device)
                with torch.no_grad():
                    context = to_native(sources[dim][rows, ci], mean, std)
                    vv, aa = speeds[rows, ci], controls[rows, ai]
                    forecast = rollout(model, context, vv, aa, h) if pilot else predict(model, context, vv, aa, h)[0]
                    raw = to_raw(forecast[:, -1], mean, std)
                    if dim == 384:
                        raw = gpu_projection(raw)
                    batches.append(raw.cpu().numpy())
            add(name, h, np.concatenate(batches), "temporal_prediction_recorded_action_query")
        del model
    # Paired uncertainty is over existing trajectories, not training runs or CP.
    for key, record in output["paths"].items():
        actor_name, name, hstr = key.split("/")
        horizon = int(hstr[1:])
        if name == "native_q32":
            continue
        reference_key = actor_name + "/native_q32/" + hstr
        record["paired_vs_native_q_speed_mae"] = paired_cluster_bootstrap(
            np.abs(residuals[key]), np.abs(residuals[reference_key]), window_sets[horizon][:, 0][targets[horizon][3]])
    for path, digest in hashes.items():
        if sha256(Path(path)) != digest:
            raise ValueError("Frozen source changed during comparison: " + path)
    if sha256(args.trajectories) != config["trajectory_sha256"]:
        raise ValueError("Reference trajectories changed")
    output.update(status="synchronized_frozen_comparison_complete_not_a_certificate",
                  runtime_seconds=time.perf_counter()-started, all_sources_unchanged=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(output, indent=2, allow_nan=False))
    print("[Exact synchronized latent-path comparison]", output["status"], flush=True)
    print("windows=%s, trajectories=%d; NO nearest lookup/training/fresh CP" %
          (config["windows_per_horizon"], len(ids)), flush=True)
    print("metrics:", args.output_dir / "metrics.json", flush=True)


if __name__ == "__main__":
    main()
