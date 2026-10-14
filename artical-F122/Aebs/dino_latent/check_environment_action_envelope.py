"""Adaptive noise IBP over empirical state-conditional environment actions.

Actions can be nominal, an empirical environment hull, or the full clipped
actuator range. IBP covers the action/noise intervals at recorded states.
This is not a continuous-state or end-to-end safety certificate.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from Aebs.dino_latent.prepare_expanded_data import sha256
from Aebs.VT.utils import MLP
from Aebs.system.env import Aebs
from Aebs.dino_latent.diagnose_environment_barrier import write_json


def interval_barrier(barrier, low, high):
    if not torch.all(low <= high):
        raise ValueError("reversed interval")
    for i, layer in enumerate(barrier.layers):
        positive, negative = layer.weight.clamp(min=0), layer.weight.clamp(max=0)
        next_low = F.linear(low, positive, layer.bias)+F.linear(high, negative, None)
        next_high = F.linear(high, positive, layer.bias)+F.linear(low, negative, None)
        low, high = next_low, next_high
        if i+1 < len(barrier.layers):
            if barrier.activation == "tanh":
                low, high = low.tanh(), high.tanh()
            elif barrier.activation == "relu":
                low, high = low.relu(), high.relu()
            else:
                raise ValueError("unsupported activation")
    if barrier.square_output:
        low, high = F.softplus(low), F.softplus(high)
    return low.flatten(), high.flatten()


def group_actions(stream, action_source="environment-hull"):
    if action_source not in ("environment-hull", "nominal", "full-actuator"):
        raise ValueError("unknown action source")
    key = np.column_stack([stream["distance_m"], stream["speed_mps"]])
    unique, inverse = np.unique(key, axis=0, return_inverse=True)
    n = len(unique)
    fields = {k: np.full(n, np.inf) for k in ("action_low", "barrier_low", "next_distance_low")}
    fields.update({k: np.full(n, -np.inf) for k in ("action_high", "barrier_high", "next_distance_high")})
    sources = ("image_action", "q_action") if action_source != "nominal" else ("q_action",)
    for source in sources:
        np.minimum.at(fields["action_low"], inverse, stream[source])
        np.maximum.at(fields["action_high"], inverse, stream[source])
    np.minimum.at(fields["barrier_low"], inverse, stream["current_barrier"])
    np.maximum.at(fields["barrier_high"], inverse, stream["current_barrier"])
    np.minimum.at(fields["next_distance_low"], inverse, stream["image_next_state"][:, 0])
    np.maximum.at(fields["next_distance_high"], inverse, stream["image_next_state"][:, 0])
    count = np.bincount(inverse)
    # Never silently treat an incomplete set of appearances as full coverage.
    appearances = sorted(set(stream["appearance"].tolist()))
    if len(appearances) != 12 or not np.all(count >= 12):
        raise ValueError("expected all 12 matched appearances at every state")
    for appearance in appearances:
        if len(np.unique(inverse[stream["appearance"] == appearance])) != n:
            raise ValueError("duplicate/missing appearance for a physical state")
    if action_source == "nominal" and not np.allclose(fields["action_low"], fields["action_high"], rtol=0, atol=1e-6):
        raise ValueError("nominal q action differs across identical states")
    if action_source == "full-actuator":
        if np.any(fields["action_low"] < -3.) or np.any(fields["action_high"] > 3.):
            raise ValueError("recorded actions not clipped to registered actuator range")
        fields["action_low"].fill(-3.)
        fields["action_high"].fill(3.)
    fields["observations_per_state"] = count
    return unique, fields, appearances


def noise_cells(env, bins, device, noise_model):
    if noise_model == "none":
        zero = torch.zeros((1, 2), device=device, dtype=torch.float64)
        return zero, zero
    if noise_model != "original-uniform":
        raise ValueError("unknown noise model")
    axes = [torch.linspace(float(lo), float(hi), bins+1, device=device, dtype=torch.float64)
            for lo, hi in zip(env.noise_bounds[0], env.noise_bounds[1])]
    nl = torch.stack(torch.meshgrid(*(a[:-1] for a in axes), indexing="ij"), dim=-1).reshape(-1, 2)
    nh = torch.stack(torch.meshgrid(*(a[1:] for a in axes), indexing="ij"), dim=-1).reshape(-1, 2)
    return nl, nh


def evaluate_cells(barrier, env, states, fields, ids, bins, device, batch_size, noise_model="original-uniform", action_splits=1):
    if action_splits < 1:
        raise ValueError("positive action subdivisions required")
    dtype = torch.float64
    nl, nh = noise_cells(env, bins, device, noise_model)
    margins, lower_margins, point_margins, outside, widths = [], [], [], [], []
    fractions = torch.linspace(0., 1., action_splits+1, device=device, dtype=dtype)
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            chosen = ids[start:start+batch_size]
            current = torch.from_numpy(states[chosen]).to(device=device, dtype=dtype)
            a_low = torch.from_numpy(fields["action_low"][chosen]).to(device)
            a_high = torch.from_numpy(fields["action_high"][chosen]).to(device)
            action_edges = a_low[:, None]+(a_high-a_low)[:, None]*fractions[None, :]
            successor_low = torch.stack((torch.from_numpy(fields["next_distance_low"][chosen]).to(device)[:, None].expand(-1, action_splits),
                                         (current[:, 1:2]-.05*action_edges[:, 1:]).clamp(0., 3.)), dim=2)
            successor_high = torch.stack((torch.from_numpy(fields["next_distance_high"][chosen]).to(device)[:, None].expand(-1, action_splits),
                                          (current[:, 1:2]-.05*action_edges[:, :-1]).clamp(0., 3.)), dim=2)
            # Covers operational float32 successor roundoff in the input arrays.
            # Ordinary floating IBP is not an outward-rounded formal verifier.
            pad = 8*torch.finfo(torch.float32).eps*torch.maximum(torch.ones_like(successor_low),
                                                                            torch.maximum(successor_low.abs(), successor_high.abs()))
            low = successor_low[:, :, None, :]+nl[None, None, :, :]-pad[:, :, None, :]
            high = successor_high[:, :, None, :]+nh[None, None, :, :]+pad[:, :, None, :]
            lb, ub = interval_barrier(barrier, low.reshape(-1, 2), high.reshape(-1, 2))
            points = barrier(((low+high)/2).reshape(-1, 2)).flatten()
            if not torch.all((lb <= points+1e-8) & (points <= ub+1e-8)):
                raise ValueError("IBP does not enclose its midpoint samples")
            # max per noise cell bounds even noise-dependent action selection;
            # never average over action intervals as if actions were random.
            expected_low = lb.reshape(len(chosen), action_splits, -1).min(dim=1).values.mean(1)
            expected_high = ub.reshape(len(chosen), action_splits, -1).max(dim=1).values.mean(1)
            expected_point = points.reshape(len(chosen), action_splits, -1).max(dim=1).values.mean(1)
            b = torch.from_numpy(fields["barrier_low"][chosen]).to(device)
            margins.extend((b-expected_high).cpu().tolist())
            lower_margins.extend((b-expected_low).cpu().tolist())
            point_margins.extend((b-expected_point).cpu().tolist())
            widths.extend((successor_high[:, 0, 1]-successor_low[:, -1, 1]).cpu().tolist())
            domain_low = torch.tensor(env.observation_space.low, device=device, dtype=dtype)
            domain_high = torch.tensor(env.observation_space.high, device=device, dtype=dtype)
            out = ((low < domain_low) | (high > domain_high)).any(dim=3).any(dim=1).double().mean(dim=1)
            outside.extend(out.cpu().tolist())
    return {"margin": np.asarray(margins), "optimistic_margin": np.asarray(lower_margins),
            "midpoint_margin": np.asarray(point_margins), "out_of_domain_cell_fraction": np.asarray(outside),
            "successor_speed_interval_width": np.asarray(widths)}


def diagnose_fixed_action_cases(result_dir, diagnostic_dir, spvc_dir, device="cuda"):
    """Read-only fixed-action checks at the two recorded worst-case states.

    Returning both expectation bounds distinguishes an inconclusive upper
    bound from a fixed action whose expected B is bounded above current B.
    Floating arithmetic is not an outward-rounded formal proof.
    """
    result_dir, diagnostic_dir, spvc_dir = map(Path, (result_dir, diagnostic_dir, spvc_dir))
    result_path = result_dir/"metrics.json"
    saved = json.loads(result_path.read_text())
    if saved.get("action_source") != "full-actuator" or saved.get("status") not in ("sampled_full_actuator_pass", "sampled_full_actuator_unresolved"):
        raise ValueError("completed full-actuator result required")
    actor_path, barrier_path = spvc_dir/"latent_ppo_spvc.pt", spvc_dir/"sbc.pt"
    for k, p in (("actor", actor_path), ("barrier", barrier_path), ("diagnostic", diagnostic_dir/"metrics.json")):
        if sha256(p) != saved["source_hashes"][k]:
            raise ValueError("source identity changed")
    torch.set_num_threads(1)
    target_device = torch.device(device)
    model = MLP([2,16,8,1], activation="tanh", square_output=True).to(target_device).double().eval()
    model.load_state_dict(torch.load(barrier_path, map_location=target_device), strict=True)
    for p in model.parameters():
        p.requires_grad_(False)
    env = Aebs(.05)
    actions = np.linspace(-3., 3., 17)
    report = dict(experiment="fixed_action_interpretation_of_full_actuator_check", training_run=False,
                  stress_metrics_sha256=sha256(result_path), helper_script_sha256=sha256(Path(__file__)),
                  scope="17 constant actions at each recorded worst state; original registered noise model",
                  arithmetic="float64 interval bounds with float32 successor padding, not outward-rounded formal proof",
                  meaning="negative margin_upper indicates expected B exceeds current B under the registered fixed action/noise calculation; not collision or actual-policy failure",
                  results={})
    for split, summary in saved["results"].items():
        pair_path = diagnostic_dir/(split+"_pairs.npz")
        if sha256(pair_path) != saved["source_hashes"][split+"_pairs"]:
            raise ValueError("paired data changed")
        with np.load(pair_path, allow_pickle=False) as stream:
            states, fields, _ = group_actions(stream, "full-actuator")
            chosen = np.flatnonzero((states == np.asarray(summary["worst"]["state"])).all(axis=1))
            if len(chosen) != 1:
                raise ValueError("worst state not uniquely matched")
            chosen = int(chosen[0])
            mask = (stream["distance_m"] == states[chosen, 0]) & (stream["speed_mps"] == states[chosen, 1])
            q, image = stream["q_action"][mask], stream["image_action"][mask]
        replicated = {k: np.repeat(v[chosen], len(actions)) for k, v in fields.items()}
        replicated["action_low"], replicated["action_high"] = actions, actions
        bins = summary["worst"]["bins"]
        out = evaluate_cells(model, env, np.repeat(states[chosen:chosen+1], len(actions), axis=0),
                             replicated, np.arange(len(actions)), bins, target_device, 4, saved["noise_model"], 1)
        cases = [dict(action=float(a), margin_lower=float(out["margin"][i]),
                      margin_upper=float(out["optimistic_margin"][i]), midpoint_margin=float(out["midpoint_margin"][i]))
                 for i, a in enumerate(actions)]
        report["results"][split] = dict(state=states[chosen].tolist(), current_barrier=float(fields["barrier_low"][chosen]),
            noise_bins=bins, recorded_q_action_range=[float(q.min()), float(q.max())],
            recorded_image_action_range=[float(image.min()), float(image.max())], actions=cases,
            fixed_actions_with_negative_margin_upper=int((out["optimistic_margin"] < 0).sum()),
            worst_fixed_action=cases[int(np.argmin(out["optimistic_margin"]))])
    report["source_models_unchanged"] = sha256(actor_path) == saved["source_hashes"]["actor"] and sha256(barrier_path) == saved["source_hashes"]["barrier"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-dir", type=Path, required=True)
    parser.add_argument("--spvc-dir", type=Path, default=Path("results/dino_spvc_minimal_v1"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--noise-bins", default="10,20,40,80")
    parser.add_argument("--noise-model", choices=("original-uniform", "none"), default="original-uniform",
                        help="none is an explicitly deterministic AEBS model diagnostic, not a real-vehicle claim")
    parser.add_argument("--action-source", choices=("environment-hull", "nominal", "full-actuator"), default="environment-hull")
    parser.add_argument("--action-splits", type=int, default=1, help="cover the same full interval with subintervals; maximize rather than average over them")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    bins_list = [int(x) for x in args.noise_bins.split(",")]
    if args.output_dir.exists() or not bins_list or bins_list != sorted(set(bins_list)) or not 2 <= min(bins_list) <= max(bins_list) <= 100 or args.batch_size < 1 or not 1 <= args.action_splits <= 128:
        parser.error("new output required; sorted unique noise resolutions 2..100 and positive batch")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    torch.set_num_threads(1)
    started = time.monotonic()
    diagnostic_path = args.diagnostic_dir/"metrics.json"
    diagnostic = json.loads(diagnostic_path.read_text())
    if diagnostic.get("status") != "environment_barrier_diagnostic_complete" or diagnostic.get("source_files_unchanged") is not True or diagnostic.get("test_used") is not False:
        raise ValueError("diagnostic incomplete or not development-only")
    barrier_path = args.spvc_dir/"sbc.pt"
    actor_path = args.spvc_dir/"latent_ppo_spvc.pt"
    if sha256(barrier_path) != diagnostic["source_hashes"]["barrier"] or sha256(actor_path) != diagnostic["source_hashes"]["actor"]:
        raise ValueError("source model identity changed")
    env = Aebs(.05)
    if not np.isclose(env.std1, diagnostic["vt_distance_scale_m"], rtol=1e-6, atol=1e-7):
        raise ValueError("VT coordinates changed")
    device = torch.device(args.device)
    barrier = MLP([2,16,8,1], activation="tanh", square_output=True).to(device).double().eval()
    barrier.load_state_dict(torch.load(barrier_path, map_location=device), strict=True)
    for p in barrier.parameters():
        p.requires_grad_(False)
    hashes = {"barrier": sha256(barrier_path), "actor": sha256(actor_path), "diagnostic": sha256(diagnostic_path), "script": sha256(Path(__file__))}
    report = dict(experiment="registered_action_range_noise_IBP_v3", status="running", training_run=False,
                  test_used=False, source_hashes=hashes, results={}, resolutions=bins_list,
                  scope="fixed original legacy band; registered action range at exact recorded states",
                  meaning="finite-state sensitivity check under the registered action/noise model; no continuous-state or unseen-environment claim",
                  arithmetic="float64 interval propagation with float32 successor input padding; not outward-rounded formal arithmetic",
                  original_noise_assumption="uniform factor .05 is the legacy model, not independently established as physical process noise",
                  action_source=args.action_source, noise_model=args.noise_model,
                  action_splits=args.action_splits,
                  action_interval_aggregation="maximum upper B bound across all action subintervals at each noise cell, then uniform noise average",
                  full_actuator_scope="includes every clipped image action at these exact states under the registered physics/noise model, not continuous-state or task-level safety",
                  fixed_band=diagnostic["fixed_band"],
                  noise_bounds_vt=[x.tolist() for x in env.noise_bounds] if args.noise_model == "original-uniform" else [[0., 0.], [0., 0.]],
                  original_backend_changed=False,
                  zero_noise_scope="analytic AEBS update only; does not assert zero CARLA/real-world dynamics error")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir/"metrics.json", report)
    try:
        for split in ("train", "validation"):
            input_path = args.diagnostic_dir/(split+"_pairs.npz")
            hashes[split+"_pairs"] = sha256(input_path)
            with np.load(input_path, allow_pickle=False) as stream:
                states, fields, appearances = group_actions(stream, args.action_source)
            band_low, band_high = diagnostic["fixed_band"]
            selected = np.flatnonzero((fields["barrier_low"] > band_low) & (fields["barrier_high"] < band_high))
            if not len(selected):
                raise ValueError("empty band is not evidence")
            pending = selected.copy()
            final_margin = np.full(len(states), np.nan)
            final_bins = np.zeros(len(states), dtype=int)
            rounds = []
            for bins in bins_list[:1] if args.noise_model == "none" else bins_list:
                result = evaluate_cells(barrier, env, states, fields, pending, bins, device, args.batch_size, args.noise_model, args.action_splits)
                if not all(np.isfinite(v).all() for v in result.values()):
                    raise ValueError("nonfinite interval output")
                margin = result["margin"]
                final_margin[pending], final_bins[pending] = margin, bins
                row = dict(bins=bins, evaluated_states=len(pending), remaining_nonpositive=int((margin <= 0).sum()),
                           noise_cells_per_state=bins*bins if args.noise_model == "original-uniform" else 1,
                           action_subintervals=args.action_splits,
                           min_lower_margin=float(margin.min()), max_action_width=float((fields["action_high"]-fields["action_low"])[pending].max()),
                           max_successor_speed_width=float(result["successor_speed_interval_width"].max()),
                           mean_out_of_domain_noise_cell_fraction=float(result["out_of_domain_cell_fraction"].mean()))
                rounds.append(row)
                print(split, "envelope refinement", row, flush=True)
                np.savez_compressed(args.output_dir/(split+"_bins%d.npz" % bins), state_index=pending, states=states[pending], **result)
                pending = pending[margin <= 0]
                if not len(pending):
                    break
            worst = selected[np.argmin(final_margin[selected])]
            summary = dict(unique_matched_states=len(states), band_states=len(selected), remaining=len(pending),
                           min_lower_margin=float(final_margin[selected].min()), rounds=rounds,
                           worst=dict(state=states[worst].tolist(), action_interval=[fields["action_low"][worst], fields["action_high"][worst]],
                                      lower_margin=float(final_margin[worst]), bins=int(final_bins[worst])),
                           appearances=appearances, all_12_appearances_per_state=True,
                           observations_per_state_range=[int(fields["observations_per_state"].min()), int(fields["observations_per_state"].max())],
                           repeated_observations_retained=True)
            report["results"][split] = summary
            write_json(args.output_dir/"metrics.json", report)
        passed = all(v["remaining"] == 0 for v in report["results"].values())
        if args.action_source == "full-actuator":
            report["status"] = "sampled_full_actuator_pass" if passed else "sampled_full_actuator_unresolved"
        else:
            report["status"] = "sampled_environment_envelope_pass" if passed else "sampled_environment_envelope_unresolved"
        report["source_models_unchanged"] = sha256(barrier_path) == hashes["barrier"] and sha256(actor_path) == hashes["actor"]
        if not report["source_models_unchanged"]:
            raise ValueError("source models changed during check")
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        report["runtime_seconds"] = time.monotonic()-started
        write_json(args.output_dir/"metrics.json", report)
    print("\n[State-conditional empirical environment envelope]")
    print("status:", report["status"], flush=True)
    print("action source:", args.action_source, "noise model:", args.noise_model, flush=True)
    for split, result in report["results"].items():
        print(split, "states=%d unresolved=%d min_lower_margin=%.9f" % (result["band_states"], result["remaining"], result["min_lower_margin"]), flush=True)
    print("Registered action range at sampled states only; no continuous-state or end-to-end safety certificate.", flush=True)
    print("metrics:", args.output_dir/"metrics.json", flush=True)


if __name__ == "__main__":
    main()
