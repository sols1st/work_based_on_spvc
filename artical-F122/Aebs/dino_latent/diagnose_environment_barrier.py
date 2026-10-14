"""Exact-image environment effects on the frozen final SPVC barrier.

All states use each image's own distance label. No nearest lookup/test data.
Original lower-corner noise quadrature and midpoint quadrature are explicitly
separate sampled diagnostics, not interval bounds or a safety certificate.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from Aebs.dino_latent.diagnose_expanded_failures import load_fixed_inputs
from Aebs.dino_latent.prepare_expanded_data import sha256
from Aebs.dino_latent.spvc_saved_actor import load_spvc_pair
from Aebs.system.env import Aebs
from Aebs.VT.utils import MLP


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def physical_next(states, actions, scale):
    return torch.stack((states[:, 0]-.05*states[:, 1]/scale,
                        torch.clamp(states[:, 1]-.05*torch.clamp(actions.reshape(-1), -3., 3.), 0., 3.)), dim=1)


def summarize(current, nominal, image, mask, tolerance=0.):
    ids = np.flatnonzero(mask)
    if not len(ids):
        return {"checked_pairs": 0}
    q_margin, image_margin = current[ids]-nominal[ids], current[ids]-image[ids]
    q_bad, image_bad = q_margin <= tolerance, image_margin <= tolerance
    extra = image[ids]-nominal[ids]
    return dict(checked_pairs=len(ids), q_violations=int(q_bad.sum()), image_violations=int(image_bad.sum()),
                image_extra_violations=int((~q_bad & image_bad).sum()),
                q_only_violations=int((q_bad & ~image_bad).sum()),
                q_min_margin=float(q_margin.min()), image_min_margin=float(image_margin.min()),
                environmental_barrier_shift_mean=float(extra.mean()),
                environmental_barrier_shift_max=float(extra.max()),
                environmental_barrier_shift_min=float(extra.min()),
                worst_image_local_index=int(ids[np.argmin(image_margin)]))


def noise_points(env, bins, device):
    axes = [torch.linspace(float(lo), float(hi), bins+1, device=device)
            for lo, hi in zip(env.noise_bounds[0], env.noise_bounds[1])]
    low = torch.stack(torch.meshgrid(*(a[:-1] for a in axes), indexing="ij"), dim=-1).reshape(-1, 2)
    high = torch.stack(torch.meshgrid(*(a[1:] for a in axes), indexing="ij"), dim=-1).reshape(-1, 2)
    return low, (low+high)/2


def quadrature(barrier, successor, noise):
    # Uniform equal-volume cells, preserving the original env.integrate_noise.
    return barrier((successor[:, None, :]+noise[None, :, :]).reshape(-1, 2)).reshape(len(successor), -1).mean(dim=1)


def main():
    root = Path("results/dino_expanded_v1")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=root/"02_dino_h5")
    parser.add_argument("--representation", type=Path, default=root/"08_R2_group_alignment/dino_safety_latent.pt")
    parser.add_argument("--ppo-dir", type=Path, default=root/"09_R2_ppo")
    parser.add_argument("--spvc-dir", type=Path, default=Path("results/dino_spvc_minimal_v1"))
    parser.add_argument("--final-recheck", type=Path, default=Path("results/dino_spvc_minimal_v1_final_recheck/metrics.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--speed-count", type=int, default=61)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--noise-bins", type=int, default=10)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.output_dir.exists() or args.speed_count < 2 or args.batch_size < 1 or not 2 <= args.noise_bins <= 80:
        parser.error("new output required and valid sizes")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    torch.set_num_threads(1)
    started = time.monotonic()
    fixed = load_fixed_inputs(args, parser)
    _, saved_actor, _, identity = load_spvc_pair(args.representation, fixed["controller_path"], args.spvc_dir, args.final_recheck)
    device = torch.device(args.device)
    policy = saved_actor.policy.to(device).eval()
    barrier = MLP([2, 16, 8, 1], activation="tanh", square_output=True).to(device).eval()
    barrier.load_state_dict(torch.load(args.spvc_dir/"sbc.pt", map_location=device), strict=True)
    for model in (policy, barrier):
        for p in model.parameters():
            p.requires_grad_(False)
    env = Aebs(.05)
    scale = float(policy.state_distance_scale_m)
    if not np.isclose(env.std1, scale, rtol=1e-6, atol=1e-7):
        raise ValueError("VT scale mismatch")
    recheck = json.loads(args.final_recheck.read_text())
    unsafe_lower, domain_lower = recheck["unsafe_lower"], recheck["domain_lower"]
    band_lower = domain_lower+.95*(unsafe_lower-domain_lower)
    lower, midpoint = noise_points(env, args.noise_bins, device)
    latents = torch.from_numpy(fixed["latents"]).to(device)
    distances = fixed["distances"]
    speeds = np.linspace(0., 3., args.speed_count, dtype=np.float32)
    source = {"representation": args.representation, "actor": args.spvc_dir/"latent_ppo_spvc.pt",
              "barrier": args.spvc_dir/"sbc.pt", "cache": fixed["cache_path"], "features": fixed["features_path"],
              "final_recheck": args.final_recheck, "script": Path(__file__)}
    hashes = {k: sha256(v) for k, v in source.items()}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = dict(experiment="final_SPVC_exact_image_environment_barrier_v1", status="running", results={},
                  model_identity=identity, source_hashes=hashes, training_run=False, test_used=False,
                  speed_grid_mps=speeds.tolist(), noise_bins=args.noise_bins,
                  barrier_coordinates=["distance / VT scale", "speed m/s"], vt_distance_scale_m=scale,
                  fixed_band=[band_lower, unsafe_lower],
                  scope="sampled exact-distance development diagnostic, no nearest lookup or interval certificate",
                  noise_interpretation="legacy independent uniform noise is a sensitivity assumption, not a calibrated environment contract",
                  inequality="margin=current B - next/expected B; original hard check uses margin<=0; no added epsilon",
                  successor_terminal_handling="raw saved B evaluated including terminal successor; current terminal states excluded")
    write_json(args.output_dir/"metrics.json", report)
    try:
        for split in ("train", "validation"):
            row_ids = np.asarray([i for i, r in enumerate(fixed["rows"]) if r["split"] == split], dtype=int)
            image_ids = np.repeat(row_ids, len(speeds))
            d, v = distances[image_ids], np.tile(speeds, len(row_ids))
            keep = (d > 6.) & (d <= 16.) & (v > 1e-6)
            image_ids, d, v = image_ids[keep], d[keep], v[keep]
            states = np.column_stack([d/scale, v]).astype(np.float32)
            arrays = {"image_row": image_ids, "state": states, "distance_m": d, "speed_mps": v}
            arrays.update({k: np.empty(len(states), dtype=np.float32) for k in (
                "current_barrier", "q_action", "image_action", "q_next_barrier", "image_next_barrier",
                "q_legacy_expected", "image_legacy_expected", "q_midpoint_expected", "image_midpoint_expected")})
            arrays.update({k: np.empty((len(states), 2), dtype=np.float32) for k in ("q_next_state", "image_next_state")})
            with torch.no_grad():
                for start in range(0, len(states), args.batch_size):
                    stop = min(start+args.batch_size, len(states))
                    x = torch.from_numpy(states[start:stop]).to(device)
                    q_input = x[:, :1]*(scale/policy.distance_scale_m)
                    q_latent = policy.q_model(q_input)
                    img_latent = latents[torch.from_numpy(image_ids[start:stop]).to(device)]
                    aq = policy.forward_latent(q_latent, x[:, 1:2]).clamp(-3., 3.)
                    ai = policy.forward_latent(img_latent, x[:, 1:2]).clamp(-3., 3.)
                    nq, ni = physical_next(x, aq, scale), physical_next(x, ai, scale)
                    values = dict(current_barrier=barrier(x).flatten(), q_action=aq.flatten(), image_action=ai.flatten(),
                                  q_next_barrier=barrier(nq).flatten(), image_next_barrier=barrier(ni).flatten(),
                                  q_legacy_expected=quadrature(barrier, nq, lower), image_legacy_expected=quadrature(barrier, ni, lower),
                                  q_midpoint_expected=quadrature(barrier, nq, midpoint), image_midpoint_expected=quadrature(barrier, ni, midpoint),
                                  q_next_state=nq, image_next_state=ni)
                    for key, value in values.items():
                        if not torch.isfinite(value).all():
                            raise ValueError("nonfinite model output: "+key)
                        arrays[key][start:stop] = value.cpu().numpy()
                    if start == 0 or start//args.batch_size % 50 == 0 or stop == len(states):
                        print("%s environment matched pairs %d/%d" % (split, stop, len(states)), flush=True)
            current = arrays["current_barrier"]
            band = (current > band_lower) & (current < unsafe_lower)
            arrays["fixed_legacy_band"] = band
            appearance = np.asarray([fixed["rows"][int(i)]["weather"]+"|"+fixed["rows"][int(i)]["color"] for i in image_ids])
            arrays["appearance"] = appearance
            scopes = {"all_ongoing": np.ones(len(states), bool), "fixed_legacy_band": band,
                      "below_unsafe": current < unsafe_lower}
            result = {"images": len(row_ids), "pairs": len(states), "scopes": {}, "by_appearance": {}}
            for name, mask in scopes.items():
                result["scopes"][name] = {mode: summarize(current, arrays[qkey], arrays[ikey], mask)
                                         for mode, qkey, ikey in (
                                             ("deterministic", "q_next_barrier", "image_next_barrier"),
                                             ("legacy_lower_corner", "q_legacy_expected", "image_legacy_expected"),
                                             ("noise_midpoint", "q_midpoint_expected", "image_midpoint_expected"))}
            for name in sorted(set(appearance.tolist())):
                mask = appearance == name
                result["by_appearance"][name] = {scope: summarize(current, arrays["q_legacy_expected"],
                                                                 arrays["image_legacy_expected"], mask & chosen)
                                                  for scope, chosen in scopes.items()}
            margin = current-arrays["image_legacy_expected"]
            priority = np.flatnonzero(band) if band.any() else np.arange(len(states))
            worst = priority[np.argsort(margin[priority])[:20]]
            details = []
            for i in worst:
                row = fixed["rows"][int(image_ids[i])]
                details.append({"filename": row["filename"], "appearance": str(appearance[i]),
                                "distance_m": float(d[i]), "speed_mps": float(v[i]), "legacy_band": bool(band[i]),
                                **{k: float(arrays[k][i]) for k in ("current_barrier", "q_action", "image_action", "q_legacy_expected", "image_legacy_expected")}})
            result["worst_cases"] = details
            np.savez_compressed(args.output_dir/(split+"_pairs.npz"), **arrays)
            report["results"][split] = result
            write_json(args.output_dir/"metrics.json", report)
        report.update(status="environment_barrier_diagnostic_complete", source_files_unchanged={k: sha256(v) for k, v in source.items()} == hashes)
        if not report["source_files_unchanged"]:
            raise ValueError("source changed during experiment")
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        report["runtime_seconds"] = time.monotonic()-started
        write_json(args.output_dir/"metrics.json", report)
    print("\n[Final SPVC environment barrier diagnostic: sampled, no certificate]")
    for split, result in report["results"].items():
        for scope in ("fixed_legacy_band", "all_ongoing"):
            for mode in ("legacy_lower_corner", "deterministic", "noise_midpoint"):
                print(split, scope, mode, result["scopes"][scope][mode], flush=True)
    print("metrics:", args.output_dir/"metrics.json", flush=True)


if __name__ == "__main__":
    main()
