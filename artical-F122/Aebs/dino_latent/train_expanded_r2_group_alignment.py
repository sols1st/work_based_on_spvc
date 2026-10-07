"""Fine-tune R1 with explicit same-distance appearance-group alignment.

R1 already contains pointwise image-to-q MSE.  R2 adds a distinct constraint:
all train images sharing one exact physical distance are encoded together and
pulled toward their appearance-group centroid, while q(distance) is pulled to
that centroid.  DINO stays frozen and test rows are forbidden.
"""

import argparse
import copy
import json
import math
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from Aebs.dino_latent.models import SafetyDecoder, SafetyProjection, PhysicalToLatent
from Aebs.dino_latent.prepare_expanded_data import sha256
from Aebs.dino_latent.train_expanded_alignment import evaluate, training_ids


def exact_distance_groups(rows, ids):
    groups = {}
    for index in ids:
        groups.setdefault(rows[index]["group_id"], []).append(index)
    ordered = [np.asarray(groups[key], dtype=np.int64) for key in sorted(groups)]
    for group in ordered:
        distances = {round(float(rows[index]["distance_m"]), 9) for index in group}
        if len(distances) != 1:
            raise ValueError("group_id contains multiple distances")
    return ordered


def grouped_loss(p, decoder, q, features, distance, scale, groups, group_ids,
                 group_weight, centroid_q_weight):
    image_latents = []
    physical_latents = []
    normalized_distances = []
    invariance_terms = []
    centroid_q_terms = []
    for group_number in group_ids:
        indices = groups[int(group_number)]
        y = distance[indices] / scale
        z = p(features[indices])
        physical = q(y[:, None])
        centroid = z.mean(dim=0, keepdim=True)
        invariance_terms.append(F.mse_loss(z, centroid.expand_as(z)))
        centroid_q_terms.append(F.mse_loss(q(y[:1, None]), centroid))
        image_latents.append(z)
        physical_latents.append(physical)
        normalized_distances.append(y)
    z = torch.cat(image_latents)
    physical = torch.cat(physical_latents)
    y = torch.cat(normalized_distances)
    permutation = torch.randperm(len(z), device=z.device)
    far = (y - y[permutation]).abs() * scale >= 1
    separation = (z - z[permutation]).norm(dim=1) / math.sqrt(z.shape[1])
    separation_loss = (
        torch.relu(.5 - separation[far]).square().mean()
        if bool(far.any()) else z.new_tensor(0.)
    )
    base = (
        F.mse_loss(decoder(z).flatten(), y)
        + F.mse_loss(decoder(physical).flatten(), y)
        + F.mse_loss(z, physical)
        + .1 * separation_loss
        + 1e-4 * z.square().mean()
    )
    group_invariance = torch.stack(invariance_terms).mean()
    centroid_q = torch.stack(centroid_q_terms).mean()
    return base + group_weight * group_invariance + centroid_q_weight * centroid_q


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("results/dino_expanded_v1/02_dino_h5"))
    parser.add_argument(
        "--r1-representation", type=Path,
        default=Path("results/dino_expanded_v1/03_alignment_r0_r1/R1_all_appearances/dino_safety_latent.pt"),
    )
    parser.add_argument(
        "--r1-metrics", type=Path,
        default=Path("results/dino_expanded_v1/03_alignment_r0_r1/R1_all_appearances/metrics.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/dino_expanded_v1/08_R2_group_alignment"))
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--q-steps", type=int, default=400)
    parser.add_argument("--groups-per-batch", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--group-weight", type=float, default=0.2)
    parser.add_argument("--centroid-q-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if (
        args.output_dir.exists() or min(
            args.steps, args.q_steps, args.groups_per_batch, args.eval_every
        ) < 1 or args.group_weight < 0 or args.centroid_q_weight < 0
    ):
        parser.error("Use a new output directory and valid positive settings")
    required = [
        args.cache_dir / "cache_manifest.json", args.cache_dir / "features.h5",
        args.r1_representation, args.r1_metrics,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error("Missing inputs: " + ", ".join(missing))
    cache_path = args.cache_dir / "cache_manifest.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    r1 = torch.load(args.r1_representation, map_location="cpu")
    r1_metrics = json.loads(args.r1_metrics.read_text(encoding="utf-8"))
    if (
        cache.get("schema") != "spvc_expanded_dino_h5_cache_v1"
        or cache.get("test_encoded") != 0
        or r1["provenance"]["cache_manifest_sha256"] != sha256(cache_path)
        or r1_metrics.get("test_evaluated") is not False
    ):
        parser.error("R1/cache provenance mismatch or test data present")
    rows = cache["records"]
    if any(row["split"] not in ("train", "validation") for row in rows):
        parser.error("Test rows forbidden")
    features_path = args.cache_dir / "features.h5"
    if sha256(features_path) != cache["feature_sha256"]:
        parser.error("Feature H5 hash changed")
    with h5py.File(features_path, "r") as stream:
        if not stream.attrs["complete"] or stream["features"].shape != (len(rows), 384):
            parser.error("Invalid feature H5")
        raw = torch.from_numpy(np.asarray(stream["features"], dtype=np.float32)).to(args.device)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    train_ids = training_ids(rows)
    validation_ids = [i for i, row in enumerate(rows) if row["split"] == "validation"]
    groups = exact_distance_groups(rows, train_ids)
    if len(groups) < args.groups_per_batch:
        parser.error("groups-per-batch exceeds available train groups")
    config = {
        "experiment": "R2_same_distance_appearance_group_alignment",
        "registered_settings": {
            "steps": args.steps, "q_steps": args.q_steps,
            "groups_per_batch": args.groups_per_batch,
            "group_weight": args.group_weight,
            "centroid_q_weight": args.centroid_q_weight,
            "seed": args.seed,
            "selection": "same R1 score: validation image RMSE/scale + 0.2 relative latent RMSE",
        },
        "test_evaluated": False,
        "dino_frozen": True,
        "cache_manifest_sha256": sha256(cache_path),
        "r1_representation_sha256": sha256(args.r1_representation),
        "r1_metrics_sha256": sha256(args.r1_metrics),
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2, allow_nan=False), encoding="utf-8"
    )
    mean = r1["feature_mean"].to(args.device)
    std = r1["feature_std"].to(args.device)
    features = (raw - mean) / std
    distance = torch.tensor(
        [row["distance_m"] for row in rows], dtype=torch.float32, device=args.device
    )
    scale = float(r1["distance_scale_m"])
    p = SafetyProjection().to(args.device)
    decoder = SafetyDecoder().to(args.device)
    q = PhysicalToLatent().to(args.device)
    p.load_state_dict(r1["projection_state_dict"])
    decoder.load_state_dict(r1["auxiliary_decoder_state_dict"])
    q.load_state_dict(r1["physical_to_latent_state_dict"])
    parameters = list(p.parameters()) + list(decoder.parameters()) + list(q.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=3e-4, weight_decay=1e-4)

    baseline = evaluate(p, decoder, q, features, distance, scale, validation_ids, rows)
    best = tuple(copy.deepcopy(net.state_dict()) for net in (p, decoder, q))
    best_score = (
        baseline["image_distance_rmse_m"] / scale
        + .2 * baseline["relative_latent_rmse"]
    )
    best_step = 0
    history = [{
        "step": 0, "score": best_score,
        "relative_latent_rmse": baseline["relative_latent_rmse"],
        "latent_rmse": baseline["latent_rmse"],
        "image_distance_rmse_m": baseline["image_distance_rmse_m"],
    }]
    started = time.monotonic()
    for step in range(1, args.steps + 1):
        selected = rng.choice(
            len(groups), size=args.groups_per_batch, replace=False
        )
        loss = grouped_loss(
            p, decoder, q, features, distance, scale, groups, selected,
            args.group_weight, args.centroid_q_weight,
        )
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite R2 loss")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 5.)
        optimizer.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            current = evaluate(
                p, decoder, q, features, distance, scale, validation_ids, rows
            )
            score = (
                current["image_distance_rmse_m"] / scale
                + .2 * current["relative_latent_rmse"]
            )
            history.append({
                "step": step, "loss": float(loss), "score": score,
                "relative_latent_rmse": current["relative_latent_rmse"],
                "latent_rmse": current["latent_rmse"],
                "image_distance_rmse_m": current["image_distance_rmse_m"],
            })
            if score < best_score:
                best_score = score
                best_step = step
                best = tuple(copy.deepcopy(net.state_dict()) for net in (p, decoder, q))
            print(
                "R2 step=%d/%d val_relative=%.6f p95=%.6f image_rmse_m=%.6f" % (
                    step, args.steps, current["relative_latent_rmse"],
                    current["residual_l2_p95"], current["image_distance_rmse_m"],
                ), flush=True,
            )
    for net, state in zip((p, decoder, q), best):
        net.load_state_dict(state)

    with torch.no_grad():
        train_target = p(features[train_ids]).detach()
        validation_target = p(features[validation_ids]).detach()
        q_best_loss = float(F.mse_loss(
            q(distance[validation_ids, None] / scale), validation_target
        ))
    q_best = copy.deepcopy(q.state_dict())
    q_best_step = 0
    q_optimizer = torch.optim.AdamW(q.parameters(), lr=1e-3, weight_decay=1e-5)
    for step in range(1, args.q_steps + 1):
        q_loss = F.mse_loss(q(distance[train_ids, None] / scale), train_target)
        q_optimizer.zero_grad()
        q_loss.backward()
        q_optimizer.step()
        with torch.no_grad():
            value = float(F.mse_loss(
                q(distance[validation_ids, None] / scale), validation_target
            ))
        if value < q_best_loss:
            q_best_loss = value
            q_best_step = step
            q_best = copy.deepcopy(q.state_dict())
        if step == 1 or step % 100 == 0:
            print("R2 q_fit=%d/%d" % (step, args.q_steps), flush=True)
    q.load_state_dict(q_best)

    validation = evaluate(
        p, decoder, q, features, distance, scale, validation_ids, rows
    )
    train_metrics = evaluate(
        p, decoder, q, features, distance, scale, train_ids, rows
    )
    result = {
        **config,
        "best_joint_step": best_step,
        "best_q_step": q_best_step,
        "runtime_seconds": time.monotonic() - started,
        "baseline_R1_validation": baseline,
        "train": train_metrics,
        "validation": validation,
        "validation_delta": {
            "relative_latent_rmse": validation["relative_latent_rmse"] - baseline["relative_latent_rmse"],
            "latent_rmse": validation["latent_rmse"] - baseline["latent_rmse"],
            "residual_l2_p95": validation["residual_l2_p95"] - baseline["residual_l2_p95"],
            "image_distance_rmse_m": validation["image_distance_rmse_m"] - baseline["image_distance_rmse_m"],
        },
        "history": history,
    }
    torch.save({
        "projection_state_dict": p.cpu().state_dict(),
        "auxiliary_decoder_state_dict": decoder.cpu().state_dict(),
        "physical_to_latent_state_dict": q.cpu().state_dict(),
        "feature_mean": mean.cpu(), "feature_std": std.cpu(),
        "distance_scale_m": scale, "latent_dimension": 32,
        "dino_frozen": True, "variant": "R2_group_alignment",
        "provenance": config,
    }, args.output_dir / "dino_safety_latent.pt")
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2, allow_nan=False), encoding="utf-8"
    )
    print("[R2 same-distance appearance-group alignment — no PPO/SBC/test]")
    print("R1 validation relative=%.6f R2=%.6f" % (
        baseline["relative_latent_rmse"], validation["relative_latent_rmse"]
    ))
    print("R1 validation image_rmse_m=%.6f R2=%.6f" % (
        baseline["image_distance_rmse_m"], validation["image_distance_rmse_m"]
    ))
    print("metrics:", args.output_dir / "metrics.json")


if __name__ == "__main__":
    main()
