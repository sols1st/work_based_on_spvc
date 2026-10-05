"""Grouped A/B representation training; frozen DINO, no PPO/SBC/CP runs."""
import argparse
import copy
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from Aebs.dino_latent.backbone import FrozenDinoV2
from Aebs.dino_latent.models import SafetyProjection, SafetyDecoder, PhysicalToLatent
from Aebs.dino_latent.train_representation import appearance_view, extract, set_seed, file_sha256


def alignment_stats(z, q):
    rmse = (z - q).square().mean().sqrt()
    scale = (z - z.mean(0)).square().mean().sqrt()
    return {"latent_rmse": float(rmse), "latent_centered_rms": float(scale),
            "relative_latent_rmse": float(rmse / scale) if float(scale) > 1e-8 else None,
            "collapsed": bool(float(scale) <= 1e-8),
            "residual_l2_p95": float(torch.quantile((z-q).norm(dim=1), 0.95))}


def evaluate_rep(p, decoder, q, features, target, scale):
    with torch.no_grad():
        z = p(features[0]); physical = q(target[:, None])
        result = alignment_stats(z, physical)
        for name, value in [("image", z), ("physical", physical)]:
            error = (decoder(value).flatten() - target) * scale
            result[name + "_distance_mae_m"] = float(error.abs().mean())
            result[name + "_distance_rmse_m"] = float(error.square().mean().sqrt())
        result["invariance_mse"] = float((F.mse_loss(z, p(features[1])) + F.mse_loss(z, p(features[2]))) / 2)
        far = (target[:, None] - target[None, :]).abs() * scale >= 1.0
        pair = torch.cdist(z, z) / np.sqrt(z.shape[1])
        result["far_pair_margin_violation_fraction"] = float((pair[far] < 0.5).float().mean()) if bool(far.any()) else None
    return result


def train_variant(name, f, target, distance_scale, split, args, common):
    set_seed(args.seed)
    device = f[0].device
    p, decoder, q = SafetyProjection().to(device), SafetyDecoder().to(device), PhysicalToLatent().to(device)
    joint = name == "B_joint"
    params = list(p.parameters()) + list(decoder.parameters()) + (list(q.parameters()) if joint else [])
    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    train = torch.tensor(split["train"], device=device)
    val = torch.tensor(split["validation"], device=device)
    best, best_score, history = None, float("inf"), []
    for epoch in range(1, args.epochs + 1):
        order = train[torch.randperm(len(train), device=device)]
        for ids in order.split(32):
            z, z1, z2 = [p(view[ids]) for view in f]
            y = target[ids]
            safety = F.mse_loss(decoder(z).flatten(), y)
            inv = (F.mse_loss(z, z1) + F.mse_loss(z, z2)) / 2
            perm = torch.randperm(len(ids), device=device)
            far = (y-y[perm]).abs()*distance_scale >= 1.0
            pairs = (z-z[perm]).norm(dim=1)/np.sqrt(32)
            sep = torch.relu(0.5-pairs[far]).square().mean() if bool(far.any()) else z.new_tensor(0.)
            loss = safety + 0.2*inv + 0.1*sep + 1e-4*z.square().mean()
            if joint:
                physical = q(y[:, None])
                loss = loss + args.align_weight*F.mse_loss(z, physical)
                loss = loss + F.mse_loss(decoder(physical).flatten(), y)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.)
            optimizer.step()
        metrics = evaluate_rep(p, decoder, q, [view[val] for view in f], target[val], distance_scale)
        # Same projection selection criterion in both arms, registered in config.
        score = metrics["image_distance_rmse_m"]/distance_scale + 0.2*np.sqrt(metrics["invariance_mse"])
        if score < best_score:
            best_score = score
            best = (copy.deepcopy(p.state_dict()), copy.deepcopy(decoder.state_dict()), copy.deepcopy(q.state_dict()), epoch)
        if epoch == 1 or epoch % 25 == 0:
            history.append({"epoch": epoch, "validation": metrics})
            print("%s epoch=%d val_image_rmse_m=%.5f latent_rmse=%.5f" % (name, epoch, metrics["image_distance_rmse_m"], metrics["latent_rmse"]), flush=True)
    p.load_state_dict(best[0]); decoder.load_state_dict(best[1]); q.load_state_dict(best[2])
    # Both arms finish with the same frozen-projection q fitting stage.
    with torch.no_grad():
        ztrain, zval = p(f[0][train]), p(f[0][val])
    optq = torch.optim.AdamW(q.parameters(), lr=1e-3, weight_decay=1e-5)
    with torch.no_grad():
        best_q_score = float(F.mse_loss(q(target[val, None]), zval))
    best_q, best_q_epoch = copy.deepcopy(q.state_dict()), 0
    for epoch in range(1, args.q_epochs + 1):
        loss = F.mse_loss(q(target[train, None]), ztrain)
        optq.zero_grad(); loss.backward(); optq.step()
        with torch.no_grad():
            score = float(F.mse_loss(q(target[val, None]), zval))
        if score < best_q_score:
            best_q_score, best_q_epoch, best_q = score, epoch, copy.deepcopy(q.state_dict())
    q.load_state_dict(best_q)
    result = {"variant": name, "best_projection_epoch": best[3], "best_q_epoch": best_q_epoch,
              "test_evaluated": False, "history": history}
    for subset in ("train", "validation"):
        idx = torch.tensor(split[subset], device=device)
        result[subset] = evaluate_rep(p, decoder, q, [view[idx] for view in f], target[idx], distance_scale)
    folder = args.output_dir / name
    folder.mkdir()
    torch.save({**common, "projection_state_dict": p.cpu().state_dict(),
                "auxiliary_decoder_state_dict": decoder.cpu().state_dict(),
                "physical_to_latent_state_dict": q.cpu().state_dict(),
                "best_projection_epoch": best[3], "variant": name}, folder / "dino_safety_latent.pt")
    (folder / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("Aebs/data/Downsampled.h5"))
    parser.add_argument("--manifest", type=Path, default=Path("results/dino_improvement/split.json"))
    parser.add_argument("--dino-repo", type=Path, default=Path("../external/dinov2"))
    parser.add_argument("--dino-weights", type=Path, default=Path("../external/dinov2_weights/dinov2_vits14_pretrain.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/dino_improvement/05_representation_alignment_ab"))
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--q-epochs", type=int, default=400)
    parser.add_argument("--align-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.epochs < 1 or args.q_epochs < 1 or not np.isfinite(args.align_weight) or args.align_weight <= 0:
        parser.error("epochs and alignment weight must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output directory is nonempty")
    torch.set_num_threads(1); set_seed(args.seed)
    manifest = json.loads(args.manifest.read_text())
    if file_sha256(args.data) != manifest["hashes"]["data"]:
        parser.error("dataset differs from split manifest")
    split = manifest["indices"]
    with h5py.File(args.data, "r") as stream:
        images = np.asarray(stream["X_train"], dtype=np.float32)
        distances = np.asarray(stream["y_train"], dtype=np.float32).reshape(-1)
    if sorted(sum(split.values(), [])) != list(range(len(images))) or any(not split[k] for k in ("train", "validation", "test")):
        parser.error("invalid split partition")
    groups = np.asarray(manifest["groups"])
    sets = [set(groups[split[k]]) for k in ("train", "validation", "test")]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        parser.error("distance groups overlap")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    config.update({"manifest_sha256": file_sha256(args.manifest), "data_sha256": file_sha256(args.data),
                   "test_used": False, "selection": "validation image normalized RMSE + 0.2 sqrt(invariance)",
                   "B_changes": "joint q updates, alignment MSE, physical decoder supervision; not single-loss ablation",
                   "historical_validation_already_inspected": True})
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Extract only train/validation images, with no test feature computation.
    used = np.asarray(split["train"] + split["validation"], dtype=int)
    local = {int(i): j for j,i in enumerate(used)}
    local_split = {k: [local[i] for i in split[k]] for k in ("train", "validation")}
    backbone = FrozenDinoV2(args.dino_repo, args.dino_weights, device)
    subset_images = images[used]
    raw = [extract(backbone, view, device, 32) for view in [subset_images, appearance_view(subset_images, args.seed+101), appearance_view(subset_images, args.seed+202)]]
    train_ids = local_split["train"]
    mean = raw[0][train_ids].mean(0); std = raw[0][train_ids].std(0).clamp_min(1e-5)
    features = [((r-mean)/std).to(device) for r in raw]
    distance_scale = float(np.std(distances[split["train"]]))
    if distance_scale <= 0:
        raise ValueError("zero training distance scale")
    target = torch.from_numpy(distances[used]/distance_scale).to(device)
    common = {"dino_model": "dinov2_vits14", "dino_repo": str(args.dino_repo),
              "dino_weights": str(args.dino_weights), "dino_weights_sha256": file_sha256(args.dino_weights),
              "distance_scale_m": distance_scale, "feature_mean": mean, "feature_std": std,
              "latent_dimension": 32, "dino_frozen": True, "manifest_sha256": config["manifest_sha256"]}
    results = {name: train_variant(name, features, target, distance_scale, local_split, args, common)
               for name in ("A_sequential", "B_joint")}
    (args.output_dir / "comparison.json").write_text(json.dumps(results, indent=2, allow_nan=False))
    print("[DINO representation alignment A/B]")
    print("DINO frozen; PPO/SBC unchanged; test not evaluated")
    for name, result in results.items():
        for subset in ("train", "validation"):
            print(name, subset, json.dumps(result[subset], allow_nan=False))
    print("comparison: %s" % (args.output_dir / "comparison.json"))
    print("New latent coordinates and distance scale: old PPO/VT adapters must not be reused without adaptation.")


if __name__ == "__main__":
    main()
