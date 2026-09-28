"""Train the 32-D safety latent around a completely frozen DINOv2 backbone.

The deployed representation is 32 dimensional.  Distance reconstruction is
only an auxiliary safety-sufficiency loss; no one-dimensional estimate is fed
to the future controller.  Mild appearance-only augmentations provide the
first nuisance-invariance signal available in the current 400-image dataset.
"""

import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from Aebs.dino_latent.backbone import FrozenDinoV2
from Aebs.dino_latent.models import PhysicalToLatent, SafetyDecoder, SafetyProjection
from Aebs.semantic.train import regression_metrics, split_indices


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def appearance_view(images, seed):
    """Brightness/contrast/noise only; geometry and distance are unchanged."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    value = torch.from_numpy(images).float().unsqueeze(1)
    brightness = 0.8 + 0.4 * torch.rand((len(value), 1, 1, 1), generator=generator)
    contrast = 0.8 + 0.4 * torch.rand((len(value), 1, 1, 1), generator=generator)
    centered = value - value.mean(dim=(2, 3), keepdim=True)
    noise = 0.015 * torch.randn(value.shape, generator=generator)
    return torch.clamp(centered * contrast + value.mean(dim=(2, 3), keepdim=True), 0, 1).mul(
        brightness
    ).add(noise).clamp(0, 1).squeeze(1).numpy()


def extract(backbone, images, device, batch_size):
    outputs = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = torch.from_numpy(images[start:start + batch_size]).to(device)
            outputs.append(backbone(batch).cpu())
    return torch.cat(outputs).float()


def projection_validation(projection, decoder, f0, f1, f2, target, index, device):
    projection.eval()
    decoder.eval()
    with torch.no_grad():
        z0 = projection(f0[index].to(device))
        z1 = projection(f1[index].to(device))
        z2 = projection(f2[index].to(device))
        prediction = decoder(z0).reshape(-1)
        rmse = torch.sqrt(F.mse_loss(prediction, target[index].to(device)))
        invariance = 0.5 * (F.mse_loss(z0, z1) + F.mse_loss(z0, z2))
    return float(rmse), float(invariance)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("Aebs/data/Downsampled.h5"))
    parser.add_argument("--dino-repo", type=Path, default=Path("../external/dinov2"))
    parser.add_argument(
        "--dino-weights", type=Path,
        default=Path("../external/dinov2_weights/dinov2_vits14_pretrain.pth"),
    )
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/dino_safety_latent_stage1"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--q-epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=32)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output directory is nonempty; choose a new directory")
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.data, "r") as data_file:
        images = np.asarray(data_file["X_train"], dtype=np.float32)
        distance_m = np.asarray(data_file["y_train"], dtype=np.float32).reshape(-1)
    distance_scale = float(np.std(distance_m))
    target = torch.from_numpy((distance_m / distance_scale).astype(np.float32))
    indices = split_indices(len(images), args.seed)
    rng = np.random.default_rng(args.seed + 1)
    shuffled = rng.permutation(indices["train"])
    validation_count = max(1, int(round(0.2 * len(shuffled))))
    validation_idx = shuffled[:validation_count]
    optimization_idx = shuffled[validation_count:]

    config = {
        "experiment": "frozen_dino_safety_latent_stage1",
        "registered_before_training": True,
        "dino_model": "dinov2_vits14",
        "dino_frozen": True,
        "dino_weights_sha256": file_sha256(args.dino_weights),
        "feature_dimension": 384,
        "latent_dimension": args.latent_dim,
        "projection": "384 -> 128 -> 32",
        "deployment_semantics": "controller consumes 32-D latent plus speed; not a 1-D distance estimate",
        "auxiliary_losses": {
            "safety_sufficiency": "latent decoder reconstructs normalized distance during training only",
            "nuisance_invariance": "two brightness/contrast/noise views should share latent",
            "state_separation": "states separated by >=1m receive a latent-margin loss",
        },
        "physical_to_latent": "q_psi(normalized distance) -> 32-D latent",
        "dataset_limitation": (
            "Current data have distance labels but no repeated CARLA weather/camera-pose groups; "
            "appearance augmentations are only a first nuisance proxy"
        ),
        "sample_counts": {name: int(len(value)) for name, value in indices.items()},
        "optimization_count": int(len(optimization_idx)),
        "validation_count": int(len(validation_idx)),
        "seed": args.seed,
        "epochs": args.epochs,
        "q_epochs": args.q_epochs,
    }
    with open(args.output_dir / "config.json", "w", encoding="utf-8") as output_file:
        json.dump(config, output_file, indent=2, ensure_ascii=False)

    started = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = FrozenDinoV2(
        args.dino_repo, args.dino_weights, device, model_name="dinov2_vits14"
    )
    view1 = appearance_view(images, args.seed + 101)
    view2 = appearance_view(images, args.seed + 202)
    raw0 = extract(backbone, images, device, args.batch_size)
    raw1 = extract(backbone, view1, device, args.batch_size)
    raw2 = extract(backbone, view2, device, args.batch_size)
    feature_mean = raw0[optimization_idx].mean(dim=0)
    feature_std = raw0[optimization_idx].std(dim=0).clamp_min(1e-5)
    f0 = (raw0 - feature_mean) / feature_std
    f1 = (raw1 - feature_mean) / feature_std
    f2 = (raw2 - feature_mean) / feature_std

    projection = SafetyProjection(latent_dim=args.latent_dim).to(device)
    decoder = SafetyDecoder(latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.AdamW(
        list(projection.parameters()) + list(decoder.parameters()),
        lr=args.learning_rate, weight_decay=1e-4,
    )
    loader = DataLoader(
        TensorDataset(
            f0[optimization_idx], f1[optimization_idx], f2[optimization_idx],
            target[optimization_idx], torch.from_numpy(distance_m[optimization_idx]),
        ),
        batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    best_score = float("inf")
    best_epoch = 0
    best_projection = copy.deepcopy(projection.state_dict())
    best_decoder = copy.deepcopy(decoder.state_dict())
    history = []
    for epoch in range(1, args.epochs + 1):
        projection.train()
        decoder.train()
        running = []
        for batch_f0, batch_f1, batch_f2, batch_y, batch_distance in loader:
            batch_f0 = batch_f0.to(device)
            batch_f1 = batch_f1.to(device)
            batch_f2 = batch_f2.to(device)
            batch_y = batch_y.to(device)
            batch_distance = batch_distance.to(device)
            z0 = projection(batch_f0)
            z1 = projection(batch_f1)
            z2 = projection(batch_f2)
            safety = F.mse_loss(decoder(z0).reshape(-1), batch_y)
            invariance = 0.5 * (F.mse_loss(z0, z1) + F.mse_loss(z0, z2))
            permutation = torch.randperm(len(z0), device=device)
            far = (batch_distance - batch_distance[permutation]).abs() >= 1.0
            pair_distance = torch.linalg.vector_norm(
                z0 - z0[permutation], dim=1
            ) / np.sqrt(args.latent_dim)
            separation = (
                torch.relu(0.5 - pair_distance[far]).square().mean()
                if bool(far.any()) else z0.new_tensor(0.0)
            )
            regularization = z0.square().mean()
            loss = safety + 0.2 * invariance + 0.1 * separation + 1e-4 * regularization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(projection.parameters()) + list(decoder.parameters()), 5.0
            )
            optimizer.step()
            running.append((float(loss), float(safety), float(invariance), float(separation)))
        validation_rmse, validation_invariance = projection_validation(
            projection, decoder, f0, f1, f2, target, validation_idx, device
        )
        score = validation_rmse + 0.2 * np.sqrt(max(validation_invariance, 0.0))
        if score < best_score:
            best_score = score
            best_epoch = epoch
            best_projection = copy.deepcopy(projection.state_dict())
            best_decoder = copy.deepcopy(decoder.state_dict())
        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            mean_loss = np.asarray(running).mean(axis=0)
            row = {
                "epoch": epoch, "loss": float(mean_loss[0]),
                "safety_loss": float(mean_loss[1]),
                "invariance_loss": float(mean_loss[2]),
                "separation_loss": float(mean_loss[3]),
                "validation_rmse_norm": validation_rmse,
                "validation_invariance_mse": validation_invariance,
            }
            history.append(row)
            print("epoch=%d loss=%.6f val_rmse=%.6f val_inv=%.6f" % (
                epoch, row["loss"], validation_rmse, validation_invariance
            ), flush=True)
    projection.load_state_dict(best_projection)
    decoder.load_state_dict(best_decoder)
    projection.eval()
    decoder.eval()

    with torch.no_grad():
        all_latent = projection(f0.to(device)).detach().cpu()
    q_model = PhysicalToLatent(physical_dim=1, latent_dim=args.latent_dim).to(device)
    q_optimizer = torch.optim.AdamW(q_model.parameters(), lr=1e-3, weight_decay=1e-5)
    q_train_x = target[optimization_idx, None].to(device)
    q_train_y = all_latent[optimization_idx].to(device)
    q_val_x = target[validation_idx, None].to(device)
    q_val_y = all_latent[validation_idx].to(device)
    best_q = copy.deepcopy(q_model.state_dict())
    best_q_loss = float("inf")
    for epoch in range(1, args.q_epochs + 1):
        q_model.train()
        prediction = q_model(q_train_x)
        loss = F.mse_loss(prediction, q_train_y)
        q_optimizer.zero_grad()
        loss.backward()
        q_optimizer.step()
        q_model.eval()
        with torch.no_grad():
            val_loss = float(F.mse_loss(q_model(q_val_x), q_val_y))
        if val_loss < best_q_loss:
            best_q_loss = val_loss
            best_q = copy.deepcopy(q_model.state_dict())
    q_model.load_state_dict(best_q)
    q_model.eval()

    test_idx = indices["test"]
    calibration_idx = indices["calibration"]
    with torch.no_grad():
        test_latent = all_latent[test_idx].to(device)
        test_distance_prediction = decoder(test_latent).cpu().numpy().reshape(-1)
        test_z_view1 = projection(f1[test_idx].to(device))
        test_z_view2 = projection(f2[test_idx].to(device))
        test_invariance = float(0.5 * (
            F.mse_loss(test_latent, test_z_view1) + F.mse_loss(test_latent, test_z_view2)
        ))
        test_q = q_model(target[test_idx, None].to(device))
        calibration_q = q_model(target[calibration_idx, None].to(device))
        q_test_rmse = float(torch.sqrt(F.mse_loss(test_q, test_latent)))
        calibration_residual_norm = torch.linalg.vector_norm(
            all_latent[calibration_idx].to(device) - calibration_q, dim=1
        ).cpu().numpy()

    checkpoint = {
        "experiment": config["experiment"],
        "dino_model": "dinov2_vits14",
        "dino_repo": str(args.dino_repo),
        "dino_weights": str(args.dino_weights),
        "dino_weights_sha256": config["dino_weights_sha256"],
        "distance_scale_m": distance_scale,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "projection_state_dict": projection.cpu().state_dict(),
        "auxiliary_decoder_state_dict": decoder.cpu().state_dict(),
        "physical_to_latent_state_dict": q_model.cpu().state_dict(),
        "latent_dimension": args.latent_dim,
        "best_projection_epoch": best_epoch,
        "dino_frozen": True,
    }
    torch.save(checkpoint, args.output_dir / "dino_safety_latent.pt")
    np.savez_compressed(
        args.output_dir / "latent_data.npz",
        split_train=indices["train"], split_calibration=calibration_idx,
        split_test=test_idx, latent=all_latent.cpu().numpy(),
        calibration_residual_norm=calibration_residual_norm,
    )
    result = {
        **config,
        "device": str(device),
        "official_dino_parameters": int(sum(p.numel() for p in backbone.backbone.parameters())),
        "trained_projection_parameters": int(sum(p.numel() for p in projection.parameters())),
        "trained_auxiliary_decoder_parameters": int(sum(p.numel() for p in decoder.parameters())),
        "trained_q_parameters": int(sum(p.numel() for p in q_model.parameters())),
        "best_projection_epoch": best_epoch,
        "best_validation_score": best_score,
        "test_auxiliary_distance": regression_metrics(
            target[test_idx].numpy(), test_distance_prediction, distance_scale
        ),
        "test_nuisance_invariance_mse": test_invariance,
        "test_q_latent_rmse_per_coordinate": q_test_rmse,
        "calibration_residual_norm": {
            "count": int(len(calibration_residual_norm)),
            "mean": float(calibration_residual_norm.mean()),
            "median": float(np.median(calibration_residual_norm)),
            "max": float(calibration_residual_norm.max()),
        },
        "history": history,
        "runtime_seconds": time.time() - started,
        "next_gate": (
            "Train/evaluate the new [32-D latent, speed] controller only after "
            "checking safety sufficiency, invariance, and q_psi approximation"
        ),
    }
    with open(args.output_dir / "metrics.json", "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2, ensure_ascii=False, allow_nan=False)
    print("[Frozen DINOv2 -> 32-D safety latent]")
    print("DINO frozen parameters=%d; projection/decoder/q trainable=%d/%d/%d" % (
        result["official_dino_parameters"], result["trained_projection_parameters"],
        result["trained_auxiliary_decoder_parameters"], result["trained_q_parameters"],
    ))
    point = result["test_auxiliary_distance"]
    print("auxiliary safety readout only: MAE=%.6f m, RMSE=%.6f m" % (
        point["mae_m"], point["rmse_m"]
    ))
    print("latent invariance MSE=%.9f; q_psi latent RMSE/coordinate=%.9f" % (
        result["test_nuisance_invariance_mse"],
        result["test_q_latent_rmse_per_coordinate"],
    ))
    print("checkpoint: %s" % (args.output_dir / "dino_safety_latent.pt"))
    print("metrics: %s" % (args.output_dir / "metrics.json"))


if __name__ == "__main__":
    main()
