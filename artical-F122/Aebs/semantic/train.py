"""Train and evaluate the certified distance abstraction used by Semantic-DR-SafePVC."""

import argparse
import copy
import json
import random
import time
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from torch.utils.data import DataLoader, TensorDataset

from Aebs.semantic.conformal import (
    MondrianSplitConformalRegressor,
    SplitConformalRegressor,
    StateConditionalErrorContract,
)
from Aebs.semantic.model import SemanticEncoder
from Combined_network.model import SubNet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_indices(n: int, seed: int, train_fraction: float = 0.6, calibration_fraction: float = 0.2):
    if n < 20:
        raise ValueError("at least 20 samples are required")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_train = int(round(n * train_fraction))
    n_calibration = int(round(n * calibration_fraction))
    return {
        "train": order[:n_train],
        "calibration": order[n_train : n_train + n_calibration],
        "test": order[n_train + n_calibration :],
    }


def predict(model: SemanticEncoder, x: np.ndarray, device: torch.device, batch_size: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch_size, shuffle=False)
    means, scales = [], []
    model.eval()
    with torch.no_grad():
        for (images,) in loader:
            mean, scale = model(images.to(device))
            means.append(mean.cpu().numpy().reshape(-1))
            scales.append(scale.cpu().numpy().reshape(-1))
    return np.concatenate(means), np.concatenate(scales)


def regression_metrics(y: np.ndarray, prediction: np.ndarray, distance_scale: float) -> Dict[str, float]:
    error_m = (prediction - y) * distance_scale
    return {
        "mae_m": float(np.mean(np.abs(error_m))),
        "rmse_m": float(np.sqrt(np.mean(np.square(error_m)))),
        "max_abs_error_m": float(np.max(np.abs(error_m))),
    }


def conformal_metrics(y, lower, upper, distance_scale) -> Dict[str, float]:
    return {
        "test_coverage": float(np.mean((y >= lower) & (y <= upper))),
        "mean_width_m": float(np.mean(upper - lower) * distance_scale),
        "median_width_m": float(np.median(upper - lower) * distance_scale),
        "max_width_m": float(np.max(upper - lower) * distance_scale),
    }


def control_sufficiency_metrics(
    true_distance: np.ndarray,
    predicted_distance: np.ndarray,
    controller_path: str,
    seed: int,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed + 101)
    speed = rng.uniform(0.0, 3.0, size=len(true_distance)).astype(np.float32)
    true_obs = np.stack((true_distance, speed), axis=1).astype(np.float32)
    semantic_obs = np.stack((predicted_distance, speed), axis=1).astype(np.float32)
    model = PPO.load(controller_path, device="cpu")
    policy = model.policy
    policy.eval()
    with torch.no_grad():
        true_tensor = torch.from_numpy(true_obs)
        semantic_tensor = torch.from_numpy(semantic_obs)
        true_action = policy.action_net(policy.mlp_extractor.policy_net(true_tensor)).numpy().reshape(-1)
        semantic_action = policy.action_net(policy.mlp_extractor.policy_net(semantic_tensor)).numpy().reshape(-1)
    action_error = semantic_action - true_action
    return {
        "action_mae": float(np.mean(np.abs(action_error))),
        "action_rmse": float(np.sqrt(np.mean(np.square(action_error)))),
        "action_max_abs_error": float(np.max(np.abs(action_error))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="Aebs/data/Downsampled.h5")
    parser.add_argument("--legacy-model", default="Aebs/controller/state_net_trained.pth")
    parser.add_argument("--controller", default="Aebs/controller/best_model/best_model.zip")
    parser.add_argument("--output-dir", default="results/semantic_stage1")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.data, "r") as data_file:
        images = np.asarray(data_file["X_train"], dtype=np.float32)
        distance_m = np.asarray(data_file["y_train"], dtype=np.float32).reshape(-1)
    distance_scale = float(np.std(distance_m))
    targets = distance_m / distance_scale
    indices = split_indices(len(images), args.seed)
    # Keep calibration/test untouched; reserve 20% of the training portion for early stopping.
    rng = np.random.default_rng(args.seed + 1)
    shuffled_train = rng.permutation(indices["train"])
    validation_size = max(1, int(round(0.2 * len(shuffled_train))))
    validation_idx = shuffled_train[:validation_size]
    optimization_idx = shuffled_train[validation_size:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SemanticEncoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(images[optimization_idx]), torch.from_numpy(targets[optimization_idx, None])),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_validation = float("inf")
    best_epoch = 0
    stale_epochs = 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_images, batch_targets in loader:
            batch_images = batch_images.to(device)
            batch_targets = batch_targets.to(device)
            mean, scale = model(batch_images)
            standardized = (batch_targets - mean) / scale
            # Gaussian NLL learns relative uncertainty; MSE keeps point predictions sharp.
            loss = (0.5 * standardized.square() + torch.log(scale)).mean() + 0.25 * F.mse_loss(mean, batch_targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        validation_mean, _ = predict(model, images[validation_idx], device)
        validation_rmse = float(np.sqrt(np.mean(np.square(validation_mean - targets[validation_idx]))))
        if validation_rmse < best_validation - 1e-6:
            best_validation = validation_rmse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch == 1 or epoch % 25 == 0:
            print(f"epoch={epoch} validation_rmse_norm={validation_rmse:.6f} best_epoch={best_epoch}", flush=True)
        if stale_epochs >= args.patience:
            break
    model.load_state_dict(best_state)

    calibration_mean, calibration_scale = predict(model, images[indices["calibration"]], device)
    calibrator = SplitConformalRegressor.fit(
        targets[indices["calibration"]], calibration_mean, calibration_scale, alpha=args.alpha
    )
    mondrian_calibrator = MondrianSplitConformalRegressor.fit(
        targets[indices["calibration"]], calibration_mean, calibration_scale, alpha=args.alpha, n_bins=4
    )
    state_contract = StateConditionalErrorContract.fit(
        targets[indices["calibration"]],
        calibration_mean,
        distance_m[indices["calibration"]],
        bin_edges=np.linspace(float(distance_m.min()), float(distance_m.max()), 5),
        alpha=args.alpha,
    )
    test_mean, test_scale = predict(model, images[indices["test"]], device)
    test_lower, test_upper = calibrator.interval(test_mean, test_scale)
    mondrian_lower, mondrian_upper = mondrian_calibrator.interval(test_mean, test_scale)
    test_y = targets[indices["test"]]
    state_lower, state_upper = state_contract.estimate_interval(
        test_y, distance_m[indices["test"]]
    )

    legacy = SubNet([1024, 256, 64, 1]).to(device)
    legacy.load_state_dict(torch.load(args.legacy_model, map_location=device))
    legacy.eval()
    with torch.no_grad():
        legacy_prediction = legacy(torch.from_numpy(images[indices["test"]]).to(device).reshape(len(test_y), -1)).cpu().numpy().reshape(-1)

    per_bin = []
    test_distance_m = distance_m[indices["test"]]
    bin_edges = np.quantile(test_distance_m, np.linspace(0.0, 1.0, 5))
    for bin_index in range(4):
        right_closed = bin_index == 3
        mask = (test_distance_m >= bin_edges[bin_index]) & (
            test_distance_m <= bin_edges[bin_index + 1] if right_closed else test_distance_m < bin_edges[bin_index + 1]
        )
        per_bin.append({
            "distance_low_m": float(bin_edges[bin_index]),
            "distance_high_m": float(bin_edges[bin_index + 1]),
            "count": int(mask.sum()),
            "coverage": float(np.mean((test_y[mask] >= test_lower[mask]) & (test_y[mask] <= test_upper[mask]))),
            "mean_width_m": float(np.mean(test_upper[mask] - test_lower[mask]) * distance_scale),
        })

    result = {
        "experiment": "semantic_stage1",
        "seed": args.seed,
        "alpha": args.alpha,
        "device": str(device),
        "sample_counts": {name: int(len(value)) for name, value in indices.items()},
        "optimization_count": int(len(optimization_idx)),
        "validation_count": int(len(validation_idx)),
        "distance_scale_m": distance_scale,
        "best_epoch": best_epoch,
        "runtime_seconds": time.time() - started,
        "legacy_real_image": regression_metrics(test_y, legacy_prediction, distance_scale),
        "semantic_point": regression_metrics(test_y, test_mean, distance_scale),
        "conformal_global": {
            **calibrator.to_dict(),
            "target_coverage": 1.0 - args.alpha,
            **conformal_metrics(test_y, test_lower, test_upper, distance_scale),
            "distance_bins": per_bin,
        },
        "conformal_mondrian": {
            **mondrian_calibrator.to_dict(),
            "target_coverage": 1.0 - args.alpha,
            **conformal_metrics(test_y, mondrian_lower, mondrian_upper, distance_scale),
        },
        "state_conditional_contract": {
            **state_contract.to_dict(),
            "units": "normalized_distance",
            "radii_m": [float(radius * distance_scale) for radius in state_contract.radii],
            "target_coverage": 1.0 - args.alpha,
            **conformal_metrics(test_mean, state_lower, state_upper, distance_scale),
        },
        "control_sufficiency": control_sufficiency_metrics(
            test_y, test_mean, args.controller, args.seed
        ),
    }

    checkpoint = {
        "model_state_dict": model.cpu().state_dict(),
        "distance_scale_m": distance_scale,
        "calibrator": calibrator.to_dict(),
        "mondrian_calibrator": mondrian_calibrator.to_dict(),
        "state_contract": state_contract.to_dict(),
        "seed": args.seed,
    }
    torch.save(checkpoint, output_dir / "semantic_encoder.pt")
    np.savez(
        output_dir / "split_indices.npz",
        train=indices["train"],
        calibration=indices["calibration"],
        test=indices["test"],
    )
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as result_file:
        json.dump(result, result_file, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
