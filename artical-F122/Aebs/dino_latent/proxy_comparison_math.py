"""Model-free paired metrics for the legacy/DINO development comparison."""
import numpy as np


def legacy_pixels(image):
    """Same operations as connect/downSample.py, without its file-writing imports."""
    gray = np.asarray(image.convert("L").resize((160, 160)), dtype=np.float64) / 255.0
    patches = gray.reshape(32, 5, 32, 5).transpose(0, 2, 1, 3).reshape(32, 32, 25)
    out = np.sort(patches, axis=-1)[..., -5:].mean(axis=-1)
    return np.clip(out-out.mean()+.5, 0, 1).astype(np.float32)


def next_speed(speed, action):
    speed, action = np.asarray(speed, dtype=np.float64), np.asarray(action, dtype=np.float64)
    return np.clip(speed - .05*np.clip(action, -3., 3.), 0., 3.)


def errors(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite error values")
    if not len(values):
        return {"count": 0}
    absolute = np.abs(values)
    return dict(count=len(values), bias=float(values.mean()), mae=float(absolute.mean()),
                rmse=float(np.sqrt(np.mean(values**2))), p95_abs=float(np.quantile(absolute, .95)),
                max_abs=float(absolute.max()))


def paired_summary(dino_error, legacy_error, groups):
    """Paired sample and distance-block summaries; no IID significance claim."""
    dino, legacy = np.asarray(dino_error, dtype=np.float64), np.asarray(legacy_error, dtype=np.float64)
    groups = np.asarray(groups)
    if dino.shape != legacy.shape or dino.shape[0] != len(groups):
        raise ValueError("paired errors must have matching shapes and row identities")
    dm, lm = errors(dino), errors(legacy)
    if not dino.size:
        return {"dino": dm, "legacy": lm, "status": "empty"}
    diff = np.abs(dino)-np.abs(legacy)
    by_group = {}
    for group in sorted(set(groups.tolist())):
        mask = groups == group
        by_group[str(group)] = float(diff[mask].mean())
    return {"dino": dm, "legacy": lm,
            "delta_mae_dino_minus_legacy": dm["mae"]-lm["mae"],
            "delta_rmse_dino_minus_legacy": dm["rmse"]-lm["rmse"],
            "fraction_dino_abs_error_lower": float((diff < -1e-12).mean()),
            "distance_block_macro_delta_mae": float(np.mean(list(by_group.values()))),
            "by_distance_block_delta_mae": by_group,
            "sampled_mae_and_rmse_lower": dm["mae"] < lm["mae"] and dm["rmse"] < lm["rmse"],
            "generalization_or_safety_claim": False}


def transition_summary(dino_delta, legacy_delta, valid, groups):
    """legacy_delta [images, draws, speeds]; use all draws, never best-of-K."""
    dino, legacy = np.asarray(dino_delta), np.asarray(legacy_delta)
    valid, groups = np.asarray(valid, dtype=bool), np.asarray(groups)
    if (dino.ndim != 2 or legacy.ndim != 3 or valid.shape != dino.shape
            or legacy.shape[0] != dino.shape[0] or legacy.shape[2] != dino.shape[1]
            or len(groups) != len(dino) or legacy.shape[1] < 1):
        raise ValueError("transition array shape mismatch")
    # Flatten in identical (image, speed, draw) order. Repeats are NOT independent observations.
    legacy_pairs = legacy.transpose(0, 2, 1)[valid]
    dino_pairs = np.broadcast_to(dino[valid, None], legacy_pairs.shape)
    group_pairs = np.broadcast_to(groups[:, None], valid.shape)[valid]
    result = paired_summary(dino_pairs, legacy_pairs, group_pairs)
    result.update(physical_state_pairs=int(valid.sum()), latent_draws=legacy.shape[1],
                  distance_successor_difference_m=0.0,
                  definition="proxy next speed minus its own image-path next speed, m/s")
    return result
