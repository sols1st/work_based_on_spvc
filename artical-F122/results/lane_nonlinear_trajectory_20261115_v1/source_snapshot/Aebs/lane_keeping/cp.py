"""Split-conformal calibration for the 2-D DINO lane observation.

The residual is a two-vector in normalized coordinates.  A single max-norm
quantile defines a joint box for both components; this is the box the SBC
verifier will range over.  Calibration scenes are disjoint from projection
training and from the final test scenes.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from . import core
from .data import load_dataset
from .perception import DINOProjection

def _load_projection(path):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    model = DINOProjection(saved['feature_dim'], saved['hidden'], saved['output_dim'])
    model.load_state_dict(saved['projection'])
    model.feature_mean.copy_(saved['feature_mean'])
    model.feature_std.copy_(saved['feature_std'])
    return model.eval()


def conformal_radius(values, alpha):
    values = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    n = len(values)
    if n < 1 or not 0 < alpha < 1 or not np.isfinite(alpha):
        raise ValueError('nonempty residuals and alpha in (0,1) required')
    if np.isnan(values).any() or (values < 0).any():
        raise ValueError('nonnegative residuals without NaN required')
    rank = int(np.ceil((n + 1) * (1 - alpha)))
    # Split CP adds an infinity atom. Clipping rank n+1 to n incorrectly
    # promises unattainable coverage when the calibration sample is too small.
    if rank > n:
        return float('inf'), rank
    return float(values[rank - 1]), rank


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--projection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--features', type=Path, help='optional cached features.pt')
    parser.add_argument('--alpha', type=float, default=.05)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f'choose a new output path: {args.output}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = load_dataset(args.data_dir)
    split = data['split']
    labels = np.stack([data['states'][:, 0] / core.E_SCALE,
                       data['states'][:, 1] / core.THETA_SCALE], axis=1)
    features_path = args.features or (args.projection.parent / 'features.pt')
    features = torch.load(features_path, map_location='cpu', weights_only=False)['features']
    model = _load_projection(args.projection)
    with torch.no_grad():
        prediction = model(features).numpy()
    residual = prediction - labels
    max_norm = np.abs(residual).max(1)
    calibration = split == 'calibration'
    test = split == 'test'
    if calibration.sum() < 10 or test.sum() < 5:
        raise ValueError('not enough calibration/test samples')
    radius, rank = conformal_radius(max_norm[calibration], args.alpha)
    coverage = {}
    for name, mask in (('calibration', calibration), ('test', test)):
        coverage[name] = dict(n=int(mask.sum()),
                              fraction_within=float(np.mean(max_norm[mask] <= radius)),
                              max_norm_quantile_95=float(np.quantile(max_norm[mask], .95)),
                              worst=float(max_norm[mask].max()))
    payload = dict(
        format='lane_keeping_cp_v1', alpha=args.alpha, n_calibration=int(calibration.sum()),
        rank=rank, radius_normalized=radius,
        error_model=dict(kind='global_box', radius=radius),
        physical=dict(e=radius * core.E_SCALE, theta=radius * core.THETA_SCALE),
        coverage=coverage,
        residual=dict(mean_e=float(residual[calibration, 0].mean()),
                      mean_theta=float(residual[calibration, 1].mean()),
                      std_e=float(residual[calibration, 0].std()),
                      std_theta=float(residual[calibration, 1].std())))
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
