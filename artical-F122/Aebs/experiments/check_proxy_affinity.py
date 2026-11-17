"""Measure the effect of removing learned curvature on the same camera readouts."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .compare_high_dim_proxies import TASKS
from ..curve_braking import plant as brake
from ..curve_braking.models import Proxy
from ..overhead_crane import plant as crane
from ..overhead_crane_proxy.train import ProxyObservation


def training_states(results, task):
    if task == 'braking':
        folder, label = 'curve_braking_5d_20261117', 'states'
    elif task == 'hover':
        folder, label = 'visual_hover_6d_20261117_v1', 'position'
    else:
        folder, label = 'overhead_crane_8d_20261117_v1', 'labels'
    with np.load(results / folder / 'static.npz') as data:
        z = data[label][data['split'] == 'train']
    return brake.normalize(z) if task == 'braking' else z


def load_model(arm, task):
    filename = {'braking': 'proxy.pt', 'hover': 'model.pt', 'crane': 'frontend.pt'}[task]
    weights = torch.load(arm / filename, map_location='cpu', weights_only=True)
    model = ProxyObservation('neural') if task == 'crane' else Proxy('neural', weights['head.weight'].shape[1])
    model.load_state_dict(weights)
    return model.eval()


def positions(task, states):
    if task == 'crane':
        return crane.observation(states)
    if task == 'braking':
        return brake.normalize(states)
    return states[..., :3]


@torch.inference_mode()
def analyze(results, task):
    root, seeds = TASKS[task]
    train = training_states(results, task)
    design = np.column_stack([train, np.ones(len(train))])
    stats = {}
    for seed in seeds:
        arm = results / root / (f'neural_{seed}' if task == 'crane' else f'{seed}_neural')
        model = load_model(arm, task)
        y_train = model.nominal(torch.tensor(train, dtype=torch.float32)).numpy().astype(float)
        affine = np.linalg.lstsq(design, y_train, rcond=None)[0]
        linear = np.linalg.lstsq(train, y_train, rcond=None)[0]
        record = {}
        with np.load(arm / 'holdout.npz') as data:
            z = positions(task, data['states'][:, :-1]).reshape(-1, train.shape[1])
            z_hat = z + data['errors'].reshape(z.shape)
            if task == 'braking':
                y = (model.nominal(torch.tensor(z, dtype=torch.float32)).numpy()
                     + data['raw_latent_errors'].reshape(z.shape))
            else:
                y = data['latent'].reshape(z.shape)
            check = model.decode(torch.tensor(y, dtype=torch.float32)).numpy()
            np.testing.assert_allclose(z_hat, check, atol=2e-6, rtol=1e-6)
            for mode, matrix, offset in [('through_origin', linear, 0.), ('affine', affine[:-1], affine[-1])]:
                z_counterfactual = np.linalg.solve(matrix.T, (y - offset).T).T
                delta = z_counterfactual - z_hat
                record[mode] = dict(normalized_rmse=float(np.sqrt(np.mean(delta**2))),
                                    normalized_max_abs=float(np.abs(delta).max()),
                                    condition=float(np.linalg.cond(matrix)))
        stats[str(seed)] = record
    return stats


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--results', type=Path, default=Path('results'))
    results = cli.parse_args().results
    torch.set_num_threads(2)
    stats = {task: analyze(results, task) for task in TASKS}
    target = results / 'high_dim_proxy_comparison_20261117' / 'same_image_affine_diagnostic.json'
    target.write_text(json.dumps(stats, indent=2, allow_nan=False) + '\n')
