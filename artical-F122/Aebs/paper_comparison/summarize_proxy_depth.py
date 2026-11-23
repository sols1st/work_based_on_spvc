"""Audit all architecture results and summarize their paired held-out errors."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .proxy_architecture_compare import Proxy, SEEDS, write
from .proxy_depth_data import ARCHITECTURES, BUDGETS


def summarize(root, task, budgets=None, metric_scale=None):
    archs = ('fixed_Cx', 'fitted_Cx', 'affine', *ARCHITECTURES)
    summary, previous = {}, set()
    reference = None
    for n in (BUDGETS[task] if budgets is None else budgets):
        path = root / str(n) / task
        inp = np.load(path/'inputs.npz')
        idx = np.load(path/'source_indices.npy')
        split, x, y, C = (inp[k] for k in ('split', 'x', 'visual', 'C'))
        test = split == 'test'
        va = split == 'development'
        train = set(idx[split == 'train'])
        assert len(train) == n and previous <= train
        if reference is None:
            reference = idx[split != 'train']
        np.testing.assert_array_equal(reference, idx[split != 'train'])
        previous = train
        rows = json.loads((path/'results.json').read_text())
        assert len(rows) == 21
        base = {}
        for seed in SEEDS:
            hashes = {r['extra']['batch_sha256'] for r in rows if r['seed'] == seed}
            assert len(hashes) == 1
        for arch in archs:
            selected = [r for r in rows if r['architecture'] == arch]
            assert [r['seed'] for r in selected] == (list(SEEDS) if arch in ARCHITECTURES else [None])
            metrics, norms, dev, p95, steps = [], [], [], [], []
            for row in selected:
                key = arch if row['seed'] is None else f"{arch}_{row['seed']}"
                p = np.load(path/(key+'.npz'))['proxy']
                if row['seed'] is not None:
                    model = Proxy(C, arch)
                    model.load_state_dict(torch.load(path/(key+'.pt'), map_location='cpu', weights_only=False)['state_dict'])
                    with torch.no_grad():
                        np.testing.assert_allclose(model(torch.from_numpy(x)).numpy(), p, atol=3e-6)
                    steps.append(row['extra']['selected_step'])
                norm = (y-p) @ np.linalg.inv(C).T
                e = abs(norm[test] * inp['physical_scales'])
                mae = e.mean(0)
                np.testing.assert_allclose(mae, row['metrics']['test']['mae_physical'], rtol=1e-5)
                np.testing.assert_allclose(np.mean(norm[va]**2), row['development_mse'], rtol=2e-5)
                conversion = metric_scale if metric_scale is not None else ([100.] if task == 'aebs' else [100., 180./np.pi])
                metrics.append(mae * conversion)
                p95.append(np.quantile(e,.95,axis=0) * conversion)
                dev.append(row['development_mse'])
                norms.append(float(np.sqrt((norm[test]**2).mean())))
            base[arch] = dict(mae=np.mean(metrics,axis=0).tolist(), mae_by_seed=np.array(metrics).tolist(),
                              p95=np.mean(p95,axis=0).tolist(), normalized_rmse=float(np.mean(norms)),
                              development_mse=float(np.mean(dev)), selected_steps=steps)
        rank = sorted(archs, key=lambda k:base[k]['development_mse'])
        assert rank == json.loads((path/'selection.json').read_text())['development_rank']
        summary[n] = dict(architectures=base, selected=rank[0], selected_nn=next(k for k in rank if k in ARCHITECTURES))
    return summary


def paired_lane_gain(root):
    path = root/'12000'/'lane'
    test = np.load(path/'inputs.npz')['split'] == 'test'

    def absolute_errors(arch):
        keys = [arch] if arch == 'fixed_Cx' else [f'{arch}_{s}' for s in SEEDS]
        return np.stack([abs(np.load(path/(k+'.npz'))['residual_physical'][test])
                         * [100, 180/np.pi] for k in keys])

    candidate = absolute_errors('relu16x16')
    records = {}
    for base in ('fixed_Cx', 'relu16'):
        gain = absolute_errors(base) - candidate
        per_frame = gain.mean(axis=0)
        rng = np.random.default_rng(2026112205)
        draws = np.concatenate([per_frame[rng.integers(len(per_frame), size=(100, len(per_frame)))].mean(1)
                                for _ in range(50)])
        records[base] = dict(mean=per_frame.mean(0).tolist(), ci95_percentile=np.percentile(draws, [2.5,97.5], axis=0).tolist(),
                             paired_seed_gains=gain.mean(1).tolist())
    write(root/'lane_paired_test.json', dict(candidate='relu16x16', budget=12000, n=int(test.sum()),
          units=['cm','degree'], paired_resample='resample static frames; average over three fixed seeds; baseline minus candidate',
          resamples=5000, seed=2026112205, results=records))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if (args.root/'lane_bank.npz').exists():
        with np.load(args.root/'lane_bank.npz') as z:
            assert len(set(z['sha256'])) == len(z['sha256']) == 16000
    write(args.root/'summary.json', {task:summarize(args.root, task) for task in BUDGETS})
    paired_lane_gain(args.root)
