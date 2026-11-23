"""Paired one- vs two-hidden-layer proxies at nested data budgets."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .proxy_architecture_compare import dataset, run, sha, write

ARCHITECTURES = tuple(act + width for act in ('tanh', 'relu', 'silu') for width in ('16', '16x16'))
BUDGETS = {'aebs': (2160, 11338, 14172), 'lane': (2160, 6000, 12000)}


def prepare(task, root):
    x, y, C, scales, split, bindings = dataset(task)
    rng = np.random.default_rng(2026112204)
    if task == 'aebs':
        # Preserve the exact previous medium training set. Only validation is repartitioned.
        old_train = rng.permutation(np.flatnonzero(split == 'train'))
        old_dev = rng.permutation(np.flatnonzero(split == 'development'))
        order = np.concatenate([old_train, old_dev])
        original_val = rng.permutation(np.flatnonzero(split == 'calibration'))
        split[old_dev] = 'train'
        split[original_val[:len(original_val)//2]] = 'development'
        scope = 'Historical CARLA RGB bank, original test; fixed frozen vision and proxy-only data budgets.'
    else:
        path = root / 'lane_bank.npz'
        with np.load(path) as bank:
            x = bank['normalized_state'].astype('float32')
            y = bank['latent'].astype('float32')
            split = bank['split']
        manifest = json.loads((root / 'lane_bank.json').read_text())
        if sha(path) != manifest['bank_sha256']:
            raise ValueError('Lane collection binding mismatch')
        bindings[str(path)] = manifest['bank_sha256']
        order = rng.permutation(np.flatnonzero(split == 'train'))
        scope = 'New deduplicated nominal static render collection, fixed frozen vision, no trajectory guarantee.'
    if len(order) != max(BUDGETS[task]) or not np.isfinite(y).all():
        raise ValueError('Invalid observation bank')
    return x, y, C, scales, split, bindings, order, scope


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=list(BUDGETS))
    parser.add_argument('output', type=Path)
    parser.add_argument('--steps', type=int, default=1500)
    args = parser.parse_args()
    torch.set_num_threads(1)
    x, y, C, scales, split, bindings, order, scope = prepare(args.task, args.output)
    for budget in BUDGETS[args.task]:
        ids = np.sort(np.r_[order[:budget], np.flatnonzero(split != 'train')])
        dest = args.output / str(budget)
        run(argparse.Namespace(output=dest, steps=args.steps), args.task, ARCHITECTURES,
            (x[ids], y[ids], C, scales, split[ids], bindings), scope)
        np.save(dest / args.task / 'source_indices.npy', ids)
    write(args.output / (args.task + '_run.json'), dict(budgets=BUDGETS[args.task], driver_sha256=sha(__file__),
          torch=torch.__version__, numpy=np.__version__, task=args.task, scope=scope))


if __name__ == '__main__':
    main()
