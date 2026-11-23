"""Run the actual SPVT training loop in an isolated, byte-recorded source copy.

The fixed arm removes a redundant normalized-action -> degree conversion only.
This is a native-training budget diagnostic, not a DINO/framework benchmark.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--arm', choices=['released', 'units_fixed'], required=True)
    p.add_argument('--states', type=int, default=1024)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--seed', type=int, default=202611231)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    import numpy as np
    import torch
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    upstream = args.upstream.resolve()
    copy = out / 'source'
    copy.mkdir()
    for name in ['model', 'spvt']:
        shutil.copytree(upstream / name, copy / name,
                        ignore=shutil.ignore_patterns('__pycache__', 'logs', '.git', '*.pyc'))
    shutil.copy2(upstream / 'train.py', copy / 'train.py')
    originals = {str(f.relative_to(copy)): sha(f) for f in sorted(copy.rglob('*.py'))}
    loss_file = copy / 'spvt/loss.py'
    if args.arm == 'units_fixed':
        text = loss_file.read_text()
        for line in ["        true_control = convert_action(true_control, 'unnormalize')\n",
                     "        lb = convert_action(lb, 'unnormalize')\n",
                     "        ub = convert_action(ub, 'unnormalize')\n"]:
            if text.count(line) != 1:
                raise ValueError('unexpected upstream action conversion')
            text = text.replace(line, '')
        loss_file.write_text(text)
    data = torch.load(upstream / 'data/carla_states.pt', map_location='cpu')
    if not 32 <= args.states <= len(data):
        raise ValueError('invalid state budget')
    ids = np.random.default_rng(20261123001).permutation(len(data))[:args.states]
    (copy / 'data').mkdir()
    torch.save(data[ids], copy / 'data/carla_states.pt')
    np.save(out / 'training_state_indices.npy', ids)
    protocol = dict(arm=args.arm, seed=args.seed, state_budget=args.states,
        epochs=args.epochs, batch=args.batch, updates_per_epoch=int(np.ceil(args.states/args.batch)),
        device=args.device, upstream=str(upstream), original_sources=originals,
        executed_loss_sha256=sha(loss_file), dataset_sha256=sha(upstream/'data/carla_states.pt'),
        selected_indices_sha256=sha(out/'training_state_indices.npy'),
        wrapper_sha256=sha(__file__), bounds='backward (upstream default)',
        retained=['native train.train', 'bound-gap loss', 'original PID action target',
                  'adaptive loss weights', 'hard-example replay', 'K=4..10 curriculum',
                  'frozen native generator', 'native learning-rate schedule'],
        modification='remove three pre-dynamics degree conversions' if args.arm == 'units_fixed' else 'none',
        selection='fixed final epoch; no selection by test outcomes',
        scope='bounded-budget native-method repair diagnostic, not current-framework comparison',
        remaining_original_caveats=['Gaussian state replacement has unbounded support',
            'corner-based propagation needs monotonicity/no-angle-wrap checks',
            'BatchNorm training bounds are not automatically deployment bounds'])
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    os.chdir(copy)
    sys.path[:0] = [str(copy), str(copy/'spvt')]
    sys.argv = [str(copy/'train.py'), '--num_epochs', str(args.epochs),
                '--batch_size', str(args.batch), '--seed', str(args.seed),
                '--device', args.device, '--run_id', 'logs/paired',
                '--tb_path', 'logs/paired/runs', '--ckpt_path', 'logs/paired/ckpts']
    from spvt.args import get_args
    from train import train
    config = get_args()
    start = time.time()
    train(config)
    final = copy / f'spvt/logs/paired/ckpts/checkpoint_epoch_{args.epochs}.pth'
    if not final.exists():
        raise RuntimeError('native loop did not save the registered final epoch')
    shutil.copy2(final, out/'final.pth')
    (out/'complete.json').write_text(json.dumps(dict(elapsed_seconds=time.time()-start,
        checkpoint_sha256=sha(out/'final.pth'), epochs=args.epochs), indent=2)+'\n')


if __name__ == '__main__':
    main()
