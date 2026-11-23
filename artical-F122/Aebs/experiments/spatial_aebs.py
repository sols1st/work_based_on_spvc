"""Development-only selection of matched-pixel AEBS readouts.

The experiment keeps the original scalar distance observation and state proxy.
No calibration/test labels enter optimization or model selection.  Predictions
over the common image bank are emitted only after each final model is frozen.
"""
import argparse
import copy
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Aebs.dino_latent.backbone import FrozenDinoV2
from Aebs.vision_sls.model import ObservationProjection
from Aebs.vision_sls.run import sha256
from Combined_network.model import SubNet
from .matched_aebs_library import SCALE, write


BUDGETS = (400, 10000, 24000)
SEEDS = (27411, 27412, 27413)
EPOCHS, BATCH = 100, 64
ARMS = ('gray32_mlp', 'gray32_cls', 'gray32_spatial')
BASE_RATES = (3e-5, 1e-4, 3e-4, 1e-3)
SPATIAL_TRIALS = ((2, 1e-4), (2, 1e-3), (4, 1e-4), (4, 1e-3))


def read(path):
    return json.loads(Path(path).read_text())


class SpatialProjection(nn.Module):
    """Shared patch-channel readout retaining the fixed spatial pool order."""
    def __init__(self, grid):
        super().__init__()
        self.grid = int(grid)
        self.register_buffer('feature_mean', torch.zeros(1, 1, 384))
        self.register_buffer('feature_std', torch.ones(1, 1, 384))
        self.channel = nn.Sequential(nn.Linear(384, 32), nn.GELU())
        self.readout = nn.Sequential(nn.Linear(grid * grid * 32, 128),
                                     nn.GELU(), nn.Linear(128, 1))

    def forward(self, feature):
        return self.readout(self.channel((feature - self.feature_mean) /
                                        self.feature_std).flatten(1))


def make_model(arm, grid=None):
    if arm == 'gray32_mlp':
        return SubNet([1024, 256, 64, 1])
    if arm == 'gray32_cls':
        return ObservationProjection()
    if arm == 'gray32_spatial':
        return SpatialProjection(grid)
    raise ValueError(arm)


def register(args):
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), args.budgets / 'protocol.json',
               args.source / 'protocol.json', args.source / 'cache.json',
               args.source / 'raw_gray.npy', args.source / 'gray32.npy',
               args.source / 'ids.npy', args.weights,
               Path('Aebs/dino_latent/backbone.py'),
               Path('Aebs/vision_sls/model.py'), Path('Combined_network/model.py')]
    budget = read(args.budgets / 'protocol.json')
    allocation = {}
    master = set(np.load(args.source / 'ids.npy').tolist())
    for n in BUDGETS:
        rec = budget[str(n)]
        assert len(rec['files']) == n and len(set(rec['files'])) == n
        assert set(rec['files']).issubset(master)
        allocation[str(n)] = {s: rec['split'].count(s) for s in ('train', 'validation', 'test')}
        assert sum(allocation[str(n)].values()) == n
    protocol = dict(
        format='aebs_same_pixel_spatial_readout_v1', source=str(args.source),
        budgets_path=str(args.budgets), weights=str(args.weights), repo=str(args.repo),
        budgets=list(BUDGETS), training_seeds=list(SEEDS), arms=list(ARMS),
        allocation=allocation, epochs=EPOCHS, batch=BATCH,
        trials={a: ([dict(grid=g, lr=r) for g, r in SPATIAL_TRIALS] if a == 'gray32_spatial'
                    else [dict(grid=None, lr=r) for r in BASE_RATES]) for a in ARMS},
        initialization='same paired torch seed; each trial reinitializes, same seeded minibatch orders',
        selection='minimum validation MSE over exactly four trials and 100 epochs; no test metrics',
        observation='scalar y = distance_m / distance_scale_m; fixed state proxy C=[1,0]',
        distance_scale_m=SCALE, loss='mean squared normalized-distance error',
        input='same audited raw_gray.npy 32x32 bytes; MLP rescales to [-1,1]; DINO repeats channels and CPU bicubic224/ImageNet preprocessing',
        spatial='frozen ViT-S/14 normalized patch tokens; adaptive average pool2x2 or4x4; shared384->32 GELU; flatten->128 GELU->1',
        normalization='DINO statistics from training rows only; spatial stats pooled across train images and grid locations; std floor .01',
        additional_prior='Both DINO arms use official pretrained DINOv2; pixel MLP has no external pretraining',
        extraction=dict(device=args.device, batch=args.extract_batch, memory_fraction=args.memory_fraction,
                        dtype='float32', cpu_preprocess=True, no_parameter_updates=True),
        frozen_before='patch feature extraction and all new head training; historical test results never used to choose any grid/rate here',
        calibration_and_test='not performed by this script; downstream evaluation must bind the frozen checkpoint identity',
        torch=str(torch.__version__), bindings={str(p): sha256(p) for p in sources})
    write(args.output / 'protocol.json', protocol)
    print(json.dumps({'registered': str(args.output), 'allocation': allocation, 'head_training_runs': 108}), flush=True)


def validate_protocol(output):
    p = read(output / 'protocol.json')
    for filename, expected in p['bindings'].items():
        if sha256(Path(filename)) != expected:
            raise ValueError('registered input/code changed: ' + filename)
    return p


def extract(args):
    p = validate_protocol(args.output)
    if (args.output / 'cache.json').exists():
        raise FileExistsError('feature cache already completed')
    torch.set_num_threads(2)
    device = p['extraction']['device']
    if device.startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(p['extraction']['memory_fraction'], torch.device(device))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    source = Path(p['source'])
    raw = np.load(source / 'raw_gray.npy', mmap_mode='r')
    original_cls = np.load(source / 'gray32.npy', mmap_mode='r')
    assert raw.shape == (24000, 1024) and raw.dtype == np.uint8
    pools = {g: np.lib.format.open_memmap(args.output / ('pool%d.npy' % g), mode='w+',
                                        dtype=np.float32, shape=(len(raw), g * g, 384)) for g in (2, 4)}
    cls = np.lib.format.open_memmap(args.output / 'cls.npy', mode='w+', dtype=np.float32,
                                   shape=(len(raw), 384))
    encoder = FrozenDinoV2(p['repo'], p['weights'], device)
    start = time.perf_counter()
    max_cls_difference, squared_difference = 0., 0.
    batch_size = p['extraction']['batch']
    with torch.inference_mode():
        for i in range(0, len(raw), batch_size):
            gray = torch.from_numpy(np.array(raw[i:i + batch_size])).float().reshape(-1, 1, 32, 32).div_(255.)
            inputs = encoder.preprocess(gray).to(device)
            features = encoder.backbone.forward_features(inputs)
            value = features['x_norm_clstoken'].cpu().numpy()
            patches = features['x_norm_patchtokens']
            assert patches.shape[1:] == (256, 384)
            spatial = patches.transpose(1, 2).reshape(-1, 384, 16, 16)
            for grid in (2, 4):
                pooled = F.adaptive_avg_pool2d(spatial, grid).flatten(2).transpose(1, 2)
                pools[grid][i:i + len(value)] = pooled.cpu().numpy()
            cls[i:i + len(value)] = value
            delta = value.astype(np.float64) - original_cls[i:i + len(value)]
            max_cls_difference = max(max_cls_difference, float(np.abs(delta).max()))
            squared_difference += float(np.square(delta).sum())
            if i % (batch_size * 100) == 0:
                print('patch_encoded', i, len(raw), round(time.perf_counter() - start, 2), flush=True)
    for values in [cls, *pools.values()]:
        values.flush()
    # This numerical check uses features only, never a held-out distance label.
    if max_cls_difference > 2e-4:
        raise ValueError('CLS differs from the same-pixel source by %.6g' % max_cls_difference)
    write(args.output / 'cache.json', dict(
        files={f.name: sha256(f) for f in args.output.glob('*.npy')},
        protocol_sha256=sha256(args.output / 'protocol.json'),
        seconds=time.perf_counter() - start, images=len(raw), dino_forwards=math.ceil(len(raw) / batch_size),
        cls_consistency=dict(max_absolute=max_cls_difference, rms=math.sqrt(squared_difference / original_cls.size),
                             tolerance=2e-4, reference_sha256=sha256(source / 'gray32.npy')),
        extraction=p['extraction'], torch=str(torch.__version__),
        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(torch.device(device)) if device.startswith('cuda') else None))
    print('patch_cache_completed', round(time.perf_counter() - start, 2), flush=True)


def model_data(source, output, arm, grid):
    filename = source / 'raw_gray.npy' if arm == 'gray32_mlp' else output / ('cls.npy' if arm == 'gray32_cls' else 'pool%d.npy' % grid)
    return np.load(filename, mmap_mode='r')


def tensor_rows(data, rows, arm):
    x = torch.from_numpy(np.array(data[rows])).float()
    if arm == 'gray32_mlp':
        x.mul_(2 / 255.).sub_(1.)
    return x


def fit(spec):
    output, arm, n, seed = spec
    torch.set_num_threads(1)
    p = read(output / 'protocol.json')
    source = Path(p['source'])
    folder = output / 'models' / ('%s_%d_%d' % (arm, n, seed))
    if (folder / 'fit.json').exists():
        return read(folder / 'fit.json')
    folder.mkdir(parents=True, exist_ok=True)
    rec = read(Path(p['budgets_path']) / 'protocol.json')[str(n)]
    master = {name: i for i, name in enumerate(np.load(source / 'ids.npy').tolist())}
    rows = np.array([master[name] for name in rec['files']])
    split = np.asarray(rec['split'])
    train = np.flatnonzero(split == 'train')
    dev = np.flatnonzero(split == 'validation')
    # Explicitly index labels before converting them to trainable tensors.
    distances = np.asarray(rec['distances'])
    y_train = torch.tensor(distances[train] / SCALE, dtype=torch.float32).reshape(-1, 1)
    y_dev = torch.tensor(distances[dev] / SCALE, dtype=torch.float32).reshape(-1, 1)
    start = time.perf_counter()
    trials, winner, winner_info = [], None, None
    cached_grid, x_train, x_dev = 'unset', None, None
    for cfg in p['trials'][arm]:
        grid, rate = cfg['grid'], cfg['lr']
        if cached_grid != grid:
            data = model_data(source, output, arm, grid)
            x_train = tensor_rows(data, rows[train], arm)
            x_dev = tensor_rows(data, rows[dev], arm)
            cached_grid = grid
        torch.manual_seed(seed)
        model = make_model(arm, grid)
        if arm != 'gray32_mlp':
            dimensions = (0, 1) if arm == 'gray32_spatial' else (0,)
            model.feature_mean.copy_(x_train.mean(dim=dimensions, keepdim=True))
            model.feature_std.copy_(x_train.std(dim=dimensions, keepdim=True).clamp_min(.01))
        order = torch.Generator().manual_seed(seed + 1100)
        optimizer = torch.optim.Adam(model.parameters(), lr=rate)
        best_loss, best_epoch = float('inf'), None
        trial_start = time.perf_counter()
        for epoch in range(1, EPOCHS + 1):
            model.train()
            for batch in torch.randperm(len(train), generator=order).split(BATCH):
                loss = (model(x_train[batch]) - y_train[batch]).square().mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.inference_mode():
                # Fixed batch evaluation avoids large temporary patch activations.
                total = 0.
                for j in range(0, len(dev), 256):
                    total += float((model(x_dev[j:j + 256]) - y_dev[j:j + 256]).square().sum())
                dev_mse = total / len(dev)
            if dev_mse < best_loss:
                best_loss, best_epoch = dev_mse, epoch
                if winner_info is None or dev_mse < winner_info['dev_mse']:
                    winner = copy.deepcopy(model.state_dict())
                    winner_info = dict(grid=grid, lr=rate, epoch=epoch, dev_mse=dev_mse)
        trials.append(dict(grid=grid, lr=rate, epoch=best_epoch, dev_mse=best_loss,
                           gradients=EPOCHS * math.ceil(len(train) / BATCH),
                           parameters=sum(v.numel() for v in model.parameters()),
                           training_seconds=time.perf_counter() - trial_start))
        print('trial_done', arm, n, seed, trials[-1], flush=True)
    model = make_model(arm, winner_info['grid'])
    model.load_state_dict(winner)
    model.eval()
    torch.save(winner, folder / 'readout.pt')
    # Checkpoint and selection are committed before any blind-image prediction.
    write(folder / 'frozen.json', dict(arm=arm, budget=n, seed=seed, rank_selection=winner_info,
          model_sha256=sha256(folder / 'readout.pt'), trials=trials,
          protocol_sha256=sha256(output / 'protocol.json'), test_metrics_accessed=False))
    training_seconds = time.perf_counter() - start
    data = model_data(source, output, arm, winner_info['grid'])
    prediction = np.lib.format.open_memmap(folder / 'prediction.npy', mode='w+', dtype=np.float32, shape=(len(data),))
    with torch.inference_mode():
        for i in range(0, len(data), 256):
            prediction[i:i + 256] = model(tensor_rows(data, slice(i, i + 256), arm)).flatten().numpy()
    prediction.flush()
    result = dict(mode=arm, budget=n, seed=seed, rank_selection=winner_info, all_trials=trials,
                  epochs=EPOCHS, batch=BATCH, n_train=len(train), n_dev=len(dev), test=int((split == 'test').sum()),
                  gradients=sum(t['gradients'] for t in trials), training_seconds=training_seconds,
                  total_seconds=time.perf_counter() - start, head_params=sum(v.numel() for v in model.parameters()),
                  prediction=sha256(folder / 'prediction.npy'), model=sha256(folder / 'readout.pt'),
                  frozen_sha256=sha256(folder / 'frozen.json'), cache_sha256=sha256(output / 'cache.json'),
                  protocol_sha256=sha256(output / 'protocol.json'), test_metrics_accessed=False,
                  distance_scale_m=SCALE)
    write(folder / 'fit.json', result)
    print('fit_done', folder.name, winner_info, 'seconds', round(training_seconds, 2), flush=True)
    return result


def train(args):
    p = validate_protocol(args.output)
    cache = read(args.output / 'cache.json')
    assert cache['protocol_sha256'] == sha256(args.output / 'protocol.json')
    for name, expected in cache['files'].items():
        if sha256(args.output / name) != expected:
            raise ValueError('cache changed: ' + name)
    jobs = [(args.output, arm, n, s) for n in args.train_budgets for s in SEEDS for arm in ARMS]
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(fit, jobs))
    write(args.output / ('train_summary_%s.json' % '_'.join(map(str, args.train_budgets))),
          dict(models=len(jobs), workers=args.workers, seconds=time.perf_counter() - start,
               protocol_sha256=sha256(args.output / 'protocol.json'), test_metrics_accessed=False))
    manifest = []
    for n in BUDGETS:
        for s in SEEDS:
            for arm in ARMS:
                key = '%s_%d_%d' % (arm, n, s)
                folder = args.output / 'models' / key
                if not (folder / 'fit.json').exists():
                    return
                fit_info = read(folder / 'fit.json')
                manifest.append(dict(key=key, prediction_path=str(folder / 'prediction.npy'),
                    ids_path=str(Path(p['source']) / 'ids.npy'), prediction_units='distance_over_scale',
                    model_path=str(folder / 'readout.pt'), fit_path=str(folder / 'fit.json'),
                    budget=n, train_seed=s, mode=arm, distance_scale_m=SCALE,
                    prediction_sha256=fit_info['prediction'], model_sha256=fit_info['model']))
    write(args.output / 'models_manifest.json', dict(models=manifest,
          protocol_sha256=sha256(args.output / 'protocol.json'), test_metrics_accessed=False,
          paths_relative_to='artical-F122 working directory'))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('phase', choices=['register', 'extract', 'train'])
    cli.add_argument('output', type=Path)
    cli.add_argument('--source', type=Path, default=Path('results/aebs_factorial_20261117'))
    cli.add_argument('--budgets', type=Path, default=Path('results/aebs_budget_scaling_20261110'))
    cli.add_argument('--repo', type=Path, default=Path('../external/dinov2'))
    cli.add_argument('--weights', type=Path, default=Path('../external/dinov2_weights/dinov2_vits14_pretrain.pth'))
    cli.add_argument('--device', default='cuda:1')
    cli.add_argument('--extract-batch', type=int, default=8)
    cli.add_argument('--memory-fraction', type=float, default=.035)
    cli.add_argument('--workers', type=int, default=6)
    cli.add_argument('--train-budgets', type=int, nargs='+', default=list(BUDGETS), choices=BUDGETS)
    args = cli.parse_args()
    {'register': register, 'extract': extract, 'train': train}[args.phase](args)


if __name__ == '__main__':
    main()
