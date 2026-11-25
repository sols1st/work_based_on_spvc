"""Prepare one distance-stratified CARLA split for both full AEBS methods."""
import argparse
import csv
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

SEED = 2026112501


def image_row(path):
    with Image.open(path) as source:
        rgb = source.convert('RGB')
        gray = rgb.convert('L').resize((32, 32), Image.Resampling.BICUBIC)
        return hashlib.sha256(rgb.tobytes()).hexdigest(), np.asarray(gray)


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def prepare(source, output, upstream, workers):
    output.mkdir(parents=True, exist_ok=True)
    metadata = source / 'samples.jsonl'
    rows = [json.loads(line) for line in metadata.read_text().splitlines()]
    files = np.array([row['filename'] for row in rows])
    distance = np.array([row['distance_m'] for row in rows])
    keys = [(row['weather'], row['color']) for row in rows]
    appearances = sorted(set(keys))
    group = np.array([appearances.index(key) for key in keys])
    grid, inverse = np.unique(distance, return_inverse=True)
    rng = np.random.default_rng(SEED)
    split_grid = np.full(len(grid), 'train', dtype='U11')
    for block in np.array_split(np.arange(1, len(grid) - 1), 100):
        shuffled = rng.permutation(block)
        split_grid[shuffled[:2]] = 'development'
        split_grid[shuffled[2:8]] = 'holdout'
    split = split_grid[inverse]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        hashes, gray = zip(*pool.map(image_row, (source / f for f in files), chunksize=64))
    if len(set(hashes)) != len(rows):
        raise ValueError('RGB pixel duplicates require assigning duplicate groups to one split')
    gray = np.stack(gray)
    train = split == 'train'
    holdout = split == 'holdout'
    gray_fit_hashes = {v.tobytes() for v in gray[~holdout]}
    gray_duplicates = [int(i) for i in np.flatnonzero(holdout) if gray[i].tobytes() in gray_fit_hashes]
    # An identical 32x32 input provides no new static or closed-loop test sample.
    split[gray_duplicates] = 'excluded'
    holdout = split == 'holdout'
    h5 = upstream / 'Aebs/data/Downsampled.h5'
    with h5py.File(h5, 'w') as data:
        data['X_train'] = gray[train].astype('float32') / 255.
        data['y_train'] = distance[train].astype('float32')[:, None]
    np.savez_compressed(output / 'manifest.npz', files=files, distance_m=distance, appearance=group,
                        split=split, rgb_sha256=np.asarray(hashes))
    with (output / 'split.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['filename', 'distance_m', 'appearance', 'split', 'rgb_sha256'])
        writer.writerows(zip(files, distance, group, split, hashes))
    gaps = {}
    for part in ['train', 'development', 'holdout']:
        gaps[part] = float(max(np.max(np.diff(np.sort(distance[(group == g) & (split == part)])))
                                for g in range(len(appearances))))
    scale = float(np.std(distance[train].astype('float32')))
    write_json(output / 'data_protocol.json', dict(
        source=str(source), samples_sha256=hashlib.sha256(metadata.read_bytes()).hexdigest(),
        split_seed=SEED, split_rule='100 blocks of adjacent unique distances, two development and six holdout distances per block; all appearances of a distance travel together; 5m and 16m train endpoints',
        counts={name: int((split == name).sum()) for name in np.unique(split)},
        distance_range_m=grid[[0, -1]].tolist(), train_distance_scale_m=scale,
        appearance_groups=appearances, max_internal_lookup_gap_m=gaps,
        excluded_gray_pixel_duplicate_count=len(gray_duplicates),
        h5_sha256=hashlib.sha256(h5.read_bytes()).hexdigest()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('upstream', type=Path)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    prepare(args.source, args.output, args.upstream, args.workers)
