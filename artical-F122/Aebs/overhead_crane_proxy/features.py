"""Recompute the same frozen DINO features for both proxy families and bind the static split."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from ..curve_braking.evaluate import digest, save
from ..overhead_crane.camera import Camera


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(3)
    static = args.base / 'static.npz'
    with np.load(static) as data:
        images, split = data['images'], data['split']
        labels, lights = data['labels'], data['lights']
    if list(map(lambda k: int((split == k).sum()), ('train', 'val', 'test'))) != [2048, 256, 256]:
        raise ValueError('Unexpected static split')
    hashes = [hashlib.sha256(image.tobytes()).hexdigest() for image in images]
    if len(set(hashes)) != len(images) or not np.isfinite(labels).all():
        raise ValueError('Static inputs must be finite with unique images')
    camera = Camera(args.repo, args.weights, args.device)
    parts = [camera(images[start:start + args.batch]).half().cpu()
             for start in range(0, len(images), args.batch)]
    tokens = torch.cat(parts)
    if not torch.isfinite(tokens).all() or tuple(tokens.shape) != (len(images), 256, 384):
        raise ValueError('Unexpected patch features')
    torch.save(tokens, args.output / 'features.pt')
    np.savez_compressed(args.output / 'static_index.npz', images_sha256=hashes, split=split,
                        labels=labels, lights=lights)
    save(args.output / 'features.json', dict(static=digest(static), weights=digest(args.weights),
         features=digest(args.output/'features.pt'), index=digest(args.output/'static_index.npz'),
         dino_sources={str(p.relative_to(args.repo)): digest(p) for p in
                       [args.repo/'hubconf.py', *sorted((args.repo/'dinov2').rglob('*.py'))]},
         frames=len(images), shape=list(tokens.shape), batch=args.batch, device=args.device))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('base', type=Path)
    cli.add_argument('output', type=Path)
    cli.add_argument('--repo', type=Path, default=Path('../external/dinov2'))
    cli.add_argument('--weights', type=Path, default=Path('../external/dinov2_weights/dinov2_vits14_pretrain.pth'))
    cli.add_argument('--device', default='cuda:0')
    cli.add_argument('--batch', type=int, default=8)
    main(cli.parse_args())
