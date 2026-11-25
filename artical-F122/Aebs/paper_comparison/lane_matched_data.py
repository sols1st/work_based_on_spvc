"""Render a new shared 192x192 RGB bank for the straight-road SPVT comparison."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path

import numpy as np
from PIL import Image

from .static_road_camera import StaticRoadCamera

PARTS = (('train', 20000), ('development', 2000), ('test', 4000))
_CAMERA = None


def render(states):
    global _CAMERA
    if _CAMERA is None:
        _CAMERA = StaticRoadCamera()
    return np.stack([_CAMERA.frame([e / 100, np.deg2rad(theta)]) for e, theta in states])


def grayscale(images):
    return np.stack([np.asarray(Image.fromarray(image).convert('L').resize(
        (16, 8), Image.Resampling.BICUBIC)).reshape(-1) for image in images])


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def collect(output, workers):
    output.mkdir(parents=True, exist_ok=False)
    seen = set()
    manifest = {}
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        for part, (name, count) in enumerate(PARTS):
            rng = np.random.default_rng(2026112600 + part)
            rgb = np.lib.format.open_memmap(output / (name + '_rgb.npy'), mode='w+',
                                          dtype=np.uint8, shape=(count, 192, 192, 3))
            gray = np.empty((count, 128), dtype=np.uint8)
            accepted = []
            digests = []
            rejected = 0
            while len(accepted) < count:
                size = min(count - len(accepted), 512)
                states = rng.uniform([-80., -12.], [80., 12.], (size, 2)).astype('float32')
                rendered = np.concatenate(list(pool.map(render, np.array_split(states, min(workers, size)))))
                for state, image in zip(states, rendered):
                    digest = hashlib.sha256(image.tobytes()).hexdigest()
                    if digest in seen:
                        rejected += 1
                        continue
                    seen.add(digest)
                    rgb[len(accepted)] = image
                    gray[len(accepted)] = grayscale(image[None])[0]
                    accepted.append(state)
                    digests.append(digest)
                if len(accepted) == count or len(accepted) % 2048 < 512:
                    print(name, len(accepted), '/', count, flush=True)
            rgb.flush()
            np.savez_compressed(output / (name + '.npz'), states_cm_degrees=np.stack(accepted),
                                gray=gray, rgb_sha256=np.array(digests))
            manifest[name] = dict(count=count, seed=2026112600 + part, rgb_duplicates_rejected=rejected)
    write_json(output / 'data.json', dict(splits=manifest, physical_units=['cm', 'degree'],
        state_domain=[[-80, -12], [80, 12]], rgb_shape=[192, 192, 3],
        scope='new fixed synthetic straight-road camera; split before any fitting; RGB pixel uniqueness across splits',
        grayscale='Pillow luminance and bicubic 16x8, row-major',
        renderer_sha256=hashlib.sha256(Path(__file__).with_name('static_road_camera.py').read_bytes()).hexdigest()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--workers', type=int, default=20)
    args = parser.parse_args()
    collect(args.output, args.workers)
