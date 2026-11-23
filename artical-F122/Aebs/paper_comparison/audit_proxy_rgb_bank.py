"""Hash decoded CARLA RGB pixels in observation-row order."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path

import numpy as np
from PIL import Image

from .proxy_architecture_compare import sha, write


def pixel_hash(path):
    with Image.open(path) as image:
        rgb = image.convert('RGB')
        return sha256(np.asarray(rgb).tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('raw', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    source = Path('results/aebs_robust_synthesis_20261117_release/package/observations.npz')
    with np.load(source) as data:
        names, split = data['filename'], data['split']
    with ThreadPoolExecutor(max_workers=8) as pool:
        hashes = np.asarray(list(pool.map(pixel_hash, (args.raw/name for name in names))))
    tr, va, te = (set(hashes[split == name]) for name in ('train', 'validation', 'test'))
    mask = (split == 'test') & ~np.isin(hashes, list(tr | va))
    np.savez_compressed(args.output/'aebs_rgb_hashes.npz', filename=names, sha256=hashes, unseen_test=mask)
    write(args.output/'aebs_rgb_hashes.json', dict(raw_root=str(args.raw), observation_sha256=sha(source),
          source_sha256=sha(__file__), total=len(hashes), unique=len(set(hashes)), train_test_unique_overlap=len(tr & te),
          validation_test_unique_overlap=len(va & te), test_rows_without_train_or_validation_pixel_match=int(mask.sum())))


if __name__ == '__main__':
    main()
