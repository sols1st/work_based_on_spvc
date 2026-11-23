"""Collect paired state and frozen visual observations for a lane proxy ablation."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ..lane_keeping import core
from ..lane_keeping.compare_proxy_cp import MatchedImageObserver
from ..lane_keeping.renderer import RoadRenderer
from .proxy_architecture_compare import sha, write

SOURCE = Path('results/lane_proxy_comparison_20261117_v1')
CHECKPOINT = SOURCE / 'train_202611171/linear/observation.pt'
SPLITS = [('train', 12000), ('development', 1000), ('calibration', 1000), ('test', 2000)]


def render(args):
    state, curvature = args
    image = RoadRenderer(192, 192).render(state, curvature)
    return image, hashlib.sha256(image.tobytes()).hexdigest()


def collect(output, device, workers):
    weights = Path('../external/dinov2_weights/dinov2_vits14_pretrain.pth')
    historical = SOURCE / 'feature_protocol.json'
    seen = set(json.loads(historical.read_text())['image_sha256'])
    output.mkdir(exist_ok=True, parents=True)
    target = output / 'lane_bank.npz'
    if target.exists():
        raise FileExistsError(target)
    torch.set_num_threads(2)
    observer = MatchedImageObserver(CHECKPOINT, 'linear', device, Path('../external/dinov2'), weights).eval()
    bank = {k: [] for k in ('state', 'curvature', 'sha256', 'split', 'latent')}
    rejected = {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=__import__('multiprocessing').get_context('spawn')) as pool:
        for part, (name, count) in enumerate(SPLITS):
            rng = np.random.default_rng(2026112200 + part)
            done = rejected[name] = 0
            while done < count:
                n = min(count - done, 128)
                states = rng.uniform([-.8, -.25], [.8, .25], (n, 2))
                curves = rng.choice([-.01, 0., .01], n)
                images = []
                for (state, curvature), (img, digest) in zip(zip(states, curves), pool.map(render, zip(states, curves), chunksize=4)):
                    if digest in seen:
                        rejected[name] += 1
                        continue
                    seen.add(digest)
                    images.append(img)
                    for key, value in [('state', state), ('curvature', curvature), ('sha256', digest), ('split', name)]:
                        bank[key].append(value)
                for start in range(0, len(images), 8):
                    batch = torch.from_numpy(np.stack(images[start:start+8]))
                    _, latent = observer.forward_with_latent(batch)
                    bank['latent'].extend(latent.cpu().numpy())
                done += len(images)
                if done == count or done % 1000 < len(images):
                    print(name, done, flush=True)
    scales = np.array([core.E_SCALE, core.THETA_SCALE])
    np.savez_compressed(target, **bank, normalized_state=np.array(bank['state']) / scales)
    write(output / 'lane_bank.json', dict(bank_sha256=sha(target), rejected=rejected,
          seeds={name:2026112200+i for i,(name,_) in enumerate(SPLITS)}, splits=dict(SPLITS),
          checkpoint_sha256=sha(CHECKPOINT), weights_sha256=sha(weights), historical_hash_list_sha256=sha(historical),
          source_sha256=sha(__file__), renderer_sha256=sha(Path(__file__).parents[1]/'lane_keeping/renderer.py')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    collect(args.output, args.device, args.workers)
