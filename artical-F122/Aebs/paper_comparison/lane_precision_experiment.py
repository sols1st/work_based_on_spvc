"""Reproducible train -> synthetic check -> fresh-camera CP/audit workflow."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys

from .lane_precision import SEEDS, digest, save
from .lane_precision_camera import packages


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--skip-training', action='store_true')
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--device', default='cuda:1')
    a = p.parse_args()
    a.root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')

    def command(arguments, log):
        with log.open('w') as stream:
            subprocess.run([sys.executable, '-m']+arguments, stdout=stream,
                           stderr=subprocess.STDOUT, env=env, check=True)

    if not a.skip_training:
        def train(seed):
            command(['Aebs.paper_comparison.lane_precision', 'train', '--root', str(a.root),
                     '--seed', str(seed)], a.root/(str(seed)+'.log'))
        with ThreadPoolExecutor(3) as pool:
            list(pool.map(train, SEEDS))
    frozen = packages(a.root)
    save(a.root/'evaluation_manifest.json', dict(source_sha256=digest(__file__),
        frozen={key: {name: digest(path/name) for name in ['actor.pt', 'sbc.pt', 'patch_projection.pt']}
                for key, path in frozen.items()},
        source_of_selection='all three prespecified training seeds and both objectives, no seed filtering'))

    def synthetic(pair):
        name, package = pair
        command(['Aebs.paper_comparison.lane_precision', 'synthetic', '--package', str(package),
                 '--output', str(a.root/'synthetic'/name)], a.root/('synthetic_'+name+'.log'))
    with ThreadPoolExecutor(3) as pool:
        list(pool.map(synthetic, frozen.items()))
    command(['Aebs.paper_comparison.lane_precision_camera', '--training-root', str(a.root),
             '--output', str(a.root/'camera'), '--batch', str(a.batch), '--device', a.device], a.root/'camera.log')
    command(['Aebs.paper_comparison.lane_precision_verify', '--camera', str(a.root/'camera')],
            a.root/'precision_verification.log')
    command(['Aebs.paper_comparison.audit_lane_precision_results', '--camera', str(a.root/'camera')],
            a.root/'independent_audit.log')
    command(['Aebs.paper_comparison.lane_precision_examples', '--camera', str(a.root/'camera')],
            a.root/'camera_examples.log')
    print('Completed: '+str(a.root/'camera/summary.json'), flush=True)


if __name__ == '__main__':
    main()
