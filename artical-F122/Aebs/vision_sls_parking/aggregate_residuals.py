"""Add explicitly reused development image trajectories to sensor-fit data."""
import argparse
import json
from pathlib import Path
import numpy as np
from .train_ppo import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-data', required=True)
    p.add_argument('--image-eval-dirs', nargs='+', required=True)
    p.add_argument('--output-dir', required=True)
    a = p.parse_args()
    out = Path(a.output_dir)
    if out.exists():
        p.error('new output directory required')
    base = Path(a.base_data)
    config = json.loads((base.parent/'config.json').read_text())
    with np.load(base) as data:
        fit = data['splits'].astype(str) == 'fit'
        states, errors = [data['states'][fit]], [data['residuals'][fit]]
    sources = {}
    for directory in a.image_eval_dirs:
        root = Path(directory)
        record = json.loads((root/'config.json').read_text())
        if record['projection_sha256'] != config['projection_sha256']:
            raise ValueError('cannot mix residuals from different visual projections')
        for path in sorted(root.glob('*.npz')):
            with np.load(path) as data:
                states.append(data['capture_states']); errors.append(data['residuals'])
            sources[str(path)] = sha(path)
    x, r = np.concatenate(states), np.concatenate(errors)
    _, ids = np.unique(x[:, :3], axis=0, return_index=True)
    x, r = x[ids], r[ids]
    out.mkdir(parents=True)
    np.savez_compressed(out/'samples.npz', states=x, residuals=r, splits=np.repeat('fit', len(x)))
    config.update(source_archives=sources, base_data_sha256=sha(base), samples=len(x),
                  training_purpose='development trajectory aggregation; excludes calibration and audit rows',
                  evaluation_role='15 original development starts now reused for training; not heldout testing')
    (out/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    print('fit rows:', len(x), flush=True)


if __name__ == '__main__':
    main()
