"""New paired DINO image rollouts for comparing CP boxes of six frozen hover policies."""
import argparse
from pathlib import Path

import numpy as np
import torch

from ..curve_braking.evaluate import digest, save
from ..curve_braking.vision import Encoder
from . import plant
from .compare_proxies import OUTPUT as PREVIOUS, REPO, SOURCE, WEIGHTS, read
from .compare_rollouts import parameters
from .evaluate import collect


OUT = Path('results/hover_cp_boxes_20261117_v1')
SPLITS = {'calibration': (199, 661721), 'holdout': (100, 661722)}


def run(args):
    spec = read(PREVIOUS / 'protocol.json')
    prior = read(PREVIOUS / 'evaluation_protocol.json')
    old_seal = read(PREVIOUS / 'all_models_frozen.json')
    if (old_seal != prior['models'] or digest(PREVIOUS / 'protocol.json') != prior['protocol']
            or any(digest(SOURCE / name) != value for name, value in spec['source_files'].items())
            or any(digest(REPO / name) != value for name, value in spec['repo_sources'].items())
            or digest(WEIGHTS) != spec['dino_weights']):
        raise ValueError('Mismatch with frozen source')
    parent = Path(__file__).resolve().parents[1]
    for name, value in prior['source'].items():
        if digest(parent / name) != value:
            raise ValueError('Frozen observer or dynamics changed: ' + name)
    for arm, key in old_seal.items():
        if digest(PREVIOUS / arm / 'model.pt') != key:
            raise ValueError('Frozen checkpoint changed: ' + arm)
    design = spec['design']
    batch = spec['image_batch']
    protocol = dict(splits={s:list(config) for s,config in SPLITS.items()}, design=design, batch=batch,
                    device=args.device, source_model_protocol=digest(PREVIOUS/'evaluation_protocol.json'),
                    source_extra={n:digest(parent/n) for n in ['dino_latent/backbone.py','vision_sls/observation_coupling.py']},
                    weights=old_seal, development={a:digest(PREVIOUS/a/'calibration.npz') for a in old_seal},
                    collection_source=digest(Path(__file__)),
                    features=['true normalized height relative to hover reference', 'infinity norm of v+0.7*p'],
                    selection='fixed six systems; same new external scenarios across arms; no retraining',
                    calibration_unit='whole trajectory; 24 image/control steps, no exclusion')
    OUT.mkdir(parents=True, exist_ok=True)
    protocol_path = OUT/'collection_protocol.json'
    if protocol_path.exists():
        if read(protocol_path) != protocol:
            raise ValueError('Collection protocol changed')
    elif args.split == 'calibration':
        save(protocol_path, protocol)
    else:
        raise ValueError('Missing calibration protocol')
    seal = None
    if args.split == 'holdout':
        if args.seal is None or not args.seal.is_file():
            raise ValueError('All six CP contract sets must be frozen before holdout')
        seal = digest(args.seal)
    n, seed = SPLITS[args.split]
    enc = Encoder(REPO, WEIGHTS, args.device)
    front = torch.load(SOURCE/'frontend.pt',map_location='cpu',weights_only=True)
    scenario = plant.scenarios(n,seed)
    for arm in old_seal:
        folder = OUT/arm
        folder.mkdir(exist_ok=True)
        file = folder/(args.split+'.npz')
        stamp = folder/(args.split+'_source.json')
        binding = dict(protocol=digest(protocol_path), model=old_seal[arm], split=args.split, n=n, seed=seed, seal=seal)
        if file.exists() and stamp.exists() and read(stamp) == {**binding, 'sha256':digest(file)}:
            continue
        model = parameters(PREVIOUS/arm,front)
        records = collect(scenario, design, enc, front, model, batch, record_observations=True)
        np.savez_compressed(file,**records)
        save(stamp,{**binding,'sha256':digest(file)})
        print(args.split,arm,flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('split',choices=SPLITS)
    cli.add_argument('--device',default='cuda:1')
    cli.add_argument('--seal',type=Path)
    args = cli.parse_args()
    torch.set_num_threads(2)
    run(args)
