"""New paired camera runs for comparing error sets after freezing all six policies."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import plant
from .compare_proxy import ImageCache, REPO, WEIGHTS, load
from .evaluate import digest, save
from .models import Actor, Proxy
from .vision import Encoder


BASE = Path('results/curve_braking_5d_20261117')
OBSERVERS = Path('results/curve_braking_proxy_comparison_20261117_v1')
OUTPUT = Path('results/curve_braking_cp_boxes_20261117_v1')
SCENARIOS = {'calibration': (199, 202611175091), 'holdout': (100, 202611175092)}


def read(path):
    return json.loads(path.read_text())


def bindings():
    models = read(OBSERVERS/'models_frozen.json')
    source = read(OBSERVERS/'protocol.json')
    actor_barrier = {name: digest(BASE/name) for name in ('actor.pt', 'barrier.pt', 'pca.pt')}
    if any(value != source['bindings'][name] for name, value in actor_barrier.items()):
        raise ValueError('changed frozen deployment components')
    if source['dino_sha256'] != digest(WEIGHTS):
        raise ValueError('changed backbone')
    for name, expected in models.items():
        if digest(OBSERVERS/name/'proxy.pt') != expected:
            raise ValueError(f'changed {name}')
    package = Path(__file__).resolve().parents[1]
    files = ['curve_braking/'+name+'.py' for name in ('box_trajectories', 'compare_proxy', 'models','evaluate','vision','plant','verify')]
    files.extend(['vision_sls/observation_coupling.py','dino_latent/backbone.py'])
    return dict(arms=models, deployment=actor_barrier, vision=digest(WEIGHTS),
                source={k:digest(package/k) for k in files},
                old_calibration_for_development={n:digest(OBSERVERS/n/'calibration.npz') for n in models})


def run(out, split, device):
    torch.set_num_threads(3)
    out.mkdir(parents=True, exist_ok=True)
    current = bindings()
    proto = out/'trajectory_protocol.json'
    if not proto.exists():
        if split != 'calibration':
            raise ValueError('collect calibration first')
        save(proto,dict(bindings=current,scenarios=SCENARIOS,horizon=plant.HORIZON,batch=8,
                        collection='render at every reached state; SHA256-identical rasters reuse DINO only',
                        alpha=.05, rank=190, state_features='true obstacle distance (m), absolute heading error (rad)',
                        component='certify minimum brake floor via maximum perceived obstacle distance on the barrier sublevel'))
    if read(proto)['bindings'] != current:
        raise ValueError('frozen artifact changed')
    if split == 'holdout' and not (out/'all_boxes_frozen.json').exists():
        raise ValueError('freeze all calibrated boxes before start of blind test')
    sealed = digest(out/'all_boxes_frozen.json') if split == 'holdout' else None
    manifest = out/(split+'_capture.json')
    if manifest.exists():
        raise FileExistsError(manifest)
    count, seed = SCENARIOS[split]
    pca = load(BASE/'pca.pt')
    actor = Actor()
    actor.load_state_dict(load(BASE/'actor.pt'))
    cache = ImageCache(Encoder(REPO,WEIGHTS,device))
    archives = {}
    for arm, h in current['arms'].items():
        target = out/arm
        target.mkdir(exist_ok=True)
        proxy = Proxy(arm.split('_')[-1], pca['basis'].shape[1])
        proxy.load_state_dict(load(OBSERVERS/arm/'proxy.pt'))
        archive = target/(split+'.npz')
        rows = cache.run(count, seed, pca, proxy.eval(), actor.eval())
        np.savez_compressed(archive, **rows)
        archives[arm] = dict(archive=digest(archive),observer=h)
        print(split,arm,flush=True)
    if split == 'holdout' and sealed != digest(out/'all_boxes_frozen.json'):
        raise ValueError('thresholds changed')
    save(manifest,dict(archives=archives,protocol=digest(proto),sealed=sealed,
                       effective_observations=len(archives)*count*plant.HORIZON,
                       distinct_rasters=len(cache.features),inference_slots=cache.encoded))


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('split',choices=SCENARIOS)
    cli.add_argument('--device',default='cuda:0')
    cli.add_argument('--output',type=Path,default=OUTPUT)
    args=cli.parse_args()
    run(args.output,args.split,args.device)
