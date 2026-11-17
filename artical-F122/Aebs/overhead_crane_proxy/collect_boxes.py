"""New on-policy crane camera trajectories for the paired CP box experiment."""
import argparse
from pathlib import Path

import numpy as np
import torch

from ..curve_braking.evaluate import digest, save
from ..overhead_crane import plant
from ..overhead_crane.camera import Camera
from ..overhead_crane.evaluate import collect
from .check import replay
from .evaluate import ARMS, ImageDigest
from .train import ProxyObservation, arm_path, read

BASE = Path('results/crane_proxy_comparison_20261117_v1')
OUT = Path('results/crane_cp_boxes_20261117_v1')
SPLITS = {'calibration': (199, 819771), 'holdout': (100, 819772)}


def bindings():
    original = read(BASE/'protocol.json')
    parts = Path(__file__).resolve().parents[1]
    for name, expected in original['source'].items():
        if digest(parts/name) != expected:
            raise ValueError(f'Frozen implementation mismatch: {name}')
    arms = {}
    for family, seed in ARMS:
        old = arm_path(BASE, family, seed)
        weight = digest(old/'frontend.pt')
        if weight != original['models'][old.name]['checkpoint']:
            raise ValueError(f'Frozen checkpoint mismatch: {old.name}')
        arms[f'{seed}_{family}'] = dict(checkpoint=weight, development=digest(old/'calibration.npz'))
    features = read(BASE/'features.json')
    return dict(arms=arms, splits=SPLITS, old_protocol=digest(BASE/'protocol.json'),
                source=digest(Path(__file__)), dino_weights=original['weights'],
                dino_source=features['dino_sources'], camera_batch=8, horizon=plant.HORIZON,
                gains=original['gains'], physical_features=['max_abs_trolley_m','max_abs_sway_over_0.5m'],
                scene_law='original nominal initial scale, independent seeded resets and gusts across trajectories')


def sealed():
    seal = read(OUT/'all_boxes_frozen.json')
    contracts = {f'{seed}_{family}':digest(OUT/f'{seed}_{family}'/'boxes.json') for family,seed in ARMS}
    if contracts != seal['contracts']:
        raise ValueError('Required all frozen box-contract hashes')
    return digest(OUT/'all_boxes_frozen.json')


@torch.inference_mode()
def main(args):
    OUT.mkdir(exist_ok=True)
    protocol = bindings()
    path = OUT/'camera_protocol.json'
    if not path.exists():
        if args.split != 'calibration':
            raise ValueError('Missing collection protocol')
        save(path, protocol)
    elif read(path) != {**protocol, 'splits': {k:list(v) for k,v in SPLITS.items()}}:
        raise ValueError('Collection protocol or inputs changed')
    inputs = dict(protocol=digest(path))
    if args.split == 'holdout':
        inputs['frozen_boxes'] = sealed()
    elif (OUT/'all_boxes_frozen.json').exists():
        raise ValueError('Calibration is sealed')
    if digest(args.weights) != protocol['dino_weights'] or any(
            digest(args.repo/k) != v for k,v in protocol['dino_source'].items()):
        raise ValueError('Backbone mismatch')
    camera = Camera(args.repo, args.weights, args.device)
    n, seed_scenes = SPLITS[args.split]
    scenarios = plant.scenarios(n, seed_scenes)
    for family, seed in ARMS:
        arm = OUT/f'{seed}_{family}'
        arm.mkdir(exist_ok=True)
        dataset, marker = arm/(args.split+'.npz'), arm/(args.split+'_camera.json')
        if marker.exists():
            past=read(marker)
            if past['inputs'] != inputs or past['data'] != digest(dataset):
                raise ValueError('Incompatible partial collection')
            continue
        old = arm_path(BASE, family, seed)
        model = ProxyObservation(family).to(args.device).eval()
        model.load_state_dict(torch.load(old/'frontend.pt', map_location=args.device, weights_only=True))
        capture = ImageDigest(camera, protocol['camera_batch'], n)
        data = collect(scenarios, protocol['gains'], capture, model, protocol['camera_batch'])
        data['image_digests'] = np.array([h.hexdigest() for h in capture.hashers])
        z = plant.observation(data['states'][:,:-1]).reshape(-1,4)
        nominal = model.nominal(torch.tensor(z,device=args.device,dtype=torch.float32)).cpu().numpy()
        data['latent_errors'] = data['latent']-nominal.reshape(n,plant.HORIZON,4)
        data['latent_scores'] = np.abs(data['latent_errors']).max(axis=(1,2))
        np.savez_compressed(dataset, **data)
        save(marker, dict(inputs=inputs,data=digest(dataset),n=n,steps=n*plant.HORIZON,gpu_calls=capture.calls))
        print(args.split,arm.name,n,flush=True)


def check():
    p = read(OUT/'camera_protocol.json')
    seal = sealed()
    rows = {}
    for family, seed in ARMS:
        old = arm_path(BASE,family,seed)
        arm = OUT/f'{seed}_{family}'
        model = ProxyObservation(family).eval()
        model.load_state_dict(torch.load(old/'frontend.pt', map_location='cpu',weights_only=True))
        for split,(n, scene_seed) in SPLITS.items():
            marker=read(arm/(split+'_camera.json'))
            if marker['data']!=digest(arm/(split+'.npz')) or (split=='holdout' and marker['inputs']['frozen_boxes']!=seal):
                raise ValueError('Stale trajectory file')
            with np.load(arm/(split+'.npz')) as data:
                replay(data,model,n,scene_seed,p['gains'],True)
                stats=plant.metrics(data['states'])
            rows[f'{arm.name}/{split}']=stats
    save(OUT/'trajectory_check.json',dict(rows=rows,transitions=6*(199+100)*plant.HORIZON,
        rerendered_last_trajectory_per_split_per_arm=12,seal=seal))


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('split',choices=['calibration','holdout','check'])
    cli.add_argument('--device',default='cuda:0')
    cli.add_argument('--repo',type=Path,default=Path('../external/dinov2'))
    cli.add_argument('--weights',type=Path,default=Path('../external/dinov2_weights/dinov2_vits14_pretrain.pth'))
    args=cli.parse_args()
    torch.set_num_threads(3)
    check() if args.split=='check' else main(args)
