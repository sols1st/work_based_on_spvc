"""Freeze trained models, calibrate their separate camera loops, then run paired holdouts."""
import argparse
import hashlib
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

from ..curve_braking.evaluate import digest, save
from ..experiments.analyze_crane_fixed_contracts import exact_geometry
from ..lane_keeping.cp import conformal_radius
from ..overhead_crane import certificate, plant
from ..overhead_crane.camera import Camera
from ..overhead_crane.evaluate import collect
from .train import FAMILIES, SEEDS, ProxyObservation, arm_path, read


SPLITS = {'calibration': (199, 817741), 'holdout': (100, 817742)}
ARMS = [(family, seed) for seed in SEEDS for family in FAMILIES]


def freeze(args):
    root = args.root
    if (root/'protocol.json').exists():
        raise FileExistsError('This protocol was already frozen')
    origin = args.base/'evaluation'/'synthesis.json'
    models = {}
    for family, seed in ARMS:
        arm = arm_path(root, family, seed)
        settings = read(arm/'training.json')
        if settings['checkpoint'] != digest(arm/'frontend.pt'):
            raise ValueError('Weights changed')
        models[arm.name] = dict(checkpoint=settings['checkpoint'], training=digest(arm/'training.json'))
    for seed in SEEDS:
        pair = [read(arm_path(root, f, seed)/'training.json') for f in FAMILIES]
        for key in ('features_manifest','warm_start','batch_order','epochs','lr','batch','updates','init_max_delta'):
            if pair[0][key] != pair[1][key]:
                raise ValueError(f'Unmatched {key}')
    package = Path(__file__).resolve().parents[1]
    code = ['overhead_crane_proxy/' + name + '.py' for name in ['features','train','evaluate']]
    code += ['overhead_crane/' + name + '.py' for name in ['camera','plant','evaluate','certificate']]
    code += ['vision_sls/observation_coupling.py', 'dino_latent/backbone.py', 'lane_keeping/cp.py',
             'experiments/analyze_crane_fixed_contracts.py', 'curve_braking/evaluate.py']
    save(root/'protocol.json',dict(source={k:digest(package/k) for k in code},
        weights=digest(args.weights), features_manifest=digest(root/'features.json'), models=models,
        synthesis=digest(origin), gains=read(origin)['selected']['gains'], alpha=.05, splits=SPLITS,
        sensors='render state to 176x176 RGB on every observation, frozen DINOv2; velocity sensors unchanged',
        primary='trajectory max absolute decoded-coordinate error, uniform decoded four-coordinate box',
        secondaries=['raw latent residual', 'exact barrier ratio and geometric scale supremum at fixed q',
                     'safe and settled', 'final payload Euclidean position error', 'last-second sway'],
        ct=plant.HORIZON, camera_batch=args.batch, init_scale=1, updates_each=read(
            arm_path(root,*ARMS[0])/'training.json')['updates']))


class ImageDigest:
    def __init__(self, camera, batch, count):
        self.camera = camera
        self.batch = batch
        self.count = count
        self.hashers = [hashlib.sha256() for _ in range(count)]
        self.calls = 0

    def __call__(self, images):
        start = self.calls // plant.HORIZON * self.batch
        for i, img in enumerate(images[:max(0, self.count-start)]):
            self.hashers[start+i].update(img.tobytes())
        self.calls += 1
        return self.camera(images)


@torch.inference_mode()
def rollout(args, family, seed, camera, protocol):
    root = args.root
    arm = arm_path(root, family, seed)
    model = ProxyObservation(family).to(args.device)
    if digest(arm/'frontend.pt') != protocol['models'][arm.name]['checkpoint']:
        raise ValueError('Frozen model changed')
    model.load_state_dict(torch.load(arm/'frontend.pt',map_location=args.device,weights_only=True))
    model.eval()
    n, scene_seed = protocol['splits'][args.phase]
    scenario = plant.scenarios(n, scene_seed)
    # Calls use the original archived collection path, with per-trajectory image digests added.
    capture = ImageDigest(camera, args.batch, n)
    data = collect(scenario, protocol['gains'], capture, model, args.batch)
    assert capture.calls == int(np.ceil(n/args.batch))*plant.HORIZON
    data['image_digests'] = np.array([h.hexdigest() for h in capture.hashers])
    z = plant.observation(data['states'][:,:-1]).reshape(-1,4)
    true_z = torch.tensor(z, device=args.device, dtype=torch.float32)
    nominal = model.nominal(true_z).cpu().numpy().reshape(n,plant.HORIZON,4)
    data['latent_errors'] = data['latent']-nominal
    data['latent_scores'] = np.abs(data['latent_errors']).max(axis=(1,2))
    scores=data['scores']
    path = arm/(args.phase+'.npz')
    np.savez_compressed(path, **data)
    if args.phase == 'calibration':
        q, rank = conformal_radius(scores, protocol['alpha'])
        audit = certificate.verify(protocol['gains'], q, exact=True)
        geometry = exact_geometry(protocol['gains'], q, Fraction(1))
        assert audit['initial_upper_exact'] == geometry['upper_exact']
        q_raw, rank_raw = conformal_radius(data['latent_scores'], protocol['alpha'])
        save(arm/'certificate.json', dict(protocol=digest(root/'protocol.json'), calibration=digest(path),
            checkpoint=digest(arm/'frontend.pt'), q=q, rank=rank, raw_q=q_raw, raw_rank=rank_raw,
            geometry=geometry, audit=audit))
        print('calibration', arm.name, q, audit['initial_upper'], flush=True)
    else:
        fit = read(arm/'certificate.json')
        report = dict(**plant.metrics(data['states']),test=digest(path),certificate=digest(arm/'certificate.json'),
            covered=int((scores<=fit['q']).sum()), raw_covered=int((data['latent_scores']<=fit['raw_q']).sum()),
            mean_peak=float(scores.mean()), raw_mean_peak=float(data['latent_scores'].mean()))
        save(arm/'summary.json', report)
        print('holdout', arm.name, report['covered'], report['settled'], flush=True)


def main(args):
    root = args.root
    if args.phase == 'freeze':
        return freeze(args)
    p = read(root/'protocol.json')
    package = Path(__file__).resolve().parents[1]
    if args.batch != p['camera_batch'] or any(digest(package/k) != val for k,val in p['source'].items()):
        raise ValueError('Frozen protocol or source changed')
    if digest(args.weights) != p['weights']:
        raise ValueError('Changed DINO weights')
    assets = read(root/'features.json')
    if digest(root/'features.json') != p['features_manifest'] or any(
            digest(args.repo/k) != h for k,h in assets['dino_sources'].items()):
        raise ValueError('Frozen backbone source changed')
    seal = root/'all_contracts.json'
    if args.phase in ('seal','holdout'):
        contracts = {arm_path(root,*item).name:digest(arm_path(root,*item)/'certificate.json') for item in ARMS}
        if args.phase=='seal':
            if seal.exists(): raise FileExistsError(seal)
            save(seal,dict(protocol=digest(root/'protocol.json'),contracts=contracts))
            return
        if read(seal) != dict(protocol=digest(root/'protocol.json'),contracts=contracts):
            raise ValueError('All calibration results must be sealed before holdout')
    elif seal.exists():
        raise ValueError('Calibration is sealed')
    torch.set_num_threads(3)
    camera = Camera(args.repo,args.weights,args.device)
    for family, seed in ARMS:
        if args.family and family != args.family or args.seed and seed != args.seed:
            continue
        arm = arm_path(root, family, seed)
        marker = arm/('certificate.json' if args.phase=='calibration' else 'summary.json')
        if not marker.exists():
            rollout(args, family, seed, camera, p)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('base',type=Path)
    cli.add_argument('root',type=Path)
    cli.add_argument('phase',choices=['freeze','calibration','seal','holdout'])
    cli.add_argument('--family', choices=FAMILIES)
    cli.add_argument('--seed', type=int, choices=SEEDS)
    cli.add_argument('--batch',type=int,default=8)
    cli.add_argument('--repo',type=Path,default=Path('../external/dinov2'))
    cli.add_argument('--weights',type=Path,default=Path('../external/dinov2_weights/dinov2_vits14_pretrain.pth'))
    cli.add_argument('--device',default='cuda:0')
    main(cli.parse_args())
