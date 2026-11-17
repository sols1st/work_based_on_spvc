"""Independent confirmation at the 1.5 operating scale selected in the grid study."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import compare_crane_contracts as grid
from ..curve_braking.evaluate import digest, save
from ..overhead_crane import plant
from ..overhead_crane.camera import Camera, Observation
from ..overhead_crane.evaluate import bindings, collect
from ..lane_keeping.cp import conformal_radius


def confirm(args):
    selection = json.loads((args.grid / 'protocol.json').read_text())
    if digest(Path(grid.__file__)) != selection['source']:
        raise ValueError('grid scoring and verification source changed')
    if bindings(args.base, args.repo, args.weights) != selection['bindings']:
        raise ValueError('baseline simulator or model changed')
    args.output.mkdir(parents=True, exist_ok=False)
    names = ['box', 'box_and_action']
    protocol = dict(scale=1.5, names=names, calibration=[999, 2211711],
                    test=[1000, 2211712], alpha=.05, batch=args.batch, device=args.device,
                    source=digest(Path(__file__)), grid_protocol=digest(args.grid/'protocol.json'),
                    selection_summary=digest(args.grid/'summary.json'),
                    primary='joint passes and box fails the exact certificate at scale 1.5')
    save(args.output/'protocol.json', protocol)
    torch.set_num_threads(3)
    camera = Camera(args.repo, args.weights, args.device)
    model = Observation().to(args.device)
    model.load_state_dict(torch.load(args.base/'frontend.pt',map_location=args.device,weights_only=True))
    model.eval()
    k = selection['k']
    def simulate(split):
        n, seed = protocol[split]
        result = collect(grid.scenario(n, seed, protocol['scale']), k, camera, model, args.batch)
        np.savez_compressed(args.output/(split+'.npz'), **result)
        return result
    cal = simulate('calibration')
    fits = {}
    for name in names:
        definition = selection['methods'][name]
        q, rank = conformal_radius(grid.trajectory_scores(cal['errors'], k, definition), protocol['alpha'])
        fits[name] = dict(**grid.certify(k, q, definition, protocol['scale']), rank=rank)
    save(args.output/'certificates.json', dict(fits=fits, protocol=digest(args.output/'protocol.json'),
                                               calibration=digest(args.output/'calibration.npz')))
    cert_hash = digest(args.output/'certificates.json')
    held = simulate('test')
    for name in names:
        scores = grid.trajectory_scores(held['errors'], k, selection['methods'][name])
        fits[name]['covered'] = int((scores <= fits[name]['q']).sum())
    if digest(args.output/'certificates.json') != cert_hash:
        raise RuntimeError('frozen certificates changed')
    result = dict(fits=fits, performance=plant.metrics(held['states']), certificates=cert_hash,
                  test=digest(args.output/'test.npz'),
                  primary_met=bool(fits['box_and_action']['passes'] and not fits['box']['passes']))
    save(args.output/'summary.json', result)
    print({name:(fits[name]['upper'],fits[name]['covered']) for name in names}, flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser()
    cli.add_argument('base', type=Path)
    cli.add_argument('grid', type=Path)
    cli.add_argument('output', type=Path)
    cli.add_argument('--device', default='cuda:0')
    cli.add_argument('--batch', type=int, default=8)
    cli.add_argument('--repo', type=Path, default=Path('../external/dinov2'))
    cli.add_argument('--weights', type=Path, default=Path('../external/dinov2_weights/dinov2_vits14_pretrain.pth'))
    confirm(cli.parse_args())
