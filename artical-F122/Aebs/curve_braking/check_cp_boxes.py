"""Independent membership, consequence and image-trajectory checks for braking boxes."""
from pathlib import Path

import numpy as np
import torch

from ..conformal.trajectory_boxes import cuts, widths
from .box_trajectories import BASE, OBSERVERS, OUTPUT, SCENARIOS, read
from .check_proxy_comparison import check_trajectories
from .compare_proxy import load
from .cp_boxes import features
from .evaluate import digest, save
from .models import Actor, Proxy


def check_guard(shape, q, report):
    # Conservative audit is a continuum calculation. Here independently look for
    # violations at all learned boundaries and the neighboring one-sided values,
    # with a dense spacing in between to exercise the interpolation and clipping.
    dcuts, hcuts = cuts(shape)
    def axis(lo, hi, edges):
        knots = np.asarray(edges)
        interior = knots[(knots >= lo) & (knots <= hi)]
        raw = np.r_[np.linspace(lo, hi, 127), interior,
                    np.nextafter(interior, -np.inf), np.nextafter(interior, np.inf)]
        return np.unique(np.clip(raw,lo,hi))
    distance, heading = np.meshgrid(axis(1.92,6.15,dcuts),axis(0,.19,hcuts),indexing='ij')
    _, positive = widths(shape, np.stack((distance,heading),axis=-1))
    actual = (distance + 2*q*positive[...,2]).max()
    ceiling = report['maximum_perceived_distance_upper_m']
    if actual > ceiling + 2e-14:
        raise ValueError('camera distance can exceed declared proof bound')
    return float(ceiling-actual)


def run(root):
    torch.set_num_threads(2)
    proto = read(root/'trajectory_protocol.json')
    frozen = read(root/'all_boxes_frozen.json')
    actor = Actor()
    actor.load_state_dict(load(BASE/'actor.pt'))
    reports = read(root/'comparison.json')['arms']
    archives = {split:read(root/(split+'_capture.json')) for split in SCENARIOS}
    if digest(root/'all_boxes_frozen.json') != archives['holdout']['sealed']:
        raise ValueError('holdout evaluated different boxes')
    paths = {}
    for arm, model_hash in proto['bindings']['arms'].items():
        folder = root/arm
        if digest(OBSERVERS/arm/'proxy.pt') != model_hash or digest(folder/'boxes.json') != frozen['contracts'][arm]:
            raise ValueError('changed weights/contract')
        model = Proxy(arm.split('_')[-1],192)
        model.load_state_dict(load(OBSERVERS/arm/'proxy.pt'))
        boxes = read(folder/'boxes.json')['boxes']
        check, per_box = {}, {}
        for split,(n,seed) in SCENARIOS.items():
            archive = folder/(split+'.npz')
            if digest(archive) != archives[split]['archives'][arm]['archive']:
                raise ValueError('archive binding changed')
            with np.load(archive, allow_pickle=False) as data:
                check[split] = check_trajectories(data,model,actor,n,seed)
                f = features(data['states'][:,:-1])
                err = data['errors']
                for name, spec in boxes.items():
                    left, right = widths(spec['shape'], f)
                    q = spec['q']
                    if split == 'calibration':
                        score = np.where(err>=0, err/right, -err/left).max(axis=(1,2))
                        np.testing.assert_equal(q, np.sort(score)[189])
                    else:
                        accepted = ((err>=-q*left)&(err<=q*right)).all(axis=(1,2))
                        report = reports[arm]['boxes'][name]
                        np.testing.assert_equal(int(accepted.sum()), report['covered'])
                        per_box[name] = dict(rank=190, covered=int(accepted.sum()),
                                extra_gap_above_numerical_perturbation_probe_m=check_guard(spec['shape'],q,report['audit']))
        paths[arm] = dict(trajectory_checks=check, box_checks=per_box)
    save(root/'independent_box_check.json',dict(passed=True,checks=paths,
            method='replay all 5D actions and successors, rerender first/last full trajectory per split '
                   'and arm, recover ranks, test membership inequalities directly, probe continuum guard around knots'))


if __name__ == '__main__':
    run(OUTPUT)
