"""Calibrate and audit matched trajectory-level perception boxes on three frozen tasks."""
import argparse
import hashlib
import importlib
import inspect
import json
from pathlib import Path

import numpy as np

from ..conformal import trajectory_boxes as method
from .compare_high_dim_proxies import TASKS, endpoints, scenario_data


PACKAGES = {'braking': 'curve_braking', 'hover': 'visual_hover', 'crane': 'overhead_crane'}
FEATURES = {'braking': ['obstacle distance (m)', 'absolute heading (rad)'],
            'hover': ['normalized vertical position', 'maximum absolute transformed velocity'],
            'crane': ['maximum absolute trolley position (m)', 'maximum normalized relative sway']}
OUTPUTS = {'braking': 'curve_braking', 'hover': 'hover', 'crane': 'crane'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    data = json.dumps(value, indent=2, allow_nan=False) + '\n'
    if path.exists() and path.read_text() != data:
        raise RuntimeError(f'Refusing to replace a different artifact at {path}')
    if not path.exists():
        tmp = path.with_suffix('.tmp')
        tmp.write_text(data)
        tmp.replace(path)


def arms(task, results, out):
    folder, seeds = TASKS[task]
    for seed in seeds:
        for kind in ('linear', 'neural'):
            name = f'{seed}_{kind}'
            old_name = f'{kind}_{seed}' if task == 'crane' else name
            yield name, results/folder/old_name, out/name


def arrays(path, backend, n, reference=None):
    with np.load(path, allow_pickle=False) as data:
        states, e = data['states'], data['errors']
        if len(states) != n or states.shape[1] != e.shape[1]+1:
            raise ValueError(f'invalid trajectory file {path}')
        if reference is not None:
            scene = scenario_data(data)
            if not reference:
                reference.update(scene)
            if set(reference) != set(scene):
                raise ValueError('mismatched exogenous scenario fields')
            for k, v in scene.items():
                np.testing.assert_array_equal(v, reference[k])
        return backend.features(states[:, :-1]), e


def freeze(task, results, out, backend):
    design = dict(task=task, features=FEATURES[task], methods=list(method.METHODS),
                  error='normalized decoded camera observation minus true physical coordinate',
                  score='max over all times, axes and both signs of error / applicable half-width',
                  shape_fit='old frozen-system calibration split reassigned as development only',
                  calibration_trajectories=199, rank=190, alpha=.05, holdout_trajectories=100,
                  feature_definition=inspect.getsource(backend.features),
                  method_sha256=digest(Path(method.__file__)))
    save(out/'design.json', design)
    seal_path = out/'all_boxes_frozen.json'
    if seal_path.exists():
        frozen = json.loads(seal_path.read_text())
        if digest(out/'design.json') != frozen['design_sha256']:
            raise ValueError('frozen design changed')
        for name, old, arm in arms(task, results, out):
            box_file = arm/'boxes.json'
            spec = json.loads(box_file.read_text())
            if (digest(box_file) != frozen['contracts'][name] or
                    digest(arm/'calibration.npz') != spec['calibration_sha256'] or
                    digest(old/'calibration.npz') != spec['development_sha256']):
                raise ValueError(f'{name}: frozen calibration inputs or contracts changed')
        return
    old_scenes, new_scenes, frozen = {}, {}, {}
    for name, old, arm in arms(task, results, out):
        dev, cal = old/'calibration.npz', arm/'calibration.npz'
        if (arm/'holdout.npz').exists() and not (out/'all_boxes_frozen.json').exists():
            raise ValueError('holdout already exists before sealing calibration')
        f_dev, e_dev = arrays(dev, backend, 199, old_scenes)
        f, e = arrays(cal, backend, 199, new_scenes)
        if np.array_equal(old_scenes['initial_states'], new_scenes['initial_states']):
            raise ValueError('development and calibration use identical starts')
        shapes = method.fit_shapes(f_dev, e_dev)
        boxes = {key: dict(shape=s, **method.calibrate(s, f, e)) for key, s in shapes.items()}
        save(arm/'boxes.json', dict(design_sha256=digest(out/'design.json'),
                                    development_sha256=digest(dev), calibration_sha256=digest(cal), boxes=boxes))
        frozen[name] = digest(arm/'boxes.json')
    save(out/'all_boxes_frozen.json', dict(design_sha256=digest(out/'design.json'), contracts=frozen))


def trajectory_metrics(f, errors, shape, q):
    neg, pos = method.widths(shape, f)
    neg, pos = q*neg, q*pos
    covered = method.trajectory_scores(shape, f, errors) <= q
    half = .5*(neg+pos)
    return dict(covered=int(covered.sum()), num_trajectories=len(f),
                mean_box_volume=float(np.prod(2*half, axis=-1).mean()),
                mean_equivalent_halfwidth=float(np.prod(half, axis=-1).mean()**(1/half.shape[-1])),
                mean_axis_fullwidth=(neg+pos).mean(axis=(0,1)).tolist(),
                max_negative=neg.max(axis=(0,1)).tolist(), max_positive=pos.max(axis=(0,1)).tolist())


def evaluate(task, results, out, backend):
    frozen = json.loads((out/'all_boxes_frozen.json').read_text())
    if digest(out/'design.json') != frozen['design_sha256']:
        raise ValueError('changed design')
    if json.loads((out/'design.json').read_text())['method_sha256'] != digest(Path(method.__file__)):
        raise ValueError('changed shape or score implementation')
    hold_scenes, records = {}, {}
    for name, old, arm in arms(task, results, out):
        box_file, hold_file = arm/'boxes.json', arm/'holdout.npz'
        spec = json.loads(box_file.read_text())
        if digest(box_file) != frozen['contracts'][name] or digest(arm/'calibration.npz') != spec['calibration_sha256']:
            raise ValueError(f'{name}: changed frozen box or calibration')
        f, errors = arrays(hold_file, backend, 100, hold_scenes)
        with np.load(hold_file) as h:
            safe, success, _ = endpoints(task, h)
        reports = {}
        for key, row in spec['boxes'].items():
            shape, q = row['shape'], row['q']
            stats = trajectory_metrics(f, errors, shape, q)
            check = backend.audit(shape, q)
            reports[key] = dict(q=q, **stats, audit=check)
        base_volume = reports['fixed_uniform']['mean_box_volume']
        for r in reports.values():
            r['volume_relative_to_uniform'] = r['mean_box_volume']/base_volume
        records[name] = dict(holdout_sha256=digest(hold_file), safe=int(safe.sum()),
                             success=int(success.sum()), boxes=reports)
    save(out/'comparison.json', dict(contracts_sha256=digest(out/'all_boxes_frozen.json'), arms=records))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('task', choices=TASKS)
    p.add_argument('stage', choices=['freeze', 'evaluate'])
    p.add_argument('--results', type=Path, default=Path('results'))
    a = p.parse_args()
    out = a.results / (OUTPUTS[a.task] + '_cp_boxes_20261117_v1')
    backend = importlib.import_module(f'Aebs.{PACKAGES[a.task]}.cp_boxes')
    if a.stage == 'freeze':
        freeze(a.task, a.results, out, backend)
    else:
        evaluate(a.task, a.results, out, backend)


if __name__ == '__main__':
    main()
