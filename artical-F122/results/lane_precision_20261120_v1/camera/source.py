"""New closed-loop rendered-image CP and precision evaluation of frozen actors.

Every arm is frozen before ANY new calibration or test collection. All arms
use paired external scenarios, but render their own state-dependent frames.
Calibration thresholds are frozen before that arm's test is collected.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import multiprocessing
from pathlib import Path
import shutil
import time

import numpy as np

from .lane_precision import (ARMS, SEEDS, digest, save, precision_metrics, summarize,
                             calibrate_performance, performance_scores)

CAL_SEED = 202611200901
TEST_SEED = 202611200902


def packages(training_root):
    from .tasks import RESULTS
    result = {'anchor': RESULTS/'lane_robust_synthesis_20261116_v2/package'}
    for seed in SEEDS:
        for arm in ARMS:
            result[str(seed)+'__'+arm] = training_root/str(seed)/arm/'package'
    return result


def run(args):
    import torch
    from Aebs.lane_keeping import robust_trajectory_cp as cp
    from Aebs.lane_keeping.policy import load_actor
    from Aebs.lane_keeping.cp import conformal_radius
    from Aebs.lane_keeping.renderer import RoadRenderer
    from . import lane_image_confirmation_parallel as parallel
    from .tasks import RESULTS, load_task
    package_map = packages(args.training_root)
    bindings = {arm: {name: digest(path/name) for name in ['actor.pt', 'sbc.pt', 'patch_projection.pt']}
                for arm, path in package_map.items()}
    if len(set(b['patch_projection.pt'] for b in bindings.values())) != 1:
        raise ValueError('observer differs: this experiment isolates backend training')
    if len(set(b['sbc.pt'] for b in bindings.values())) != 1:
        raise ValueError('barrier differs between arms')
    args.output.mkdir(parents=True, exist_ok=False)
    repo = RESULTS.parent.parent/'external/dinov2'
    weights = RESULTS.parent.parent/'external/dinov2_weights/dinov2_vits14_pretrain.pth'
    protocol = dict(format='lane_precision_camera_v1', date='2026-11-20', arms=list(package_map),
        training_seeds=list(SEEDS), calibration_seed=CAL_SEED, test_seed=TEST_SEED,
        calibration_n=args.calibration, test_n=args.holdout, horizon=80, dt=.1, alpha=.05,
        strict_goal='whole episode safe; all sampled states t=2.0 through 8.0 seconds within lateral tolerance',
        tolerances_m=[.02, .05, .10, .20], speed_mps=4.,
        scope='fixed original DINO spatial observer; backend controlled ablation, NOT native SPVC or real-road cameras',
        source_sha256=digest(__file__), metric_source_sha256=digest(Path(__file__).with_name('lane_precision.py')),
        collector_sha256=digest(Path(parallel.__file__)), cp_source_sha256=digest(Path(cp.__file__)),
        bindings=bindings, dino_sha256=digest(weights), gpu_memory_fraction=args.memory_fraction,
        device=args.device, batch=args.batch, renderer_workers=args.workers,
        no_model_selection=True, unsafe_trajectories_retained=True,
        safety='new trajectory CP and fresh fixed neural barrier numerical audit per arm; no precision certificate',
        performance_cp='separate prespecified joint safety/precision score calibrated before holdout; not a conjunction with visual-error coverage at 95 percent',
        simultaneous_claim='Individual marginal CP only; not a joint 95 percent guarantee over all arms',
        uncertainty='scene-paired bootstrap conditional on these three trained seeds; no claim of training-population confidence')
    save(args.output/'protocol.json', protocol)
    shutil.copyfile(__file__, args.output/'source.py')
    torch.set_num_threads(4)
    if args.device.startswith('cuda'):
        torch.cuda.set_device(torch.device(args.device))
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, torch.device(args.device))
    observer = cp.DevicePatchObserver(next(iter(package_map.values()))/'patch_projection.pt', args.device, repo, weights)
    renderer = RoadRenderer(192, 192)
    started = time.monotonic()
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        parallel._POOL = pool
        save(args.output/'pixel_equivalence.json', parallel.pixel_check())
        for arm, source in package_map.items():
            out = args.output/arm
            package = out/'package'
            package.mkdir(parents=True)
            for name in bindings[arm]:
                shutil.copyfile(source/name, package/name)
                if digest(package/name) != bindings[arm][name]:
                    raise ValueError('frozen package changed before collection')
            actor = load_actor(package/'actor.pt').requires_grad_(False)
            config = argparse.Namespace(output_dir=out, calibration_episodes=args.calibration,
                holdout_episodes=args.holdout, calibration_seed=CAL_SEED, holdout_seed=TEST_SEED,
                batch_size=args.batch, horizon=80)
            binding = digest(args.output/'protocol.json')
            calibration = parallel.collect_parallel(config, 'calibration', actor, observer, renderer, binding)
            radius, rank = conformal_radius(calibration['scores'], .05)
            if not np.isfinite(radius):
                raise ValueError('finite radius required; increase calibration count')
            save(out/'calibration.json', dict(radius=radius, rank=rank, n=args.calibration, alpha=.05,
                protocol_sha256=binding, scores_sha256=digest(out/'calibration.npz'),
                scope='marginal whole-trajectory coverage under registered frozen-policy nominal simulator law'))
            precision_cal = calibrate_performance(precision_metrics(calibration['states'], calibration['actions']))
            save(out/'precision_calibration.json', precision_cal)
            threshold_hash = digest(out/'calibration.json')
            precision_hash = digest(out/'precision_calibration.json')
            print(json.dumps(dict(arm=arm, event='threshold_frozen_before_holdout', radius=radius, rank=rank)), flush=True)
            test = parallel.collect_parallel(config, 'holdout', actor, observer, renderer, binding)
            if (threshold_hash != digest(out/'calibration.json')
                    or precision_hash != digest(out/'precision_calibration.json')):
                raise ValueError('threshold changed during test')
            for name, expected in bindings[arm].items():
                if digest(package/name) != expected or digest(source/name) != expected:
                    raise ValueError('model changed during collection')
            values = precision_metrics(test['states'], test['actions'])
            np.savez_compressed(out/'precision_metrics.npz', **values)
            save(out/'performance.json', dict(n=args.holdout, metrics=summarize(values),
                 coverage=cp.coverage_summary(test['scores'], radius, independent=True),
                 precision_bound=precision_cal,
                 precision_covered=(int(((~values['unsafe']) & (performance_scores(values) <= precision_cal['radius_m'])).sum())
                     if precision_cal['finite_bound'] else None)))
            # Fresh q and fresh actor, never an inherited safety pass.
            shadow = out/'shadow/results/lane_robust_synthesis_20261116_v2'
            (shadow/'package').mkdir(parents=True)
            (shadow/'final_cp').mkdir()
            for name in ['actor.pt', 'sbc.pt']:
                shutil.copyfile(package/name, shadow/'package'/name)
            shutil.copyfile(out/'calibration.json', shadow/'final_cp/calibration.json')
            torch.set_num_threads(1)
            task = load_task('lane', root=out/'shadow')
            audit = task.current_verify(args.verify_seconds)
            audit.update(metadata=task.metadata, precision_certificate=False,
                         safety_scope='conditional on full error support, model and floating-point audit limitations')
            save(out/'barrier_audit.json', audit)
            torch.set_num_threads(4)
            print(json.dumps(dict(arm=arm, event='complete', barrier=audit.get('status'),
                rmse_cm=100*np.mean(values['rmse_e_m']), strict_5cm=int(values['strict_05cm'].sum()),
                unsafe=int(values['unsafe'].sum()), elapsed=time.monotonic()-started)), flush=True)
    aggregate(args.output)


def aggregate(root):
    protocol = json.loads((root/'protocol.json').read_text())
    metrics, rows, reference = {}, [], None
    for arm in protocol['arms']:
        data = dict(np.load(root/arm/'holdout.npz', allow_pickle=False))
        signature = (data['episode_seeds'], data['kappas'], data['states'][:, 0])
        if reference is None:
            reference = signature
        elif not all(np.array_equal(a, b) for a, b in zip(signature, reference)):
            raise AssertionError('external scenarios differ between arms')
        metric = precision_metrics(data['states'], data['actions'])
        metrics[arm] = metric
        for i in range(len(data['states'])):
            row = dict(arm=arm, trajectory=i, curvature=float(data['kappas'][i]))
            row.update({key: float(value[i]) for key, value in metric.items()})
            rows.append(row)
    with (root/'per_trajectory.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    n = protocol['test_n']
    rng = np.random.default_rng(202611200903)
    indices = rng.integers(n, size=(5000, n))
    aggregate_metrics = {'anchor': metrics['anchor']}
    for arm in ARMS:
        aggregate_metrics[arm] = {key: np.mean([metrics[str(seed)+'__'+arm][key] for seed in SEEDS], axis=0)
                                  for key in metrics['anchor']}
    comparisons = {}
    for left, right in [('anchor', ARMS[0]), ('anchor', ARMS[1]), (ARMS[0], ARMS[1])]:
        pair = {}
        for key in metrics['anchor']:
            a, b = np.asarray(aggregate_metrics[left][key], float), np.asarray(aggregate_metrics[right][key], float)
            difference = b-a
            pair[key] = dict(left_mean=float(a.mean()), right_mean=float(b.mean()),
                right_minus_left=float(difference.mean()),
                scene_paired_ci95=np.quantile(difference[indices].mean(1), [.025, .975]).tolist())
        comparisons[right+'__minus__'+left] = pair
    summary = dict(protocol_sha256=digest(root/'protocol.json'), paired_comparisons=comparisons,
        seed_aggregate={key: summarize(value) for key, value in aggregate_metrics.items()},
        per_arm={arm: dict(metrics=summarize(metrics[arm]),
            calibration=json.loads((root/arm/'calibration.json').read_text()),
            coverage=json.loads((root/arm/'performance.json').read_text())['coverage'],
            precision_calibration=json.loads((root/arm/'precision_calibration.json').read_text()),
            precision_covered=json.loads((root/arm/'performance.json').read_text())['precision_covered'],
            barrier=json.loads((root/arm/'barrier_audit.json').read_text())) for arm in protocol['arms']},
        no_precision_certificate=True, no_native_spvc_comparison=True,
        statistical_unit='independent paired scenes, NOT frames; seeds averaged per scene before bootstrap',
        caveat='do not condition CP claims on certificate acceptance; no real-road or continuous-time guarantee')
    save(root/'summary.json', summary)
    print(json.dumps(dict(event='all_complete', output=str(root/'summary.json'))), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--memory-fraction', type=float, default=.035)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--calibration', type=int, default=199)
    p.add_argument('--holdout', type=int, default=100)
    p.add_argument('--verify-seconds', type=float, default=120.)
    p.add_argument('--aggregate-only', action='store_true')
    args = p.parse_args()
    if args.aggregate_only:
        aggregate(args.output)
    else:
        if args.calibration < 19 or min(args.holdout, args.batch, args.workers) < 1:
            p.error('need at least 19 calibration trajectories and positive counts')
        run(args)


if __name__ == '__main__':
    main()
