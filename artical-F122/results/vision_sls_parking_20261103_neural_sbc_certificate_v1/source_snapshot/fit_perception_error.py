"""Collect fresh independent simulated poses and calibrate a 3D error box.

Coverage refers to the registered uniform pose sampler and fixed renderer,
not to states visited by a policy or complete closed-loop trajectories.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from .core import C, RENDER_OFFSET
from .ppo_env import FrozenImageSensor
from .perception_error import fit_scale, calibrate, radius, joint_scores
from .train_ppo import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--projection', default='results/vision_sls_parking_20261025_adaptation_v1/projection_adapted.pt')
    p.add_argument('--vision-root', default='../VISION-SLS')
    p.add_argument('--dino-repo', default='../external/dinov2')
    p.add_argument('--dino-weights', default='../external/dinov2_weights/dinov2_vitg14_pretrain.pth')
    p.add_argument('--fit-count', type=int, default=256)
    p.add_argument('--cal-count', type=int, default=128)
    p.add_argument('--audit-count', type=int, default=128)
    p.add_argument('--seed', type=int, default=20261102)
    a = p.parse_args()
    out = Path(a.output_dir)
    if out.exists() or min(a.fit_count, a.cal_count, a.audit_count) < 1:
        p.error('new directory and positive split sizes required')
    out.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    low = np.array([-3.5, -2., -np.pi]); high = np.array([.5, 1.9, np.pi])
    counts = [a.fit_count, a.cal_count, a.audit_count]
    rng = np.random.default_rng(a.seed)
    states = np.column_stack([rng.uniform(low, high, (sum(counts), 3)), np.zeros(sum(counts))])
    splits = np.repeat(['fit', 'calibration', 'audit'], counts)
    cfg = dict(vars(a), low=low.tolist(), high=high.tolist(),
               sampler='independent uniform poses, speed zero, fixed OpenGL scene',
               projection_sha256=sha(a.projection), dino_sha256=sha(a.dino_weights),
               marginal_scope='one new pose from this same uniform simulator distribution',
               policy_distribution_coverage=False, trajectory_coverage=False,
               physical_speed_invisible_in_renderer=True)
    (out/'config.json').write_text(json.dumps(cfg, indent=2)+'\n')
    np.savez_compressed(out/'sampling_plan.npz', states=states, splits=splits)
    sensor = FrozenImageSensor(a.vision_root, a.dino_repo, a.dino_weights, a.projection)
    predictions = []
    started = time.monotonic()
    try:
        for i, state in enumerate(states):
            predictions.append(sensor(state))
            if (i+1) % 64 == 0:
                print('fresh error sample %d/%d' % (i+1, len(states)), flush=True)
    finally:
        sensor.close()
    y = np.asarray(predictions)
    residuals = y-(states+RENDER_OFFSET)@C.T
    fit = splits == 'fit'; cal = splits == 'calibration'; audit = splits == 'audit'
    model = fit_scale(states[fit], residuals[fit], low, high)
    quantile = calibrate(states[cal], residuals[cal], model)
    if not quantile['finite']:
        raise RuntimeError('too few calibration units for finite 95% box')
    model.update(q=quantile['q'], calibration=quantile, provenance=cfg,
                 guarantee='marginal CP under registered uniform-pose protocol; not policy or trajectory safety')
    (out/'error_model.json').write_text(json.dumps(model, indent=2)+'\n')
    np.savez_compressed(out/'samples.npz', states=states, observations=y, residuals=residuals, splits=splits)
    metrics = dict(config=cfg, calibration=quantile, runtime_seconds=time.monotonic()-started,
                   per_split={}, safety_certificate=False)
    for name in ('fit', 'calibration', 'audit'):
        mask = splits == name; r = radius(states[mask], model)
        metrics['per_split'][name] = dict(count=int(mask.sum()),
            jointly_covered=int((joint_scores(states[mask], residuals[mask], model) <= model['q']).sum()),
            mean_radius=r.mean(0).tolist(), max_radius=r.max(0).tolist(),
            latent_rmse=float(np.sqrt(np.mean(residuals[mask]**2))))
    (out/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    print(json.dumps(metrics['per_split'], indent=2), flush=True)


if __name__ == '__main__':
    main()
