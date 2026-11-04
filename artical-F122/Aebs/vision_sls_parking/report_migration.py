"""Audit and package a parking development candidate without certifying it."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from .core import C, RENDER_OFFSET, OBSTACLES, STATE_LOW, STATE_HIGH, GOAL_LOW, GOAL_HIGH
from .train_ppo import sha
from .report_ppo import audit_rollouts
from .perception_error import joint_scores


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy-dir', required=True)
    p.add_argument('--image-result-dir', required=True)
    p.add_argument('--barrier-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--projection-dir', default='results/vision_sls_parking_20261025_adaptation_v1')
    p.add_argument('--error-dir', default='results/vision_sls_parking_20261102_error_v1')
    p.add_argument('--baseline-dir', default='results/vision_sls_parking_20261102_image_eval_v1')
    a = p.parse_args()
    out = Path(a.output_dir); policy_dir = Path(a.policy_dir); images = Path(a.image_result_dir)
    projection_dir = Path(a.projection_dir); error_dir = Path(a.error_dir); barrier_dir = Path(a.barrier_dir)
    if out.exists():
        p.error('new output directory required')
    image_summary = json.loads((images/'summary.json').read_text())
    candidate = image_summary['results'][policy_dir.name]
    error_model = json.loads((error_dir/'error_model.json').read_text())
    barrier = json.loads((barrier_dir/'metrics.json').read_text())
    actor_sha = sha(policy_dir/'ppo.zip'); projection_sha = sha(projection_dir/'projection_adapted.pt')
    if candidate['checkpoint_sha256'] != actor_sha or barrier['config']['ppo_sha256'] != actor_sha:
        raise ValueError('actor/image/barrier checkpoint mismatch')
    if (image_summary['config']['projection_sha256'] != projection_sha or
            error_model['provenance']['projection_sha256'] != projection_sha or
            barrier['config']['error_model_sha256'] != sha(error_dir/'error_model.json')):
        raise ValueError('perception/calibration/barrier provenance mismatch')
    with np.load(images/(policy_dir.name+'.npz')) as z:
        data = {k:z[k].copy() for k in z.files}
    audit = audit_rollouts({'proxy':candidate['proxy'], 'image':candidate['image']}, data)
    capture = data['capture_states']
    expected = data['image_states'][:, np.maximum(np.arange(31)-1, 0)].reshape(-1, 4)
    np.testing.assert_array_equal(capture, expected)
    np.testing.assert_allclose(data['targets'], (capture+RENDER_OFFSET)@C.T, atol=1e-12)
    np.testing.assert_allclose(data['residuals'], data['observations']-data['targets'], atol=1e-12)
    domain = ((capture[:, :3] >= error_model['low']) & (capture[:, :3] <= error_model['high'])).all(1)
    covered = np.zeros(len(capture), bool)
    covered[domain] = joint_scores(capture[domain], data['residuals'][domain], error_model) <= error_model['q']
    audit.update(delayed_image_state_pairing=True, checkpoint_hashes_match=True,
                 registered_error_domain_frames=int(domain.sum()), total_recorded_observations=len(capture),
                 covered_observations=int(covered.sum()),
                 covered_development_trajectories=int(covered.reshape(15, 31).all(1).sum()),
                 independent_trajectory_test=False, safety_certificate=False)
    failures = []
    for row in candidate['image']['per_scenario']:
        if not row['success']:
            i = row['scenario']; final = data['image_states'][i, -1]
            violation = np.maximum(GOAL_LOW[i]-final, 0)+np.maximum(final-GOAL_HIGH[i], 0)
            failures.append(dict(scenario=i, collision=row['collision'], out_of_bounds=row['out_of_bounds'],
                                 final_state=final.tolist(), goal_boundary_excess=violation.tolist()))
    out.mkdir(parents=True)
    copies = {'ppo.zip':policy_dir/'ppo.zip', 'projection.pt':projection_dir/'projection_adapted.pt',
              'error_model.json':error_dir/'error_model.json', 'sbc_candidate.pt':barrier_dir/'sbc.pt',
              'sbc_diagnostic.json':barrier_dir/'metrics.json'}
    for name, source in copies.items():
        shutil.copy2(source, out/name)
    projection_cfg = json.loads((projection_dir/'config.json').read_text())
    (out/'config.json').write_text(json.dumps(dict(world_domain=projection_cfg['world_domain'],
        projection_sha256=projection_sha, status='development_candidate_not_certified'), indent=2)+'\n')
    summary = dict(status='multidimensional_parking_development_candidate_not_certified',
                   sources=vars(a), nominal=candidate['proxy'], actual_images=candidate['image'],
                   audit=audit, remaining_failed_scenarios=failures,
                   barrier_diagnostic={k:v for k,v in barrier.items() if k not in ('config', 'history')},
                   files={name:sha(out/name) for name in copies},
                   dino_sha256=image_summary['config']['dino_sha256'],
                   safety_certificate=False, certificate_probability=None,
                   scope='original 15 development scenarios, deterministic simulator and fixed renderer')
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (out/'README.md').write_text(
        '# Parking development candidate\n\n'
        'The actor uses the original 16D observation/history input. '
        'The paired SBC is an unverified training candidate; no safety probability is certified.\n\n'
        f'Actual images: {candidate["image"]["success_count"]}/15 success, '
        f'{candidate["image"]["collision_count"]} collision, '
        f'{candidate["image"]["out_of_bounds_count"]} out-of-bounds episodes. '
        'These are reused development tasks, not independent testing.\n\n'
        'From artical-F122 on the GPU server, run (choose a new evaluation output directory):\n\n'
        '```bash\nCUDA_VISIBLE_DEVICES=0 xvfb-run -a /opt/miniconda3/envs/vt/bin/python '
        '-m Aebs.vision_sls_parking.evaluate_images '
        f'--ppo-dirs {out} --projection {out}/projection.pt '
        '--output-dir results/parking_candidate_recheck\n```\n', encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle
    base_name = 'vision_sls_parking_20261025_spvc_backend_v2'
    base_dir = Path(a.baseline_dir)
    base = json.loads((base_dir/'summary.json').read_text())['results'][base_name]['image']
    with np.load(base_dir/(base_name+'.npz')) as z:
        base_states = z['image_states'].copy()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), constrained_layout=True)
    for ax, states, metrics, title in zip(axes,
        [base_states, data['proxy_states'], data['image_states']],
        [base, candidate['proxy'], candidate['image']],
        ['Previous candidate / actual images', 'New candidate / nominal observations', 'New candidate / actual images']):
        for obstacle in OBSTACLES:
            ax.add_patch(Circle(obstacle[:2], obstacle[2], facecolor='#ecc8c2', edgecolor='#a64436'))
        for i, path in enumerate(states):
            color = '#16796b' if metrics['per_scenario'][i]['success'] else '#c17719'
            ax.plot(path[:, 0], path[:, 1], color=color, alpha=.7, lw=1.2)
            ax.scatter(*path[-1, :2], s=18, color=color)
        for i in (0, 1, 7, 11):
            ax.add_patch(Rectangle(GOAL_LOW[i, :2], *(GOAL_HIGH[i, :2]-GOAL_LOW[i, :2]),
                                   facecolor='none', edgecolor='#667788', linestyle='--'))
        ax.set(xlim=(-3.7, .8), ylim=(-2.2, 2.1), xlabel='x (m)', ylabel='y (m)')
        ax.set_aspect('equal'); ax.grid(alpha=.15)
        ax.set_title(title+'\n%d/15 success; %d collision; %d out of bounds' % (
            metrics['success_count'], metrics['collision_count'], metrics['out_of_bounds_count']), fontsize=10)
    fig.suptitle('Original 30-step parking task — development rollouts, not a safety certificate', fontsize=12)
    fig.savefig(out/'trajectory_comparison.png', dpi=170); plt.close(fig)
    print(json.dumps(dict(actual_success=candidate['image']['success_count'], audit=audit,
                          failed_scenarios=failures), indent=2), flush=True)


if __name__ == '__main__':
    main()
