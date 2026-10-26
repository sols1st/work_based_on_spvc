"""Instrument the original quadrotor experiment without editing upstream files."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--weights', type=Path, required=True)
parser.add_argument('--dinov2-source', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--rollouts', type=int, default=50)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()
args.source, args.output = args.source.resolve(), args.output.resolve()
assert args.rollouts > 0
args.output.mkdir(parents=True, exist_ok=True)
(args.output / 'data').mkdir(exist_ok=True)
(args.output / 'frames').mkdir(exist_ok=True)
os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.insert(0, str(args.source))
os.chdir(args.output)
import numpy as np
import torch
import imageio.v2 as imageio
from matplotlib import pyplot as plt
import perception_utils.eval_dino as dino

torch.set_num_threads(4)
torch.set_num_interop_threads(1)
np.random.seed(args.seed)
assert args.weights.stat().st_size == 1217586395
def load_local_dino(model_type='vitl14'):
    assert model_type == 'vitl14'
    model = torch.hub.load(str(args.dinov2_source), 'dinov2_vitl14',
                           source='local', pretrained=False)
    model.load_state_dict(torch.load(args.weights, map_location='cpu'), strict=True)
    model.eval()
    model.patch_embed.forward = dino.patch_embed_forward.__get__(model.patch_embed)
    return model
dino.load_dino_v2_model = load_local_dino
records, video_frames = [], []
started = time.perf_counter()
def capture_step(rollout, t, states, observations, u, disturbance, m, rgb):
    index = 0 if t == 1 else t - 2
    x = states[:, index].copy()
    y = observations[:, t - 1].copy()
    residual = y - m.C @ x
    bound = np.broadcast_to(np.asarray(m.get_disturbance(x)).reshape(-1), (m.ny,)).copy()
    records.append((u.copy(), y, x, residual, bound, disturbance.copy()))
    if rollout == 0:
        frame = np.asarray(rgb, dtype=np.uint8).copy()
        imageio.imwrite(args.output / 'frames' / f'{t-1:03d}.png', frame)
        video_frames.append(frame)
    if t == states.shape[1] - 1:
        covered = [np.all(np.abs(r[3]) <= r[4] + 1e-8) for r in records]
        progress = {'completed_observation_sequences': rollout + 1,
                    'observations': len(records), 'joint_bound_coverage': float(np.mean(covered)),
                    'wall_seconds': time.perf_counter() - started}
        (args.output / 'progress.json').write_text(json.dumps(progress, indent=2) + '\n')
        print(f'completed rollout {rollout+1}/{args.rollouts} at '
              f'{progress["wall_seconds"]:.1f}s; joint bound coverage '
              f'{progress["joint_bound_coverage"]:.3f}', flush=True)

entry = args.source / 'expe/quad_perception/run_quad_dino.py'
source = entry.read_text()
original_sha = hashlib.sha256(source.encode()).hexdigest()
def replace_once(old, new):
    global source
    assert source.count(old) == 1, old
    source = source.replace(old, new, 1)
replace_once('ROLLOUT_NUM = 50', f'ROLLOUT_NUM = {args.rollouts}')
replace_once('data = np.load(f"data/quad10d_perception_{eval_idx}.npz")',
             'data = np.load(os.path.join(base_path, f"data/quad10d_perception_{eval_idx}.npz"))')
replace_once('        u = primal_u[:,t-1] + delta_u',
             '        u = primal_u[:,t-1] + delta_u\n'
             '        capture_step(rollout, t, states, observations, u, '
             'disturbance_state[:,t-1], m, rgb_im)')
ns = {'__name__': '__main__', '__file__': str(entry), 'capture_step': capture_step}
exec(compile(source, str(entry), 'exec'), ns)
wall = time.perf_counter() - started
states, nominal, nominal_u, m = [ns[k] for k in ['rollout_states', 'primal_x', 'primal_u', 'm']]
R, nx, times = states.shape
N, nu, ny = times - 1, m.nu, m.ny
assert len(records) == R * N
def stack(index, width):
    return np.stack([r[index] for r in records]).reshape(R, N, width).transpose(0, 2, 1)
controls, observations, observation_states = stack(0, nu), stack(1, ny), stack(2, nx)
residuals, bounds, disturbances = stack(3, ny), stack(4, ny), stack(5, nx)
assert all(np.isfinite(a).all() for a in [states, controls, observations, residuals, bounds])
xmin, xmax = ns['x_min'], ns['x_max']
fmin, fmax = ns['x_min_f'], ns['x_max_f']
umax, umin = m.g[nx:nx+nu], -m.g[2*nx+nu:2*(nx+nu)]
path_ok = ((states[:, :, :-1] >= xmin[None, :, None] - 1e-3)
           & (states[:, :, :-1] <= xmax[None, :, None] + 1e-3)).all(axis=(1, 2))
terminal_ok = ((states[:, :, -1] >= fmin[None] - 1e-3)
               & (states[:, :, -1] <= fmax[None] + 1e-3)).all(axis=1)
input_ok = ((controls >= umin[None, :, None] - 1e-3)
            & (controls <= umax[None, :, None] + 1e-3)).all(axis=(1, 2))
centers, r2 = ns['obstacle_centers'], ns['obstacle_r2']
xy = states[:, :2, :].transpose(0, 2, 1)
d2 = np.sum((xy[:, :, None] - centers[None, None])**2, axis=-1)
clearance = np.sqrt(d2) - np.sqrt(r2)[None, None]
collision = (d2 <= r2[None, None] + 1e-4).any(axis=(1, 2))
plan = np.load(args.source / 'expe/quad_perception/data/quad10d_perception_0.npz')
state_half = np.vstack([plan['backoff'][:, :nx], plan['backoff_f'][None, :nx]])
input_half = plan['backoff'][:, nx:nx+nu]
state_excess = np.abs(states - nominal[None]) - state_half.T[None]
input_excess = np.abs(controls - nominal_u[None]) - input_half.T[None]
component_covered = np.abs(residuals) <= bounds + 1e-8
joint_covered = component_covered.all(axis=1)
ratio = np.abs(residuals) / np.maximum(bounds, 1e-12)
target = ns['xG']
terminal_distance = np.linalg.norm(states[:, :3, -1] - target[None, :3], axis=1)
np.savez_compressed(args.output / 'diagnostic_traces.npz', states=states, inputs=controls,
                    observations=observations, observation_states=observation_states,
                    residuals=residuals, error_bounds=bounds, disturbances=disturbances,
                    nominal_states=nominal, nominal_inputs=nominal_u,
                    state_halfwidth=state_half, input_halfwidth=input_half, target=target,
                    terminal_position_distances=terminal_distance)
summary = {
    'experiment': 'VISION-SLS quadrotor visual obstacle avoidance, initial condition 0',
    'host': platform.node(), 'gpu': torch.cuda.get_device_name(0),
    'source_revision': '89b5896a326a57dc8bd7edf7e93f13105a47cdd4',
    'original_rollout_source_sha256': original_sha,
    'rendering': 'Original GUI/OpenGL path under Xvfb, Mesa software rendering',
    'model': 'dinov2_vitl14, float32, original 224 preprocessing, 1024 CLS features',
    'seed': args.seed, 'rollouts': R, 'steps_per_rollout': N, 'dt': m.dt,
    'wall_seconds': wall, 'state_dimension': nx, 'input_dimension': nu, 'observation_dimension': ny,
    'author_successes': int(ns['success']), 'author_constraint_violations': int(ns['constr_violation']),
    'full_state_input_collision_constraint_violations': int((~(path_ok & terminal_ok & input_ok & ~collision)).sum()),
    'input_constraint_violating_rollouts': int((~input_ok).sum()), 'collision_rollouts': int(collision.sum()),
    'minimum_sampled_obstacle_clearance_m': float(clearance.min()),
    'target_position': target[:3].tolist(),
    'nominal_terminal_position': nominal[:3, -1].tolist(),
    'terminal_position_distance_mean_m': float(terminal_distance.mean()),
    'terminal_position_distance_max_m': float(terminal_distance.max()),
    'within_0_25m_of_target_and_no_collision': int(((terminal_distance <= .25) & ~collision).sum()),
    'terminal_constraint_lower': fmin.tolist(), 'terminal_constraint_upper': fmax.tolist(),
    'state_tube_violating_rollouts_tolerance_1e-6': int((state_excess > 1e-6).any(axis=(1,2)).sum()),
    'input_tube_violating_rollouts_tolerance_1e-6': int((input_excess > 1e-6).any(axis=(1,2)).sum()),
    'max_state_tube_excess': float(state_excess.max()), 'max_input_tube_excess': float(input_excess.max()),
    'joint_absolute_perception_bound_coverage': float(joint_covered.mean()),
    'per_component_absolute_perception_bound_coverage': component_covered.mean(axis=(0,2)).tolist(),
    'perception_bound_violating_observations': int((~joint_covered).sum()),
    'max_normalized_absolute_residual': float(ratio.max()),
    'normalized_residual_l2_le_1_fraction': float((np.linalg.norm(ratio, axis=1) <= 1+1e-8).mean()),
    'minimum_bound': float(bounds.min()),
    'average_original_image_pipeline_seconds': float(ns['avg_inference_time']),
    'terminal_states': states[:, :, -1].tolist(),
    'versions': {k: importlib.metadata.version(k) for k in
                 ['torch','torchvision','numpy','scipy','drake','casadi','numba','pybullet']},
    'limitations': ['One of 15 initial conditions; no representation or envelope retraining.',
                    'Original success predicate uses broad terminal position bounds, not target distance.',
                    '0.25 m target check is an added diagnostic, not the paper criterion.',
                    'Collision checks at discrete times against the five original XY circles.',
                    'Original solver omits nonlinear remainder; empirical results are not a safety proof.',
                    'Original quad script was unseeded; this reproduction sets NumPy seed 42.'],
}
(args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
with imageio.get_writer(args.output / 'first_rollout.mp4', fps=1/m.dt, codec='libx264') as writer:
    for frame in video_frames:
        writer.append_data(frame)
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
ax = axes[0]
for center, radius in zip(centers, np.sqrt(r2)):
    ax.add_patch(plt.Circle(center, radius, color='#d85b5b', alpha=.7))
for k in range(times):
    ax.add_patch(plt.Rectangle(nominal[:2,k]-state_half[k,:2], 2*state_half[k,0],
                               2*state_half[k,1], facecolor='#60ba74', alpha=.2))
for path in states:
    ax.plot(path[0], path[1], color='#1769aa', alpha=.25, linewidth=.7)
ax.plot(nominal[0], nominal[1], color='#168039', linewidth=2, label='Nominal')
ax.scatter(*nominal[:2,0], c='black', label='Start')
ax.scatter(*target[:2], c='#bd6900', marker='*', s=130, label='Target')
ax.set(xlabel='x (m)', ylabel='y (m)', title=f'{R} image-based quadrotor rollouts')
ax.set_aspect('equal', adjustable='datalim')
ax.legend()
t = np.arange(times)*m.dt
for path in states:
    axes[1].plot(t, path[2], color='#1769aa', alpha=.25, linewidth=.7)
axes[1].plot(t, nominal[2], color='#168039', linewidth=2, label='Nominal altitude')
axes[1].axhline(target[2], color='#bd6900', linestyle='--', label='Target altitude')
axes[1].set(xlabel='Time (s)', ylabel='z (m)', title='Altitude')
axes[1].legend()
for ax in axes:
    ax.grid(alpha=.25)
fig.tight_layout()
fig.savefig(args.output / 'rollout_trajectories.png', dpi=170)
plt.close(fig)
print(json.dumps({k:v for k,v in summary.items() if k != 'terminal_states'}, indent=2), flush=True)
