"""Run the original parking-lot rollout with local weights and extra diagnostics.

The controller, dynamics, image preprocessing, rendering and disturbances are
unchanged. The source is instrumented in memory; upstream files are not edited.
Run under xvfb-run to retain the author's GUI/OpenGL rendering path.
"""
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
parser.add_argument('--seed',type=int,default=42)
parser.add_argument('--random-only',action='store_true')
parser.add_argument('--plan',type=Path)
parser.add_argument('--state-cp-model',type=Path)
args = parser.parse_args()
assert args.rollouts > 0
args.source = args.source.resolve()
args.output = args.output.resolve()
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
if args.state_cp_model is not None:
    from state_cp_model import install_state_cp
    install_state_cp(args.source,args.state_cp_model.resolve())

torch.set_num_threads(4)
torch.set_num_interop_threads(1)
assert args.weights.stat().st_size == 4546108579, 'Checkpoint download is incomplete'


def load_local_dino(model_type='vitg14'):
    assert model_type == 'vitg14'
    model = torch.hub.load(str(args.dinov2_source), 'dinov2_vitg14',
                           source='local', pretrained=False)
    model.load_state_dict(torch.load(args.weights, map_location='cpu'), strict=True)
    model.eval()
    model.patch_embed.forward = dino.patch_embed_forward.__get__(model.patch_embed)
    return model


dino.load_dino_v2_model = load_local_dino
records = []
video_frames = []
started = time.perf_counter()


def capture_step(rollout, t, states, observations, u, process_noise, m, nominal, offset, rgb):
    source_index = 0 if t == 1 else t - 2
    source_state = states[:, source_index].copy()
    observed = observations[:, t - 1].copy()
    residual = observed - m.C @ (source_state + offset)
    bound = float(m.get_disturbance(source_state))
    records.append((u.copy(), observed, source_state, residual, bound, process_noise.copy()))
    if rollout == 0:
        frame = np.asarray(rgb, dtype=np.uint8).copy()
        imageio.imwrite(args.output / 'frames' / f'{t-1:03d}.png', frame)
        video_frames.append(frame)
    if t == states.shape[1] - 1:
        print(f'completed rollout {rollout+1}/{args.rollouts} '
              f'at {time.perf_counter()-started:.1f}s', flush=True)


entry = args.source / 'expe/parking_lot/run_parking_lot_dino.py'
source = entry.read_text()
original_sha = hashlib.sha256(source.encode()).hexdigest()


def replace_once(old, new):
    global source
    assert source.count(old) == 1, f'Expected exactly one source anchor: {old!r}'
    source = source.replace(old, new, 1)


replace_once('ROLLOUT_NUM = 50', f'ROLLOUT_NUM = {args.rollouts}')
replace_once('np.random.seed(42)',f'np.random.seed({args.seed})')
if args.plan:
    replace_once('data = np.load(os.path.join(base_path, f"data/unicycle_of_perception_{eval_idx}.npz"))',
                 'data = np.load('+repr(str(args.plan.resolve()))+')')
if args.random_only:
    start=source.index('    if rollout == 0: # some adversarial disturbance')
    end=source.index('    for t in range(1, N+1):',start)
    source=source[:start]+('    for i in range(N):\n'
        '        disturbance_state[:,i] = np.random.random(m.nx)-0.5\n'
        '        disturbance_state[:,i] /= np.linalg.norm(disturbance_state[:,i])\n\n')+source[end:]
replace_once(
    '        u = primal_u[:,t-1] + delta_u',
    '        u = primal_u[:,t-1] + delta_u\n'
    '        capture_step(rollout, t, states, observations, u, '
    'disturbance_state[:,t-1], m, primal_x, state_offset, rgb_im)'
)
namespace = {'__name__': '__main__', '__file__': str(entry), 'capture_step': capture_step}
exec(compile(source, str(entry), 'exec'), namespace)
wall_time = time.perf_counter() - started
states = namespace['rollout_states']
nominal = namespace['primal_x']
nominal_input = namespace['primal_u']
R, nx, times = states.shape
N = times - 1
nu = nominal_input.shape[0]
controls = np.stack([r[0] for r in records]).reshape(R, N, nu).transpose(0, 2, 1)
observations = np.stack([r[1] for r in records]).reshape(R, N, -1).transpose(0, 2, 1)
observation_states = np.stack([r[2] for r in records]).reshape(R, N, nx).transpose(0, 2, 1)
residuals = np.stack([r[3] for r in records]).reshape(R, N, -1).transpose(0, 2, 1)
bounds = np.array([r[4] for r in records]).reshape(R, N)
disturbances = np.stack([r[5] for r in records]).reshape(R, N, nx).transpose(0, 2, 1)
assert np.isfinite(states).all() and np.isfinite(controls).all()
assert len(records) == R * N

xmin, xmax = namespace['x_min'], namespace['x_max']
umin, umax = namespace['u_min'], namespace['u_max']
fmin, fmax = namespace['x_min_f'], namespace['x_max_f']
path_ok = ((states[:, :, :-1] >= xmin[None, :, None] - 1e-3)
           & (states[:, :, :-1] <= xmax[None, :, None] + 1e-3)).all(axis=(1, 2))
terminal_ok = ((states[:, :, -1] >= fmin[None] - 1e-3)
               & (states[:, :, -1] <= fmax[None] + 1e-3)).all(axis=1)
input_ok = ((controls >= umin[None, :, None] - 1e-3)
            & (controls <= umax[None, :, None] + 1e-3)).all(axis=(1, 2))
centers = namespace['obstacle_centers']
radii = np.sqrt(namespace['obstacle_r2'])
xy = states[:, :2, :].transpose(0, 2, 1)
distance_squared = np.sum((xy[:, :, None] - centers[None, None])**2, axis=-1)
clearance = np.sqrt(distance_squared) - radii[None, None]
collision = (distance_squared <= namespace['obstacle_r2'][None, None] + 1e-4).any(axis=(1, 2))
all_constraints_ok = path_ok & terminal_ok & input_ok & ~collision
plan = np.load(args.plan if args.plan else args.source / 'expe/parking_lot/data/unicycle_of_perception_0.npz')
state_halfwidth = np.concatenate([plan['backoff'][:, :nx], plan['backoff_f'][None, :nx]])
input_halfwidth = plan['backoff'][:, nx:nx + nu]
state_excess = np.abs(states - nominal[None]) - state_halfwidth.T[None]
input_excess = np.abs(controls - nominal_input[None]) - input_halfwidth.T[None]
residual_norm = np.max(np.abs(residuals), axis=1)
coverage = residual_norm <= bounds + 1e-8
l2_covered=np.linalg.norm(residuals,axis=1)<=bounds+1e-8
target=np.array([.25,.5,0.,0.])
target_distances=np.linalg.norm(states[:,:2,-1]-target[None,:2],axis=1)
np.savez_compressed(args.output / 'diagnostic_traces.npz',
                    states=states, inputs=controls, observations=observations,
                    observation_states=observation_states, residuals=residuals,
                    error_bounds=bounds, disturbances=disturbances,
                    nominal_states=nominal, nominal_inputs=nominal_input,
                    state_halfwidth=state_halfwidth, input_halfwidth=input_halfwidth,
                    target=target,terminal_position_distances=target_distances)
summary = {
    'experiment': 'VISION-SLS parking lot, initial condition 0',
    'source_revision': '89b5896a326a57dc8bd7edf7e93f13105a47cdd4',
    'original_rollout_source_sha256': original_sha,
    'host': platform.node(), 'gpu': torch.cuda.get_device_name(0),
    'rendering': 'author GUI/OpenGL renderer under Xvfb',
    'model': 'dinov2_vitg14, original float32 and original 224 preprocessing',
    'rollouts': R, 'steps_per_rollout': N, 'wall_seconds': wall_time,
    'seed':args.seed,'random_only':args.random_only,'state_cp_model':str(args.state_cp_model) if args.state_cp_model else None,
    'own_bound_l2_point_coverage':float(l2_covered.mean()),
    'own_bound_l2_trajectories_covered':int(l2_covered.all(axis=1).sum()),
    'terminal_position_distance_mean_m':float(target_distances.mean()),
    'terminal_position_distance_max_m':float(target_distances.max()),
    'author_successes': int(namespace['success']),
    'author_constraint_violations': int(namespace['constr_violation']),
    'full_state_input_collision_constraint_violations': int((~all_constraints_ok).sum()),
    'input_constraint_violating_rollouts': int((~input_ok).sum()),
    'collision_rollouts': int(collision.sum()),
    'minimum_sampled_obstacle_clearance_m': float(clearance.min()),
    'state_tube_violating_rollouts_tolerance_1e-6': int((state_excess > 1e-6).any(axis=(1, 2)).sum()),
    'input_tube_violating_rollouts_tolerance_1e-6': int((input_excess > 1e-6).any(axis=(1, 2)).sum()),
    'max_state_tube_excess': float(state_excess.max()),
    'max_input_tube_excess': float(input_excess.max()),
    'actual_absolute_perception_bound_coverage': float(coverage.mean()),
    'perception_bound_violating_observations': int((~coverage).sum()),
    'maximum_perception_residual_over_bound': float(np.max(residual_norm / np.maximum(bounds, 1e-12))),
    'terminal_states': states[:, :, -1].tolist(),
    'versions': {name: importlib.metadata.version(name) for name in
                 ['torch', 'torchvision', 'numpy', 'scipy', 'drake', 'casadi', 'numba', 'pybullet']},
    'limitations': ['One initial condition; not all 15 initial conditions in the paper.',
                    'No new representation training; direct CP replaces the envelope when enabled.',
                    'Original solver omits nonlinear remainder and uses Euclidean tightening.',
                    'Collision checks are at discrete trajectory samples.'],
}
(args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
with imageio.get_writer(args.output / 'first_rollout.mp4', fps=1 / .15, codec='libx264') as writer:
    for frame in video_frames:
        writer.append_data(frame)

fig, ax = plt.subplots(figsize=(8, 8))
for center, radius in zip(centers, radii):
    ax.add_patch(plt.Circle(center, radius, color='#dd5555', alpha=.75))
for k in range(times):
    half = state_halfwidth[k]
    ax.add_patch(plt.Rectangle(nominal[:2, k] - half[:2], 2*half[0], 2*half[1],
                               facecolor='#60ba74', edgecolor='#399650', alpha=.22))
for r in range(R):
    ax.plot(states[r, 0], states[r, 1], color='#1769aa', alpha=.3, linewidth=.8)
ax.plot(nominal[0], nominal[1], color='#168039', linewidth=2, label='Nominal plan')
ax.scatter(nominal[0, 0], nominal[1, 0], marker='o', c='black', label='Start')
ax.add_patch(plt.Rectangle(fmin[:2], *(fmax[:2]-fmin[:2]),
                           fill=False, edgecolor='black', linewidth=1.5, label='Goal set'))
ax.set(xlabel='x (m)', ylabel='y (m)', title=f'VISION-SLS: {R} visual closed-loop rollouts')
ax.set_aspect('equal', adjustable='datalim')
ax.grid(alpha=.25)
ax.legend()
fig.tight_layout()
fig.savefig(args.output / 'rollout_trajectories.png', dpi=180)
plt.close(fig)
print(json.dumps(summary, indent=2), flush=True)
