"""Render and dynamics connection test; no DINO/PPO/SBC training or certificate."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np

from . import core
from .env import ParkingEnv


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--vision-sls-root', type=Path,
                   default=Path(__file__).resolve().parents[3] / 'VISION-SLS')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--renderer', choices=['tiny', 'opengl', 'egl', 'none'], default='tiny')
    p.add_argument('--observation-delay', type=int, choices=[0, 1], default=1)
    args = p.parse_args()
    root = args.vision_sls_root.resolve()
    required = ['dyn/unicycle.py', 'dyn/model.py', 'dyn/unicycle_of_perception.py',
                'expe/parking_lot/run_parking_lot_dino.py', 'expe/parking_lot/main_parking_lot.py',
                'LICENSE']
    if args.output_dir.exists() or any(not (root/s).is_file() for s in required):
        p.error('new output directory and complete upstream scene sources required')
    upstream_files = [root/s for s in required] + list((root/'expe/parking_lot/plane_v2').glob('*'))
    hashes = {str(f): sha(f) for f in upstream_files if f.is_file()}
    own_hashes = {str(f): sha(f) for f in Path(__file__).parent.glob('*.py')}
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'config.json').write_text(json.dumps(dict(
        upstream_sha256=hashes, adapter_sha256=own_hashes, dt=core.DT, horizon=core.HORIZON,
        renderer=args.renderer, observation_delay=args.observation_delay,
        process_disturbance='explicit zero; no stochastic law introduced',
        actions='fixed open-loop yaw=.08*sin(t*.2), acceleration=.25; NOT a controller',
        training=False, python=platform.python_version()), indent=2), encoding='utf-8')
    started = time.monotonic()
    renderer, env = None, None
    metrics = dict(status='failed', training=False, safety_certificate=False,
                   DINO_loaded=False, PPO_loaded=False, SBC_loaded=False,
                   full_upstream_controller_reproduced=False,
                   scene_adapter_only=True, renderer=args.renderer,
                   observation_delay=args.observation_delay,
                   collision_definition='sampled state center within planning circles; no swept car body',
                   process_disturbance='zero')
    try:
        if args.renderer != 'none':
            from .renderer import ParkingRenderer
            renderer = ParkingRenderer(root, args.renderer)
        env = ParkingEnv(renderer, observation_delay=args.observation_delay)
        states, observations, actions, indices, scenarios, flags = [], [], [], [], [], []
        outcomes, frames = Counter(), []
        pose_errors, repeat_error, speed_error = [], None, None
        if renderer is not None:
            a = renderer.render(core.INITIAL_STATES[0])
            b = renderer.render(core.INITIAL_STATES[0])
            speed_state = core.INITIAL_STATES[0].copy(); speed_state[3] = 1.
            c = renderer.render(speed_state)
            repeat_error = int(np.max(np.abs(a.astype(int)-b.astype(int))))
            speed_error = int(np.max(np.abs(a.astype(int)-c.astype(int))))
            if repeat_error != 0 or speed_error != 0:
                raise RuntimeError('rendering is not deterministic or static image unexpectedly encodes speed')
        for scenario in range(15):
            obs = env.reset(scenario)
            for t in range(core.HORIZON+1):
                states.append(obs['current_state']); observations.append(obs['observation_state'])
                indices.append(obs['observation_index']); scenarios.append(scenario)
                flags.append(core.constraints(obs['current_state'], scenario))
                if renderer is not None:
                    pose_errors.append(renderer.last_pose_error)
                    if t == 0 or (scenario in (0, 7, 14) and t == core.HORIZON):
                        from PIL import Image
                        name = 'scene_%02d_step_%02d.png' % (scenario, t)
                        Image.fromarray(obs['rgb']).save(args.output_dir/name)
                        if t == 0:
                            frames.append((scenario, obs['rgb']))
                if t == core.HORIZON:
                    break
                action = np.array([.08*np.sin(t*.2), .25])
                obs, done, info = env.step(action)
                actions.append(info['action_applied'])
            outcomes[info['outcome']] += 1
            print('scenario %02d/14: 30 open-loop steps, %s (NOT policy performance)' %
                  (scenario, info['outcome']), flush=True)
        np.savez_compressed(args.output_dir/'samples.npz',
                            states=np.asarray(states), observation_states=np.asarray(observations),
                            observation_indices=np.asarray(indices), scenarios=np.asarray(scenarios),
                            actions=np.asarray(actions).reshape(15, core.HORIZON, 2),
                            nominal_observations=np.array([core.nominal_observation(s) for s in observations]))
        if frames:
            from PIL import Image, ImageDraw
            sheet = Image.new('RGB', (5*256, 3*280), 'white')
            draw = ImageDraw.Draw(sheet)
            for i, (scenario, image) in enumerate(frames):
                col, row = i % 5, i // 5
                sheet.paste(Image.fromarray(image).resize((256, 256)), (col*256, row*280+24))
                draw.text((col*256+4, row*280+4), 'scenario %02d (world x + 3)' % scenario, fill='black')
            sheet.save(args.output_dir/'contact_sheet.png')
        # A second state needs a different image: guards flat/missing rendering.
        image_delta = None
        if renderer is not None:
            a = renderer.render(core.INITIAL_STATES[0])
            b = renderer.render(core.INITIAL_STATES[7])
            image_delta = float(np.abs(a.astype(float)-b.astype(float)).mean())
            if image_delta < .01:
                raise RuntimeError('different physical poses yielded indistinguishable images')
        if any(sha(f) != old for f, old in hashes.items()):
            raise RuntimeError('upstream files changed during smoke test')
        metrics.update(status='scene_connection_pass' if renderer else 'dynamics_only_pass',
                       trajectories=15, dynamics_steps=len(actions), saved_state_rows=len(states),
                       sampled_constraint_violations=sum(f['collision'] or not f['in_bounds'] for f in flags),
                       open_loop_outcomes=dict(outcomes),
                       renderer_pose_max_error_m=max(pose_errors) if pose_errors else None,
                       repeated_pose_max_pixel_difference=repeat_error,
                       speed_only_max_pixel_difference=speed_error,
                       different_pose_mean_pixel_difference=image_delta,
                       actual_rgb_rendered=renderer is not None,
                       upstream_sources_unchanged=True,
                       visual_review='pending human inspection of contact_sheet.png',
                       renderer_pixel_equivalent_to_upstream=False if args.renderer == 'tiny' else None,
                       next_step='inspect scene; only then connect frozen DINO/projection')
    except BaseException as exc:
        metrics['error'] = '%s: %s' % (type(exc).__name__, exc)
        raise
    finally:
        if env is not None:
            env.close()
        elif renderer is not None:
            renderer.close()
        metrics['runtime_seconds'] = time.monotonic()-started
        (args.output_dir/'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print('[VISION-SLS parking scene connection - no training]')
    print('status=%s trajectories=15 steps=%d renderer=%s delay=%d' %
          (metrics['status'], metrics['dynamics_steps'], args.renderer, args.observation_delay))
    print('No PPO performance result or SBC certificate. Inspect contact_sheet.png.')
    print('metrics: %s' % (args.output_dir/'metrics.json'))


if __name__ == '__main__': main()
