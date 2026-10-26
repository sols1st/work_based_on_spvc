"""Minimal proxy PPO training and paired actual-image parking evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def evaluate(model, sensor=None):
    from .ppo_env import ParkingPPOEnv
    rows, paths, actions = [], [], []
    env = ParkingPPOEnv(image_sensor=sensor)
    try:
        for scenario in range(15):
            obs, _ = env.reset(options={'scenario': scenario})
            states, us, rewards = [env.task.state.copy()], [], []
            while True:
                u, _ = model.predict(obs, deterministic=True)
                obs, r, _, done, info = env.step(u)
                states.append(env.task.state.copy())
                us.append(info['action_applied'])
                rewards.append(r)
                if done:
                    break
            row = dict(scenario=scenario, success=info['episode_success'],
                       collision=info['collision_seen'], out_of_bounds=info['out_of_bounds_seen'],
                       goal_at_horizon=info['goal_at_horizon'],
                       final_goal_distance=info['final_goal_distance'],
                       return_sum=float(sum(rewards)), steps=len(us))
            if sensor is not None:
                row.update(rendered_frames=sensor.calls,
                           outside_adaptation_domain=sensor.outside_adaptation_domain)
            rows.append(row)
            paths.append(states)
            actions.append(us)
            print('path=%s scenario=%02d success=%s collision=%s bounds=%s distance=%.3f' %
                  ('image' if sensor else 'Cx', scenario, row['success'], row['collision'],
                   row['out_of_bounds'], row['final_goal_distance']), flush=True)
    finally:
        env.close()
    return dict(episodes=15, success_count=sum(r['success'] for r in rows),
                collision_count=sum(r['collision'] for r in rows),
                out_of_bounds_count=sum(r['out_of_bounds'] for r in rows),
                goal_at_horizon_count=sum(r['goal_at_horizon'] for r in rows),
                mean_final_goal_distance=float(np.mean([r['final_goal_distance'] for r in rows])),
                per_scenario=rows), np.asarray(paths), np.asarray(actions)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--timesteps', type=int, default=300000)
    p.add_argument('--load', help='Resume PPO training without changing task or perception')
    p.add_argument('--evaluate-only', action='store_true')
    p.add_argument('--image-eval', action='store_true')
    p.add_argument('--vision-root', default='../VISION-SLS')
    p.add_argument('--dino-repo', default='../external/dinov2')
    p.add_argument('--dino-weights', default='../external/dinov2_weights/dinov2_vitg14_pretrain.pth')
    p.add_argument('--projection', default='results/vision_sls_parking_20261025_adaptation_v1/projection_adapted.pt')
    args = p.parse_args()
    out = Path(args.output_dir)
    if out.exists() or args.timesteps < 1 or (args.evaluate_only and not args.load):
        p.error('new output directory / positive timesteps / checkpoint for evaluate-only required')
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import BaseCallback
    from .ppo_env import ParkingPPOEnv, FrozenImageSensor
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out.mkdir(parents=True)
    cfg = vars(args).copy()
    cfg.update(seed=7, delay=1, history_measurements=3, history_actions=2,
               horizon=30, dt=.15, true_speed_in_observation=False,
               true_position_in_observation=False, process_disturbance='zero',
               training_path='analytic Cx only', task_starts='original 15, no jitter',
               projection_sha256=sha(args.projection),
               status='connection/development diagnostic, not a safety certificate')
    (out/'config.json').write_text(json.dumps(cfg, indent=2)+'\n')
    started = time.monotonic()
    vec = make_vec_env(ParkingPPOEnv, n_envs=8, seed=7)
    class Progress(BaseCallback):
        def __init__(self):
            super().__init__()
            self.next_report = 50000
        def _on_step(self):
            if self.num_timesteps >= self.next_report:
                print('PPO steps=%d elapsed=%.1fs' %
                      (self.num_timesteps, time.monotonic()-started), flush=True)
                self.next_report += 50000
            return True
    try:
        if args.load:
            model = PPO.load(args.load, env=vec, device='cpu')
        else:
            model = PPO('MlpPolicy', vec, device='cpu', seed=7, n_steps=256,
                        batch_size=256, n_epochs=10, learning_rate=3e-4,
                        gamma=.99, gae_lambda=.95, ent_coef=.005,
                        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
                        verbose=0)
        if not args.evaluate_only:
            model.learn(args.timesteps, callback=Progress(), reset_num_timesteps=False)
        model.save(str(out/'ppo.zip'))
    finally:
        vec.close()
    frozen_actor = sha(out/'ppo.zip')
    metrics = dict(config=cfg, trained_total_timesteps=int(model.num_timesteps),
                   checkpoint_sha256=frozen_actor, training_seconds=time.monotonic()-started)
    data = {}
    metrics['proxy'], data['proxy_states'], data['proxy_actions'] = evaluate(model)
    (out/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    np.savez_compressed(out/'rollouts.npz', **data)
    if args.image_eval:
        sensor = FrozenImageSensor(args.vision_root, args.dino_repo, args.dino_weights,
                                   args.projection)
        try:
            metrics['image'], data['image_states'], data['image_actions'] = evaluate(model, sensor)
        finally:
            sensor.close()
        metrics['projection_unchanged'] = sha(args.projection) == cfg['projection_sha256']
        metrics['checkpoint_unchanged_during_evaluation'] = sha(out/'ppo.zip') == frozen_actor
    metrics['runtime_seconds'] = time.monotonic()-started
    (out/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    np.savez_compressed(out/'rollouts.npz', **data)
    print('[Parking PPO, unchanged 30-step task, no privileged policy state]')
    for name in ('proxy', 'image'):
        if name in metrics:
            m = metrics[name]
            print('%s: success=%d/15 collision=%d/15 out_of_bounds=%d/15 distance=%.4fm' %
                  (name, m['success_count'], m['collision_count'], m['out_of_bounds_count'],
                   m['mean_final_goal_distance']))
    print('metrics: %s' % (out/'metrics.json'), flush=True)


if __name__ == '__main__':
    main()
