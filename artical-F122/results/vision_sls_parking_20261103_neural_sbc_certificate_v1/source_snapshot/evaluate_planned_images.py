"""Matched original/perturbed-start image ablation for planning and feedback."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from .core import C, RENDER_OFFSET, INITIAL_STATES, OBSTACLES
from .ppo_env import FrozenImageSensor, ParkingPPOEnv
from .planned_env import evaluate_planned, perturbed_initial_state
from .train_migration import load_ppo
from .train_ppo import sha


def evaluate_legacy(policy, sensor, jitter, seed):
    env = ParkingPPOEnv(image_sensor=sensor)
    rows, paths, commands = [], [], []
    try:
        for scenario in range(15):
            initial = perturbed_initial_state(scenario, np.random.default_rng(seed+scenario), jitter)
            obs, _ = env.reset(options=dict(scenario=scenario, state=initial))
            states, actions = [initial.copy()], []
            for _ in range(30):
                obs, _, _, _, info = env.step(policy.predict(obs, deterministic=True)[0])
                states.append(env.task.state.copy()); actions.append(info['action_applied'])
            rows.append(dict(scenario=scenario, success=info['episode_success'],
                             collision=info['collision_seen'], out_of_bounds=info['out_of_bounds_seen'],
                             final_goal_distance=info['final_goal_distance']))
            paths.append(states); commands.append(actions)
    finally:
        env.close()
    return dict(episodes=15, success_count=sum(r['success'] for r in rows),
                collision_count=sum(r['collision'] for r in rows),
                out_of_bounds_count=sum(r['out_of_bounds'] for r in rows),
                mean_final_goal_distance=float(np.mean([r['final_goal_distance'] for r in rows])),
                per_scenario=rows), dict(states=np.asarray(paths), actions=np.asarray(commands))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--references', required=True); p.add_argument('--output-dir', required=True)
    p.add_argument('--modes', nargs='+', choices=['open_loop', 'feedback', 'residual', 'legacy'],
                   default=['open_loop', 'feedback'])
    p.add_argument('--residual-policy'); p.add_argument('--legacy-policy')
    p.add_argument('--jitters', nargs='+', type=float, default=[0., .04]); p.add_argument('--seed', type=int, default=701)
    p.add_argument('--projection', default='results/vision_sls_parking_20261025_adaptation_v1/projection_adapted.pt')
    p.add_argument('--vision-root', default='../VISION-SLS'); p.add_argument('--dino-repo', default='../external/dinov2')
    p.add_argument('--dino-weights', default='../external/dinov2_weights/dinov2_vitg14_pretrain.pth')
    a = p.parse_args(); out = Path(a.output_dir)
    if out.exists() or any(not np.isfinite(j) or j < 0 for j in a.jitters):
        p.error('new output directory and nonnegative finite jitters required')
    for mode in ('residual', 'legacy'):
        if mode in a.modes and not getattr(a, mode+'_policy'):
            p.error(mode+' requires its policy checkpoint')
    torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out.mkdir(parents=True)
    cfg = dict(vars(a), references_sha256=sha(a.references), projection_sha256=sha(a.projection),
               dino_sha256=sha(a.dino_weights), independent_test=False, safety_certificate=False,
               perturbations='independent uniform bounds on x,y,heading,velocity; reject initially unsafe',
               reference_selection='initial image measurement plus commanded goal only',
               physics='unchanged RK4, DT=.15, H=30, one-frame delay, zero process noise')
    for mode in ('residual', 'legacy'):
        path = getattr(a, mode+'_policy')
        if path:
            cfg[mode+'_sha256'] = sha(path)
    (out/'config.json').write_text(json.dumps(cfg, indent=2)+'\n')
    class Recorder(FrozenImageSensor):
        def __call__(self, state):
            y = super().__call__(state)
            records_x.append(state.copy()); records_y.append(y.copy())
            return y
    sensor = Recorder(a.vision_root, a.dino_repo, a.dino_weights, a.projection)
    started = time.monotonic(); results = {}
    try:
        for mode in a.modes:
            path = getattr(a, mode+'_policy', None)
            policy = load_ppo(path, device='cpu') if path else None
            for jitter in a.jitters:
                records_x, records_y = [], []
                if mode == 'legacy':
                    nominal, _ = evaluate_legacy(policy, None, jitter, a.seed)
                    actual, data = evaluate_legacy(policy, sensor, jitter, a.seed)
                else:
                    controller_mode = 'open_loop' if mode == 'open_loop' else 'feedback'
                    nominal, _ = evaluate_planned(a.references, policy=policy, mode=controller_mode,
                                                   jitter=jitter, seed=a.seed)
                    actual, data = evaluate_planned(a.references, policy=policy, sensor=sensor,
                                                     mode=controller_mode, jitter=jitter, seed=a.seed)
                capture, y = np.asarray(records_x), np.asarray(records_y)
                errors = y-(capture+RENDER_OFFSET)@C.T
                key = mode+'_jitter_'+str(jitter)
                np.savez_compressed(out/(key+'.npz'), **data, capture_states=capture,
                                    capture_observations=y, residuals=errors)
                clearance = np.linalg.norm(data['states'][:, :, None, :2]-OBSTACLES[None, None, :, :2], axis=-1)-OBSTACLES[:, 2]
                results[key] = dict(nominal=nominal, image=actual, latent_rmse=float(np.sqrt(np.mean(errors**2))),
                                    minimum_clearance=float(clearance.min()))
                if 'estimates' in data:
                    results[key]['position_estimate_rmse'] = float(np.sqrt(np.mean(
                        np.sum((data['states'][..., :2]-data['estimates'][..., :2])**2, axis=-1))))
                (out/'metrics.json').write_text(json.dumps(results, indent=2)+'\n')
                print(key, {k:v for k,v in actual.items() if k!='per_scenario'}, flush=True)
    finally:
        sensor.close()
    (out/'summary.json').write_text(json.dumps(dict(config=cfg, results=results,
        runtime_seconds=time.monotonic()-started, projection_unchanged=sha(a.projection)==cfg['projection_sha256']), indent=2)+'\n')


if __name__ == '__main__':
    main()
