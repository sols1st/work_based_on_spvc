"""Resume parking PPO with safety shaping and explicit development selection.

The original 15 scenarios are development tasks, not independent test data.
No image inference or safety certificate is implied by nominal Cx training.
"""
import argparse
import json
from pathlib import Path
import time
import itertools
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from .ppo_env import ParkingPPOEnv
from .train_ppo import sha, evaluate


def quality(metrics):
    rows = metrics['per_scenario']
    unsafe = sum(r['collision'] or r['out_of_bounds'] for r in rows)
    return (-unsafe, metrics['success_count'], -metrics['mean_final_goal_distance'])


def load_ppo(path, **kwargs):
    # Cross-Python-version cloudpickle schedules are not portable. Training
    # explicitly installs its own schedules below; evaluation needs no schedule.
    return PPO.load(path, custom_objects={'lr_schedule': lambda _: 1e-4,
                                         'clip_range': lambda _: .15}, **kwargs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--load', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--timesteps', type=int, default=400000)
    p.add_argument('--eval-every', type=int, default=25000)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--residual-data', help='Fresh samples.npz; uses fit rows only as a sensor surrogate')
    p.add_argument('--reward-mode', choices=['safety_v2', 'safety_v3'], default='safety_v2')
    p.add_argument('--sensor-jitter', type=float, default=0.)
    p.add_argument('--focus-scenarios', nargs='*', type=int, default=[])
    p.add_argument('--projection', default='results/vision_sls_parking_20261025_adaptation_v1/projection_adapted.pt')
    a = p.parse_args()
    out = Path(a.output_dir)
    if (out.exists() or min(a.timesteps, a.eval_every) < 1 or not np.isfinite(a.learning_rate)
            or a.learning_rate <= 0 or not np.isfinite(a.sensor_jitter) or a.sensor_jitter < 0
            or any(s not in range(15) for s in a.focus_scenarios)):
        p.error('new output directory and positive training settings required')
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    out.mkdir(parents=True)
    cfg = dict(vars(a), source_sha256=sha(a.load), projection_sha256=sha(a.projection),
               physical_dimension=4, visual_dimension=3,
               actor_input_dimension=16, horizon=30, delay=1, process_noise='zero',
               training_sensor='analytic Cx, not actual images',
               selection='development 15 starts: fewer unsafe, more successes, lower final distance',
               independent_test=False, visual_weights_changed=False)
    stress_sensor = None
    if a.residual_data:
        from .residual_replay import ResidualReplaySensor
        data_config = json.loads((Path(a.residual_data).parent/'config.json').read_text())
        if data_config['projection_sha256'] != cfg['projection_sha256']:
            raise ValueError('residual data and frozen projection provenance mismatch')
        cfg.update(training_sensor='50% nominal episodes + 50% fit-only nearest-residual surrogate',
                   residual_sha256=sha(a.residual_data),
                   selection='joint nominal/surrogate development: fewer unsafe, higher worst success, total success, distance')
        stress_sensor = ResidualReplaySensor(a.residual_data, nominal_fraction=0, seed=a.seed)
    (out/'config.json').write_text(json.dumps(cfg, indent=2)+'\n')
    started = time.monotonic()
    scenario_weights = None
    if a.focus_scenarios:
        scenario_weights = np.ones(15)
        scenario_weights[a.focus_scenarios] = 3.
    if a.residual_data:
        ids = itertools.count()
        vec = make_vec_env(lambda: ParkingPPOEnv(reward_mode=a.reward_mode, scenario_weights=scenario_weights,
            image_sensor=ResidualReplaySensor(a.residual_data, seed=a.seed+next(ids), jitter=a.sensor_jitter)),
            n_envs=8, seed=a.seed)
    else:
        vec = make_vec_env(ParkingPPOEnv, n_envs=8, seed=a.seed,
                           env_kwargs={'reward_mode': a.reward_mode, 'scenario_weights': scenario_weights})
    def assess(candidate):
        nominal, _, _ = evaluate(candidate, quiet=True)
        if stress_sensor is None:
            return quality(nominal), nominal, None
        stress, _, _ = evaluate(candidate, stress_sensor, quiet=True)
        a_score, b_score = quality(nominal), quality(stress)
        score = (a_score[0]+b_score[0], min(a_score[1], b_score[1]),
                 a_score[1]+b_score[1], a_score[2]+b_score[2])
        return score, nominal, stress
    history = []
    try:
        model = load_ppo(a.load, env=vec, device='cpu', seed=a.seed)
        model.learning_rate = a.learning_rate
        model.lr_schedule = lambda _: a.learning_rate
        model.clip_range = lambda _: .15
        model.ent_coef = .001
        source_steps = int(model.num_timesteps)
        best, base, base_stress = assess(model)
        best_steps = 0
        model.save(out/'ppo.zip')
        class Select(BaseCallback):
            def __init__(self):
                super().__init__()
                self.next_eval = a.eval_every
                self.start_steps = model.num_timesteps
            def _on_step(self):
                nonlocal best, best_steps
                added = self.num_timesteps-self.start_steps
                if added < self.next_eval:
                    return True
                self.next_eval += a.eval_every
                score, m, stress = assess(self.model)
                if score > best:
                    best, best_steps = score, added
                    self.model.save(out/'ppo.zip')
                row = dict(added_steps=added, elapsed_seconds=time.monotonic()-started,
                           success=m['success_count'], collision=m['collision_count'],
                           out_of_bounds=m['out_of_bounds_count'],
                           final_distance=m['mean_final_goal_distance'], selected=score == best)
                if stress is not None:
                    row.update(surrogate_success=stress['success_count'],
                               surrogate_collision=stress['collision_count'],
                               surrogate_bounds=stress['out_of_bounds_count'])
                history.append(row)
                (out/'training_history.json').write_text(json.dumps(history, indent=2)+'\n')
                print(json.dumps(row), flush=True)
                return True
        model.learn(a.timesteps, callback=Select(), reset_num_timesteps=False)
        model.save(out/'last_ppo.zip')
        last_score, last, _ = assess(model)
        if last_score > best:
            best_steps = int(model.num_timesteps)-source_steps
            model.save(out/'ppo.zip')
        selected = load_ppo(out/'ppo.zip', device='cpu')
        proxy, states, actions = evaluate(selected, quiet=True)
        np.savez_compressed(out/'rollouts.npz', proxy_states=states, proxy_actions=actions)
        metrics = dict(config=cfg, baseline=base, proxy=proxy, last=last,
                       best_added_steps=best_steps, checkpoint_sha256=sha(out/'ppo.zip'),
                       source_unchanged=sha(a.load) == cfg['source_sha256'],
                       projection_unchanged=sha(a.projection) == cfg['projection_sha256'],
                       runtime_seconds=time.monotonic()-started, image_evaluated=False,
                       safety_certificate=False, independent_test=False,
                       status='nominal_controller_development_training_complete')
        if stress_sensor is not None:
            metrics['baseline_surrogate'] = base_stress
            metrics['surrogate'], _, _ = evaluate(selected, stress_sensor, quiet=True)
            metrics['status'] = 'nominal_and_residual_surrogate_training_complete_not_image_evaluation'
        (out/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
        print('Selected:', {k: v for k, v in proxy.items() if k != 'per_scenario'}, flush=True)
    finally:
        vec.close()


if __name__ == '__main__':
    main()
