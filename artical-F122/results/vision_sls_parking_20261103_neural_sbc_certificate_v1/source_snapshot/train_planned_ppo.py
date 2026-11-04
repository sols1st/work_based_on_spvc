"""Train a small residual PPO around frozen references and visual feedback."""
import argparse
import itertools
import json
from pathlib import Path
import time
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from .planned_env import PlannedParkingEnv, evaluate_planned
from .planning_feedback import ReferenceBank
from .residual_replay import ResidualReplaySensor
from .train_migration import quality
from .train_ppo import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--references', required=True); p.add_argument('--output-dir', required=True)
    p.add_argument('--residual-data', required=True)
    p.add_argument('--timesteps', type=int, default=200000)
    p.add_argument('--projection', default='results/vision_sls_parking_20261025_adaptation_v1/projection_adapted.pt')
    a = p.parse_args(); out = Path(a.output_dir)
    if out.exists() or a.timesteps < 1:
        p.error('new output directory and positive step budget required')
    cfg = dict(vars(a), references_sha256=sha(a.references), projection_sha256=sha(a.projection),
               residual_data_sha256=sha(a.residual_data), residual_action_scale=[.3, .5],
               observation_dimension=13, initial_jitter=.04,
               seed=103, true_state_to_actor=False, safety_certificate=False,
               selection='nominal, fit-residual sensor, perturbed nominal initial states; development only')
    source = json.loads((Path(a.residual_data).parent/'config.json').read_text())
    if source['projection_sha256'] != cfg['projection_sha256']:
        raise ValueError('residual data/projection mismatch')
    torch.set_num_threads(1); torch.manual_seed(103)
    out.mkdir(parents=True); (out/'config.json').write_text(json.dumps(cfg, indent=2)+'\n')
    bank = ReferenceBank(a.references); ids = itertools.count()
    vec = make_vec_env(lambda: PlannedParkingEnv(bank,
        sensor=ResidualReplaySensor(a.residual_data, seed=103+next(ids), jitter=.02), initial_jitter=.04),
        n_envs=8, seed=103)
    stress = ResidualReplaySensor(a.residual_data, nominal_fraction=0)
    def assess(model):
        results = {name:evaluate_planned(bank, model, sensor=sensor, jitter=jitter)[0]
                   for name, sensor, jitter in [('nominal', None, 0), ('surrogate', stress, 0), ('jitter', None, .04)]}
        scores = [quality(m) for m in results.values()]
        rank = (sum(s[0] for s in scores), min(s[1] for s in scores), sum(s[1] for s in scores), sum(s[2] for s in scores))
        return rank, results
    started = time.monotonic(); history = []
    try:
        model = PPO('MlpPolicy', vec, device='cpu', seed=103, n_steps=256, batch_size=256, n_epochs=5,
                    learning_rate=1e-4, gamma=.99, ent_coef=0.,
                    policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64]), log_std_init=-2.))
        with torch.no_grad():
            model.policy.action_net.weight.zero_(); model.policy.action_net.bias.zero_()
        best, baseline = assess(model); selected_steps = 0; model.save(out/'ppo.zip')
        class Select(BaseCallback):
            def __init__(self):
                super().__init__(); self.next_check = 25000
            def _on_step(self):
                nonlocal best, selected_steps
                if self.num_timesteps < self.next_check:
                    return True
                self.next_check += 25000
                score, results = assess(self.model)
                if score > best:
                    best = score; selected_steps = self.num_timesteps; self.model.save(out/'ppo.zip')
                row = dict(steps=self.num_timesteps, seconds=time.monotonic()-started,
                           results={k:{f:v[f] for f in ('success_count','collision_count','out_of_bounds_count')} for k,v in results.items()},
                           selected_steps=selected_steps)
                history.append(row); print(json.dumps(row), flush=True)
                return True
        model.learn(a.timesteps, callback=Select()); model.save(out/'last_ppo.zip')
        score, final = assess(model)
        if score > best:
            selected_steps = model.num_timesteps; model.save(out/'ppo.zip')
        selected = PPO.load(out/'ppo.zip', device='cpu')
        _, results = assess(selected)
        metrics = dict(config=cfg, baseline=baseline, selected=results, last=final,
                       selected_steps=selected_steps, checkpoint_sha256=sha(out/'ppo.zip'), history=history,
                       runtime_seconds=time.monotonic()-started, actual_images_evaluated=False,
                       safety_certificate=False)
        (out/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
        print('selected steps', selected_steps, 'successes', {k:v['success_count'] for k,v in results.items()}, flush=True)
    finally:
        vec.close()


if __name__ == '__main__':
    main()
