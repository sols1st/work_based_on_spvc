"""Fresh robust AEBS synthesis initialized by the newly trained common PPO."""
import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from Combined_network.model import CombinedPolicyNetwork
from Aebs.vision_sls.model import VisionSLSPolicy
from Aebs.vision_sls.conformal import cp_quantile
from .aebs_intuitive_evaluate import collect
from .aebs_matched_data import write_json
from .aebs_matched_native import sha

SEED = 2026112502
STAGES = ('Aebs.vision_sls.train_sbc_cp', 'Aebs.vision_sls.refine_sbc_actor',
          'Aebs.vision_sls.verify_sbc_cells')


def fresh_policy(ppo_path, scale):
    ppo = PPO.load(ppo_path, device='cpu').policy
    network = CombinedPolicyNetwork(copy.deepcopy(ppo.mlp_extractor.policy_net),
                                    copy.deepcopy(ppo.action_net))
    return VisionSLSPolicy(network, scale).eval().requires_grad_(False)


def scenario(n, seed, groups=12):
    rng = np.random.default_rng(seed)
    return rng.uniform([15., 2.5], [16., 3.], (n, 2)), rng.integers(0, groups, n)


def library(data, mask):
    return {key: data[key][mask] for key in ('distance_m', 'appearance', 'files')}


def contract(package, q, **kwargs):
    return dict(format='vision_sls_state_cp_reference_v1', coefficients=[1.] * 4,
                envelope_units='m', score_units='m', calibration=dict(q_hat=q),
                source_policy_sha256=sha(package / 'policy.pt'),
                source_observation_sha256=sha(package / 'observation.pt'), **kwargs)


def run_stage(output, module, *args):
    with (output.parent / (output.name + '.log')).open('w') as stream:
        subprocess.run([sys.executable, '-u', '-m', module, *map(str, args)],
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    return json.loads((output / 'report.json').read_text())


def synthesis(experiment, rounds, minimum_radius, output):
    start = time.monotonic()
    output.mkdir(exist_ok=False)
    original = output / 'initial'
    original.mkdir()
    for name in ('observation.pt', 'observations.npz', 'splits.npz'):
        shutil.copy2(experiment / 'own_initial' / name, original / name)
    ppo_path = experiment / 'upstream/Aebs/controller/best_model/best_model.zip'
    protocol = json.loads((experiment / 'data_protocol.json').read_text())
    scale = protocol['train_distance_scale_m']
    saved = fresh_policy(ppo_path, scale)
    torch.save(dict(format='vision_sls_policy_v1', state_dict=saved.state_dict(),
                    observation_sha256=sha(original / 'observation.pt'),
                    metadata=saved.metadata(), actor_hidden=[64, 64], activation='Tanh'),
               original / 'policy.pt')
    dev = dict(np.load(original / 'observations.npz'))
    mask = dev['split'] == 'development'
    pred = dev['observation'][mask]
    initial, appearance = scenario(249, SEED)
    panel = collect(saved, library(dev, mask), pred, initial, appearance, scale=scale)
    dev_q = cp_quantile(panel['scores'], .05)['q_hat']
    design_radius = max(minimum_radius, np.ceil((dev_q + .05) * 20) / 20)
    write_json(output / 'development.json', dict(
        nominal_ppo_sha256=sha(ppo_path), train_distance_scale_m=scale,
        dev_q_249=dev_q, radius_m=design_radius, max_rounds=rounds,
        dev_q_source='separate development-image replay; not the final CP calibration',
        radius_rule=f'max({minimum_radius}, ceil((dev_q+.05)*20)/20), m',
        dev_outcomes={str(k): int(v) for k, v in zip(*np.unique(panel['outcomes'], return_counts=True))}))
    write_json(original / 'perception_contract.json', contract(original, design_radius,
               independent_trajectory_calibration=False, statistical_status='development only'))
    fit, refine, verify = STAGES
    candidate = original
    prev_barrier = None
    prev_audit = None
    history = []
    chosen = None
    for iteration in range(rounds + 1):
        trained = output / f'barrier_{iteration:02d}'
        audit = output / f'audit_{iteration:02d}'
        reference = candidate / 'perception_contract.json'
        args = ['--artifact-dir', candidate, '--perception-contract', reference,
                '--output-dir', trained, '--epochs', 4000, '--seed', 41,
                '--barrier-distance-scale', scale, '--nonnegative-activation', 'relu',
                '--counterexample-feedback', '--raw-region-loss', '--drift-loss', 'linear',
                '--learning-rate', '.003' if prev_barrier is None else '.0003']
        if prev_barrier:
            args.extend(['--resume', prev_barrier / 'sbc.pt'])
            extra = prev_audit / 'counterexamples.npz'
            if extra.exists():
                args.extend(['--extra-states', extra])
        run_stage(trained, fit, *args)
        result = run_stage(audit, verify, '--artifact-dir', candidate, '--barrier-dir', trained,
                           '--perception-contract', reference, '--output-dir', audit,
                           '--stop-on-witness', '--actor-bound-method', 'affine',
                           '--max-depth', 20, '--max-cells', 1000000)
        history.append(dict(actor=str(candidate), barrier=str(trained), verification=result))
        write_json(output / 'history.json', history)
        chosen = (candidate, trained, result)
        if result['continuous_state_numerical_interval_pass'] and iteration > 0:
            break
        if iteration == rounds:
            break
        actor = output / f'actor_{iteration:02d}'
        run_stage(actor, refine, '--artifact-dir', candidate, '--barrier-dir', trained,
                  '--perception-contract', reference, '--output-dir', actor,
                  '--rounds', 8, '--steps-per-round', 100, '--line-search')
        candidate, prev_barrier, prev_audit = actor, trained, audit
    # Prefer the most recent passing actor; preserve the start if updates all fail.
    passing = [row for row in history if row['verification']['continuous_state_numerical_interval_pass']]
    if passing:
        last = passing[-1]
        chosen = (Path(last['actor']), Path(last['barrier']), last['verification'])
    actor, barrier, result = chosen
    package = output / 'package'
    package.mkdir()
    for name in ('policy.pt', 'observation.pt', 'observations.npz'):
        shutil.copy2(actor / name, package / name)
    for name in ('sbc.pt', 'config.json'):
        shutil.copy2(barrier / name, package / name)
    write_json(output / 'synthesis.json', dict(
        package=str(package), selected_actor=str(actor), selected_barrier=str(barrier),
        design_radius_m=design_radius, checked_pairs=len(history),
        design_verification=result, actual_transition_training='worst-case one-step barrier drift; train-only task preservation',
        seconds=time.monotonic() - start, weights={f.name: sha(f) for f in package.glob('*.pt')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', type=Path)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--minimum-radius', type=float, default=.45)
    parser.add_argument('--output', type=Path, help='An independent result directory for a development run')
    args = parser.parse_args()
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '2')))
    synthesis(args.experiment, args.rounds, args.minimum_radius,
              args.output or args.experiment / 'own_synthesis')


if __name__ == '__main__':
    main()
