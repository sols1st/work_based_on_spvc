"""Frozen-model, paired AEBS accuracy and task-completion experiment.

No training, model selection, online CARLA rendering, or certification is done
here. Register first, then execute the byte-bound protocol. Extra manifests add
all models in a later independently registered stage on the same scenario law.
"""
import argparse
import csv
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from stable_baselines3 import PPO
from torch import nn

from Aebs.system.outcomes import EPISODE_OUTCOMES
from Aebs.vision_sls.model import VisionSLSPolicy
from Aebs.vision_sls.rollout import rollout
from Combined_network.model import CombinedPolicyNetwork

SCALE = 3.18272066116333
CAL_SEED, TEST_SEED = 202611189101, 202611189102
SMOKE_SEED = 202611189099
CAL_N, TEST_N = 199, 1000
ACTORS = ('original_ppo', 'current_robust')
FACTORIAL = Path('results/aebs_factorial_20261117')
LIBRARY = Path('results/aebs_matched_replay_20261117/library')
FULL = Path('results/aebs_native_cross_20261117_v1/full_release_bank')
PACKAGE = Path('results/aebs_robust_synthesis_20261117_release/package')
PPO_PATH = Path('Aebs/controller/best_model/best_model.zip')
H5_PATH = Path('Aebs/data/Downsampled.h5')


def read(p):
    return json.loads(Path(p).read_text())


def write(p, obj):
    p = Path(p)
    p.parent.mkdir(exist_ok=True, parents=True)
    p.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def episodes(n, seed):
    rng = np.random.default_rng(seed)
    return rng.uniform([15., 2.5], [16., 3.], (n, 2)), rng.integers(0, 12, size=n)


class OriginalActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('C', torch.ones(1))
        self.agent = PPO.load(PPO_PATH, device='cpu')
        with h5py.File(H5_PATH) as f:
            self.original_scale = float(f['y_train'][:].std())

    def forward_observation(self, y, v):
        # Same physical observed distance, converted to the released PPO units.
        z = torch.cat([y * (SCALE / self.original_scale), v], dim=1)
        return torch.from_numpy(self.agent.predict(z.cpu().numpy(), deterministic=True)[0])


def load_actors():
    saved = torch.load(PACKAGE / 'policy.pt', map_location='cpu', weights_only=False)
    obs = torch.load(PACKAGE / 'observation.pt', map_location='cpu', weights_only=False)
    if saved['observation_sha256'] != sha(PACKAGE / 'observation.pt'):
        raise ValueError('robust actor/observation binding')
    widths = [2] + saved['actor_hidden']
    layers = []
    for a, b in zip(widths, widths[1:]):
        layers.extend([nn.Linear(a, b), nn.Tanh()])
    policy = VisionSLSPolicy(CombinedPolicyNetwork(nn.Sequential(*layers), nn.Linear(widths[-1], 1)),
                            obs['distance_scale_m'])
    policy.load_state_dict(saved['state_dict'])
    policy.set_state_distance_scale(saved['metadata']['state_distance_scale_m'])
    if policy.distance_scale_m != SCALE or policy.state_distance_scale_m != SCALE:
        raise ValueError('robust physical normalization changed')
    return {'original_ppo': OriginalActor().eval(), 'current_robust': policy.eval()}


def design(extra_manifest=None):
    inputs = [Path(__file__), Path('Aebs/vision_sls/rollout.py'), Path('Aebs/system/outcomes.py'),
              Path('Aebs/vision_sls/model.py'), Path('Combined_network/model.py'),
              PPO_PATH, H5_PATH, PACKAGE / 'policy.pt', PACKAGE / 'observation.pt',
              LIBRARY / 'manifest.json', LIBRARY / 'libraries.json', LIBRARY / 'predictions.npz',
              LIBRARY / 'image_hashes.npz', FACTORIAL / 'protocol.json']
    models = []
    if extra_manifest is None:
        for budget in (400, 10000, 24000):
            for seed in (27411, 27412, 27413):
                for mode in ('gray32_mlp', 'gray32'):
                    key = '%s_%s_%s' % (mode, budget, seed)
                    path = FACTORIAL / 'models' / key
                    fit = read(path / 'fit.json')
                    if fit['prediction'] != sha(path / 'prediction.npy'):
                        raise ValueError('changed predictions: ' + key)
                    if fit['protocol_sha256'] != sha(FACTORIAL / 'protocol.json'):
                        raise ValueError('changed training protocol: ' + key)
                    model_path = path / 'readout.pt'
                    if fit['model'] != sha(model_path):
                        raise ValueError('changed model: ' + key)
                    row = dict(key=key, mode=mode, budget=budget, train_seed=seed,
                               category='fair_frozen_same_gray32', library='common',
                               prediction_path=str(path / 'prediction.npy'), ids_path=str(FACTORIAL / 'ids.npy'),
                               prediction_units='distance_over_scale', model_path=str(model_path),
                               fit_path=str(path / 'fit.json'))
                    models.append(row)
                    inputs.extend(Path(row[k]) for k in ('prediction_path', 'ids_path', 'model_path', 'fit_path'))
        for bank in ('common', 'full24k'):
            for field, key in [('native_m', 'published_native_head'), ('dino_m', 'release_rgb_dino')]:
                models.append(dict(key=key, mode=key, budget=None, train_seed=None,
                                   category='release_transfer_unequal_training', library=bank,
                                   prediction_path=str(FULL / 'estimates.npz'), field=field,
                                   prediction_units='meters', ids_field='files'))
        inputs.extend([FULL / 'estimates.npz', FULL / 'protocol.json', FULL / 'groups.json'])
        for key in ('selected_image_label_oracle', 'current_state_oracle'):
            models.append(dict(key=key, mode=key, budget=None, train_seed=None,
                               category='diagnostic_oracle', library='common', oracle=key))
    else:
        inputs.append(Path(extra_manifest))
        manifest = read(extra_manifest)
        for item in manifest['models']:
            row = dict(item, category='new_equal_tuning_same_gray32', library='common')
            if row['prediction_units'] not in ('distance_over_scale', 'meters'):
                raise ValueError('unsupported prediction units')
            for key in ('prediction_path', 'ids_path', 'model_path', 'fit_path'):
                inputs.append(Path(row[key]))
            models.append(row)
    keys = [m['library'] + '__' + m['key'] for m in models]
    if len(set(keys)) != len(keys):
        raise ValueError('duplicate model/library keys')
    arms = [dict(model=i, actor=actor, key=models[i]['library'] + '__' + models[i]['key'] + '__' + actor)
            for i in range(len(models)) for actor in ACTORS]
    token = dict(version='aebs_frozen_accuracy_and_completion_v1', models=models, arms=arms,
        inputs={str(p): sha(p) for p in sorted(set(inputs))},
        seeds={'calibration': CAL_SEED, 'holdout': TEST_SEED}, n={'calibration': CAL_N, 'holdout': TEST_N},
        alpha=.05, horizon=400, dt_seconds=.05, state_storage='float32 [distance/SCALE, speed]',
        scale_m=SCALE, action='braking acceleration clipped [-3,3] m/s^2',
        scenarios='IID uniform [15,16] m x [2.5,3] m/s; one IID uniform fixed appearance out of 12 per episode',
        pairing='identical initial states and appearance draws for every arm; individual closed-loop states evolve separately',
        camera='fixed nearest-distance image within appearance; common library removes train/dev IDs and pixel collisions of all original budgets; full24k includes fitting images',
        common_n=3229, full24k_n=24000, static='uniform over each fixed bank image, descriptive finite-bank metrics; frames are not independent experimental trials',
        stopping='first outcome under Aebs/system/outcomes.py; success is d in [5,6] m and v<=0.5 m/s; unsafe d<=6 m and v>0.5; early safe stop d>6 and v<=1e-6; other exits distinct; max400 steps',
        terminal_metrics='all episodes including failures: distance to target interval [5,6], excess speed max(v-.5,0); |d-5.5| is auxiliary and not the original objective; goal time is steps*.05 if successful and20s otherwise, actual termination time retained separately',
        errors='total observation error relative to current physical distance includes image lookup mismatch; network-only error relative to selected image label reported separately',
        cp='per-arm new199 calibration trajectory maxima; rank190; freeze all thresholds before evaluating any1000 holdout; no certificate transferred between actors',
        inference='paired whole-episode bootstrap4000, seed202611189103; binomial exact intervals per outcome; same1000 scenarios per training seed, never pool three seeds as3000 independent scenarios',
        limitations=['fixed camera library conditional experiment, not new independent camera capture',
                     'DINO pretraining cost external and not equalized',
                     'published PPO versus robust actor has unequal training objectives and histories',
                     'oracle arms are diagnostics, not feasible sensor baselines',
                     'CP coverage is not a new safety certificate'],
        extras_manifest=None if extra_manifest is None else str(extra_manifest))
    return token


def load_banks():
    with np.load(LIBRARY / 'predictions.npz') as z:
        table = {k: z[k].copy() for k in ('filename', 'distance_m', 'weather', 'color')}
    lookup = {s: i for i, s in enumerate(table['filename'])}
    selected = np.array([lookup[f] for f in read(LIBRARY / 'libraries.json')['common']])
    groups = sorted(set(zip(table['weather'], table['color'])))
    if len(groups) != 12 or len(selected) != 3229:
        raise ValueError('changed common camera law')
    groupmap = {key: i for i, key in enumerate(groups)}
    common = dict(filename=table['filename'][selected], distance_m=table['distance_m'][selected],
                  appearance=np.array([groupmap[p] for p in zip(table['weather'][selected], table['color'][selected])]))
    with np.load(FULL / 'estimates.npz') as z:
        full = dict(filename=z['files'].copy(), distance_m=z['distance_m'].copy(), appearance=z['appearance'].copy())
    if read(FULL / 'groups.json') != [list(g) for g in groups] or len(full['filename']) != 24000:
        raise ValueError('full library appearance identities')
    return {'common': common, 'full24k': full}


def predictions(model, bank):
    if 'oracle' in model:
        return bank['distance_m'].astype(float) / SCALE
    if 'field' in model:
        with np.load(model['prediction_path']) as z:
            ids, pred = z[model['ids_field']], z[model['field']]
    else:
        ids, pred = np.load(model['ids_path']), np.load(model['prediction_path'])
    pred = np.asarray(pred).reshape(-1)
    if len(ids) != len(pred) or len(set(ids.tolist())) != len(ids) or not np.isfinite(pred).all():
        raise ValueError('invalid prediction cache')
    index = {f: i for i, f in enumerate(ids)}
    output = pred[[index[f] for f in bank['filename']]].astype(float)
    return output / SCALE if model['prediction_units'] == 'meters' else output


def static_metrics(bank, pred, oracle=None, *, scale=SCALE):
    residual = pred.astype(float) * scale - bank['distance_m'].astype(float)
    absolute = np.abs(residual)
    return dict(n=len(absolute), mae_m=float(absolute.mean()), rmse_m=float(np.sqrt((residual ** 2).mean())),
                p95_abs_m=float(np.quantile(absolute, .95)), max_abs_m=float(absolute.max()),
                signed_bias_m=float(residual.mean()),
                scope='diagnostic true-label oracle' if oracle else 'fixed-bank frame descriptive statistic')


@torch.inference_mode()
def collect(policy, bank, pred, initial, appearances, oracle=None, *, scale=SCALE):
    n = len(initial)
    peak, errsum, sqsum = np.zeros(n), np.zeros(n), np.zeros(n)
    lookup_peak, lookup_sum, net_peak, net_sum = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    steps = np.zeros(n, int)
    outcomes = np.empty(n, dtype='U32')
    terminal = np.empty_like(initial)
    for group in range(12):
        ids = np.flatnonzero(bank['appearance'] == group)
        ep = np.flatnonzero(appearances == group)
        if len(ids) == 0:
            raise ValueError('empty appearance group')
        if len(ep) == 0:
            continue
        labels = bank['distance_m'][ids]
        observed = pred[ids]
        def monitor(indices, x, nxt, action, y, end):
            active = ep[indices]
            nearest = np.abs(x[:, :1] - labels[None, :]).argmin(axis=1)
            selected_label = labels[nearest]
            error = y.astype(float) * scale - x[:, 0]
            neterror = np.abs(y.astype(float) * scale - selected_label)
            peak[active] = np.maximum(peak[active], np.abs(error))
            errsum[active] += np.abs(error)
            sqsum[active] += error ** 2
            lookup_error = np.abs(x[:, 0] - selected_label)
            lookup_peak[active] = np.maximum(lookup_peak[active], lookup_error)
            lookup_sum[active] += lookup_error
            net_peak[active] = np.maximum(net_peak[active], neterror)
            net_sum[active] += neterror
        is_state_oracle = oracle == 'current_state_oracle'
        result = rollout(policy, scale, initial[ep], None if is_state_oracle else labels,
                         None if is_state_oracle else observed, monitor=monitor)
        outcomes[ep], steps[ep], terminal[ep] = result['outcomes'], result['steps'], result['terminal_states']
    if not np.isfinite(peak).all() or not (steps > 0).all():
        raise ValueError('invalid episode')
    # Label discrepancy is not a visual-network error for the non-camera state oracle.
    if oracle == 'current_state_oracle':
        net_peak.fill(0); net_sum.fill(0); lookup_peak.fill(0); lookup_sum.fill(0)
    return dict(initial=initial, appearance=appearances, scores=peak, outcomes=outcomes, steps=steps,
                terminal_states=terminal, episode_mae_m=errsum / steps,
                episode_rmse_m=np.sqrt(sqsum / steps), max_lookup_error_m=lookup_peak,
                lookup_episode_mae_m=lookup_sum / steps,
                network_peak_error_m=net_peak, network_episode_mae_m=net_sum / steps,
                terminal_distance_to_goal_m=np.maximum(np.maximum(5. - terminal[:, 0], terminal[:, 0] - 6.), 0),
                terminal_abs_deviation_5p5_m=np.abs(terminal[:, 0] - 5.5),
                terminal_signed_deviation_5p5_m=terminal[:, 0] - 5.5,
                terminal_excess_speed_mps=np.maximum(terminal[:, 1] - .5, 0.),
                time_to_goal_capped_seconds=np.where(outcomes == 'success', steps * .05, 400 * .05),
                goal_event_observed=outcomes == 'success')


def distribution(x):
    return dict(mean=float(np.mean(x)), median=float(np.median(x)), p95=float(np.quantile(x, .95)), max=float(np.max(x)))


def summary(data, q):
    from scipy.stats import beta
    n = len(data['scores'])
    counts = {key: int(np.sum(data['outcomes'] == key)) for key in EPISODE_OUTCOMES}
    if sum(counts.values()) != n:
        raise ValueError('nonexhaustive outcomes')
    intervals = {key: [0. if k == 0 else float(beta.ppf(.025, k, n - k + 1)),
                       1. if k == n else float(beta.ppf(.975, k + 1, n - k))] for key, k in counts.items()}
    return dict(n=n, q_m=q, covered=int(np.sum(data['scores'] <= q)), outcomes=counts,
                outcome_binomial_95=intervals, in_fixed_0p45m_tube=int(np.sum(data['scores'] <= .45)),
                trajectory_peak_error_m=distribution(data['scores']),
                trajectory_mae_m=distribution(data['episode_mae_m']),
                trajectory_rmse_m=distribution(data['episode_rmse_m']),
                trajectory_network_peak_error_m=distribution(data['network_peak_error_m']),
                trajectory_network_mae_m=distribution(data['network_episode_mae_m']),
                trajectory_lookup_mae_m=distribution(data['lookup_episode_mae_m']),
                lookup_peak_m=distribution(data['max_lookup_error_m']),
                terminal_distance_to_goal_m=distribution(data['terminal_distance_to_goal_m']),
                terminal_abs_deviation_5p5_m=distribution(data['terminal_abs_deviation_5p5_m']),
                terminal_signed_deviation_5p5_m=distribution(data['terminal_signed_deviation_5p5_m']),
                terminal_speed_mps=distribution(data['terminal_states'][:, 1]),
                terminal_excess_speed_mps=distribution(data['terminal_excess_speed_mps']),
                time_to_goal_capped_seconds=distribution(data['time_to_goal_capped_seconds']),
                steps=distribution(data['steps']))


def run(args):
    torch.set_num_threads(1)
    token = design(args.extra_manifest)
    if args.register:
        args.output.mkdir(parents=True, exist_ok=False)
        write(args.output / 'protocol.json', token)
        print(json.dumps(dict(registered=str(args.output), arms=len(token['arms']))), flush=True)
        return
    if args.smoke:
        args.output.mkdir(parents=True, exist_ok=False)
        write(args.output / 'protocol.json', dict(token, mode='smoke_not_formal', smoke_seed=SMOKE_SEED, smoke_n=32))
    elif read(args.output / 'protocol.json') != token:
        raise ValueError('registered protocol/input hashes differ')
    if (args.output / 'summary.json').exists():
        raise ValueError('completed experiment exists')
    start = time.perf_counter()
    banks, actors = load_banks(), load_actors()
    static = {}
    for m in token['models']:
        label = m['library'] + '__' + m['key']
        static[label] = static_metrics(banks[m['library']], predictions(m, banks[m['library']]), m.get('oracle'))
    write(args.output / 'static_metrics.json', static)
    thresholds, results = {}, {}
    arms = token['arms']
    if args.smoke:
        # Cover both actors, both fairness fronts, both release fronts, and both oracles.
        arms = [a for a in arms if token['models'][a['model']]['key'] in
                ('gray32_mlp_400_27411', 'gray32_400_27411', 'published_native_head',
                 'release_rgb_dino', 'selected_image_label_oracle', 'current_state_oracle')]
        stages = [('smoke', 32, SMOKE_SEED)]
    else:
        stages = [('calibration', CAL_N, CAL_SEED), ('holdout', TEST_N, TEST_SEED)]
    for phase, n, seed in stages:
        initial, appearance = episodes(n, seed)
        np.savez_compressed(args.output / (phase + '_scenarios.npz'), initial=initial, appearance=appearance)
        for arm in arms:
            model = token['models'][arm['model']]
            bank = banks[model['library']]
            pred = predictions(model, bank)
            one_start = time.perf_counter()
            data = collect(actors[arm['actor']], bank, pred, initial, appearance, model.get('oracle'))
            dest = args.output / arm['key']
            dest.mkdir(exist_ok=True)
            np.savez_compressed(dest / (phase + '.npz'), **data)
            elapsed = time.perf_counter() - one_start
            if phase == 'calibration':
                rank = int(math.ceil((n + 1) * .95))
                thresholds[arm['key']] = dict(q_m=float(np.sort(data['scores'])[rank - 1]), rank=rank, n=n,
                                               seconds=elapsed, calibration_sha256=sha(dest / 'calibration.npz'))
            else:
                q = 0. if phase == 'smoke' else thresholds[arm['key']]['q_m']
                results[arm['key']] = dict(summary(data, q), seconds=elapsed, arm=arm, model=model,
                                          result_sha256=sha(dest / (phase + '.npz')))
                print(json.dumps(dict(phase=phase, arm=arm['key'], seconds=elapsed,
                                      outcomes=results[arm['key']]['outcomes'])), flush=True)
        if phase == 'calibration':
            write(args.output / 'frozen_thresholds.json', thresholds)
    if design(args.extra_manifest) != token:
        raise ValueError('inputs changed during evaluation')
    write(args.output / 'summary.json', dict(arms=results, total_seconds=time.perf_counter() - start,
        static_metrics_sha256=sha(args.output / 'static_metrics.json'), protocol_sha256=sha(args.output / 'protocol.json'),
        environment=dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                         cpu_threads=1, device='cpu'), formal=not args.smoke))
    print('complete', str(args.output), time.perf_counter() - start, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--extra-manifest', type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--register', action='store_true')
    group.add_argument('--smoke', action='store_true')
    run(parser.parse_args())
