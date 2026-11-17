"""Cross frozen camera front ends with released AEBS actors, including SPVC's z=0 generator.

Run from artical-F122 on the server where the two frozen image banks are present.
This comparison scores native image-to-proxy discrepancy separately from physical
distance error. No policy, model, threshold or rollout is selected on holdout.
"""
import argparse
import copy
import json
import math
from pathlib import Path

import h5py
import numpy as np
import torch
from stable_baselines3 import PPO
from torch import nn

from cGAN.taxi_models_and_data import AebsMLPGenerator
from Combined_network.model import CombinedPolicyNetwork, SubNet
from Aebs.system.outcomes import classify_terminal_outcome, EPISODE_OUTCOMES
from Aebs.vision_sls.finalize_robust_aebs import inclusion_evidence, constant_radius
from Aebs.vision_sls.model import VisionSLSPolicy
from Aebs.vision_sls.run import sha256
from Aebs.vision_sls.sbc_cp import load_sbc_checkpoint
from Aebs.vision_sls.verify_sbc_cells import actual_witnesses


RELEASE = Path('results/aebs_robust_synthesis_20261117_release')
PACKAGE = RELEASE / 'package'
GRAY = Path('results/vision_sls_spvc_sbc_pass_20261029')
H5 = Path('Aebs/data/Downsampled.h5')
RGB_GRAY = Path('results/aebs_budget_scaling_20261110/cgan_24000_gray.npz')
GEN = Path('Aebs/cGAN/mlp_supervised_ld4/mlp_supervised.pth')
STATE = Path('Aebs/controller/state_net_trained.pth')
PPO_FILE = Path('Aebs/controller/best_model/best_model.zip')
FRONTS = ('generated', 'h5_native', 'h5_dino', 'rgb_native', 'rgb_dino')
ACTORS = ('published_ppo', 'robust_actor')
SEEDS = {'calibration': 202611170511, 'holdout': 202611170512}


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def tensor(array):
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))


def meters(normalized, scale):
    return normalized.astype(np.float64) * scale


def net_load(path):
    return torch.load(path, weights_only=False, map_location='cpu')


def robust_policy(model, scale):
    widths = [2, *model['actor_hidden']]
    layers = []
    for nin, nout in zip(widths, widths[1:]):
        layers.extend([nn.Linear(nin, nout), nn.Tanh()])
    controller = CombinedPolicyNetwork(nn.Sequential(*layers), nn.Linear(widths[-1], 1))
    policy = VisionSLSPolicy(controller, scale)
    policy.load_state_dict(model['state_dict'])
    policy.set_state_distance_scale(model['metadata']['state_distance_scale_m'])
    return policy.eval()


class Models:
    def __init__(self):
        self.generator = AebsMLPGenerator(4, 1).eval()
        self.generator.load_state_dict(net_load(GEN))
        self.state = SubNet([1024, 256, 64, 1]).eval()
        self.state.load_state_dict(net_load(STATE))
        self.ppo = PPO.load(PPO_FILE, device='cpu')
        saved = net_load(PACKAGE / 'policy.pt')
        if saved['observation_sha256'] != sha256(PACKAGE / 'observation.pt'):
            raise ValueError('robust actor/observation binding')
        self.robust = robust_policy(saved, net_load(PACKAGE / 'observation.pt')['distance_scale_m'])
        with h5py.File(H5) as h:
            self.original_scale = float(h['y_train'][:].std())

    def estimate_native(self, inputs):
        return self.state(inputs.reshape(len(inputs), -1)).flatten().numpy()

    def generated_m(self, d):
        images = self.generator(torch.zeros((len(d), 4)), tensor(d / self.original_scale)[:, None])
        return meters(self.estimate_native(images), self.original_scale)

    def action(self, actor, observed, speed):
        if actor == 'published_ppo':
            inputs = np.column_stack([observed / self.original_scale, speed]).astype(np.float32)
            return self.ppo.predict(inputs, deterministic=True)[0].reshape(-1).clip(-3, 3)
        z = tensor(observed / self.robust.distance_scale_m)[:, None]
        return self.robust.forward_observation(z, tensor(speed)[:, None]).flatten().numpy()


def banks(models, out):
    h5_ckpt = net_load(GRAY / 'observation.pt')
    with h5py.File(H5) as h, np.load(GRAY / 'observations.npz') as z:
        h5_d = h['y_train'][:].reshape(-1)
        if not np.array_equal(h5_d, z['distance_m']) or sha256(H5) != h5_ckpt['provenance']['data_sha256']:
            raise ValueError('gray image identities')
        h5_native = meters(models.estimate_native(tensor(h['X_train'][:] * 2.0 - 1.0)), models.original_scale)
        h5_dino = meters(z['observation'], h5_ckpt['distance_scale_m'])
    with np.load(GRAY / 'splits.npz') as z:
        h5_ids = np.sort(z['test'])
    # Keep original file order for stable distance ties inside each appearance.
    with np.load(RGB_GRAY) as raw, np.load(PACKAGE / 'observations.npz') as z:
        for left, right in [('files', 'filename'), ('split', 'split'), ('distances', 'distance_m')]:
            if not np.array_equal(raw[left], z[right]):
                raise ValueError(f'RGB image identities misaligned: {left}')
        ids = np.flatnonzero(z['split'] == 'test')
        d = z['distance_m'][ids]
        dino = meters(z['observation'][ids], models.robust.distance_scale_m)
        groups = sorted(set(zip(z['weather'][ids].tolist(), z['color'][ids].tolist())))
        tuples = dict(zip(groups, range(len(groups))))
        g = np.array([tuples[k] for k in zip(z['weather'][ids], z['color'][ids])])
        native = meters(np.concatenate([models.estimate_native(chunk) for chunk in
            tensor(raw['images'][ids]).split(256)]), models.original_scale)
        filenames = z['filename'][ids]
    if len(ids) != 4920 or len(groups) != 12 or len(h5_ids) != 80:
        raise ValueError('unexpected image-bank size')
    np.savez_compressed(out / 'frame_estimates.npz', h5_distances_m=h5_d, h5_native_m=h5_native,
        h5_dino_m=h5_dino, h5_test_indices=h5_ids, rgb_filenames=filenames, rgb_distances_m=d,
        rgb_native_m=native, rgb_dino_m=dino, rgb_groups=g)
    data = {}
    for front, values in [('h5_native', h5_native), ('h5_dino', h5_dino)]:
        data[front] = [(h5_d[h5_ids], values[h5_ids])]
    for front, values in [('rgb_native', native), ('rgb_dino', dino)]:
        data[front] = [(d[g == i], values[g == i]) for i in range(len(groups))]
    return data, groups


def observed_frames(front, data, d, appearances):
    observed = np.empty_like(d)
    mismatch = np.empty_like(d)
    group_ids = appearances if front.startswith('rgb') else np.zeros_like(appearances)
    for i in np.unique(group_ids):
        idx = np.flatnonzero(group_ids == i)
        labels, values = data[front][i]
        closest = np.abs(d[idx, None] - labels[None, :]).argmin(axis=1)
        observed[idx] = values[closest]
        mismatch[idx] = np.abs(d[idx] - labels[closest])
    return observed, mismatch


def collect(models, data, front, actor, initial, groups):
    n = len(initial)
    state = np.column_stack([initial[:, 0] / models.original_scale, initial[:, 1]]).astype(np.float32)
    outcomes = np.full(n, '', dtype='U32')
    steps = np.zeros(n, int)
    max_truth = np.zeros(n)
    max_proxy = np.zeros(n)
    max_lookup = np.zeros(n)
    sum_truth = np.zeros(n)
    for _ in range(400):
        active = np.flatnonzero(outcomes == '')
        if not len(active):
            break
        d = meters(state[active, 0], models.original_scale)
        v = state[active, 1].astype(np.float64)
        nominal = models.generated_m(d) if front == 'generated' or front.endswith('native') else d
        if front == 'generated':
            observed = nominal
        else:
            observed, mismatch = observed_frames(front, data, d, groups[active])
            max_lookup[active] = np.maximum(max_lookup[active], mismatch)
        err = np.abs(observed - d)
        max_truth[active] = np.maximum(max_truth[active], err)
        max_proxy[active] = np.maximum(max_proxy[active], np.abs(observed - nominal))
        sum_truth[active] += err
        act = models.action(actor, observed, v).astype(np.float64)
        if not np.isfinite(act).all():
            raise ValueError('nonfinite action')
        d_next = d - v * .05
        v_next = np.clip(v - act * .05, 0, 3)
        state[active, 0] = d_next / models.original_scale
        state[active, 1] = v_next
        steps[active] += 1
        for j, idx in enumerate(active):
            label = classify_terminal_outcome(d_next[j], v_next[j])
            if label:
                outcomes[idx] = label
    outcomes[outcomes == ''] = 'timeout'
    return dict(initial=initial, appearance=groups, outcomes=outcomes, steps=steps,
        max_truth_m=max_truth, max_proxy_m=max_proxy, mean_truth_m=sum_truth / steps,
        max_lookup_m=max_lookup, terminal_d_m=meters(state[:, 0], models.original_scale), terminal_v_mps=state[:, 1])


def barrier_status(q, policy, barrier, cfg, parent, outer):
    if q <= constant_radius(outer):
        return {'status': 'pass_by_inclusion', 'outer_m': constant_radius(outer), 'checked_cells': parent['checked_cells']}
    contract = dict(outer, calibration={'q_hat': q})
    scale = cfg['barrier_distance_scale_m']
    div = torch.tensor([scale, 1.], dtype=torch.float64)
    for lo, hi, vhi in [(6.00001, 16., 3.), (6.00001, 7.5, .8)]:
        d, v = torch.meshgrid(torch.linspace(lo, hi, 200, dtype=torch.float64),
            torch.linspace(1e-5, vhi, 160, dtype=torch.float64), indexing='ij')
        for x in torch.stack([d.flatten(), v.flatten()], dim=1).split(1024):
            bad, drift, observed = actual_witnesses(policy, barrier, scale, contract, x)
            if bad.any():
                j = int(torch.nonzero(bad)[0, 0])
                b = float(barrier(x[j:j + 1] / div).item())
                return dict(status='positive_drift_witness', d_m=float(x[j, 0]), v_mps=float(x[j, 1]),
                    error_m=float(observed[j] * policy.distance_scale_m - x[j, 0]), barrier=b,
                    successor_barrier=b + float(drift[j]), drift=float(drift[j]))
    return {'status': 'outside_existing_tube_no_grid_witness'}


@torch.inference_mode()
def run(out):
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    files = [GEN, STATE, PPO_FILE, H5, RGB_GRAY, GRAY / 'observations.npz', GRAY / 'observation.pt',
        GRAY / 'splits.npz', *[PACKAGE / n for n in ('policy.pt','observation.pt','observations.npz','sbc.pt','config.json')],
        RELEASE / 'development_certificate.json', RELEASE / 'development_contract.json', Path(__file__),
        Path('Aebs/system/outcomes.py'),Path('Aebs/vision_sls/verify_sbc_cells.py')]
    manifest = {str(f): sha256(f) for f in files}
    write(out / 'protocol.json',dict(files=manifest, calibrated_n=199, holdout_n=500, seeds=SEEDS,
        alpha=.05, rank=190, horizon=400, actors=ACTORS,fronts=FRONTS,
        initial='uniform [15,16] x [2.5,3]; one uniform weather/color per RGB episode; paired across all arms',
        generated='original released G(z=0,d) -> state_net; synthetic nominal model (not paper stochastic law)',
        h5='80 fixed test image IDs of historical DINO400 split; native generator trained with all 400 H5 images',
        rgb='4,920 fixed test image IDs of 24,000; native estimator receives downsampled gray, DINO RGB',
        score='max pre-action |observation-d| meters; also |observation-g(d)| for native and |observation-d| for DINO',
        stopping='first common terminal outcome; include all episodes and timeouts',
        inference='paired episode bootstrap of mean within-episode max score difference, 4000 resamples; conditional on banks and models'))
    models = Models()
    data, groups = banks(models, out)
    write(out / 'groups.json', groups)
    cfg = read(PACKAGE / 'config.json')
    barrier, _ = load_sbc_checkpoint(PACKAGE / 'sbc.pt',cfg)
    barrier = barrier.double().eval()
    outer = read(RELEASE / 'development_contract.json')
    parent = read(RELEASE / 'development_certificate.json')
    if sha256(RELEASE / 'development_contract.json') != parent['source_contract_sha256']:
        raise ValueError('development certificate contract mismatch')
    inclusion_evidence(parent,outer,outer,PACKAGE/'policy.pt',PACKAGE/'sbc.pt',PACKAGE/'observation.pt')
    rank = math.ceil((199 + 1) * .95)
    thresholds = {}
    report = {}
    for phase, count in [('calibration', 199), ('holdout', 500)]:
        rng = np.random.default_rng(SEEDS[phase])
        initial = rng.uniform([15, 2.5], [16, 3], size=(count, 2))
        appearance = rng.integers(len(groups), size=count)
        for actor in ACTORS:
            for front in FRONTS:
                key = f'{actor}__{front}'
                r = collect(models, data, front, actor, initial, appearance)
                np.savez_compressed(out / f'{key}_{phase}.npz', **r)
                if phase == 'calibration':
                    thresholds[key] = {k: float(np.sort(r[k])[rank - 1]) for k in ('max_truth_m','max_proxy_m')}
                else:
                    q = thresholds[key]
                    report[key] = dict(thresholds=q, test_covered={k:int((r[k]<=v).sum()) for k,v in q.items()},
                        outcomes={k:int((r['outcomes']==k).sum()) for k in EPISODE_OUTCOMES},
                        mean_episode_max_truth_m=float(r['max_truth_m'].mean()),
                        mean_frame_truth_m=float(np.average(r['mean_truth_m'],weights=r['steps'])),
                        max_lookup_m=float(r['max_lookup_m'].max()), max_steps=int(r['steps'].max()))
                print(phase, key, thresholds[key], flush=True)
        if phase == 'calibration':
            write(out / 'frozen_thresholds.json',thresholds)
    actor64 = copy.deepcopy(models.robust).double()
    for front in FRONTS:
        key = 'robust_actor__' + front
        report[key]['fixed_barrier_on_physical_tube'] = barrier_status(
            thresholds[key]['max_truth_m'], actor64, barrier, cfg, parent, outer)
    contrasts = {}
    for actor in ACTORS:
        for bank in ('h5','rgb'):
            names = [actor+'__'+bank+'_'+family for family in ('native','dino')]
            with np.load(out / (names[0]+'_holdout.npz')) as old, np.load(out/(names[1]+'_holdout.npz')) as new:
                if not np.array_equal(old['initial'],new['initial']) or not np.array_equal(old['appearance'],new['appearance']):
                    raise ValueError('episode pairing')
                delta = old['max_truth_m'] - new['max_truth_m']
                boot = np.random.default_rng(202611170513).choice(delta,(4000,len(delta)),replace=True).mean(axis=1)
                contrasts[actor+'__'+bank] = dict(mean_paired_reduction_m=float(delta.mean()),
                    percentile_interval=np.quantile(boot,[.025,.975]).tolist(), favorable=int((delta>0).sum()),
                    ties=int((delta==0).sum()), success_old_only=int(((old['outcomes']=='success')&(new['outcomes']!='success')).sum()),
                    success_new_only=int(((old['outcomes']!='success')&(new['outcomes']=='success')).sum()))
    if any(sha256(f)!=manifest[str(f)] for f in files):
        raise RuntimeError('inputs changed')
    write(out/'report.json', dict(arms=report, paired=contrasts))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    run(parser.parse_args().output_dir)
