"""Pair complete controllers on a fresh, live-rendered straight road."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import sys
import time
from collections import OrderedDict

import numpy as np
from PIL import Image
import torch

from Aebs.lane_keeping.cp import conformal_radius
from Aebs.lane_keeping.policy import load_actor
from .lane_matched_data import grayscale, render, write_json
from .lane_matched_native import sha

CAL_SEED = 2026112621
TEST_SEED = 2026112622
DEV_SEED = 2026112620
COUNTS = dict(calibration=199, test=500)


def read(path):
    return json.loads(path.read_text())


def initial(seed, count):
    return np.random.default_rng(seed).uniform([-60., -7.], [60., 7.], (count, 2)).astype('float32')


def plant(states, u):
    e, heading = states.T
    rad = torch.deg2rad(heading)
    steer = torch.deg2rad((u + 1) / 2 * 60 - 30)
    next_heading = (rad + 5 * torch.tan(steer) * .05 / 2.5) % (2 * torch.pi)
    next_heading = torch.where(next_heading > torch.pi, next_heading - 2 * torch.pi, next_heading)
    next_e = (e / 100 + 5 * torch.sin(rad) * .05) * 100
    return torch.stack([next_e, torch.rad2deg(next_heading)], dim=1)


def metrics(states, actions):
    lateral = abs(states[:, :, 0])
    unsafe = lateral.max(axis=1) > 80
    post = lateral[:, 40:].max(axis=1)
    future = np.maximum.accumulate(lateral[:, ::-1], axis=1)[:, ::-1]
    enters = (~unsafe[:, None]) & (future <= 5)
    return dict(unsafe=unsafe, rmse_cm=np.sqrt((states[:, 1:, 0] ** 2).mean(axis=1)),
                strict_5cm=(~unsafe) & (post <= 5), strict_10cm=(~unsafe) & (post <= 10),
                settled_s=np.where(enters.any(axis=1), enters.argmax(axis=1) * .05, 8.),
                steering_change_deg=(30 * abs(np.diff(actions, axis=1))).mean(axis=1),
                performance_score_cm=np.where(unsafe, np.inf, post))


def render_parallel(pool, x, workers):
    chunks = np.array_split(x, min(len(x), workers))
    return np.concatenate(list(pool.map(render, chunks)))


class OwnCamera:
    def __init__(self, base, device):
        from Aebs.dino_latent.backbone import FrozenDinoV2
        from .lane_matched_own import REPO, WEIGHTS
        self.device = device
        self.head = torch.load(base/'patch_projection.pt', map_location=device, weights_only=True)
        assert self.head['format'] == 'lane_pooled_rgb_dino_vision_sls_v1'
        assert self.head['weights_sha256'] == sha(WEIGHTS)
        self.backbone = FrozenDinoV2(REPO, WEIGHTS, device)
        self.actor = load_actor(base/'actor.pt').requires_grad_(False)
        self.cache = OrderedDict()

    @torch.inference_mode()
    def project(self, rgb):
        from .lane_matched_own import pool_tokens
        image = rgb.permute(0, 3, 1, 2).to(self.device).float() / 255
        patches = self.backbone.backbone.forward_features(self.backbone.preprocess(image))['x_norm_patchtokens']
        x = (pool_tokens(patches) - self.head['mean']) / self.head['std']
        return (x @ self.head['weight'] + self.head['bias']).cpu()

    def observation(self, pictures):
        hashes = [picture.tobytes() for picture in pictures]
        missing = {}
        for key, picture in zip(hashes, pictures):
            if key not in self.cache:
                missing[key] = picture
        if missing:
            values = torch.cat([self.project(chunk) for chunk in torch.from_numpy(
                np.stack(list(missing.values()))).split(32)])
            self.cache.update((key, value.clone()) for key, value in zip(missing, values))
        output = torch.stack([self.cache[key] for key in hashes])
        while len(self.cache) > 2048:
            self.cache.popitem(last=False)
        return output

    def __call__(self, pictures, states):
        y = self.observation(pictures)
        nominal = torch.stack([states[:, 0]/85, torch.deg2rad(states[:, 1])/.30], 1)
        u = self.actor(torch.cat([y, torch.zeros_like(y[:, :1])], 1)).flatten()
        nominal_u = self.actor(torch.cat([nominal, torch.zeros_like(y[:, :1])], 1)).flatten()
        return u, (u-nominal_u)*30, y-nominal


class NativeCamera:
    def __init__(self, model):
        self.model = model.eval().requires_grad_(False)

    def __call__(self, pictures, states):
        image = torch.from_numpy(grayscale(pictures)).float()/127.5-1
        u = self.model.controller(image).flatten()
        nominal = self.model(torch.zeros(len(states), 10), states).flatten()
        return u, (u-nominal)*30, None


@torch.inference_mode()
def collect(camera, states, pool, workers, batch, label, output=None):
    records = []
    start = time.monotonic()
    for first in range(0, len(states), batch):
        x = torch.from_numpy(states[first:first+batch].copy())
        path = [x.numpy().copy()]
        us, gaps, residuals = [], [], []
        for t in range(160):
            rgb = render_parallel(pool, x.numpy(), workers)
            u, gap, residual = camera(rgb, x)
            us.append(u.numpy())
            gaps.append(gap.numpy())
            if residual is not None:
                residuals.append(residual.numpy())
            x = plant(x, u)
            path.append(x.numpy())
            if output is not None and first == 0 and t in (0, 40, 159):
                Image.fromarray(rgb[0]).save(output / f'{label}_{t:03d}.png')
        xs = np.stack(path, axis=1)
        actions = np.stack(us, axis=1)
        err = np.stack(gaps, axis=1)
        record = dict(states=xs, actions=actions, action_gap_deg=err,
                      action_score_deg=abs(err).max(axis=1), **metrics(xs, actions))
        if residuals:
            obs = np.stack(residuals, axis=1)
            record.update(observation_residual=obs, observation_score=abs(obs).max(axis=(1,2)))
        records.append(record)
        print(f'{label}: {min(first+batch,len(states))}/{len(states)}, {time.monotonic()-start:.1f}s', flush=True)
    return {key: np.concatenate([row[key] for row in records]) for key in records[0]}


def thresholds(rows):
    result = {}
    for name in ('action_score_deg','performance_score_cm','observation_score'):
        if name in rows:
            q, rank = conformal_radius(rows[name], .05)
            result[name] = dict(q=q if np.isfinite(q) else None, rank=rank)
    return result


def summarize(rows, q):
    n = len(rows['unsafe'])
    return dict(n=n, safe=int((~rows['unsafe']).sum()), strict_5cm=int(rows['strict_5cm'].sum()),
        strict_10cm=int(rows['strict_10cm'].sum()), rmse_cm=float(rows['rmse_cm'].mean()),
        settled_s=float(rows['settled_s'].mean()), mean_step_deg=float(rows['steering_change_deg'].mean()),
        p95_step_deg=float(np.quantile(30*abs(np.diff(rows['actions'],axis=1)),.95)),
        thresholds={key: {**t, 'holdout_covered': int((rows[key]<=t['q']).sum()) if t['q'] is not None else None}
                    for key,t in q.items()})


def run_arm(name, camera, output, pool, workers, batch):
    for phase, seed in [('calibration',CAL_SEED),('test',TEST_SEED)]:
        rows = collect(camera, initial(seed, COUNTS[phase]), pool, workers, batch,
                       name+'_'+phase, output if phase == 'test' else None)
        np.savez_compressed(output / f'{name}_{phase}.npz', **rows)
        if phase == 'calibration':
            q = thresholds(rows)
            write_json(output / f'{name}_thresholds.json', q)
    write_json(output / f'{name}_summary.json', summarize(rows,q))


def select_native(experiment, output, pool, workers, batch):
    root = (experiment/'native').resolve()
    sys.path[:0] = [str(root), str(root/'spvt')]
    from model.model import get_model
    from spvt.utils import dynamics
    from spvt.loss import calculate_action
    from .lane_matched_native import pid_target
    random = np.random.default_rng(2026112630)
    sample = torch.tensor(random.uniform([-80,-20,-1],[80,20,1],(512,3)),dtype=torch.float32)
    assert torch.equal(pid_target(sample[:,:2]),calculate_action(sample[:,:2][:,None]).reshape(-1,1))
    expected = torch.stack(dynamics(sample[:,0],sample[:,1],sample[:,2]),1)
    assert torch.equal(plant(sample[:,:2],sample[:,2]),expected), 'native dynamics adapter mismatch'
    generator = torch.load(root/'generator.pth', map_location='cpu')['generator_state_dict']
    anchor = get_model()
    anchor.generator.load_state_dict(generator)
    anchor.controller.load_state_dict(torch.load(root/'anchor.pth',map_location='cpu'))
    selected, selection, winner = None, [], None
    for epoch in (20,40,60,80,100):
        candidate = root / f'spvt/logs/matched/ckpts/checkpoint_epoch_{epoch}.pth'
        model = get_model()
        model.load_state_dict(torch.load(candidate, map_location='cpu', weights_only=False)['model_state_dict'])
        assert all(torch.equal(value, generator[key]) for key,value in model.generator.state_dict().items())
        camera = NativeCamera(model)
        rows = collect(camera, initial(DEV_SEED,64), pool, workers, batch, f'dev_{epoch}')
        record = summarize(rows,{})
        record.update(epoch=epoch, path=str(candidate), sha256=sha(candidate))
        selection.append(record)
        order = (-record['safe'],record['rmse_cm'],epoch)
        if selected is None or order < selected:
            selected, winner = order, camera
    chosen = min(selection,key=lambda r:(-r['safe'],r['rmse_cm'],r['epoch']))
    write_json(output/'native_selection.json',dict(selected=chosen,candidates=selection,
        criterion='64 disjoint live RGB development episodes, maximize safe then minimize lateral RMSE',
        exact_native_plant_check_cases=512))
    return winner, NativeCamera(anchor)


def evaluate(args):
    output = args.experiment / ('evaluation_own' if args.arm=='own' else 'evaluation_native')
    output.mkdir(exist_ok=False)
    bank = args.experiment/'bank'
    write_json(output/'protocol.json',dict(seed_calibration=CAL_SEED,seed_test=TEST_SEED,
        counts=COUNTS, alpha=.05, initial_box_cm_degrees=[[-60,-7],[60,7]], horizon_steps=160,
        evaluation='live RGB each physical step; both steering caps 30 degrees; separate CP by arm',
        camera_bank_protocol_sha256=sha(bank/'data.json'), device=args.device,
        source_sha256=sha(Path(__file__))))
    with ProcessPoolExecutor(args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        if args.arm == 'own':
            base=args.experiment/'own'
            model = OwnCamera(base,args.device)
            rgb=np.load(bank/'test_rgb.npy',mmap_mode='r')
            manifest=dict(np.load(bank/'test.npz'))
            y=torch.cat([model.observation(np.array(part)) for part in np.array_split(rgb,125)]).numpy()
            physical=y*np.array([85,np.rad2deg(.30)])
            e=abs(physical-manifest['states_cm_degrees'])
            write_json(output/'static.json',dict(n=len(e),mae_cm_deg=e.mean(axis=0).tolist(),
                p95_cm_deg=np.quantile(e,.95,axis=0).tolist(),
                projection_sha256=sha(base/'patch_projection.pt'), actor_sha256=sha(base/'actor.pt')))
            run_arm('own',model,output,pool,args.workers,args.batch)
        else:
            baseline,anchor=select_native(args.experiment,output,pool,args.workers,args.batch)
            run_arm('spvt',baseline,output,pool,args.workers,args.batch)
            run_arm('anchor',anchor,output,pool,args.workers,args.batch)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('experiment',type=Path)
    p.add_argument('arm',choices=['own','spvt'])
    p.add_argument('--device',default='cpu')
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--batch',type=int,default=96)
    a=p.parse_args()
    torch.set_num_threads(2)
    evaluate(a)
