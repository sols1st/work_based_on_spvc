"""Recompute paired five-state trajectories, tube ranks, and saved numerical certificates."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from . import plant
from .compare_proxy import load, metrics
from .evaluate import digest, save
from .models import Actor, Proxy
from .verify import verify_checkpoint
from .vision import render


def read(path):
    return json.loads(path.read_text())


def require_equal(actual, expected, label):
    if actual != expected:
        raise ValueError(f'{label}: {actual} != {expected}')


@torch.inference_mode()
def check_trajectories(roll, model, actor, n, seed):
    xs = roll['states']
    rng = np.random.default_rng(seed)
    start = rng.uniform(plant.INIT_LO, plant.INIT_HI, (n,5))
    k = rng.choice(plant.KAPPAS,n)
    np.testing.assert_array_equal(k,roll['kappas'])
    np.testing.assert_array_equal(xs[:,0],start)
    measured = np.abs(roll['errors']).max(axis=(1,2))
    np.testing.assert_array_equal(measured,roll['scores'])
    lat, gap = plant.margins(xs,k[:,None])
    unsafe = ((lat<=0)|(gap<=0)).any(axis=1)
    stopped = xs[:,-1,3]==0
    success = stopped & ~unsafe & (xs[:,-1,2]>=plant.TARGET[0]) & (xs[:,-1,2]<=plant.TARGET[1])
    for key, value in [('unsafe',unsafe),('parked',stopped),('success',success),
                       ('min_body_lane_margin',lat.min(1)),('min_body_obstacle_margin',gap.min(1))]:
        np.testing.assert_allclose(roll[key],value,rtol=0,atol=3e-14)
    residual_max = 0.
    action_max = 0.
    for t in range(plant.HORIZON):
        x = xs[:,t]
        norm = torch.from_numpy(plant.normalize(x)).float()
        obs = model.decode(model.nominal(norm) + torch.from_numpy(roll['raw_latent_errors'][:,t]))
        residual_max = max(residual_max,float(np.abs(obs.numpy()-plant.normalize(x)-roll['errors'][:,t]).max()))
        u = actor(obs,torch.from_numpy(x[:,3]).float(),torch.from_numpy(x[:,4]).float(),torch.from_numpy(k).float())
        action_max = max(action_max,float(np.abs(u.numpy()-roll['actions'][:,t]).max()))
        np.testing.assert_allclose(plant.step(x,k,roll['actions'][:,t]),xs[:,t+1],rtol=0,atol=3e-14)
    if max(residual_max,action_max)>3e-5:
        raise ValueError('Residual/action inconsistent with the frozen models')
    for i in [0,n-1]:
        for t in range(plant.HORIZON):
            h = hashlib.sha256(render(xs[i,t],k[i]).tobytes()).hexdigest()
            require_equal(h,roll['image_sha256'][i,t].decode(),'image hash')
    return dict(trajectories=n,steps=n*plant.HORIZON,rerendered_hashes=2*plant.HORIZON,
                decoded_max_abs_difference=residual_max,action_max_abs_difference=action_max)


def main(root):
    torch.set_num_threads(2)
    protocol = read(root/'protocol.json')
    source = Path(protocol['source'])
    for name, value in protocol['bindings'].items():
        require_equal(digest(source/name),value,name)
    files = {f.name:f for f in Path(__file__).parent.glob('*.py')}
    files['backbone.py'] = Path(__file__).parent.parent/'dino_latent/backbone.py'
    for name, value in protocol['code'].items():
        require_equal(digest(files[name]),value,name)
    orders = set()
    tests = {}
    reports = read(root/'summary.json')['arms']
    frozen = read(root/'models_frozen.json')
    contracts = read(root/'contracts_frozen.json')
    actor = Actor()
    actor.load_state_dict(load(source/'actor.pt'))
    for seed in protocol['train_seeds']:
        for kind in protocol['arms']:
            name = f'{seed}_{kind}'
            arm = root/name
            proxy = Proxy(kind,192)
            proxy.load_state_dict(load(arm/'proxy.pt'))
            contract = read(arm/'contract.json')
            stats = read(arm/'training.json')
            require_equal(digest(arm/'proxy.pt'), frozen[name], 'frozen model')
            require_equal(stats['model_sha256'], frozen[name], 'trained model')
            require_equal(contract['model_sha256'], frozen[name], 'calibration model')
            require_equal(digest(arm/'contract.json'), contracts[name], 'frozen CP')
            require_equal(digest(root/'protocol.json'),contract['protocol_sha256'],'protocol')
            require_equal(digest(arm/'calibration.npz'), contract['calibration_sha256'], 'calibration data')
            require_equal(digest(arm/'holdout.npz'), reports[name]['holdout_sha256'], 'blind data')
            orders.add((seed, stats['batch_order_sha256']))
            q = contract['radius']
            cert = verify_checkpoint(source/'barrier.pt',q)
            stored = read(arm/'certificate.json')
            require_equal(stored['actor_sha256'],protocol['bindings']['actor.pt'],'actor in certificate')
            require_equal(stored['barrier_sha256'],protocol['bindings']['barrier.pt'],'barrier in certificate')
            require_equal(stored['contract_sha256'],contracts[name],'certified tube')
            require_equal(cert['passed'],stored['passed'],'numerical certificate')
            for check, value in cert['checks'].items():
                np.testing.assert_allclose(value,stored['checks'][check],atol=1e-12,rtol=1e-12)
            tests[name] = {}
            for split,n,rng_seed in [('calibration',protocol['n_cal'],protocol['calibration_seed']),
                                     ('holdout',protocol['n_hold'],protocol['blind_seed'])]:
                with np.load(arm/f'{split}.npz') as roll:
                    tests[name][split] = check_trajectories(roll,proxy,actor,n,rng_seed)
                    if split=='calibration':
                        rank = int(np.ceil((n+1)*(1-protocol['alpha'])))
                        require_equal(rank,contract['rank'],'CP order statistic')
                        require_equal(float(np.sort(roll['scores'])[rank-1]),q,'CP radius')
                    else:
                        for key,value in metrics(roll,q).items():
                            np.testing.assert_allclose(value,reports[name][key],rtol=1e-12,atol=1e-12)
    require_equal(len(orders),len(protocol['train_seeds']),'identical batch order within each pair')
    helper = Path(__file__).parent.parent/'vision_sls/observation_coupling.py'
    return dict(reconstructed=tests, coupling_implementation_sha256=digest(helper),
                checker_sha256=digest(Path(__file__)), passed=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--save',action='store_true')
    args=p.parse_args()
    checks=main(args.root)
    if args.save:
        save(args.root/'independent_check.json',checks)
    print('Verified',len(checks['reconstructed']),'frozen arms.')
