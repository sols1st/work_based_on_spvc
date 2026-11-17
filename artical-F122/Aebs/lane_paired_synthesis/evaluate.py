"""Independent trajectory calibration and paired blind tests on five registered curves."""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import torch

from ..lane_contract_comparison.collect import ARMS, maxima, parameters, scores
from ..lane_keeping import core
from ..lane_keeping.cp import conformal_radius
from ..lane_keeping.learned_sbc import sha256
from ..lane_keeping.policy import load_actor
from ..lane_keeping.renderer import RoadRenderer
from ..lane_keeping.robust_trajectory_cp import DevicePatchObserver


KAPPAS = (-.02, -.01, 0., .01, .02)
HORIZON = 80
SPLITS = {'calibration':199,'holdout':100}
SEEDS = {'fixed':(202611178291,202611178292),'trained':(202611178301,202611178302)}
REFERENCE = Path('results/lane_contract_comparison_20261117_v1')
PACKAGE = Path('results/lane_robust_synthesis_20261116_v2/package')
REPO = Path('../external/dinov2')
WEIGHTS = Path('../external/dinov2_weights/dinov2_vits14_pretrain.pth')


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    # On resume, never replace an earlier protocol or threshold.
    content = json.dumps(value, indent=2, allow_nan=False) + '\n'
    if path.exists():
        if path.read_text() != content:
            raise ValueError(f'frozen file differs: {path}')
        return
    tmp = path.with_suffix('.partial')
    tmp.write_text(content)
    tmp.replace(path)


def save_arrays(path, data):
    tmp = path.with_suffix('.partial.npz')
    np.savez_compressed(tmp, **data)
    tmp.replace(path)


def draws(seed, n):
    children = np.random.SeedSequence(seed).spawn(n)
    initial, curvatures = [], []
    for child in children:
        rng = np.random.default_rng(child)
        initial.append(rng.uniform(core.INITIAL.low, core.INITIAL.high))
        curvatures.append(rng.choice(KAPPAS))
    return np.array(initial), np.array(curvatures)


def render(frame):
    x, kappa = frame
    return RoadRenderer(192,192).render(x, kappa)


def register(args, actors):
    reference = read(REFERENCE/'protocol.json')
    bindings = {}
    for arm, folder in actors.items():
        bindings[arm] = {name:sha256(folder/(name+'.pt')) for name in ('actor','sbc')}
    for name, path in [('observer',PACKAGE/'patch_projection.pt'),('dino_weights',WEIGHTS)]:
        if sha256(path) != reference['binding'][name+'_sha256']:
            raise ValueError(f'{name} differs from frozen reference')
    sources = [Path(__file__), Path('Aebs/lane_contract_comparison/collect.py'),
               *[Path('Aebs/lane_keeping')/f for f in
                 ['core.py','renderer.py','robust_trajectory_cp.py','policy.py','patch_perception.py','cp.py']],
               Path('Aebs/dino_latent/backbone.py')]
    protocol = dict(mode=args.mode, actor_barrier=bindings, horizon=HORIZON, kappas=KAPPAS,
                    counts=SPLITS,seeds=SEEDS[args.mode],alpha=.05,batch=args.batch,
                    device=args.device, reference=sha256(REFERENCE/'protocol.json'),
                    development=reference['development'], observer=reference['binding']['observer_sha256'],
                    dino_weights=reference['binding']['dino_weights_sha256'],
                    sources={str(p):sha256(p) for p in sources},
                    observation='float32 camera; float64 policy and plant; no stopping or rejection',
                    sampling='independent uniform initial rectangle and one uniform registered curvature per trajectory; paired across policies',
                    outcomes='primary mean per-trajectory mean absolute lateral error on abs(k)=0.02; all-curve counterpart and max lateral offset, violations, CP coverage',
                    paired_intervals='whole-episode paired bootstrap of mean lateral error difference, 4000 reps, one interval per comparison, conditional on frozen models',
                    audit=dict(extra_rad=[0.,.045,.0675,.072,.081,.0855,.09],curvatures=KAPPAS,
                               grid=40,max_depth=16,error_subdivisions=2,max_pending=30000,
                               heading_requirement_deg=2.,heading_scope='one-step change caused by observation at identical state, same curvature, versus nominal observation; separate from full safety'),
                    calibration_rule='freeze all final tubes before sampling any blind episode')
    if args.mode == 'trained':
        protocol['training_protocol'] = sha256(args.root/'protocol.json')
        protocol['frozen_models'] = {name:sha256(folder/'frozen.json') for name,folder in actors.items()}
    save(args.root/'image_protocol.json',protocol)
    return protocol,sha256(args.root/'image_protocol.json')


@torch.inference_mode()
def collect(root, source, folder, split, actor, observer, pool, batch, token):
    n = SPLITS[split]
    start, curves = draws(SEEDS[source][list(SPLITS).index(split)], n)
    target = folder/(split+'.npz')
    if target.exists():
        with np.load(target) as z:
            if str(z['protocol']) != token:
                raise ValueError('archived split belongs to a different protocol')
            return dict(z)
    chunks = []
    cache = folder/(split+'_batches')
    cache.mkdir(exist_ok=True, parents=True)
    for first in range(0,n,batch):
        path = cache/f'{first:04}.npz'
        if path.exists():
            with np.load(path) as z:
                if str(z['protocol']) != token:
                    raise ValueError('archived batch belongs to a different protocol')
                chunks.append({k:z[k] for k in z.files if k!='protocol'})
            continue
        ids=np.minimum(np.arange(first,first+batch),n-1)
        count=min(n-first,batch)
        x, ks = start[ids].copy(), curves[ids]
        states, observations, actions, errors = [x[:count].copy()], [], [], []
        digests = [hashlib.sha256() for _ in range(count)]
        for _ in range(HORIZON):
            images = np.stack(pool.map(render, zip(x,ks)))
            y = observer(torch.from_numpy(images)).cpu().numpy()
            obs = torch.from_numpy(np.column_stack([y, ks/core.KAPPA_SCALE])).double()
            u = actor(obs).flatten().numpy()
            e = y-x/[core.E_SCALE,core.THETA_SCALE]
            for h,frame in zip(digests,images):
                h.update(frame.tobytes())
            x = core.dynamics_batch(x, u*core.PHI_MAX, ks)
            states.append(x[:count].copy())
            observations.append(y[:count].copy())
            actions.append(u[:count].copy())
            errors.append(e[:count].copy())
        data = dict(states=np.stack(states,1),observations=np.stack(observations,1),
                    actions=np.stack(actions,1),errors=np.stack(errors,1),kappas=ks[:count],
                    image_sha256=np.array([h.hexdigest() for h in digests]))
        if not all(np.isfinite(data[key]).all() for key in ('states','observations','actions','errors')):
            raise ValueError('nonfinite rollout')
        save_arrays(path,dict(**data,protocol=np.array(token)))
        chunks.append(data)
        if first%(batch*4)==0:
            print(folder.name,split,first+count,n,flush=True)
    data = {k:np.concatenate([part[k] for part in chunks]) for k in chunks[0]}
    np.testing.assert_array_equal(data['states'][:,0],start)
    np.testing.assert_array_equal(data['kappas'],curves)
    save_arrays(target,dict(**data,protocol=np.array(token)))
    return data


def stats(data, s, q):
    x = data['states'][:,1:]
    e = np.abs(x[:,:,0])
    unsafe = (np.abs(x)/[core.E_SCALE,core.THETA_SCALE]>1).any(axis=(1,2))
    out = {}
    for name, ids in [('all',np.ones(len(x),bool)),('higher_curvature',np.abs(data['kappas'])==.02)]:
        out[name] = dict(n=int(ids.sum()),covered=int((s[ids]<=q).sum()),unsafe=int(unsafe[ids].sum()),
                         mean_abs_e_m=float(e[ids].mean()),mean_peak_e_m=float(e[ids].max(axis=1).mean()),
                         mean_abs_heading_rad=float(np.abs(x[ids,:,1]).mean()))
    return out


def paired(all_data):
    out = {}
    for baseline,candidate in [('uniform','diagonal'),('uniform','joint'),('diagonal','joint')]:
        a,b = all_data[baseline],all_data[candidate]
        np.testing.assert_array_equal(a['states'][:,0],b['states'][:,0])
        np.testing.assert_array_equal(a['kappas'],b['kappas'])
        delta=np.abs(b['states'][:,1:,0]).mean(1)-np.abs(a['states'][:,1:,0]).mean(1)
        for group, d in [('all',delta), ('higher_curvature',delta[np.abs(a['kappas'])==.02])]:
            rng=np.random.default_rng(202611178305)
            boot=rng.choice(d,size=(4000,len(d)),replace=True).mean(1)
            out[f'{candidate}_minus_{baseline}_{group}']=dict(n=len(d),delta_mean_abs_e_m=float(d.mean()),
                bootstrap95=np.quantile(boot,[.025,.975]).tolist(),improved=int((d<0).sum()),
                tied=int((d==0).sum()))
    return out


def main(args):
    args.root.mkdir(exist_ok=True,parents=True)
    folders = {'fixed':PACKAGE} if args.mode=='fixed' else {a:args.root/a for a in ARMS}
    protocol, token = register(args,folders)
    scale = protocol['development']['scale']
    networks = {k:load_actor(f/'actor.pt').double().eval().requires_grad_(False) for k,f in folders.items()}
    for key, folder in folders.items():
        if sha256(folder/'actor.pt')!=protocol['actor_barrier'][key]['actor']:
            raise ValueError('registered actor changed')
    # Pool workers only run the original CPU renderer. Start them before loading CUDA.
    with mp.get_context('fork').Pool(args.workers) as pool:
        torch.set_num_threads(2)
        observer=DevicePatchObserver(PACKAGE/'patch_projection.pt',args.device,REPO,WEIGHTS).eval()
        holdout, arms, locks = {}, {}, {}
        for split in SPLITS:
            for key,net in networks.items():
                output=args.root/key
                output.mkdir(exist_ok=True)
                data = collect(args.root,args.mode,output,split,net,observer,pool,args.batch,token)
                scored = scores(maxima(data,net),scale)
                for arm in ARMS if key=='fixed' else (key,):
                    result_dir=args.root/arm
                    result_dir.mkdir(exist_ok=True)
                    path=result_dir/'final_contract.json'
                    if split=='calibration':
                        q,rank=conformal_radius(scored[arm],.05)
                        fit=dict(q=q,rank=rank,**parameters(arm,q,scale),arm=arm,
                                 actor_sha256=protocol['actor_barrier'][key]['actor'],
                                 sbc_sha256=protocol['actor_barrier'][key]['sbc'],
                                 calibration=sha256(output/'calibration.npz'),protocol=token)
                        save(path,fit)
                        locks[arm]=sha256(path)
                    else:
                        fit=read(path)
                        if sha256(path)!=locks[arm]:
                            raise ValueError('frozen threshold changed')
                        arms[arm]=stats(data,scored[arm],fit['q'])
                        holdout[arm]=data
            if split=='calibration':
                save(args.root/'all_tubes_frozen.json',dict(contracts=locks, protocol=token))
                print('all thresholds frozen',flush=True)
        save(args.root/'blind_summary.json',dict(protocol=token,contracts=locks,arms=arms,
            paired=paired(holdout),holdouts={k:sha256(args.root/k/'holdout.npz') for k in networks}))
    if sha256(args.root/'image_protocol.json')!=token:
        raise ValueError('protocol changed')


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('mode',choices=['fixed','trained'])
    cli.add_argument('root',type=Path)
    cli.add_argument('--batch',type=int,default=8)
    cli.add_argument('--workers',type=int,default=6)
    cli.add_argument('--device',default='cuda:1')
    main(cli.parse_args())
