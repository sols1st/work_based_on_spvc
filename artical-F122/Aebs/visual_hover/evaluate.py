"""Frozen image-feedback trajectories, split CP, independent certificate and controls."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..curve_braking.evaluate import digest, save
from ..curve_braking.models import Proxy
from ..curve_braking.vision import Encoder
from ..lane_keeping.cp import conformal_radius
from . import certificate, plant
from .perception import render

CAL_SEED = 171701
TEST_SEED = 171702


@torch.inference_mode()
def collect(scenario, design, encoder, frontend, model, batch):
    initial, winds, lights = scenario
    all_states, all_errors, all_raw, all_actions = [], [], [], []
    for first in range(0, len(initial), batch):
        # Pad the last batch for the same inference shape in calibration and holdout.
        ids = np.minimum(np.arange(first, first+batch), len(initial)-1)
        count = min(len(initial)-first, batch)
        state = initial[ids].copy()
        states, errors, raw, actions = [state], [], [], []
        for k in range(plant.HORIZON):
            images = [render(s[:3], light) for s,light in zip(state, lights[ids])]
            features = encoder(images)
            coords = ((features-frontend['mean']) @ frontend['basis']) / frontend['std']
            latent = model.head(coords)
            obs = model.decode(latent).numpy().astype(np.float64)
            error = obs - state[:,:3]
            # In exact arithmetic raw = C @ decoded_error; log the actual latent too.
            raw.append(latent.numpy() - state[:,:3] @ model.matrix().numpy().T)
            errors.append(error)
            action = plant.control(obs, state[:,3:], design)
            actions.append(action)
            state = plant.step(state, action, winds[ids,k])
            states.append(state)
        for dest, items in [(all_states,states),(all_errors,errors),(all_raw,raw),(all_actions,actions)]:
            dest.append(np.stack(items, axis=1)[:count])
    states, errors, raw, actions = [np.concatenate(v) for v in (all_states,all_errors,all_raw,all_actions)]
    if not all(np.isfinite(v).all() for v in (states, errors, raw, actions)):
        raise ValueError('nonfinite image rollout')
    return dict(states=states, errors=errors, raw=raw, actions=actions, winds=winds, lights=lights,
                scores=np.abs(errors).max(axis=(1,2)))


def baseline(scenario, design, use_position):
    state, winds, _ = scenario
    states = [state]
    for k in range(plant.HORIZON):
        action = plant.control(state[:,:3], state[:,3:], design, use_vision=use_position)
        state = plant.step(state, action, winds[:, k])
        states.append(state)
    return np.stack(states, axis=1)


def metrics(states):
    safe, hover, distance = plant.outcomes(states)
    return dict(n=len(states), safe=int(safe.sum()), hover=int(hover.sum()),
                mean_final_distance_m=float(distance.mean()), max_final_distance_m=float(distance.max()))


def bindings(root, weights, repo):
    code = Path(__file__).resolve().parent
    files = list(code.glob('*.py')) + [
        code.parent/'curve_braking/vision.py', code.parent/'curve_braking/models.py',
        code.parent/'curve_braking/evaluate.py', code.parent/'dino_latent/backbone.py',
        code.parent/'lane_keeping/cp.py']
    return dict(source={str(f.relative_to(code.parent)):digest(f) for f in sorted(files)},
                dino_source={str(f.relative_to(repo)):digest(f) for f in
                             [repo/'hubconf.py', *sorted((repo/'dinov2').rglob('*.py'))]},
                frontend=digest(root/'frontend.pt'), weights=digest(weights),
                static=digest(root/'static.npz'))


def run(args):
    torch.set_num_threads(3)
    root = args.root
    out = root/'evaluation'
    out.mkdir(exist_ok=False)
    search = certificate.synthesize()
    save(out/'synthesis.json', search)
    design = search['selected']
    frozen = bindings(root,args.weights,args.repo)
    protocol = dict(bindings=frozen, design=design, synthesis=digest(out/'synthesis.json'),
                    calibration=[199,CAL_SEED], holdout=[100,TEST_SEED],
                    dt=plant.DT, horizon=plant.HORIZON, alpha=.05, batch=args.batch,
                    score='max absolute normalized decoded error over 24 observations and all 3 axes',
                    controls=['image feedback','velocity damping','perfect position'],
                    torch=str(torch.__version__), device=args.device)
    save(out/'protocol.json',protocol)
    enc = Encoder(args.repo,args.weights,args.device)
    front = torch.load(root/'frontend.pt', map_location='cpu',weights_only=True)
    model = Proxy('linear',front['basis'].shape[1])
    model.load_state_dict(front['model'])
    model.eval()
    cal = collect(plant.scenarios(199,CAL_SEED),design,enc,front,model,args.batch)
    np.savez_compressed(out/'calibration.npz',**cal)
    q,rank = conformal_radius(cal['scores'],.05)
    save(out/'contract.json',dict(q=q,rank=rank,n=199,protocol=digest(out/'protocol.json'),
                                  calibration=digest(out/'calibration.npz')))
    audit = certificate.verify(design,q)
    contract_hash = digest(out/'contract.json')
    audit['contract_sha256'] = contract_hash
    save(out/'certificate.json',audit)
    print('frozen q and audit', q, audit['passed'],flush=True)
    scenario = plant.scenarios(100,TEST_SEED)
    held = collect(scenario,design,enc,front,model,args.batch)
    np.savez_compressed(out/'holdout.npz',**held)
    report = dict(image=metrics(held['states']), damping=metrics(baseline(scenario,design,False)),
                  perfect=metrics(baseline(scenario,design,True)),
                  covered=int(np.count_nonzero(held['scores']<=q)), q=q, rank=rank,
                  physical_halfwidth_m=(q*plant.SCALE).tolist(),
                  audit_passed=audit['passed'], holdout=digest(out/'holdout.npz'),
                  certificate=digest(out/'certificate.json'),contract=contract_hash)
    if (bindings(root,args.weights,args.repo)!=frozen or digest(out/'contract.json')!=contract_hash):
        raise RuntimeError('frozen artifacts changed')
    save(out/'summary.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    cli=argparse.ArgumentParser()
    cli.add_argument('root',type=Path)
    cli.add_argument('--repo',type=Path,default=Path('../external/dinov2'))
    cli.add_argument('--weights',type=Path,default=Path('../external/dinov2_weights/dinov2_vits14_pretrain.pth'))
    cli.add_argument('--device',default='cuda:0')
    cli.add_argument('--batch',type=int,default=8)
    run(cli.parse_args())
