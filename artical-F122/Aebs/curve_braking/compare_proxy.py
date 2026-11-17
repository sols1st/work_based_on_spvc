"""Train and blind-test matched Cx and invertible neural observation models in the braking task."""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import plant
from .evaluate import collect, digest, save
from .models import Actor, Proxy
from .verify import verify_checkpoint
from .vision import Encoder

SEEDS = (20261117501, 20261117502, 20261117503)
KINDS = ('linear', 'neural')
CAL_SEED, HOLD_SEED = 202611175081, 202611175082
REPO = Path('../external/dinov2')
WEIGHTS = Path('../external/dinov2_weights/dinov2_vits14_pretrain.pth')


def load(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def features_and_start(source):
    with np.load(source/'static.npz') as data:
        x = torch.tensor(plant.normalize(data['states']), dtype=torch.float32)
        parts = [torch.from_numpy(np.flatnonzero(data['split'] == key)) for key in ('train', 'val', 'test')]
    pca = load(source/'pca.pt')
    z = (load(source/'features.pt') - pca['mean']) @ pca['basis'] / pca['std']
    train = parts[0]
    design = torch.cat([z[train], torch.ones(len(train), 1)], dim=1)
    ridge = torch.linalg.solve(design.T @ design + 2 * torch.eye(design.shape[1]), design.T @ x[train])
    return x, z, parts, ridge


@torch.inference_mode()
def geometry(proxy):
    axis = torch.linspace(-1., 1., 17)
    grid = torch.cartesian_prod(axis, axis, axis)
    values = proxy.nominal(grid)
    design = torch.cat([grid, torch.ones_like(grid[:, :1])], 1).double()
    fitted = design @ torch.linalg.lstsq(design, values.double()).solution
    residual = values.double() - fitted
    return dict(grid='17 x 17 x 17 in normalized [-1,1]^3',
                best_affine_rmse=float(residual.square().mean().sqrt()),
                best_affine_max=float(residual.abs().max()),
                inverse_roundtrip_max=float((proxy.decode(values)-grid).abs().max()))


def train_one(arm, kind, seed, training):
    x, z, (train, val, test), ridge = training
    torch.manual_seed(seed)
    model = Proxy(kind, z.shape[1])
    model.bias.requires_grad_(False)
    with torch.no_grad():
        model.head.weight.copy_(ridge[:-1].T)
        model.head.bias.copy_(ridge[-1])
        torch.testing.assert_close(model.nominal(x[train]), x[train], rtol=0, atol=0)
    opt = torch.optim.Adam(model.parameters(), lr=.00015)
    gen = torch.Generator().manual_seed(seed)
    batch_history = hashlib.sha256()
    best = None
    validation = []
    started = time.perf_counter()
    for epoch in range(1, 36):
        order = train[torch.randperm(len(train), generator=gen)]
        batch_history.update(order.numpy().tobytes())
        for ids in order.split(96):
            y = model.head(z[ids])
            loss = (y-model.nominal(x[ids])).norm(dim=1).mean() + .1*(model.decode(y)-x[ids]).square().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            mse = float((model.decode(model.head(z[val]))-x[val]).square().mean())
            validation.append(mse)
            cond = float(torch.linalg.cond(model.matrix()))
            if cond < 10 and (best is None or mse < best[0]):
                best = mse, epoch, copy.deepcopy(model.state_dict())
    if best is None:
        raise ValueError('No invertible development checkpoint')
    model.load_state_dict(best[2])
    torch.save(model.state_dict(), arm/'proxy.pt')
    with torch.no_grad():
        y = model.head(z[test])
        err = model.decode(y) - x[test]
        save(arm/'training.json', dict(seed=seed, kind=kind, updates=35*(len(train)//96),
             trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
             batch_order_sha256=batch_history.hexdigest(), selected_epoch=best[1], development_mse=best[0],
             validation=validation, elapsed_seconds=time.perf_counter()-started,
             static_decoded_mae_physical=(err.abs().mean(0)*torch.from_numpy(plant.SCALE)).tolist(),
             static_decoded_mse=float(err.square().mean()), condition=float(torch.linalg.cond(model.matrix())),
             geometry=geometry(model), model_sha256=digest(arm/'proxy.pt')))
    return model.eval()


class ImageCache:
    """Reuse DINO only for bit-identical RGB rasters. Every visited state still calls render()."""
    def __init__(self, encoder):
        self.encoder = encoder
        self.features = {}
        self.calls = []
        self.encoded = 0

    def __call__(self, images):
        keys = [hashlib.sha256(im.tobytes()).digest() for im in images]
        self.calls.append([k.hex() for k in keys])
        missing = {}
        for key, image in zip(keys, images):
            if key not in self.features:
                missing[key] = image
        if missing:
            unique = list(missing.values())
            batch = unique + [unique[-1]]*(8-len(unique))
            values = self.encoder(batch)
            self.encoded += 8
            for key, value in zip(missing, values):
                self.features[key] = value.clone()
        return torch.stack([self.features[key] for key in keys])

    def run(self, n, seed, pca, model, actor):
        self.calls.clear()
        row = collect(n, seed, self, pca, model, actor)
        keys = np.array(self.calls, dtype='S64').reshape(-1, plant.HORIZON, 8)
        row['image_sha256'] = keys.transpose(0, 2, 1).reshape(-1, plant.HORIZON)[:n]
        return row


def metrics(hold, q):
    return dict(covered=int((hold['scores'] <= q).sum()), unsafe=int(hold['unsafe'].sum()),
                success=int(hold['success'].sum()), stopped=int(hold['parked'].sum()),
                lateral_mae_m=float(np.abs(hold['states'][:,:,0]).mean()),
                final_center_target_mae_m=float(np.abs(hold['states'][:,-1,2]-np.mean(plant.TARGET)).mean()),
                decoded_mae_physical=(np.abs(hold['errors']).mean(axis=(0,1))*plant.SCALE).tolist(),
                raw_latent_mean_l2=float(np.linalg.norm(hold['raw_latent_errors'],axis=2).mean()),
                score_mean=float(hold['scores'].mean()), min_lane_margin_m=float(hold['min_body_lane_margin'].min()),
                min_obstacle_margin_m=float(hold['min_body_obstacle_margin'].min()))


def main(source, out, device):
    torch.set_num_threads(3)
    out.mkdir(parents=True, exist_ok=False)
    base = json.loads((source/'evaluation/protocol.json').read_text())
    inputs = {name: digest(source/name) for name in ('features.pt','static.npz','pca.pt','actor.pt','barrier.pt')}
    if any(base['bindings'][name] != value for name, value in inputs.items()) or base['dino_sha256'] != digest(WEIGHTS):
        raise ValueError('Source package differs from the original registered artifacts')
    source_dir = Path(__file__).parent
    code = [source_dir/name for name in ('compare_proxy.py','evaluate.py','models.py','verify.py','plant.py','vision.py')]
    code += [source_dir.parent/'dino_latent/backbone.py']
    save(out/'protocol.json', dict(source=str(source), bindings=inputs, dino_sha256=base['dino_sha256'],
         software=dict(torch=torch.__version__, numpy=np.__version__), code={f.name:digest(f) for f in code},
         device=device, train_seeds=SEEDS, arms=KINDS, n_cal=199, n_hold=100, alpha=.05, cp_rank=190,
         calibration_seed=CAL_SEED, blind_seed=HOLD_SEED, horizon=plant.HORIZON, curvatures=plant.KAPPAS.tolist(),
         training='same train-only PCA, ridge visual start, C=I, bias fixed zero, neural initialized as identity; Adam 0.00015, 35 epochs, batch 96, select lowest val decode MSE with cond(C)<10',
         image_law='render the actual state at every step; exact SHA256 memoization only for frozen DINO; fixed 8-frame inference shape',
         score='max over all 14 time instants and all 3 normalized decoded residuals',
         outcomes=['trajectory CP radius','holdout trajectory coverage','goal stop and hazard counts','mean absolute lateral offset','terminal distance error from center of target interval','geometry after removing affine part'],
         fixed_components='actor, barrier, dynamics, PCA, backbone, camera, scenario law'))
    training = features_and_start(source)
    arms = {}
    for seed in SEEDS:
        for kind in KINDS:
            arm = out/f'{seed}_{kind}'
            arm.mkdir()
            arms[arm] = train_one(arm, kind, seed, training)
            print('trained', arm.name, flush=True)
    frozen = {arm.name: digest(arm/'proxy.pt') for arm in arms}
    save(out/'models_frozen.json', frozen)
    encoder = Encoder(REPO, WEIGHTS, device)
    with np.load(source/'static.npz') as data:
        static = data['images']
    reference = load(source/'features.pt')
    # Independently bind the archived feature rows to the original camera rasters.
    maximum = 0.
    for start in range(0,len(static),8):
        maximum = max(maximum,float((encoder(static[start:start+8])-reference[start:start+8]).abs().max()))
    if maximum > .001:
        raise ValueError(f'Static feature reconstruction maximum differs {maximum}')
    save(out/'static_feature_check.json',dict(rows=len(static),max_abs_feature_difference=maximum,
         source_sha256=inputs['static.npz'],reference_sha256=inputs['features.pt']))
    cache = ImageCache(encoder)
    pca = load(source/'pca.pt')
    actor = Actor()
    actor.load_state_dict(load(source/'actor.pt'))
    actor.eval()
    contracts = {}
    for arm, model in arms.items():
        cal = cache.run(199, CAL_SEED, pca, model, actor)
        np.savez_compressed(arm/'calibration.npz', **cal)
        q = float(np.sort(cal['scores'])[189])
        checks = verify_checkpoint(source/'barrier.pt',q)
        save(arm/'contract.json',dict(radius=q,rank=190,model_sha256=frozen[arm.name],
             calibration_sha256=digest(arm/'calibration.npz'),protocol_sha256=digest(out/'protocol.json')))
        contracts[arm.name] = digest(arm/'contract.json')
        save(arm/'certificate.json',dict(**checks, contract_sha256=contracts[arm.name],
             actor_sha256=inputs['actor.pt'],barrier_sha256=inputs['barrier.pt']))
        print('calibrated',arm.name,'q',q,'passed',checks['passed'],flush=True)
    save(out/'contracts_frozen.json',contracts)
    report = {}
    for arm, model in arms.items():
        hold = cache.run(100,HOLD_SEED,pca,model,actor)
        np.savez_compressed(arm/'holdout.npz',**hold)
        q = json.loads((arm/'contract.json').read_text())['radius']
        report[arm.name] = dict(q=q, **metrics(hold,q),
              passed=json.loads((arm/'certificate.json').read_text())['passed'],
              holdout_sha256=digest(arm/'holdout.npz'))
        print('blind',arm.name,report[arm.name]['success'],flush=True)
    if any(digest(arm/'contract.json') != contracts[arm.name] or digest(arm/'proxy.pt') != frozen[arm.name] for arm in arms):
        raise ValueError('Frozen components changed')
    save(out/'summary.json',dict(arms=report, useful_camera_frames=len(arms)*(199+100)*plant.HORIZON,
         unique_closed_loop_rasters=len(cache.features), closed_loop_dino_frame_slots_with_padding=cache.encoded))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--device',default='cuda:0')
    options = parser.parse_args()
    main(options.source,options.output,options.device)
