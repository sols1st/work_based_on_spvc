"""Matched warm-start training of the overhead crane camera readout and state proxy."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ..curve_braking.evaluate import digest, save
from ..overhead_crane.camera import Observation, SWAY_SCALE
from ..vision_sls.observation_coupling import apply_shears, make_shears


SEEDS = (817731, 817732, 817733)
FAMILIES = ('linear', 'neural')
EPOCHS = 12
BATCH = 32
LR = .0002


class ProxyObservation(Observation):
    def __init__(self, family):
        super().__init__()
        if family not in FAMILIES:
            raise ValueError(family)
        self.shears = make_shears(4) if family == 'neural' else torch.nn.ModuleList()

    def nominal(self, z):
        return apply_shears(z @ self.matrix().T, self.shears)

    def decode(self, y):
        return super().decode(apply_shears(y, self.shears, inverse=True))


def arm_path(root, family, seed):
    return root / f'{family}_{seed}'


def read(path):
    return json.loads(path.read_text())


@torch.no_grad()
def geometry(model, train_x, test_x):
    # Fit on training states only. Evaluate departure from the best fitted affine map
    # on a separate set of states; rotations and offsets alone do not count as nonlinearity.
    a = lambda x: np.column_stack([x.cpu().double().numpy(), np.ones(len(x))])
    coef = np.linalg.lstsq(a(train_x), model.nominal(train_x).cpu().double().numpy(), rcond=None)[0]
    target = model.nominal(test_x)
    residual = target.cpu().double().numpy() - a(test_x) @ coef
    rt = (model.decode(target)-test_x).abs()
    return dict(best_affine_test_rmse=float(np.sqrt(np.mean(residual**2))),
                best_affine_test_max_abs=float(np.abs(residual).max()), inverse_roundtrip_max=float(rt.max()))


def train(args):
    root, family, seed = args.root, args.family, args.seed
    out = arm_path(root, family, seed)
    out.mkdir(exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.manual_seed(seed)
    pack = read(root/'features.json')
    for name, key in [('static_index.npz','index'), ('features.pt','features')]:
        if digest(root/name) != pack[key]:
            raise ValueError(f'Changed feature input {name}')
    ids_data = np.load(root/'static_index.npz')
    features = torch.load(root/'features.pt', weights_only=True, map_location='cpu')
    x = torch.from_numpy(ids_data['labels']).float().to(args.device)
    tr, va, te = [torch.from_numpy(np.flatnonzero(ids_data['split'] == k)) for k in ('train','val','test')]
    model = ProxyObservation(family).to(args.device)
    saved = torch.load(args.base/'frontend.pt', weights_only=True, map_location=args.device)
    layout = model.load_state_dict(saved, strict=False)
    if layout.unexpected_keys or set(layout.missing_keys) != {n for n in model.state_dict() if n.startswith('shears.')}:
        raise ValueError('Invalid source initialization')
    with torch.no_grad():
        init_diff = (x[:64] @ model.matrix().T-model.nominal(x[:64])).abs().max().item()
    if init_diff != 0:
        raise ValueError('Paired state proxy must start identically')
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=.0001)
    order = torch.Generator().manual_seed(seed)
    batches = hashlib.sha256()
    def predict(ids):
        return model(features[ids].to(args.device))
    @torch.no_grad()
    def residuals(ids):
        errs = []
        raws = []
        for ix in ids.split(BATCH):
            z = x[ix.to(args.device)]
            y, _ = predict(ix)
            errs.append(model.decode(y)-z)
            raws.append(y-model.nominal(z))
        return torch.cat(errs).cpu().numpy(), torch.cat(raws).cpu().numpy()
    best = None
    curve = []
    for epoch in range(1, EPOCHS+1):
        perm = tr[torch.randperm(len(tr), generator=order)]
        batches.update(perm.numpy().tobytes())
        for ix in perm.split(BATCH):
            z = x[ix.to(args.device)]
            y, logits = predict(ix)
            points = torch.stack([z[:,:2]+model.offset, z[:,:2]+SWAY_SCALE*z[:,2:]], dim=1)
            heat = (-((model.grid-points[:,:,None])**2).sum(-1)/(.10**2*2)).softmax(-1)
            landmark = -(heat*logits.log_softmax(-1)).sum(-1).mean()
            loss = (y-model.nominal(z)).abs().mean() + (model.decode(y)-z).square().mean() + .10*landmark
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        err, _ = residuals(va)
        val = float(np.mean(np.square(err, dtype=np.float64)))
        curve.append(val)
        if best is None or val < best[0]:
            best = (val, epoch, copy.deepcopy(model.state_dict()))
    model.load_state_dict(best[2])
    torch.save(model.state_dict(), out/'frontend.pt')
    err, raw = residuals(te)
    np.savez_compressed(out/'static_test.npz', rows=te.numpy(), decoded_error=err, raw_error=raw)
    save(out/'training.json', dict(family=family, seed=seed, warm_start=digest(args.base/'frontend.pt'),
         features_manifest=digest(root/'features.json'), checkpoint=digest(out/'frontend.pt'),
         batch_order=batches.hexdigest(), epochs=EPOCHS, batch=BATCH, updates=EPOCHS*len(tr)//BATCH,
         lr=LR, device=args.device, init_max_delta=init_diff, n_parameters=sum(p.numel() for p in model.parameters()),
         best_epoch=best[1], validation_mse=best[0], validation_curve=curve, test_count=len(te),
         physical_test_mae_m=(np.abs(err).mean(0)*[1,1,SWAY_SCALE,SWAY_SCALE]).tolist(),
         raw_latent_test_mae=float(np.abs(raw).mean()), normalized_decoded_test_rmse=float(np.sqrt((err**2).mean())),
         **geometry(model,x[tr.to(args.device)],x[te.to(args.device)])))
    print(family, seed, 'best', best[:2], flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('base', type=Path)
    cli.add_argument('root', type=Path)
    cli.add_argument('family', choices=FAMILIES)
    cli.add_argument('seed', type=int, choices=SEEDS)
    cli.add_argument('--device', default='cpu')
    cli.add_argument('--threads', type=int, default=3)
    train(cli.parse_args())
