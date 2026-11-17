"""Paired visual projection and state-proxy training on the frozen hover benchmark."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ..curve_braking.evaluate import digest, save
from ..curve_braking.models import Proxy
from ..curve_braking.vision import Encoder
from . import plant


SOURCE = Path('results/visual_hover_6d_20261117_v1')
OUTPUT = Path('results/hover_proxy_comparison_20261117_v1')
REPO = Path('../external/dinov2')
WEIGHTS = Path('../external/dinov2_weights/dinov2_vits14_pretrain.pth')
SEEDS = (617201, 617202, 617203)
CALIBRATION = (199, 617211)
HOLDOUT = (100, 617212)
KINDS = ('linear', 'neural')
EPOCHS = 35
BATCH = 96


def read(path):
    return json.loads(path.read_text())


def stage_features(source, output, repo, weights, device, batch):
    output.mkdir(parents=True, exist_ok=True)
    if (output/'protocol.json').exists():
        raise FileExistsError('use the existing frozen protocol or a new result directory')
    protocol = dict(
        seeds=list(SEEDS), calibration=list(CALIBRATION), holdout=list(HOLDOUT),
        static_batches=BATCH, train_epochs=EPOCHS, lr=.00018, image_batch=batch,
        score='max over 24 observations and three normalized decoded position coordinates',
        score_raw='max over 24 observations and three raw latent residual coordinates, diagnostic only',
        selection='lowest validation decoded MSE subject to cond(C)<10; earliest epoch wins ties',
        geometry='best affine least squares residual on a 13^3 normalized position grid in [-1,1]^3',
        optimizer='Adam; common minibatch order per pair; identical ridge warm start',
        source_files={name:digest(source/name) for name in ['static.npz','frontend.pt','evaluation/synthesis.json']},
        dino_weights=digest(weights), torch=str(torch.__version__),
        encoder_device=device, repo_sources={str(p.relative_to(repo)):digest(p) for p in
                                            [repo/'hubconf.py', *sorted((repo/'dinov2').rglob('*.py'))]},
        train_source=digest(Path(__file__)), design=read(source/'evaluation/synthesis.json')['selected'])
    save(output/'protocol.json', protocol)
    data = np.load(source/'static.npz', allow_pickle=False)
    images = data['images']
    frontend = torch.load(source/'frontend.pt', map_location='cpu', weights_only=True)
    encoder = Encoder(repo, weights, device)
    coords = []
    for first in range(0, len(images), batch):
        ids = np.minimum(np.arange(first,first+batch),len(images)-1)
        features = encoder(images[ids])
        scores = ((features-frontend['mean']) @ frontend['basis']) / frontend['std']
        coords.append(scores[:min(len(images)-first,batch)])
    static = torch.cat(coords)
    rows = [hashlib.sha256(im.tobytes()).hexdigest() for im in images]
    if len(rows) != len(set(rows)) or len(static) != len(images):
        raise ValueError('duplicate source image or mismatched number of feature rows')
    torch.save(dict(coords=static, image_digests=rows),output/'features.pt')
    save(output/'features.json',dict(features=digest(output/'features.pt'), protocol=digest(output/'protocol.json'),
                                  distinct_images=len(rows), shape=list(static.shape)))


def make_pair(dim, ridge, seed):
    torch.manual_seed(seed)
    models = {k:Proxy(k, dim) for k in KINDS}
    for m in models.values():
        m.bias.requires_grad_(False)
        with torch.no_grad():
            m.head.weight.copy_(ridge[:-1].T)
            m.head.bias.copy_(ridge[-1])
    return models


@torch.no_grad()
def geometry(model):
    m = copy.deepcopy(model).double().eval()
    axis = torch.linspace(-1,1,13,dtype=torch.float64)
    grid = torch.cartesian_prod(axis,axis,axis)
    latent = m.nominal(grid)
    features = torch.column_stack((grid,torch.ones_like(grid[:,0])))
    residual = latent - features @ torch.linalg.lstsq(features,latent).solution
    return dict(affine_rmse=float(residual.square().mean().sqrt()),
                affine_max=float(residual.abs().max()),
                inverse_max=float((m.decode(latent)-grid).abs().max()),
                condition=float(torch.linalg.cond(m.matrix())))


def train(source, output):
    protocol=read(output/'protocol.json')
    if (digest(source/'static.npz') != protocol['source_files']['static.npz']
            or digest(output/'features.pt') != read(output/'features.json')['features']):
        raise ValueError('training artifacts changed')
    features=torch.load(output/'features.pt',map_location='cpu',weights_only=True)
    data=np.load(source/'static.npz',allow_pickle=False)
    rows = [hashlib.sha256(im.tobytes()).hexdigest() for im in data['images']]
    if features['image_digests'] != rows:
        raise ValueError('static features do not match source image rows')
    z=features['coords']
    x=torch.tensor(data['position'],dtype=torch.float32)
    tr,va,te=[torch.from_numpy(np.flatnonzero(data['split']==part)) for part in ['train','val','test']]
    inputs=torch.column_stack((z[tr],torch.ones(len(tr))))
    ridge=torch.linalg.solve(inputs.T@inputs+2*torch.eye(inputs.shape[1]), inputs.T@x[tr])
    for seed in SEEDS:
        models=make_pair(z.shape[1],ridge,seed)
        with torch.no_grad():
            if not (torch.equal(models['linear'].nominal(x),models['neural'].nominal(x)) and
                    torch.equal(models['linear'].head(z),models['neural'].head(z))):
                raise ValueError('paired initial predictions differ')
        gen=torch.Generator().manual_seed(seed)
        optim={k:torch.optim.Adam(m.parameters(),lr=protocol['lr']) for k,m in models.items()}
        best={}; history={k:[] for k in KINDS}; schedule=hashlib.sha256()
        for epoch in range(1,EPOCHS+1):
            for ids in tr[torch.randperm(len(tr),generator=gen)].split(BATCH):
                schedule.update(ids.numpy().tobytes())
                for k,model in models.items():
                    latent=model.head(z[ids])
                    loss=((latent-model.nominal(x[ids])).norm(dim=1).mean()
                          +.1*(model.decode(latent)-x[ids]).square().mean())
                    optim[k].zero_grad()
                    loss.backward()
                    optim[k].step()
            with torch.no_grad():
                for k,model in models.items():
                    mse=float((model.decode(model.head(z[va]))-x[va]).square().mean())
                    condition=float(torch.linalg.cond(model.matrix()))
                    history[k].append(dict(epoch=epoch,mse=mse,condition=condition))
                    if condition<10 and (k not in best or mse<best[k][0]):
                        best[k]=(mse,epoch,copy.deepcopy(model.state_dict()))
        for k,model in models.items():
            folder=output/f'{seed}_{k}'; folder.mkdir(exist_ok=True)
            mse,epoch,state=best[k]
            model.load_state_dict(state)
            torch.save(state,folder/'model.pt')
            with torch.no_grad():
                y=model.head(z[te]); err=model.decode(y)-x[te]
                report=dict(seed=seed,kind=k,chosen_epoch=epoch,validation_mse=mse,
                            test_mae_m=(err.abs().mean(0).numpy()*plant.SCALE).tolist(),
                            test_decoded_mse=float(err.square().mean()), test_raw_mae=float((y-model.nominal(x[te])).abs().mean()),
                            geometry=geometry(model), trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
                            updates=EPOCHS*((len(tr)+BATCH-1)//BATCH), history=history[k],
                            batches=schedule.hexdigest(), model=digest(folder/'model.pt'))
            save(folder/'training.json',report)
    save(output/'all_models_frozen.json',{f'{s}_{k}':digest(output/f'{s}_{k}'/'model.pt') for s in SEEDS for k in KINDS})


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('stage',choices=['features','train'])
    cli.add_argument('--source',type=Path,default=SOURCE)
    cli.add_argument('--output',type=Path,default=OUTPUT)
    cli.add_argument('--repo',type=Path,default=REPO)
    cli.add_argument('--weights',type=Path,default=WEIGHTS)
    cli.add_argument('--device',default='cuda:0')
    cli.add_argument('--batch',type=int,default=8)
    args=cli.parse_args()
    torch.set_num_threads(2)
    if args.stage=='features':
        stage_features(args.source,args.output,args.repo,args.weights,args.device,args.batch)
    else:
        train(args.source,args.output)
