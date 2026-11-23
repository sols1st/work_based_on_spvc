"""Frozen three-coordinate perception proxy ablation for the five-state braking task."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ..curve_braking import plant
from ..curve_braking.models import Proxy as VisualProxy
from ..curve_braking.vision import Encoder, render
from .proxy_architecture_compare import run, sha, write
from .proxy_depth_data import ARCHITECTURES

SOURCE = Path('results/curve_braking_5d_20261117')
MODEL = Path('results/curve_braking_proxy_comparison_20261117_v1/20261117501_linear/proxy.pt')
BUDGETS = (1920, 6000, 12000)
PARTS = (('train',12000),('development',1000),('calibration',1000),('test',2000))


def load_vision():
    pca = torch.load(SOURCE/'pca.pt', map_location='cpu', weights_only=True)
    proxy = VisualProxy('linear', pca['basis'].shape[1])
    proxy.load_state_dict(torch.load(MODEL, map_location='cpu', weights_only=True))
    return pca, proxy.eval().requires_grad_(False)


def collect(out, device):
    out.mkdir(parents=True, exist_ok=True)
    path = out/'bank.npz'
    if path.exists():
        raise FileExistsError(path)
    with np.load(SOURCE/'static.npz') as old:
        historical = {hashlib.sha256(im.tobytes()).hexdigest() for im in old['images']}
    seen = set(historical)
    pca, proxy = load_vision()
    encoder = Encoder('../external/dinov2', '../external/dinov2_weights/dinov2_vits14_pretrain.pth', device)
    data = {key:[] for key in ('states','curvatures','sha256','split','visual')}
    rejected = {}
    examples = []
    for i,(name,count) in enumerate(PARTS):
        rng = np.random.default_rng(2026112230+i)
        done = rejected[name] = 0
        while done < count:
            images = []
            while len(images) < min(8, count-done):
                state = rng.uniform([-.54,-.15,2.5],[.54,.15,6.3])
                kappa = rng.choice(plant.KAPPAS)
                im = render(state,kappa)
                digest = hashlib.sha256(im.tobytes()).hexdigest()
                if digest in seen:
                    rejected[name] += 1
                    continue
                seen.add(digest)
                images.append(im)
                if done == 0 and len(images) == 1:
                    examples.append(im)
                for key,val in [('states',state),('curvatures',kappa),('sha256',digest),('split',name)]:
                    data[key].append(val)
            # Use the same 8-frame inference shape as the historical protocol.
            batch = images+[images[-1]]*(8-len(images))
            features = encoder(batch)[:len(images)]
            z = (features-pca['mean']) @ pca['basis'] / pca['std']
            with torch.no_grad():
                data['visual'].extend(proxy.head(z).numpy())
            done += len(images)
            if done % 1000 == 0:
                print(name,done,flush=True)
    np.savez_compressed(path, **data, x=plant.normalize(np.array(data['states'])).astype('float32'),
                        C=proxy.matrix().numpy(), scales=plant.SCALE)
    np.save(out/'examples.npy',np.stack(examples))
    write(out/'bank.json',dict(counts=dict(PARTS), seeds=[2026112230+i for i in range(4)], rejected=rejected,
          bank_sha256=sha(path), bindings={str(p):sha(p) for p in (MODEL,SOURCE/'pca.pt',SOURCE/'static.npz',
          Path('../external/dinov2_weights/dinov2_vits14_pretrain.pth'))},
          historical_image_count=len(historical), total_unique_images=len(seen)-len(historical),
          source_sha256=sha(__file__), renderer_sha256=sha(Path(__file__).parents[1]/'curve_braking/vision.py'),
          image_law='same uniform static state domain as original, kappa +/-0.02, fixed nominal camera'))


def train(out):
    manifest=json.loads((out/'bank.json').read_text())
    if manifest['bank_sha256'] != sha(out/'bank.npz'):
        raise ValueError('Observation identity mismatch')
    with np.load(out/'bank.npz') as z:
        x,y,C,units,split=(z[k] for k in ('x','visual','C','scales','split'))
        assert len(set(z['sha256'])) == len(x) == 16000
    order=np.random.default_rng(2026112234).permutation(np.flatnonzero(split=='train'))
    for n in BUDGETS:
        ids=np.sort(np.r_[order[:n],np.flatnonzero(split!='train')])
        run(SimpleNamespace(output=out/str(n),steps=1500),'curve_braking',ARCHITECTURES,
            (x[ids],y[ids],C,units,split[ids],manifest['bindings']),
            scope='New nominal static RGB observations; five-state task, three-coordinate proxy; frozen original PCA, visual head, C, controller and barrier; no new trajectory CP or certificate.')
        np.save(out/str(n)/'curve_braking/source_indices.npy',ids)
    write(out/'run.json',dict(torch=torch.__version__,numpy=np.__version__,budgets=BUDGETS,
          steps=1500,source_sha256=sha(__file__)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['collect','train'])
    p.add_argument('output',type=Path)
    p.add_argument('--device',default='cuda:1')
    a=p.parse_args()
    torch.set_num_threads(1)
    if a.stage=='collect':
        collect(a.output,a.device)
    else:
        train(a.output)
