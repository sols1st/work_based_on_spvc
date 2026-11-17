"""Build strict-budget and common held-image AEBS replay libraries."""
import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from Combined_network.model import SubNet
from Aebs.dino_latent.backbone import FrozenDinoV2
from Aebs.vision_sls.model import ObservationProjection
from Aebs.vision_sls.run import sha256

BUDGETS = (400, 10000, 24000)
SCALE = 3.18272066116333


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def gray_pixels(image):
    return np.asarray(image.convert('L').resize((32, 32), Image.Resampling.BICUBIC))


def gray_hash(root, filename):
    with Image.open(root / filename) as image:
        return hashlib.sha256(gray_pixels(image).tobytes()).hexdigest()


def clean_test(records, filename_to_hash):
    """Exclude shared IDs and identical gray inputs with any fit or dev record.

    Identical RGB images necessarily produce identical deterministic gray pixels.
    Keep different held images even if duplicates within the held library: they
    are part of the declared replay law, not independent statistical units.
    """
    forbidden = {filename_to_hash[f] for rec in records for f, s in zip(rec['files'], rec['split']) if s != 'test'}
    # 'common' uses all 24k test images which have never been used for fitting or selection by any arm.
    return sorted(f for f, s in zip(records[-1]['files'], records[-1]['split'])
                  if s == 'test' and filename_to_hash[f] not in forbidden)


def indexed_checkpoint(budgets, original, b, arm):
    if arm == 'gray':
        return budgets / f'cgan_realhead_{b}' / 'state_net.pt'
    if b == 24000:
        return original / 'observation.pt'
    return budgets / f'dino_{b}' / 'observation.pt'


def load_heads(a):
    heads = {}
    for b in BUDGETS:
        for arm in ('gray', 'dino'):
            checkpoint = indexed_checkpoint(a.budgets, a.original, b, arm)
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            if arm == 'gray':
                model = SubNet([1024,256,64,1])
                model.load_state_dict(saved)
            else:
                if saved['distance_scale_m'] != SCALE:
                    raise ValueError('incompatible distance output units')
                model = ObservationProjection()
                model.load_state_dict(saved['projection'])
            heads[f'{arm}_{b}'] = model.eval()
    return heads


def read_rgb(root, filename):
    with Image.open(root / filename) as im:
        rgb = im.convert('RGB')
        return np.array(rgb), gray_pixels(rgb)


def main(a):
    a.out.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((a.budgets/'protocol.json').read_text())
    master = protocol['24000']
    mapping = {f:i for i, f in enumerate(master['files'])}
    if len(mapping) != len(master['files']):
        raise ValueError('nonunique image IDs')
    for rec in protocol.values():
        for file,d,w,c in zip(rec['files'],rec['distances'],rec['weather'],rec['color']):
            i = mapping[file]
            if (d,w,c)!=(master['distances'][i],master['weather'][i],master['color'][i]):
                raise ValueError('conflicting labels for one image')
    with ThreadPoolExecutor(12) as pool:
        hashes = dict(zip(master['files'], pool.map(partial(gray_hash, a.images), master['files'])))
    libraries = {f'strict_{b}':clean_test([protocol[str(b)]],hashes) for b in BUDGETS}
    libraries['common'] = clean_test([protocol[str(b)] for b in BUDGETS],hashes)
    files = sorted(set().union(*libraries.values()))
    if not all(libraries.values()):
        raise ValueError('empty replay library')
    write(a.out/'libraries.json',libraries)
    np.savez_compressed(a.out/'image_hashes.npz', filename=list(hashes), gray_sha256=list(hashes.values()))
    print('library sizes', {k:len(v) for k,v in libraries.items()}, flush=True)
    torch.set_num_threads(3)
    heads = load_heads(a)
    dino = FrozenDinoV2(a.dino_repo,a.dino_weights,a.device)
    predictions = {name:[] for name in heads}
    with torch.inference_mode(), ThreadPoolExecutor(6) as pool:
        for start in range(0,len(files),8):
            pairs = list(pool.map(partial(read_rgb, a.images),files[start:start+8]))
            gray = (2*(np.stack([g for _,g in pairs]).astype(np.float32)/255.)-1).reshape(len(pairs),-1)
            rgb = torch.from_numpy(np.stack([r for r,_ in pairs])).permute(0,3,1,2).float().div_(255.)
            # CPU preprocessing matches the original budget feature extractor.
            f = dino.backbone(dino.preprocess(rgb).to(a.device)).cpu()
            for name,head in heads.items():
                x = torch.from_numpy(gray) if name.startswith('gray_') else f
                predictions[name].extend(head(x).flatten().tolist())
    ids = [mapping[f] for f in files]
    np.savez_compressed(a.out/'predictions.npz',filename=files,distance_m=np.array(master['distances'])[ids],
                        weather=np.array(master['weather'])[ids],color=np.array(master['color'])[ids],**predictions)
    paths = {f'{arm}_{b}':indexed_checkpoint(a.budgets,a.original,b,arm) for b in BUDGETS for arm in ('gray','dino')}
    paths.update(protocol=a.budgets/'protocol.json',backbone=a.dino_weights,source=Path(__file__))
    write(a.out/'manifest.json',dict(inputs={k:dict(path=str(v),sha256=sha256(v)) for k,v in paths.items()},
          outputs={f:sha256(a.out/f) for f in ('predictions.npz','libraries.json','image_hashes.npz')},
          device=a.device,torch=str(torch.__version__),scale=SCALE,library_sizes={k:len(v) for k,v in libraries.items()}))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    parser.add_argument('--images',type=Path,default=Path('../carla_data_spvc_expanded/full01'))
    parser.add_argument('--budgets',type=Path,default=Path('results/aebs_budget_scaling_20261110'))
    parser.add_argument('--original',type=Path,default=Path('results/real_aebs_full_v1'))
    parser.add_argument('--dino-repo',type=Path,default=Path('../external/dinov2'))
    parser.add_argument('--dino-weights',type=Path,default=Path('../external/dinov2_weights/dinov2_vits14_pretrain.pth'))
    parser.add_argument('--device',default='cuda:0')
    main(parser.parse_args())
