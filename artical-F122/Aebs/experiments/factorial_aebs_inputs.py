"""Freeze same-image 2x2 color/resolution DINO inputs and their source identities."""
import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from Aebs.dino_latent.backbone import FrozenDinoV2
from Aebs.vision_sls.run import sha256
from .matched_aebs_library import write

MODES = ('gray32', 'rgb32', 'gray640', 'rgb640')
BUDGETS = (400, 10000, 24000)
TRAIN_SEEDS = (27411, 27412, 27413)
CAL_SEED, TEST_SEED = 711841, 711842
LRATES = (.0001, .001)
EPOCHS, BATCH = 100, 64


def pixels(path):
    with Image.open(path) as source:
        rgb = source.convert('RGB')
        if rgb.size != (640, 640):
            raise ValueError(f'unexpected image size at {path}')
        gray = rgb.convert('L')
        return (np.array(gray.resize((32,32), Image.Resampling.BICUBIC)),
                np.array(rgb.resize((32,32), Image.Resampling.BICUBIC)),
                np.array(gray), np.array(rgb))


def extract(args):
    args.out.mkdir(exist_ok=False, parents=True)
    torch.set_num_threads(2)
    splitfile = args.budgets / 'protocol.json'
    splits = json.loads(splitfile.read_text())
    with np.load(args.previous / 'library/image_hashes.npz') as z:
        gray_sha = dict(zip(z['filename'],z['gray_sha256']))
    files = splits['24000']['files']
    protocol = dict(modes=list(MODES), budgets=list(BUDGETS), seeds=list(TRAIN_SEEDS),
                    heads=['gray32_mlp', *MODES], rates=list(LRATES), batches=BATCH, epochs=EPOCHS,
                    select='lowest development MSE over all epochs and both learning rates per seed',
                    cal=[199,CAL_SEED], holdout=[100,TEST_SEED], alpha=.05,
                    replay='each source strict budget held library plus common held 3229',
                    primary='MLP versus DINO on identical gray32; DINO 2x2 color by source resolution',
                    bindings={str(p):sha256(p) for p in [splitfile,args.weights,
                        args.previous/'library/manifest.json', args.previous/'library/libraries.json',
                        Path(__file__), Path(__file__).with_name('factorial_aebs_fit.py'),
                        Path(__file__).with_name('factorial_aebs_evaluate.py')]},
                    inference_batch=args.batch, preprocessing='PIL gray/resize(32,BICUBIC) before CPU DINO preprocessing 224; same frozen ViT-S/14',
                    device=args.device, torch=str(torch.__version__))
    write(args.out/'protocol.json',protocol)
    start=time.perf_counter()
    image_features={m:np.lib.format.open_memmap(args.out/(m+'.npy'),mode='w+',dtype='float32',
                                                shape=(len(files),384)) for m in MODES}
    raw_gray=np.lib.format.open_memmap(args.out/'raw_gray.npy',mode='w+',dtype=np.uint8,shape=(len(files),1024))
    ids=np.array(files)
    np.save(args.out/'ids.npy',ids)
    encoder=FrozenDinoV2(args.repo,args.weights,args.device)
    with ThreadPoolExecutor(8) as pool, torch.inference_mode():
        for pos in range(0,len(files),args.batch):
            names=files[pos:pos+args.batch]
            batch=list(pool.map(pixels,[args.images / name for name in names]))
            if any(hashlib.sha256(item[0].tobytes()).hexdigest()!=gray_sha[name]
                   for name,item in zip(names,batch)):
                raise ValueError('source gray32 pixels differ from audited split lookup')
            raw_gray[pos:pos+len(batch)] = np.stack([item[0].ravel() for item in batch])
            for j,mode in enumerate(MODES):
                x=torch.from_numpy(np.stack([item[j] for item in batch])).float().div_(255.)
                rgb=x.permute(0,3,1,2) if x.ndim==4 else x[:,None]
                feature=encoder.backbone(encoder.preprocess(rgb).to(args.device)).cpu().numpy()
                image_features[mode][pos:pos+len(batch)]=feature
            if pos % (args.batch*100) == 0:
                print('encoded',pos,'/',len(files),round(time.perf_counter()-start,1),'s',flush=True)
    for x in [raw_gray,*image_features.values()]:
        x.flush()
    write(args.out/'cache.json',dict(files={p.name:sha256(p) for p in sorted(args.out.glob('*.npy'))},
            seconds=time.perf_counter()-start,raw_rgb_decodes=len(files),cached_features=len(files)*len(MODES),
            source_gray_audit='every image is identical to the matched raw 32x32 hash',
            per_file_shape=[640,640], protocol_sha256=sha256(args.out/'protocol.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('out',type=Path)
    p.add_argument('--budgets',type=Path,default=Path('results/aebs_budget_scaling_20261110'))
    p.add_argument('--previous',type=Path,default=Path('results/aebs_matched_replay_20261117'))
    p.add_argument('--images',type=Path,default=Path('/root/carla_data_spvc_expanded/full01'))
    p.add_argument('--repo',type=Path,default=Path('../external/dinov2'))
    p.add_argument('--weights',type=Path,default=Path('../external/dinov2_weights/dinov2_vits14_pretrain.pth'))
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--batch',type=int,default=8)
    extract(p.parse_args())
