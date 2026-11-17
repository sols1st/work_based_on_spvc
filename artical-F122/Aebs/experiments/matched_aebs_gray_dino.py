"""Train a 400-image DINO readout on exactly the same gray pixels as SPVC's state head."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch

from Aebs.dino_latent.backbone import FrozenDinoV2
from Aebs.vision_sls.model import ObservationProjection
from Aebs.vision_sls.run import sha256
from .matched_aebs_library import SCALE,write


def main(a):
    a.out.mkdir(exist_ok=False)
    torch.set_num_threads(3)
    seed = 20261110 + 400
    torch.manual_seed(seed)
    with np.load(a.budgets/'cgan_24000_gray.npz') as cache:
        files=cache['files'].astype(str)
        index={f:i for i,f in enumerate(files)}
        gray=np.rint((cache['images']+1)*127.5).clip(0,255).astype(np.uint8).reshape(-1,32,32)
    rec=json.loads((a.budgets/'protocol.json').read_text())['400']
    selected=np.load(a.library/'predictions.npz')['filename'].astype(str)
    needed=list(dict.fromkeys(rec['files']+selected.tolist()))
    # Require the preprocessing cache and the source-image hashes from this run to agree.
    import hashlib
    with np.load(a.library/'image_hashes.npz') as z:
        hashes=dict(zip(z['filename'],z['gray_sha256']))
    ids=np.array([index[f] for f in needed])
    if any(hashlib.sha256(gray[i].tobytes()).hexdigest()!=hashes[f] for i,f in zip(ids,needed)):
        raise ValueError('raw gray pixel mismatch between SPVC and DINO')
    backbone=FrozenDinoV2(a.repo,a.weights,a.device)
    features=[]
    with torch.inference_mode():
        for group in np.array_split(ids,int(np.ceil(len(ids)/8))):
            x=torch.from_numpy(gray[group]).float().div_(255.)[:,None].repeat(1,3,1,1)
            features.append(backbone.backbone(backbone.preprocess(x).to(a.device)).cpu())
    f=torch.cat(features)
    lookup={s:i for i,s in enumerate(needed)}
    split=np.array(rec['split'])
    x=f[[lookup[s] for s in rec['files']]]
    target=torch.tensor(np.asarray(rec['distances'])/SCALE,dtype=torch.float32)[:,None]
    train,dev=[torch.tensor(np.flatnonzero(split==name)) for name in ('train','validation')]
    model=ObservationProjection()
    model.feature_mean.copy_(x[train].mean(0,keepdim=True))
    model.feature_std.copy_(x[train].std(0,keepdim=True).clamp_min(.01))
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    best=None
    for epoch in range(100):
        for subset in train[torch.randperm(len(train))].split(64):
            loss=(model(x[subset])-target[subset]).square().mean()
            opt.zero_grad();loss.backward();opt.step()
        with torch.no_grad():
            val=float((model(x[dev])-target[dev]).square().mean())
        if best is None or val<best[0]:
            best=val,epoch+1,copy.deepcopy(model.state_dict())
    model.load_state_dict(best[2]);model.eval()
    torch.save(model.state_dict(),a.out/'projection.pt')
    with torch.no_grad():
        obs=model(f[[lookup[s] for s in selected]]).flatten().numpy()
    np.savez_compressed(a.out/'predictions.npz',filename=selected,observation=obs)
    write(a.out/'manifest.json',dict(inputs=dict(raw_library=sha256(a.library/'manifest.json'),
          budget=sha256(a.budgets/'protocol.json'),backbone=sha256(a.weights),source=sha256(Path(__file__))),
          checkpoint=sha256(a.out/'projection.pt'),predictions=sha256(a.out/'predictions.npz'),
          train=len(train),development=len(dev),epochs=100,total=400,best_epoch=best[1],training_seed=seed,scale=SCALE,
          comparison='both see identical 32x32 8-bit grayscale; DINO repeats channels and resizes; DINO pretrained'))
    print('grayscale DINO trained',best[:2],flush=True)


if __name__=='__main__':
    cli=argparse.ArgumentParser()
    cli.add_argument('library',type=Path);cli.add_argument('out',type=Path)
    cli.add_argument('--budgets',type=Path,default=Path('results/aebs_budget_scaling_20261110'))
    cli.add_argument('--repo',type=Path,default=Path('../external/dinov2'))
    cli.add_argument('--weights',type=Path,default=Path('../external/dinov2_weights/dinov2_vits14_pretrain.pth'))
    cli.add_argument('--device',default='cuda:1')
    main(cli.parse_args())
