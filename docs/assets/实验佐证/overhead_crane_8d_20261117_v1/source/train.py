"""Static data and joint fitting; no trajectory calibration data enters training."""
import argparse
import copy
import hashlib
from pathlib import Path
import numpy as np
import torch
from .camera import Camera, Observation, SWAY_SCALE, render
from ..curve_braking.evaluate import save


def fit(root, camera, batch):
    torch.manual_seed(81117)
    rng = np.random.default_rng(81118)
    images, labels, lights, seen = [], [], [], set()
    while len(images)<2560:
        z = rng.uniform(-1.05,1.05,4)
        light = rng.integers(3)
        img = render(z,light)
        key = hashlib.sha256(img.tobytes()).digest()
        if key not in seen:
            seen.add(key)
            images.append(img)
            labels.append(z)
            lights.append(light)
    split=np.repeat(['train','val','test'],[2048,256,256])
    np.savez_compressed(root/'static.npz',images=np.stack(images),labels=labels,lights=lights,split=split)
    features=torch.cat([camera(images[i:i+batch]).half().cpu() for i in range(0,len(images),batch)])
    device=camera.device
    obs=Observation().to(device)
    x=torch.tensor(np.stack(labels),device=device,dtype=torch.float32)
    tr,va,te=[torch.from_numpy(np.flatnonzero(split==k)) for k in ['train','val','test']]
    opt=torch.optim.AdamW(obs.parameters(),lr=.001,weight_decay=.0001)
    def estimate(ids):
        return obs(features[ids].to(device))
    def error(ids):
        with torch.no_grad():
            return torch.cat([obs.decode(estimate(i)[0])-x[i.to(device)] for i in ids.split(32)])
    best=None
    for epoch in range(28):
        for ids in tr[torch.randperm(len(tr))].split(32):
            z=x[ids.to(device)]
            y,logits=estimate(ids)
            points=torch.stack([z[:,:2]+obs.offset,z[:,:2]+SWAY_SCALE*z[:,2:]],dim=1)
            heat=(-((obs.grid-points[:,:,None])**2).sum(-1)/(.10**2*2)).softmax(-1)
            landmark_loss=-(heat*logits.log_softmax(-1)).sum(-1).mean()
            loss=(y-z@obs.matrix().T).abs().mean()+(obs.decode(y)-z).square().mean()+.10*landmark_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
        mse=float(error(va).square().mean())
        if best is None or mse<best[0]:
            best=(mse,epoch+1,copy.deepcopy(obs.state_dict()))
    obs.load_state_dict(best[2])
    err=error(te).abs().cpu().numpy()
    torch.save(obs.cpu().state_dict(),root/'frontend.pt')
    save(root/'static.json',dict(best_epoch=best[1],val_mse=best[0],train=2048,val=256,test=256,
        unique_images=len(seen),C=obs.matrix().detach().tolist(),
        test_mae_m=(err.mean(0)*[1,1,.5,.5]).tolist(),test_frame_score_p95=float(np.quantile(err.max(1),.95))))
    print((root/'static.json').read_text(),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    parser.add_argument('--repo',default='../external/dinov2')
    parser.add_argument('--weights',default='../external/dinov2_weights/dinov2_vits14_pretrain.pth')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--batch',type=int,default=8)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(3)
    fit(args.output,Camera(args.repo,args.weights,args.device),args.batch)
