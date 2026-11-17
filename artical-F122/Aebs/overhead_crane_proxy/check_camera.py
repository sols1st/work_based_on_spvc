"""Replay DINO on the complete last (padded) image batch of each split and frozen arm."""
import argparse
from pathlib import Path

import numpy as np
import torch

from ..curve_braking.evaluate import digest, save
from ..overhead_crane import plant
from ..overhead_crane.camera import Camera, render
from .train import FAMILIES, SEEDS, ProxyObservation, arm_path, read


@torch.inference_mode()
def check(args):
    p=read(args.root/'protocol.json')
    assert digest(args.weights)==p['weights']
    camera=Camera(args.repo,args.weights,args.device)
    batch=p['camera_batch']
    rows={}
    for seed in SEEDS:
        for family in FAMILIES:
            arm=arm_path(args.root,family,seed)
            assert digest(arm/'frontend.pt')==p['models'][arm.name]['checkpoint']
            model=ProxyObservation(family).to(args.device)
            model.load_state_dict(torch.load(arm/'frontend.pt',weights_only=True,map_location=args.device))
            model.eval()
            for split in ('calibration','holdout'):
                with np.load(arm/(split+'.npz')) as data:
                    n=len(data['states'])
                    start=(n-1)//batch*batch
                    ids=np.minimum(np.arange(start,start+batch),n-1)
                    z=plant.observation(data['states'][ids,:-1])
                    lights=data['lights'][ids]
                    latent=data['latent'][ids]
                peak=0.
                for t in range(plant.HORIZON):
                    images=[render(obs,light) for obs,light in zip(z[:,t],lights)]
                    reproduced,_=model(camera(images).half())
                    peak=max(peak,float(np.max(np.abs(reproduced.cpu().numpy()-latent[:,t]))))
                if peak>2e-6:
                    raise AssertionError((arm.name,split,peak))
                rows[arm.name+'/'+split]=dict(real_trajectories=n-start,frames_evaluated=batch*plant.HORIZON,
                                               max_latent_difference=peak)
    save(args.output,dict(protocol=digest(args.root/'protocol.json'),code=digest(Path(__file__)),
                          rows=rows,physical_gpu=args.device))
    print('verified',len(rows),'full DINO batches', flush=True)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('root',type=Path)
    cli.add_argument('output',type=Path)
    cli.add_argument('--repo',type=Path,default=Path('../external/dinov2'))
    cli.add_argument('--weights',type=Path,default=Path('../external/dinov2_weights/dinov2_vits14_pretrain.pth'))
    cli.add_argument('--device',default='cuda:0')
    args=cli.parse_args()
    torch.set_num_threads(2)
    check(args)
