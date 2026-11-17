import argparse,json
from pathlib import Path
import numpy as np
import torch
from .plant import KAPPAS
from .vision import render,Encoder

def main():
    p=argparse.ArgumentParser()
    p.add_argument('out',type=Path);p.add_argument('--device',default='cuda:0')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    seed=17112026;r=np.random.default_rng(seed)
    # Split after sampling. Enforce pixel-disjoint static splits even for raster ties.
    seen=set();images=[];states=[];curves=[]
    import hashlib
    while len(images)<2560:
        s=r.uniform([-.54,-.15,2.5],[.54,.15,6.3]);k=r.choice(KAPPAS)
        im=render(s,k);digest=hashlib.sha256(im.tobytes()).hexdigest()
        if digest in seen:continue
        seen.add(digest);images.append(im);states.append(s);curves.append(k)
    ids=r.permutation(len(images));split=np.empty(len(images),dtype='U5')
    split[ids[:1920]]='train';split[ids[1920:2240]]='val';split[ids[2240:]]='test'
    np.savez_compressed(a.out/'static.npz',images=images,states=states,kappas=curves,split=split)
    enc=Encoder('../external/dinov2','../external/dinov2_weights/dinov2_vits14_pretrain.pth',a.device)
    f=torch.cat([enc(images[i:i+8]) for i in range(0,len(images),8)])
    torch.save(f,a.out/'features.pt')
    (a.out/'data.json').write_text(json.dumps(dict(seed=seed,n=len(images),unique_images=len(seen),train=1920,val=320,test=320)))
if __name__=='__main__':main()
