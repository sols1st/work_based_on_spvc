"""Paired single-frame CPU timings of the two deployed camera readouts."""
import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from Combined_network.model import SubNet
from Aebs.curve_braking.evaluate import digest,save
from Aebs.dino_latent.backbone import FrozenDinoV2
from Aebs.vision_sls.model import ObservationProjection


@torch.inference_mode()
def measure(args):
    args.output.mkdir(parents=True,exist_ok=False)
    cpu=max(os.sched_getaffinity(0))
    os.sched_setaffinity(0,{cpu})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    rec=json.loads((args.budget/'protocol.json').read_text())['24000']
    pool=[f for f,s in zip(rec['files'],rec['split']) if s=='test']
    rng=np.random.default_rng(1711998)
    selected=rng.choice(pool,size=120,replace=False).tolist()
    frames=[]
    for filename in selected:
        with Image.open(args.images/filename) as im:
            frames.append(im.convert('RGB'))
    head=SubNet([1024,256,64,1]).eval()
    head.load_state_dict(torch.load(args.original,map_location='cpu',weights_only=True))
    saved=torch.load(args.projection,map_location='cpu',weights_only=False)
    proj=ObservationProjection().eval()
    proj.load_state_dict(saved['projection'])
    dino=FrozenDinoV2(args.repo,args.weights,'cpu')
    def original(im):
        gray=im.convert('L').resize((32,32),Image.Resampling.BICUBIC)
        x=2*(np.asarray(gray,dtype=np.float32)/255.)-1
        return float(head(torch.from_numpy(x).reshape(1,-1)).item())
    def current(im):
        x=np.asarray(im,dtype=np.float32)/255.
        rgb=torch.from_numpy(x).permute(2,0,1).unsqueeze(0)
        return float(proj(dino(rgb)).item())
    readers={'original':original,'current':current}
    for im in frames[:12]:
        for read in readers.values():
            read(im)
    records=[]
    for i,im in enumerate(frames[12:],12):
        for name in rng.permutation(list(readers)):
            w=time.perf_counter_ns()
            t=time.thread_time_ns()
            y=readers[name](im)
            thread=time.thread_time_ns()-t
            wall=time.perf_counter_ns()-w
            records.append(dict(frame=selected[i],reader=name,prediction=y,thread_ms=thread/1e6,wall_ms=wall/1e6))
    params=lambda m:sum(p.numel() for p in m.parameters())
    results={}
    for name in readers:
        sub=[r for r in records if r['reader']==name]
        results[name]={key:np.quantile([r[key] for r in sub],[.5,.9]).tolist() for key in ['thread_ms','wall_ms']}
    save(args.output/'timings.json',records)
    save(args.output/'summary.json',dict(results=results,unit='ms [median,p90]',batch=1,
         included='decoded in-memory PIL RGB to scalar camera observation, including preprocessing; no image I/O, generator, actor or simulation',
         warmup=12,n=108,seed=1711998,pinned_cpu=cpu,source=digest(Path(__file__)),
         cpu_description=next(line.strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),
         params=dict(original=params(head),current=params(dino.backbone)+params(proj)),torch=str(torch.__version__),
         platform=platform.platform(),cpu_threads=1,bindings={str(p):digest(p) for p in
         (args.original,args.projection,args.weights,args.budget/'protocol.json')},timings=digest(args.output/'timings.json')))
    print(results,flush=True)


if __name__=='__main__':
    cli=argparse.ArgumentParser()
    cli.add_argument('output',type=Path)
    cli.add_argument('--images',type=Path,default=Path('/root/carla_data_spvc_expanded/full01'))
    cli.add_argument('--budget',type=Path,default=Path('results/aebs_budget_scaling_20261110'))
    cli.add_argument('--original',type=Path,default=Path('Aebs/controller/state_net_trained.pth'))
    cli.add_argument('--projection',type=Path,default=Path('results/real_aebs_full_v1/observation.pt'))
    cli.add_argument('--repo',type=Path,default=Path('../external/dinov2'))
    cli.add_argument('--weights',type=Path,default=Path('../external/dinov2_weights/dinov2_vits14_pretrain.pth'))
    measure(cli.parse_args())
