"""Encode 32x32 grayscale versions of the real images with frozen DINO."""
import argparse, json
from pathlib import Path
import h5py, numpy as np, torch
from PIL import Image
from Aebs.dino_latent.backbone import FrozenDinoV2

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True); ap.add_argument('--data-root', required=True)
    ap.add_argument('--output', required=True); ap.add_argument('--dino-repo', default='../external/dinov2')
    ap.add_argument('--dino-weights', default='../external/dinov2_weights/dinov2_vits14_pretrain.pth')
    ap.add_argument('--batch-size', type=int, default=16); ap.add_argument('--threads', type=int, default=8)
    args=ap.parse_args(); torch.set_num_threads(args.threads)
    recs=json.load(open(args.manifest))['records']; print('records',len(recs),flush=True)
    model=FrozenDinoV2(args.dino_repo,args.dino_weights,'cpu').eval()
    feats=[]; done=0
    for i in range(0,len(recs),args.batch_size):
        batch=recs[i:i+args.batch_size]; imgs=[]
        for r in batch:
            im=Image.open(Path(args.data_root)/r['filename']).convert('L').resize((32,32),Image.Resampling.BICUBIC)
            a=np.asarray(im,dtype=np.float32)/255.
            imgs.append(torch.from_numpy(a))
        with torch.no_grad(): f=model(torch.stack(imgs).unsqueeze(1)).cpu().numpy().astype(np.float32)
        feats.append(f); done+=len(batch)
        if (i//args.batch_size)%20==0: print('encoded',done,'/',len(recs),flush=True)
    feats=np.concatenate(feats,0)
    with h5py.File(args.output,'w') as h:
        h.create_dataset('features',data=feats)
        h.create_dataset('distance_m',data=np.array([r['distance_m'] for r in recs],dtype=np.float64))
        h.create_dataset('distance_bin_025m',data=np.array([r['distance_bin_025m'] for r in recs],dtype=np.int32))
        h.create_dataset('filename',data=np.array([r['filename'] for r in recs],dtype=h5py.string_dtype()))
        h.create_dataset('weather',data=np.array([r['weather'] for r in recs],dtype=h5py.string_dtype()))
        h.create_dataset('color',data=np.array([r['color'] for r in recs],dtype=h5py.string_dtype()))
        h.create_dataset('split',data=np.array([r['split'] for r in recs],dtype=h5py.string_dtype()))
        h.attrs['complete']=True; h.attrs['schema']='real_gray32_dino_cls_v1'; h.attrs['n']=len(recs)
    print('saved',feats.shape,flush=True)
if __name__=='__main__': main()
