"""Convert selected RGB images to 32x32 grayscale cache for cGAN training."""
import argparse,json
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from PIL import Image

def one(args):
    root,f=args
    im=Image.open(Path(root)/f).convert('L').resize((32,32),Image.Resampling.BICUBIC)
    a=np.asarray(im,dtype=np.float32)/255.; return (2*a-1).reshape(-1)
def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--budget',required=True);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=16)
    a=p.parse_args();rec=json.load(open(a.protocol))[a.budget]
    with Pool(a.workers) as pool: arr=pool.map(one,[(str(a.data_root),f) for f in rec['files']])
    np.savez_compressed(a.output,images=np.asarray(arr,dtype=np.float32),files=np.asarray(rec['files']),distances=np.asarray(rec['distances'],dtype=np.float32),split=np.asarray(rec['split']))
    print('saved',a.output,len(arr))
if __name__=='__main__':main()
