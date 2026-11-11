"""Strict-total AEBS budget protocol: 400, 10k, 24k images.

For 400/10k, sample by distance bin and split 60/20/20 within each bin.
For 24k, retain the existing project split (14,172/4,908/4,920).
The three test sets differ by design (user chose total-package budgets).
"""
import argparse, json
from pathlib import Path
import numpy as np

def split_strata(meta, n, seed):
    rng=np.random.default_rng(seed); bins=np.array(meta['bin']); n_total=len(bins)
    # proportional target per bin, at least 1 when possible
    unique,counts=np.unique(bins,return_counts=True)
    target=np.maximum(1,np.floor(n*counts/n_total).astype(int))
    while target.sum()<n:
        i=np.argmax(counts/np.maximum(target,1)); target[i]+=1
    while target.sum()>n:
        i=np.argmax(target); target[i]-=1
    chosen=[]
    for b,t in zip(unique,target):
        ids=np.flatnonzero(bins==b); t=min(int(t),len(ids))
        chosen.extend(rng.choice(ids,size=t,replace=False).tolist())
    chosen=np.array(chosen)
    if len(chosen)<n:
        rest=np.setdiff1d(np.arange(n_total),chosen); chosen=np.r_[chosen,rng.choice(rest,size=n-len(chosen),replace=False)]
    chosen=chosen[:n]
    # stratified 60/20/20 by bin
    tr=[];va=[];te=[]
    for b in np.unique(bins[chosen]):
        ids=chosen[bins[chosen]==b]; rng.shuffle(ids)
        ntr=int(round(.6*len(ids))); nva=int(round(.2*len(ids)))
        ntr=max(1,min(len(ids)-2,ntr)); nva=max(1,min(len(ids)-ntr-1,nva))
        tr.extend(ids[:ntr]); va.extend(ids[ntr:ntr+nva]); te.extend(ids[ntr+nva:])
    tr=np.array(tr);va=np.array(va);te=np.array(te)
    # exact counts may differ by rounding; rebalance deterministically if needed
    return tr,va,te

def main():
    p=argparse.ArgumentParser();p.add_argument('--metadata',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); z=np.load(a.metadata,allow_pickle=False)
    meta=dict(distance_m=z['distance_m'].astype(float),filename=z['filename'].astype(str),weather=z['weather'].astype(str),color=z['color'].astype(str),bin=z['bin'].astype(str),split=z['split'].astype(str))
    out={}
    for n in [400,10000,24000]:
        if n==24000:
            split=meta['split']; ids=np.arange(len(split))
        else:
            tr,va,te=split_strata(meta,n,20261110+n); ids=np.r_[tr,va,te]; split=np.array(['train']*len(tr)+['validation']*len(va)+['test']*len(te))
        rec=dict(total=int(len(ids)),counts={s:int((split==s).sum()) for s in ['train','validation','test']},
                 files=meta['filename'][ids].tolist(),distances=meta['distance_m'][ids].tolist(),split=split.tolist(),
                 weather=meta['weather'][ids].tolist(),color=meta['color'][ids].tolist())
        out[str(n)]=rec
        print(n,rec['counts'])
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2))
if __name__=='__main__':main()
