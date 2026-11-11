"""Train the cGAN/state_net perception head on REAL 32x32 grayscale images.

This fixes the earlier transfer baseline: the state_net is optimized on real
images, not only on cGAN-generated images. The cGAN generator is retained as
the original proxy path and is not used for the real-image head.
"""
import argparse,json,math
from pathlib import Path
import numpy as np, torch
from Combined_network.model import SubNet
S=3.18272066116333

def cp(vals,alpha=.05):
    v=np.sort(np.asarray(vals,float));n=len(v);r=math.ceil((n+1)*(1-alpha));return (float(v[r-1]) if r<=n else float('inf')),r
def stats(pred,y):
    e=pred-y;return dict(n=len(y),mae=float(np.abs(e).mean()),rmse=float(np.sqrt((e**2).mean())),p95=float(np.quantile(np.abs(e),.95)),max=float(np.abs(e).max()),bias=float(e.mean()))

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--budget',required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--epochs',type=int,default=100);p.add_argument('--seed',type=int,default=20261110)
    a=p.parse_args();out=a.output_dir
    if out.exists():raise SystemExit('output exists')
    out.mkdir(parents=True);torch.set_num_threads(8);torch.manual_seed(a.seed);rng=np.random.default_rng(a.seed)
    z=np.load(a.cache,allow_pickle=False);x=torch.from_numpy(z['images']);d=np.asarray(z['distances'],dtype=np.float64);y=torch.tensor(d/S,dtype=torch.float32)[:,None];split=np.asarray(z['split'])
    tr=np.flatnonzero(split=='train');va=np.flatnonzero(split=='validation');te=np.flatnonzero(split=='test')
    net=SubNet([1024,256,64,1]);opt=torch.optim.Adam(net.parameters(),lr=1e-4);lossf=torch.nn.MSELoss();best=1e9;best_state=None
    for ep in range(1,a.epochs+1):
        net.train();perm=torch.from_numpy(rng.permutation(tr));tot=0
        for ids in perm.split(64):
            ids=ids.long();loss=lossf(net(x[ids]),y[ids]);opt.zero_grad();loss.backward();opt.step();tot+=float(loss)*len(ids)
        net.eval()
        with torch.no_grad():vl=float(lossf(net(x[va]),y[va]))
        if vl<best:best=vl;best_state={k:v.clone() for k,v in net.state_dict().items()}
        if ep%20==0:print(json.dumps(dict(budget=a.budget,epoch=ep,train=tot/len(tr),val=vl)),flush=True)
    net.load_state_dict(best_state);net.eval()
    with torch.no_grad():pv=net(x[va]).numpy().flatten()*S;pt=net(x[te]).numpy().flatten()*S
    q,rank=cp(np.abs(pv-d[va]));rep=dict(format='aebs_cgan_realhead_budget_v1',budget=a.budget,epochs=a.epochs,seed=a.seed,
        validation=stats(pv,d[va]),test=stats(pt,d[te]),cp=dict(q=q,rank=rank,n=len(va),test_covered=int((np.abs(pt-d[te])<=q).sum())),
        counts={'train':len(tr),'validation':len(va),'test':len(te)},
        scope='cGAN/state_net perception head trained on REAL 32x32 grayscale images; generator not used for this head')
    (out/'report.json').write_text(json.dumps(rep,indent=2));torch.save(net.state_dict(),out/'state_net.pt');print(json.dumps(rep,indent=2))
if __name__=='__main__':main()
