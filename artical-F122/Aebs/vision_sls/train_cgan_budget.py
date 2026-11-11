"""Train the original-SPVC-style 32x32 generator + state_net at one budget.

This is the supervised generator used by the shipped original SPVC path
(AebsMLPGenerator), not a new adversarial GAN. It consumes RGB images converted
to 32x32 grayscale and reports both:
  - real-image state_net error (common image-input comparison)
  - native generated-image error (original proxy deployment path)
"""
import argparse,json,math
from pathlib import Path
import numpy as np, torch
from PIL import Image
from cGAN.cGAN_common import Settings
from cGAN.taxi_models_and_data import AebsMLPGenerator
from Combined_network.model import SubNet
S=3.18272066116333

def load_images(root,files):
    out=[]
    for f in files:
        im=Image.open(root/f).convert('L').resize((32,32),Image.Resampling.BICUBIC)
        a=np.asarray(im,dtype=np.float32)/255.; a=2*a-1
        out.append(torch.from_numpy(a).reshape(-1))
    return torch.stack(out)

def cp(vals,alpha=.05):
    v=np.sort(np.asarray(vals,float));n=len(v);r=math.ceil((n+1)*(1-alpha));return (float(v[r-1]) if r<=n else float('inf')),r

def stats(pred,y):
    e=pred-y; return dict(n=len(y),mae=float(np.abs(e).mean()),rmse=float(np.sqrt((e**2).mean())),p95=float(np.quantile(np.abs(e),.95)),max=float(np.abs(e).max()),bias=float(e.mean()))

def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--budget',required=True);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--epochs',type=int,default=300);p.add_argument('--state-epochs',type=int,default=50);p.add_argument('--cache',type=Path)
    a=p.parse_args();rec=json.load(open(a.protocol))[a.budget];out=a.output_dir
    if out.exists():raise SystemExit('output exists')
    out.mkdir(parents=True);torch.set_num_threads(8);torch.manual_seed(20261110+int(a.budget));rng=np.random.default_rng(20261110+int(a.budget))
    if a.cache:
        z=np.load(a.cache,allow_pickle=False); split=np.asarray(z['split']);d=np.asarray(z['distances'],dtype=np.float64);dn=d/S;allimgs=torch.from_numpy(z['images'])
        data={}
        for s in ['train','validation','test']:
            ids=np.flatnonzero(split==s);data[s]=(allimgs[ids],torch.tensor(dn[ids],dtype=torch.float32)[:,None],d[ids])
    else:
        split=np.asarray(rec['split']);files=rec['files'];d=np.asarray(rec['distances'],dtype=np.float64);dn=d/S
        data={}
        for s in ['train','validation','test']:
            ids=np.flatnonzero(split==s);data[s]=(load_images(a.data_root,[files[i] for i in ids]),torch.tensor(dn[ids],dtype=torch.float32)[:,None],d[ids])
    g=AebsMLPGenerator(4,1);opt=torch.optim.Adam(g.parameters(),lr=1e-4);lossf=torch.nn.MSELoss()
    tr_x,tr_y,_=data['train'];best=1e9
    for ep in range(1,a.epochs+1):
        g.train();perm=torch.randperm(len(tr_x));tot=0
        for ids in perm.split(64):
            z=torch.randn(len(ids),4);pred=g(z,tr_y[ids]).view(len(ids),-1);loss=lossf(pred,tr_x[ids]);opt.zero_grad();loss.backward();opt.step();tot+=float(loss)*len(ids)
        if ep%20==0:print(json.dumps(dict(budget=a.budget,stage='generator',epoch=ep,loss=tot/len(tr_x))),flush=True)
    g.eval()
    sn=SubNet([1024,256,64,1]);opt=torch.optim.Adam(sn.parameters(),lr=1e-4)
    for ep in range(1,a.state_epochs+1):
        sn.train();perm=torch.randperm(len(tr_x));tot=0
        for ids in perm.split(64):
            z=torch.randn(len(ids),4)
            with torch.no_grad(): imgs=g(z,tr_y[ids]).view(len(ids),-1)
            pred=sn(imgs);loss=lossf(pred,tr_y[ids]);opt.zero_grad();loss.backward();opt.step();tot+=float(loss)*len(ids)
        if ep%10==0:print(json.dumps(dict(budget=a.budget,stage='state_net',epoch=ep,loss=tot/len(tr_x))),flush=True)
    sn.eval();outd={}
    with torch.no_grad():
        for s in ['validation','test']:
            x,y,phys=data[s]
            # real image input
            real=sn(x).flatten().numpy()*S
            # native generated image input at true distance, z=0
            z=torch.zeros(len(x),4);imgs=g(z,y).view(len(x),-1);native=sn(imgs).flatten().numpy()*S
            outd[s]=dict(real=stats(real,phys),native=stats(native,phys),real_abs=np.abs(real-phys).tolist(),native_abs=np.abs(native-phys).tolist())
    qr,rankr=cp(outd['validation']['real_abs']);qn,rankn=cp(outd['validation']['native_abs'])
    report=dict(format='aebs_cgan_budget_v1',budget=a.budget,total=rec['total'],counts=rec['counts'],epochs=a.epochs,state_epochs=a.state_epochs,
        validation=dict(real=outd['validation']['real'],native=outd['validation']['native']),
        test=dict(real=outd['test']['real'],native=outd['test']['native']),
        cp=dict(real=dict(q=qr,rank=rankr,n=len(outd['validation']['real_abs']),test_covered=int((np.array(outd['test']['real_abs'])<=qr).sum())),
                native=dict(q=qn,rank=rankn,n=len(outd['validation']['native_abs']),test_covered=int((np.array(outd['test']['native_abs'])<=qn).sum()))),
        scope='Original-SPVC-style supervised generator + state_net; RGB->32x32 gray; each total budget has its own split')
    (out/'report.json').write_text(json.dumps(report,indent=2));torch.save(g.state_dict(),out/'generator.pt');torch.save(sn.state_dict(),out/'state_net.pt')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
