"""Train DINO projection at 400/10k budgets; 24k reuses the existing model."""
import argparse,json,math
from pathlib import Path
import numpy as np, torch
from PIL import Image
from Aebs.dino_latent.backbone import FrozenDinoV2
from Aebs.vision_sls.model import ObservationProjection
S=3.18272066116333

def load_features(root,files,backbone):
    feats=[]
    for i in range(0,len(files),16):
        ims=[]
        for f in files[i:i+16]:
            im=Image.open(root/f).convert('RGB'); a=np.asarray(im,dtype=np.float32)/255.
            ims.append(torch.from_numpy(a).permute(2,0,1))
        with torch.no_grad(): feats.append(backbone(torch.stack(ims)).cpu())
        if i%160==0: print(json.dumps(dict(processed=min(i+16,len(files)),total=len(files))),flush=True)
    return torch.cat(feats)

def cp(vals,alpha=.05):
    v=np.sort(np.asarray(vals,float));n=len(v);r=math.ceil((n+1)*(1-alpha));return (float(v[r-1]) if r<=n else float('inf')),r
def stats(pred,y):
    e=pred-y;return dict(n=len(y),mae=float(np.abs(e).mean()),rmse=float(np.sqrt((e**2).mean())),p95=float(np.quantile(np.abs(e),.95)),max=float(np.abs(e).max()),bias=float(e.mean()))

def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--budget',required=True);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--epochs',type=int,default=250);p.add_argument('--dino-repo',type=Path,default=Path('../external/dinov2'));p.add_argument('--dino-weights',type=Path,default=Path('../external/dinov2_weights/dinov2_vits14_pretrain.pth'));p.add_argument('--existing-artifact',type=Path,default=Path('results/real_aebs_full_v1'))
    a=p.parse_args();rec=json.load(open(a.protocol))[a.budget];out=a.output_dir
    if out.exists():raise SystemExit('output exists')
    out.mkdir(parents=True);torch.set_num_threads(8);torch.manual_seed(20261110+int(a.budget));torch.set_num_threads(8)
    split=np.asarray(rec['split']);files=rec['files'];d=np.asarray(rec['distances'],dtype=float);dn=d/S
    if int(a.budget)==24000:
        z=np.load(a.existing_artifact/'observations.npz',allow_pickle=False);obs=np.asarray(z['observation'],dtype=float);sp=np.asarray(z['split']).astype(str);dd=np.asarray(z['distance_m'],dtype=float)
        q,rank=cp(np.abs(obs[sp=='validation']*S-dd[sp=='validation']))
        test=np.abs(obs[sp=='test']*S-dd[sp=='test'])
        report=dict(format='aebs_dino_budget_v1',budget=24000,total=24000,counts={'train':14172,'validation':4908,'test':4920},model='existing 24k observation.pt',validation=stats(obs[sp=='validation']*S,dd[sp=='validation']),test=stats(obs[sp=='test']*S,dd[sp=='test']),cp=dict(q=q,rank=rank,n=int((sp=='validation').sum() if False else (sp=='validation').sum()),test_covered=int((test<=q).sum())))
        (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));return
    backbone=FrozenDinoV2(a.dino_repo,a.dino_weights,'cpu')
    feats={}
    for s in ['train','validation','test']:
        ids=np.flatnonzero(split==s);feats[s]=load_features(a.data_root,[files[i] for i in ids],backbone)
    tr=feats['train'];va=feats['validation'];te=feats['test'];ytr=torch.tensor(dn[split=='train'],dtype=torch.float32)[:,None];yva=torch.tensor(dn[split=='validation'],dtype=torch.float32)[:,None];yte=torch.tensor(dn[split=='test'],dtype=torch.float32)[:,None]
    model=ObservationProjection();model.feature_mean.copy_(tr.mean(0,keepdim=True));model.feature_std.copy_(tr.std(0,keepdim=True).clamp_min(.01));opt=torch.optim.Adam(model.parameters(),lr=1e-3);lossf=torch.nn.MSELoss();best=1e9;best_state=None
    for ep in range(1,a.epochs+1):
        model.train();perm=torch.randperm(len(tr));tot=0
        for ids in perm.split(64):
            pred=model(tr[ids]);loss=lossf(pred,ytr[ids]);opt.zero_grad();loss.backward();opt.step();tot+=float(loss)*len(ids)
        model.eval()
        with torch.no_grad(): vl=float(lossf(model(va),yva))
        if vl<best:best=vl;best_state={k:v.clone() for k,v in model.state_dict().items()}
        if ep%25==0:print(json.dumps(dict(budget=a.budget,epoch=ep,train=tot/len(tr),val=vl)),flush=True)
    model.load_state_dict(best_state);model.eval()
    with torch.no_grad(): pv=model(va).numpy().flatten()*S;pt=model(te).numpy().flatten()*S
    q,rank=cp(np.abs(pv-d[split=='validation']))
    report=dict(format='aebs_dino_budget_v1',budget=a.budget,total=rec['total'],counts=rec['counts'],epochs=a.epochs,validation=stats(pv,d[split=='validation']),test=stats(pt,d[split=='test']),cp=dict(q=q,rank=rank,n=int((split=='validation').sum()),test_covered=int((np.abs(pt-d[split=='test'])<=q).sum())),scope='RGB real images; frozen DINO; same total budget split')
    (out/'report.json').write_text(json.dumps(report,indent=2));torch.save(dict(format='dino_budget_projection',projection=model.state_dict(),distance_scale_m=S),out/'observation.pt');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
