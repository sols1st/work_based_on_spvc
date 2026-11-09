"""Train/evaluate a DINO distance head on 32x32 grayscale real images."""
import argparse, json, math
from pathlib import Path
import h5py, numpy as np, torch
from Aebs.vision_sls.model import ObservationProjection
S=3.18272066116333

def dec(a): return np.array([x.decode() if isinstance(x,bytes) else str(x) for x in a])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--features',required=True); ap.add_argument('--output',required=True)
    ap.add_argument('--epochs',type=int,default=300); ap.add_argument('--threads',type=int,default=8); ap.add_argument('--seed',type=int,default=7)
    args=ap.parse_args(); torch.set_num_threads(args.threads); torch.manual_seed(args.seed); np.random.seed(args.seed)
    with h5py.File(args.features,'r') as f:
        X=np.array(f['features'],dtype=np.float32); y=np.array(f['distance_m'],dtype=np.float32)
        split=dec(np.array(f['split'])); weather=dec(np.array(f['weather'])); color=dec(np.array(f['color']))
    tr=np.where(split=='train')[0]; cal=np.where(split=='validation')[0]; test=np.where(split=='test')[0]
    mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6
    xt=torch.from_numpy(X[tr]); yt=torch.from_numpy(y[tr]/S)
    model=ObservationProjection(384,128); model.feature_mean.copy_(torch.from_numpy(mu)[None]); model.feature_std.copy_(torch.from_numpy(sd)[None])
    opt=torch.optim.Adam(model.parameters(),lr=1e-3); best=(1e9,None,0)
    idx=np.arange(len(tr)); rng=np.random.default_rng(args.seed); rng.shuffle(idx); nval=max(500,int(.1*len(idx))); vsel=idx[:nval]; fsel=idx[nval:]
    for ep in range(1,args.epochs+1):
        model.train(); order=torch.randperm(len(fsel))
        for b in order.split(64):
            j=torch.from_numpy(fsel[b.numpy()]); loss=(model(xt[j])-yt[j,None]).square().mean(); opt.zero_grad(); loss.backward(); opt.step()
        if ep%5==0 or ep==1:
            model.eval()
            with torch.no_grad(): pv=model(xt[torch.from_numpy(vsel)]); mae=float((pv-yt[torch.from_numpy(vsel),None]).abs().mean()*S)
            if mae<best[0]: best=(mae,{k:v.detach().clone() for k,v in model.state_dict().items()},ep)
    model.load_state_dict(best[1]); model.eval()
    with torch.no_grad(): obs=model(torch.from_numpy(X)).numpy().reshape(-1)
    def st(mask):
        e=obs[mask]*S-y[mask]; return dict(n=int(len(mask)),mae_m=float(np.abs(e).mean()),rmse_m=float(np.sqrt((e**2).mean())),p95_abs_m=float(np.quantile(np.abs(e),.95)),max_abs_m=float(np.abs(e).max()),bias_m=float(e.mean()))
    cal_res=np.abs(obs[cal]-y[cal]/S); n=len(cal_res); rank=min(n,max(1,int(math.ceil((n+1)*.95)))); q=float(np.sort(cal_res)[rank-1])
    test_res=np.abs(obs[test]-y[test]/S)
    by={}
    for w in sorted(set(weather[test])):
        m=(weather[test]==w); e=obs[test][m]*S-y[test][m]; by[w]=dict(n=int(m.sum()),mae_m=float(np.abs(e).mean()),p95_abs_m=float(np.quantile(np.abs(e),.95)))
    out=dict(method='32x32 grayscale real images -> frozen DINO CLS 384 -> 384-128-128-1',features=str(Path(args.features).resolve()),best_epoch=best[2],
             calibration=st(cal),test=st(test),cp=dict(q_norm=q,radius_m=q*S,cal_coverage=float((cal_res<=q).mean()),test_coverage=float((test_res<=q).mean()),n=n,rank=rank),by_weather_test=by)
    Path(args.output).write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
