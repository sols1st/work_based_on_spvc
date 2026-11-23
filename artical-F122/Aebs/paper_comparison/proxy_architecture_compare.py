"""Small state-proxy ablation against identical frozen visual observations.

Residuals are mapped through the SAME original C inverse for every arm. This
does not change perception, deploy a new inverse, or inherit a safety certificate.
"""
import argparse, copy, hashlib, json, math, time
from pathlib import Path
import numpy as np
import torch
from torch import nn

SEEDS=(2026112101,2026112102,2026112103)
ARCHS=('tanh16','relu16','silu16','tanh16x16')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def dataset(task):
    if task=='aebs':
        root=Path('results/aebs_robust_synthesis_20261117_release/package')
        p=root/'observations.npz';z=np.load(p)
        scale=float(torch.load(root/'observation.pt',map_location='cpu')['distance_scale_m'])
        x=z['distance_m'][:,None]/scale;y=z['observation'][:,None]
        C=np.eye(1);units=np.array([scale]);split=z['split'].astype('U16')
        # Original validation is reserved for calibration, not architecture choice.
        split[split=='validation']='calibration'
        tr=np.flatnonzero(split=='train');rng=np.random.default_rng(2026112100)
        dev=rng.permutation(tr)[:len(tr)//5];split[dev]='development'
        bindings={str(p):sha(p),str(root/'observation.pt'):sha(root/'observation.pt')}
    else:
        root=Path('results/lane_proxy_comparison_20261117_v1/train_202611171/linear')
        p=root/'observations.npz';z=np.load(p)
        x=z['normalized_state'];y=z['latent'];split=z['split'].copy();split[split=='validation']='development'
        C=np.array(json.loads((root/'report.json').read_text())['geometry']['C']);units=np.array([.85,.30])
        bindings={str(p):sha(p),str(root/'report.json'):sha(root/'report.json')}
    assert set(np.unique(split))=={'train','development','calibration','test'}
    assert np.isfinite(x).all() and np.isfinite(y).all()
    return x.astype('float32'),y.astype('float32'),C.astype('float32'),units,split,bindings

class Proxy(nn.Module):
    def __init__(self,C,arch):
        super().__init__();self.register_buffer('C',torch.tensor(C));d=len(C)
        kind, width = arch.split('16', 1)
        if width not in ('', 'x16'):
            raise ValueError(f'Unsupported architecture {arch}')
        activation={'tanh':nn.Tanh,'relu':nn.ReLU,'silu':nn.SiLU}[kind]
        layers=[nn.Linear(d,16),activation()]
        if width=='x16':layers += [nn.Linear(16,16),activation()]
        layers += [nn.Linear(16,d)];self.residual=nn.Sequential(*layers)
        nn.init.zeros_(self.residual[-1].weight);nn.init.zeros_(self.residual[-1].bias)
    def forward(self,x):return x@self.C.T+self.residual(x)

def run(a,task,architectures=ARCHS,prepared_data=None,scope=None,model_factory=Proxy,neural_description=None):
    x,y,C,units,split,bindings=dataset(task) if prepared_data is None else prepared_data
    out=a.output/task;out.mkdir(parents=True,exist_ok=False)
    masks={k:np.flatnonzero(split==k) for k in np.unique(split)}
    np.savez_compressed(out/'inputs.npz',x=x,visual=y,C=C,physical_scales=units,split=split)
    inverse=np.linalg.inv(C).T
    protocol=dict(task=task,seeds=list(SEEDS),architectures=list(architectures),updates=a.steps,batch=128,lr=.001,
        source_sha256=sha(__file__),inputs_sha256=sha(out/'inputs.npz'),bindings=bindings,
        counts={k:len(v) for k,v in masks.items()},C=C.tolist(),scales=units.tolist(),
        train_rows=masks['train'].tolist(),development_rows=masks['development'].tolist(),
        objective='MSE((frozen_visual - proxy) @ inverse(C).T); same units and latent coordinates for all arms',
        selection='minimum development objective every50 steps including initialization; architecture selected by mean development objective over three seeds',
        neural=neural_description or 'h(x)=fixed_Cx + small MLP(x); fixed DINO/projection/decoder/controller',
        linear='fixed original Cx plus least-squares fitted Cx and affine controls; exact linear fits use no iterative budget',
        scope=scope or 'historical static held-out image residuals; not new blind data, joint visual learning, trajectory CP or safety certificate',
        AEBS_state='distance only; image bank has no measured speed labels; original C=[1,0] so omit speed',
        calibration='separate frame max-coordinate score; no assumption of exchangeable trajectory samples')
    write(out/'protocol.json',protocol)
    X=torch.tensor(x);Y=torch.tensor(y);I=torch.tensor(inverse);rows=[]
    baseline=x@C.T;total=(y-baseline)@inverse*units
    def record(name,seed,pred,params,dev,seconds,extra=None):
        residual=(y-pred)@inverse*units;bias=(pred-baseline)@inverse*units
        np.testing.assert_allclose(residual+bias,total,atol=1e-6,rtol=1e-5)
        scores=np.max(abs((y-pred)@inverse),axis=1)
        cal=np.sort(scores[masks['calibration']]);rank=math.ceil((len(cal)+1)*.95);q=float(cal[rank-1])
        metrics={}
        for part,ids in masks.items():
            e=residual[ids];metrics[part]=dict(n=len(ids),mae_physical=abs(e).mean(0).tolist(),
                rmse_physical=np.sqrt((e*e).mean(0)).tolist(),p95_abs_physical=np.quantile(abs(e),.95,axis=0).tolist(),
                normalized_rmse=float(np.sqrt(np.mean(((y[ids]-pred[ids])@inverse)**2))))
        row=dict(architecture=name,seed=seed,parameters=params,development_mse=dev,seconds=seconds,metrics=metrics,
            frame_quantile=dict(q=q,rank=rank,n=len(cal),test_fraction=float(np.mean(scores[masks['test']]<=q)),
                 scope='static diagnostic, not a trajectory coverage guarantee'),extra=extra or {})
        key=name if seed is None else name+'_'+str(seed)
        np.savez_compressed(out/(key+'.npz'),proxy=pred,residual_physical=residual,nominal_bias_physical=bias,total_error_physical=total)
        rows.append(row);write(out/'results.json',rows)
        print(json.dumps(dict(task=task,arm=key,test_mae=row['metrics']['test']['mae_physical'])),flush=True)
    tr=masks['train'];va=masks['development']
    for name,affine in [('fixed_Cx',None),('fitted_Cx',False),('affine',True)]:
        if affine is None:pred=baseline;params=0
        else:
            design=np.c_[x,np.ones(len(x))] if affine else x
            coef=np.linalg.lstsq(design[tr],y[tr],rcond=None)[0];pred=(design@coef).astype('float32');params=int(coef.size)
            np.save(out/(name+'_coefficients.npy'),coef)
        dev=float(np.mean(((y[va]-pred[va])@inverse)**2));record(name,None,pred,params,dev,0.)
    for seed in SEEDS:
        for arch in architectures:
            torch.manual_seed(seed);model=model_factory(C,arch);opt=torch.optim.Adam(model.parameters(),lr=.001)
            rng=np.random.default_rng(seed);best=(float('inf'),None,0);start=time.monotonic()
            history=[];batchhash=hashlib.sha256()
            for step in range(a.steps+1):
                if step%50==0 or step==a.steps:
                    with torch.no_grad():loss=float((((model(X[va])-Y[va])@I)**2).mean())
                    history.append(dict(step=step,development_mse=loss))
                    if loss<best[0]:best=(loss,copy.deepcopy(model.state_dict()),step)
                if step==a.steps:break
                ids=rng.choice(tr,128,replace=True);batchhash.update(ids.tobytes())
                loss=(((model(X[ids])-Y[ids])@I)**2).mean()
                if not torch.isfinite(loss):raise ValueError('nonfinite loss')
                opt.zero_grad();loss.backward();opt.step()
            key=arch+'_'+str(seed)
            with torch.no_grad():final_pred=model(X).numpy()
            final_metrics={part:dict(mae_physical=np.mean(abs((y[ids]-final_pred[ids])@inverse*units),axis=0).tolist(),
                normalized_rmse=float(np.sqrt(np.mean(((y[ids]-final_pred[ids])@inverse)**2)))) for part,ids in masks.items()}
            torch.save(dict(architecture=arch,C=C,state_dict=model.state_dict()),out/(key+'_final.pt'))
            model.load_state_dict(best[1]);model.eval()
            torch.save(dict(architecture=arch,C=C,state_dict=model.state_dict()),out/(key+'.pt'))
            with torch.no_grad():pred=model(X).numpy()
            record(arch,seed,pred,sum(p.numel() for p in model.parameters()),best[0],time.monotonic()-start,
                   dict(selected_step=best[2],batch_sha256=batchhash.hexdigest(),history=history,final_iterate_metrics=final_metrics))
    rankings={name:float(np.mean([r['development_mse'] for r in rows if r['architecture']==name])) for name in ['fixed_Cx','fitted_Cx','affine',*architectures]}
    write(out/'selection.json',dict(development_rank=sorted(rankings,key=rankings.get),development_mse=rankings,
        selected_architecture=min(rankings,key=rankings.get),test_not_used_for_selection=True))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=1500)
    a=p.parse_args();torch.set_num_threads(1)
    if a.steps<1:raise ValueError('positive training budget required')
    for task in ['aebs','lane']:run(a,task)

if __name__=='__main__':main()
