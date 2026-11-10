"""Train AEBS p_theta and C together without claiming measured speed labels."""
import argparse
import copy
import json
from pathlib import Path
import h5py
import numpy as np
import torch

from .joint_observation import JointCLSProjection
from .real_artifact import load_h5,sha256


def write(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',choices=['gray','rgb'],required=True)
    p.add_argument('--features',type=Path,required=True)
    p.add_argument('--dataset',type=Path)
    p.add_argument('--splits',type=Path)
    p.add_argument('--test-features',type=Path)
    p.add_argument('--warm-start',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=60)
    p.add_argument('--seed',type=int,default=20261109)
    p.add_argument('--lr',type=float,default=1e-4)
    args=p.parse_args(argv)
    if args.epochs<1 or args.lr<=0:raise ValueError('positive epochs/lr required')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(args.seed)
    old=torch.load(args.warm_start,map_location='cpu',weights_only=False)
    S=old['distance_scale_m']
    if args.kind=='gray':
        if args.dataset is None or args.splits is None:raise ValueError('gray dataset and splits required')
        cache=torch.load(args.features,map_location='cpu',weights_only=False)
        if cache['provenance']['data_sha256']!=sha256(args.dataset):raise ValueError('gray feature/data mismatch')
        X=cache['features'].numpy()
        with h5py.File(args.dataset) as f:y=np.array(f['y_train']).reshape(-1)
        split=np.full(len(y),'',dtype='U16')
        for name,ids in dict(np.load(args.splits,allow_pickle=False)).items():split[ids]=name
        extra=dict(weather=np.full(len(y),'gray'),color=np.full(len(y),'gray'))
    else:
        d=load_h5(args.features)
        if args.test_features:
            td=load_h5(args.test_features,'test')
            d={k:np.concatenate([d[k],td[k]]) for k in d}
        X=d['features'];y=d['distance_m'];split=d['split'].astype('U16')
        # Existing validation bins remain untouched by checkpoint selection.
        split[split=='validation']='calibration'
        extra={k:d[k] for k in ['weather','color','filename']}
    if len(X)!=len(y) or not np.isfinite(X).all():raise ValueError('invalid features/data')
    fit=np.flatnonzero(split=='train')
    # Independent fixed internal train/validation split, not calibration/test.
    rng=np.random.default_rng(args.seed);rng.shuffle(fit)
    nv=max(20,int(.1*len(fit)));va=torch.from_numpy(fit[:nv]);tr=torch.from_numpy(fit[nv:])
    if len(tr)<1:raise ValueError('not enough training images')
    f=torch.tensor(X,dtype=torch.float32);distance=torch.tensor(y/S,dtype=torch.float32)
    model=JointCLSProjection(old['feature_dim'],old['hidden_dim'])
    state={k:v for k,v in old['projection'].items() if k!='C'}
    missing=model.load_state_dict(state,strict=False).missing_keys
    if missing!=['raw_C']:raise ValueError('warm-start architecture mismatch')
    C0=model.matrix().detach().clone()
    with torch.no_grad():
        model.net[-1].weight.mul_(C0[0,0])
        model.net[-1].bias.mul_(C0[0,0]).add_(C0[0,1]*1.5)
    initial=copy.deepcopy(model.state_dict())
    bindings={str(v):sha256(v) for v in [args.features,args.warm_start,args.dataset,args.splits,args.test_features] if v is not None}
    protocol=dict(kind=args.kind,seed=args.seed,epochs=args.epochs,lr=args.lr,
        state='[distance / visual_distance_scale_m, speed_mps]',visual_distance_scale_m=S,
        synthetic_velocities=[0.,1.5,3.],image_law='image depends on distance and appearance, not instantaneous speed',
        velocity_label_status='synthetic replicated states; NOT measured velocity annotations',
        loss='mean absolute latent matching error over all three synthetic speeds',
        selection='minimum internal validation mean decoded squared physical error over three speeds; abs(C_d)>=.2',
        train_rows=tr.tolist(),internal_validation_rows=va.tolist(),bindings=bindings,
        validation_status='development selection only; warm-start model may already have seen these internal validation images',
        source_sha256={str(s):sha256(s) for s in [Path(__file__),Path(__file__).with_name('joint_observation.py')]},
        initial_C=C0.tolist(),limitations='static frozen-image observation model; not velocity observability from a single image')
    write(args.output_dir/'protocol.json',protocol)
    opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    speeds=torch.tensor([0.,1.5,3.]);best=None;history=[]
    for epoch in range(1,args.epochs+1):
        total=0.;gradient=0.
        for ids in tr[torch.randperm(len(tr))].split(64):
            latent=model(f[ids]);C=model.matrix()
            target=distance[ids,None]*C[0,0]+speeds[None,:]*C[0,1]
            loss=(latent-target).abs().mean()
            opt.zero_grad();loss.backward();gradient+=float(model.raw_C.grad.norm());opt.step()
            total+=float(loss.detach())*len(ids)
        with torch.no_grad():
            C=model.matrix();pred=model(f[va]);decoded=(pred-C[0,1]*speeds[None,:])/C[0,0]
            mse=float(((decoded-distance[va,None])*S).square().mean())
        row=dict(epoch=epoch,train_loss=total/len(tr),validation_mse_m2=mse,C=C.tolist(),C_gradient_norm_sum=gradient)
        if np.isfinite(mse) and abs(C[0,0])>=.2 and (best is None or mse<best[0]):
            best=(mse,copy.deepcopy(model.state_dict()),epoch)
        history.append(row)
        if epoch%5==0:print(json.dumps(row),flush=True)
    if best is None:raise RuntimeError('no usable learned observation matrix')
    model.load_state_dict(best[1]);model.eval().requires_grad_(False);C=model.matrix().detach()
    torch.save(dict(format='aebs_joint_C_static_image_v1',state_dict=model.state_dict(),C=C,
        feature_dim=old['feature_dim'],hidden_dim=old['hidden_dim'],distance_scale_m=S,
        provenance=old.get('provenance',{}),dino_repo=old['dino_repo'],dino_weights=old['dino_weights'],
        best_epoch=best[2],protocol_sha256=sha256(args.output_dir/'protocol.json')),args.output_dir/'joint_observation.pt')
    with torch.no_grad():latent=torch.cat([model(b) for b in f.split(256)]).numpy().reshape(-1)
    # Report speed-dependent effective state error; no calibration threshold yet.
    decoded=(latent[:,None]-float(C[0,1])*speeds.numpy()[None,:])/float(C[0,0])
    error=(decoded-y[:,None]/S)*S
    metrics={name:dict(n=int((split==name).sum()),
             mean_abs_error_m=float(abs(error[split==name]).mean()),
             max_abs_error_m=float(abs(error[split==name]).max())) for name in np.unique(split)}
    report=dict(C_initial=C0.tolist(),C_final=C.tolist(),C_change=float((C-C0).norm()),
        C_gradient_nonzero=any(r['C_gradient_norm_sum']>0 for r in history),best_epoch=best[2],
        visual_parameter_change_l2=float(sum((model.state_dict()[k]-v).square().sum() for k,v in initial.items() if k.startswith('net.')).sqrt()),
        metrics=metrics,protocol=protocol,safety_status='joint front end fitted only; no inherited CP/SBC certificate',
        checkpoint_sha256=sha256(args.output_dir/'joint_observation.pt'))
    np.savez_compressed(args.output_dir/'observations.npz',distance_m=y,observation=latent,split=split,**extra)
    write(args.output_dir/'history.json',history);write(args.output_dir/'report.json',report)
    print(json.dumps({k:report[k] for k in ['C_final','C_change','best_epoch','metrics']}),flush=True)


if __name__=='__main__':main()
