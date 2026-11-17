"""Matched linear-vs-invertible-neural observation experiment, three paired seeds.

Equal data, visual initialization, minibatches, objective, optimizer updates and
validation-only selection. Parameter counts and time differ and are reported.
"""
import argparse,copy,hashlib,json,time,shutil
from pathlib import Path
import numpy as np
import torch
from . import core
from .joint_observation import JointPatchProjection,require_invertible
from .nonlinear_observation import NonlinearPatchProjection
from .learned_sbc import sha256
from ..dino_latent.backbone import FrozenDinoV2


def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def make_pair(saved,seed):
    torch.manual_seed(seed)
    models={'linear':JointPatchProjection(saved['feature_dim'],saved['hidden'],saved['grid']),
            'neural':NonlinearPatchProjection(saved['feature_dim'],saved['hidden'],saved['grid'])}
    C0=torch.tensor([[3**.5/2,.5],[-.5,3**.5/2]],dtype=torch.float32)
    for m in models.values():
        m.visual.load_state_dict(saved['projection'])
        with torch.no_grad():
            m.raw_C.copy_(C0);m.visual.out.weight.copy_(C0@m.visual.out.weight.clone());m.visual.out.bias.copy_(C0@m.visual.out.bias.clone())
    for k,v in models['linear'].visual.state_dict().items():assert torch.equal(v,models['neural'].visual.state_dict()[k])
    return models


def objective(model,features,states):
    y=model(features)
    return (y-model.nominal(states)).norm(dim=1).mean()+.1*(model.decode(y)-states).square().mean()


def extract(a):
    target=a.output/'features.pt';manifest=a.output/'feature_protocol.json'
    if target.exists():
        meta=json.loads(manifest.read_text())
        if meta['dataset_sha256']!=sha256(a.dataset) or meta['cache_sha256']!=sha256(target):raise ValueError('cache binding changed')
        return target
    data=np.load(a.dataset,allow_pickle=False);images=data['images']
    saved=torch.load(a.warm_start,map_location='cpu',weights_only=False)
    backbone=FrozenDinoV2(a.dino_repo,a.dino_weights,a.feature_device,saved['dino_model'])
    parts=[];started=time.monotonic();torch.set_num_threads(4)
    with torch.no_grad():
        for start in range(0,len(images),8):
            x=torch.from_numpy(images[start:start+8]).to(a.feature_device).float().permute(0,3,1,2)/255.
            parts.append(backbone.backbone.forward_features(backbone.preprocess(x))['x_norm_patchtokens'].cpu())
            if start%400==0:print(json.dumps(dict(stage='features',completed=start,total=len(images))),flush=True)
    rows=[hashlib.sha256(im.tobytes()).hexdigest() for im in images]
    torch.save(dict(patches=torch.cat(parts),image_sha256=rows,dataset_sha256=sha256(a.dataset),
                    dino_weights_sha256=sha256(a.dino_weights)),target)
    write(manifest,dict(dataset_sha256=sha256(a.dataset),cache_sha256=sha256(target),
          image_sha256=rows,dino_weights_sha256=sha256(a.dino_weights),device=a.feature_device,
          batch_size=8,rows=len(images),shape=list(images.shape),seconds=time.monotonic()-started,
          policy='fresh extraction in exact dataset row order; no positional reuse of unbound old cache'))
    del backbone,parts
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return target


def audit_geometry(model):
    m=copy.deepcopy(model).double().requires_grad_(False)
    axis=torch.linspace(-1,1,41,dtype=torch.float64);x=torch.cartesian_prod(axis,axis)
    with torch.no_grad():
        y=m.nominal(x);design=torch.cat([x,torch.ones(len(x),1,dtype=x.dtype)],1)
        fit=torch.linalg.lstsq(design,y).solution;r=y-design@fit
        inverse=float((m.decode(y)-x).abs().max())
        C=m.matrix();s=torch.linalg.svdvals(C)
    # Physical deformation after removing the best affine map, not just a C rotation.
    return dict(best_affine_fit_rmse=float(r.square().mean().sqrt()),best_affine_fit_max_abs=float(r.abs().max()),
                max_shift_from_Cx=float((y-x@C.T).abs().max()),inverse_roundtrip_max_abs=inverse,
                condition=float(s[0]/s[-1]),C=C.tolist(),domain='normalized [-1,1]^2, deterministic 41x41 diagnostic grid')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['dataset','warm-start','output','dino-repo','dino-weights']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[202611171,202611172,202611173])
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--feature-device',default='cuda:0')
    p.add_argument('--feature-only',action='store_true');p.add_argument('--train-only',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    if not a.train_only:extract(a)
    if a.feature_only:return
    data=np.load(a.dataset,allow_pickle=False);saved=torch.load(a.warm_start,map_location='cpu',weights_only=False)
    cache=torch.load(a.output/'features.pt',map_location='cpu',weights_only=False)
    if cache['dataset_sha256']!=sha256(a.dataset) or cache['dino_weights_sha256']!=sha256(a.dino_weights):raise ValueError('feature identity mismatch')
    images=data['images'];rows=[hashlib.sha256(im.tobytes()).hexdigest() for im in images]
    if rows!=cache['image_sha256']:raise ValueError('image/feature row mismatch')
    del images
    features=cache['patches'].float();x=torch.tensor(data['states']/[core.E_SCALE,core.THETA_SCALE],dtype=torch.float32)
    split=data['split'];tr=torch.from_numpy(np.flatnonzero(split=='train'));va=torch.from_numpy(np.flatnonzero(split=='validation'))
    if not len(tr) or not len(va) or len(features)!=len(x):raise ValueError('invalid data splits')
    bindings=dict(dataset=sha256(a.dataset),features=sha256(a.output/'features.pt'),warm_start=sha256(a.warm_start),source=sha256(Path(__file__)))
    for seed in a.seeds:
        out=a.output/f'train_{seed}';out.mkdir(exist_ok=False)
        snap=out/'source_snapshot';snap.mkdir()
        for name in ['compare_proxy_train.py','joint_observation.py','nonlinear_observation.py','patch_perception.py']:shutil.copy2(Path(__file__).with_name(name),snap/name)
        models=make_pair(saved,seed);initial={k:copy.deepcopy(m.state_dict()) for k,m in models.items()}
        optim={k:torch.optim.Adam(m.parameters(),lr=1e-4) for k,m in models.items()}
        g=torch.Generator().manual_seed(seed);history={k:[] for k in models};best={k:None for k in models};timing={k:0. for k in models};batches=hashlib.sha256()
        write(out/'protocol.json',dict(seed=seed,epochs=a.epochs,batch_size=64,learning_rate=1e-4,
             bindings=bindings,objective='mean latent L2 + 0.1 mean decoded normalized state MSE (both arms)',
             train_rows=tr.tolist(),validation_rows=va.tolist(),selection='minimum validation decoded MSE; condition<=10',
             minibatches='shared explicit generator independent of model construction',test_or_cp_selection=False,
             mode='matched warm-start fine-tuning; trainable C vs invertible neural h; same visual head and fixed DINO',
             parameters={k:sum(p.numel() for p in m.parameters()) for k,m in models.items()}))
        for epoch in range(1,a.epochs+1):
            totals={k:0. for k in models}
            for ids in tr[torch.randperm(len(tr),generator=g)].split(64):
                batches.update(ids.numpy().tobytes())
                for k,m in models.items():
                    t=time.monotonic();m.train();loss=objective(m,features[ids],x[ids])
                    if not torch.isfinite(loss):raise FloatingPointError('nonfinite observation loss')
                    optim[k].zero_grad();loss.backward()
                    if not all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()):raise FloatingPointError('nonfinite gradient')
                    optim[k].step()
                    if k=='neural':m.constrain()
                    timing[k]+=time.monotonic()-t;totals[k]+=float(loss.detach())*len(ids)
            for k,m in models.items():
                m.eval()
                with torch.no_grad():
                    pred=torch.cat([m.decode(m(features[i])) for i in va.split(64)]);mse=float((pred-x[va]).square().mean());sv=torch.linalg.svdvals(m.matrix());cond=float(sv[0]/sv[-1])
                    if np.isfinite(mse) and cond<=10 and (best[k] is None or mse<best[k][0]):best[k]=(mse,copy.deepcopy(m.state_dict()),epoch)
                history[k].append(dict(epoch=epoch,train_loss=totals[k]/len(tr),validation_mse=mse,condition=cond))
            write(out/'history.json',history)
            print(json.dumps(dict(seed=seed,epoch=epoch,validation={k:history[k][-1]['validation_mse'] for k in models})),flush=True)
        results={}
        for k,m in models.items():
            if best[k] is None:raise RuntimeError('no valid checkpoint; do not select using test results')
            target=out/k;target.mkdir();m.load_state_dict(best[k][1]);m.eval().requires_grad_(False);C=m.matrix().detach();require_invertible(C)
            checkpoint=dict(format='lane_joint_observation_v1' if k=='linear' else 'lane_nonlinear_observation_v1',
                feature_dim=saved['feature_dim'],hidden=saved['hidden'],grid=saved['grid'],state_dict=m.state_dict(),C=C,
                state_scale=[core.E_SCALE,core.THETA_SCALE],max_condition=10.,dino_repo=str(a.dino_repo),dino_weights=str(a.dino_weights),
                dino_model=saved['dino_model'],dino_weights_sha256=sha256(a.dino_weights),best_epoch=best[k][2],training_protocol_sha256=sha256(out/'protocol.json'))
            torch.save(checkpoint,target/'observation.pt')
            with torch.no_grad():
                latent=torch.cat([m(f) for f in features.split(64)]);decoded=m.decode(latent);raw=latent-m.nominal(x);error=decoded-x
            metrics={}
            for part in np.unique(split):
                ids=np.flatnonzero(split==part);e=error[ids];score=e.abs().amax(1).numpy();r=raw[ids]
                metrics[part]=dict(n=len(ids),decoded_mae_physical=(e.abs().mean(0)*torch.tensor([core.E_SCALE,core.THETA_SCALE])).tolist(),
                    decoded_mse=float(e.square().mean()),decoded_score_quantiles=np.quantile(score,[.5,.95,.99,1]).tolist(),
                    raw_latent_l2_mean=float(r.norm(dim=1).mean()),raw_latent_max_abs=float(r.abs().max()))
            geometry=audit_geometry(m)
            report=dict(arm=k,seed=seed,best_epoch=best[k][2],validation_mse=best[k][0],metrics=metrics,geometry=geometry,
                optimization_seconds=timing[k],optimizer_updates=a.epochs*((len(tr)+63)//64),minibatch_sequence_sha256=batches.hexdigest(),
                visual_change_l2=float(sum((m.state_dict()[n]-v).square().sum() for n,v in initial[k].items() if n.startswith('visual.')).sqrt()),
                neural_change_l2=float(sum((m.state_dict()[n]-v).square().sum() for n,v in initial[k].items() if n.startswith(('f.','g.'))).sqrt()) if k=='neural' else 0.,
                checkpoint_sha256=sha256(target/'observation.pt'),scope='static held-out errors, not trajectory coverage or certificate')
            np.savez_compressed(target/'observations.npz',latent=latent.numpy(),decoded=decoded.numpy(),normalized_state=x.numpy(),raw_error=raw.numpy(),decoded_error=error.numpy(),split=split)
            write(target/'report.json',report);results[k]=report
        write(out/'report.json',results)
        print(json.dumps(dict(event='pair_frozen',seed=seed)),flush=True)


if __name__=='__main__':main()
