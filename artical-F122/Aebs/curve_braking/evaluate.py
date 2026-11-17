import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np
import torch
from . import plant
from .vision import Encoder,render
from .verify import verify_checkpoint
from .models import Actor,Proxy,Barrier

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def save(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

@torch.inference_mode()
def collect(n,seed,enc,pca,model,actor):
    r=np.random.default_rng(seed)
    start=r.uniform(plant.INIT_LO,plant.INIT_HI,(n,5));curves=r.choice(plant.KAPPAS,n)
    rows=[]
    for first in range(0,n,8):
        idx=np.minimum(np.arange(first,first+8),n-1)
        x=start[idx].copy();k=curves[idx];xs=[x.copy()];actions=[];errors=[];raw=[]
        for _ in range(plant.HORIZON):
            feat=enc([render(s,c) for s,c in zip(x,k)])
            z=(feat-pca['mean'])@pca['basis']/pca['std'];latent=model.head(z);obs=model.decode(latent)
            norm=torch.tensor(plant.normalize(x),dtype=torch.float32)
            u=actor(obs,torch.tensor(x[:,3],dtype=torch.float32),torch.tensor(x[:,4],dtype=torch.float32),torch.tensor(k,dtype=torch.float32)).numpy()
            errors.append(obs.numpy()-plant.normalize(x));raw.append((latent-model.nominal(norm)).numpy())
            actions.append(u);x=plant.step(x,k,u);xs.append(x.copy())
        m=min(8,n-first)
        rows.append([np.stack(values,1)[:m] for values in (xs,actions,errors,raw)])
    states,actions,errors,raw=[np.concatenate(v) for v in zip(*rows)]
    # Include every registered observation, even after absorbing stop.
    scores=np.max(np.abs(errors),axis=(1,2))
    if not np.isfinite(states).all() or not np.isfinite(scores).all():raise ValueError('nonfinite rollout')
    lat,gap=plant.margins(states,curves[:,None])
    unsafe=((lat<=0)|(gap<=0)).any(axis=1)
    parked=states[:,-1,3]==0
    on_target=(states[:,-1,2]>=plant.TARGET[0])&(states[:,-1,2]<=plant.TARGET[1])
    return dict(states=states,actions=actions,errors=errors,raw_latent_errors=raw,scores=scores,kappas=curves,
                unsafe=unsafe,parked=parked,success=parked&on_target&~unsafe,
                min_body_lane_margin=lat.min(axis=1),min_body_obstacle_margin=gap.min(axis=1))

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();root=a.root
    out=root/'evaluation';out.mkdir(exist_ok=False);torch.set_num_threads(3)
    names=['pca.pt','actor.pt','barrier.pt','linear.pt','affine.pt','neural.pt','static.npz','features.pt']
    bindings={n:digest(root/n) for n in names}
    repo='../external/dinov2';weights='../external/dinov2_weights/dinov2_vits14_pretrain.pth'
    src=Path(__file__).parent
    snap=out/'source';snap.mkdir()
    for f in src.glob('*.py'):shutil.copy2(f,snap/f.name)
    shutil.copy2(src.parent/'dino_latent/backbone.py',snap/'backbone.py')
    protocol=dict(arms=['linear','affine','neural'],bindings=bindings,dino_sha256=digest(weights),
        code={f.name:digest(f) for f in snap.glob('*.py')},device='cuda:0',batch_size=8,
        seed_cal=202611170511,seed_hold=202611170512,alpha=.05,n_cal=199,n_hold=100,
        horizon=plant.HORIZON,dt=plant.DT,curvatures=plant.KAPPAS.tolist(),initial='independent uniform box; equiprobable fixed curvature',
        initial_low=plant.INIT_LO.tolist(),initial_high=plant.INIT_HI.tolist(),parking_target_m=plant.TARGET,
        score='max over all 14 observation instants and all three normalized decoded errors; no discards',
        version=dict(torch=torch.__version__,numpy=np.__version__),candidate_selection='report all arms; no tuning on CP or holdout')
    save(out/'protocol.json',protocol)
    enc=Encoder(repo,weights,'cuda:0');load=lambda n:torch.load(root/n,map_location='cpu',weights_only=True)
    pca=load('pca.pt');actor=Actor();actor.load_state_dict(load('actor.pt'));actor.eval()
    barrier=Barrier().double();barrier.load_state_dict(load('barrier.pt'))
    results={}
    for kind in protocol['arms']:
        target=out/kind;target.mkdir();m=Proxy(kind,pca['basis'].shape[1]);m.load_state_dict(load(kind+'.pt'));m.eval()
        cal=collect(199,protocol['seed_cal'],enc,pca,m,actor);np.savez_compressed(target/'calibration.npz',**cal)
        rank=int(np.ceil((len(cal['scores'])+1)*.95));q=float(np.sort(cal['scores'])[rank-1])
        save(target/'contract.json',dict(radius=q,rank=rank,n=199,calibration_sha256=digest(target/'calibration.npz'),protocol_sha256=digest(out/'protocol.json')))
        frozen=digest(target/'contract.json')
        held=collect(100,protocol['seed_hold'],enc,pca,m,actor);np.savez_compressed(target/'holdout.npz',**held)
        cert=verify_checkpoint(root/'barrier.pt',q);cert.update(bindings={n:bindings[n] for n in [kind+'.pt','actor.pt','barrier.pt','pca.pt']},contract_sha256=frozen)
        save(target/'certificate.json',cert)
        metrics={}
        for name,data in [('calibration',cal),('holdout',held)]:
            b=barrier(torch.tensor(data['states'].reshape(-1,5))).detach().reshape(len(data['scores']),-1).numpy()
            metrics[name]=dict(n=len(b),covered=int((data['scores']<=q).sum()),success=int(data['success'].sum()),
                unsafe=int(data['unsafe'].sum()),parked=int(data['parked'].sum()),max_B=float(b.max()),
                min_body_lane_margin_m=float(data['min_body_lane_margin'].min()),min_body_obstacle_margin_m=float(data['min_body_obstacle_margin'].min()),
                final_distance_range_m=[float(data['states'][:,-1,2].min()),float(data['states'][:,-1,2].max())],
                normalized_score_p95=float(np.quantile(data['scores'],.95)),raw_latent_error_mean_l2=float(np.linalg.norm(data['raw_latent_errors'],axis=2).mean()))
        if frozen!=digest(target/'contract.json'):raise RuntimeError('contract changed')
        results[kind]=dict(radius=q,certificate_passed=cert['passed'],**metrics)
        save(target/'report.json',results[kind]);print(kind,results[kind],flush=True)
    if any(digest(root/n)!=v for n,v in bindings.items()):raise RuntimeError('bound artifacts changed')
    save(out/'summary.json',results)
if __name__=='__main__':main()
