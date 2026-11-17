"""Freeze the selected policy, calibrate complete crane trajectories and compare controls."""
import argparse
from pathlib import Path

import numpy as np
import torch

from ..curve_braking.evaluate import digest,save
from ..lane_keeping.cp import conformal_radius
from . import certificate,plant
from .camera import Camera,Observation,render

CAL_SEED=81121
TEST_SEED=81122


def bindings(root,repo,weights):
    package=Path(__file__).resolve().parent
    code=[*package.glob('*.py'),package.parent/'dino_latent/backbone.py',
          package.parent/'curve_braking/evaluate.py',package.parent/'lane_keeping/cp.py']
    return dict(source={str(p.relative_to(package.parent)):digest(p) for p in code},
                dino_source={str(p.relative_to(repo)):digest(p) for p in
                             [repo/'hubconf.py',*sorted((repo/'dinov2').rglob('*.py'))]},
                weights=digest(weights),frontend=digest(root/'frontend.pt'),static=digest(root/'static.npz'))


@torch.inference_mode()
def collect(scenario,k,camera,model,batch):
    initial,noise,lights=scenario
    records={key:[] for key in ['states','actions','latent','errors']}
    for start in range(0,len(initial),batch):
        # Equal inference shapes even for the last calibration/holdout batch.
        ids=np.minimum(np.arange(start,start+batch),len(initial)-1)
        n=min(batch,len(initial)-start)
        state=initial[ids].copy()
        trail={'states':[state],'actions':[],'latent':[],'errors':[]}
        for t in range(plant.HORIZON):
            z=plant.observation(state)
            images=[render(v,light) for v,light in zip(z,lights[ids])]
            latent,_=model(camera(images).half())
            observed=model.decode(latent).cpu().numpy().astype(float)
            action=plant.control(observed,state,k)
            state=plant.step(state,action,noise[ids,t])
            trail['states'].append(state)
            trail['errors'].append(observed-z)
            trail['latent'].append(latent.cpu().numpy())
            trail['actions'].append(action)
        for key,values in trail.items():
            records[key].append(np.stack(values,axis=1)[:n])
    data={key:np.concatenate(chunks) for key,chunks in records.items()}
    if not all(np.isfinite(v).all() for v in data.values()):
        raise ValueError('nonfinite crane rollout')
    data.update(noise=noise,lights=lights,scores=np.abs(data['errors']).max(axis=(1,2)))
    return data


def ideal(scenario,k):
    state,noise,_=scenario
    states=[state]
    for t in range(plant.HORIZON):
        action=plant.control(plant.observation(state),state,k)
        state=plant.step(state,action,noise[:,t])
        states.append(state)
    return np.stack(states,axis=1)


def evaluate(args):
    root=args.root
    out=root/'evaluation'
    out.mkdir(exist_ok=False)
    torch.set_num_threads(3)
    search=certificate.synthesize()
    save(out/'synthesis.json',search)
    k=search['selected']['gains']
    variants={'image_full':k,'image_no_sway_position':[k[0],k[1],0.,k[3]],
              'image_cart_only':[k[0],k[1],0.,0.]}
    bound=bindings(root,args.repo,args.weights)
    protocol=dict(bindings=bound,synthesis=digest(out/'synthesis.json'),gains=variants,oracle_gains=k,
                  calibration=[199,CAL_SEED],holdout=[100,TEST_SEED],alpha=.05,
                  horizon=plant.HORIZON,dt=plant.DT,limits=plant.LIMITS,target=plant.TARGET,
                  dwell=plant.DWELL,score='max over time and four normalized decoded errors',
                  batch=args.batch,device=args.device,torch=str(torch.__version__))
    save(out/'protocol.json',protocol)
    enc=Camera(args.repo,args.weights,args.device)
    model=Observation().to(args.device)
    model.load_state_dict(torch.load(root/'frontend.pt',map_location=args.device,weights_only=True))
    model.eval()
    calibration=collect(plant.scenarios(199,CAL_SEED),k,enc,model,args.batch)
    np.savez_compressed(out/'calibration.npz',**calibration)
    q,rank=conformal_radius(calibration['scores'],.05)
    save(out/'contract.json',dict(q=q,rank=rank,n=199,scope='image_full',protocol=digest(out/'protocol.json'),
                                  calibration=digest(out/'calibration.npz')))
    contract=digest(out/'contract.json')
    audit=certificate.verify(k,q)
    audit.update(q=q,gains=k,contract=contract)
    save(out/'certificate.json',audit)
    print('CP and robust barrier',q,audit['passed'],audit['initial_upper'],flush=True)
    scenario=plant.scenarios(100,TEST_SEED)
    report={}
    for name,gains in variants.items():
        data=collect(scenario,gains,enc,model,args.batch)
        path=out/(name+'.npz')
        np.savez_compressed(path,**data)
        report[name]=plant.metrics(data['states'])
        report[name]['trajectory_sha256']=digest(path)
        if name=='image_full':
            report[name]['covered']=int((data['scores']<=q).sum())
        print(name,report[name],flush=True)
    exact_states=ideal(scenario,k)
    np.save(out/'ideal.npy',exact_states)
    report['ideal_position']=plant.metrics(exact_states)
    report['ideal_position']['trajectory_sha256']=digest(out/'ideal.npy')
    if bindings(root,args.repo,args.weights)!=bound or digest(out/'contract.json')!=contract:
        raise RuntimeError('bound inputs changed during evaluation')
    save(out/'summary.json',dict(arms=report,q=q,rank=rank,barrier_passed=audit['passed'],
                                 contract=contract,certificate=digest(out/'certificate.json')))


if __name__=='__main__':
    cli=argparse.ArgumentParser()
    cli.add_argument('root',type=Path)
    cli.add_argument('--repo',type=Path,default=Path('../external/dinov2'))
    cli.add_argument('--weights',type=Path,default=Path('../external/dinov2_weights/dinov2_vits14_pretrain.pth'))
    cli.add_argument('--batch',type=int,default=8)
    cli.add_argument('--device',default='cuda:0')
    evaluate(cli.parse_args())
