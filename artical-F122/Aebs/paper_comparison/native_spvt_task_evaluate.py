"""Evaluate actual SPVT checkpoints in their own generator and plant interface.

This establishes a native-method baseline. Generated observations are NOT real
camera rollouts, and the published PID initializer is NOT a trained SPVT model.
No current-framework controller is silently evaluated with oracle observations.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import torch
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--episodes',type=int,default=1000)
    p.add_argument('--trained-checkpoint',type=Path)
    p.add_argument('--trained-epoch',type=int,default=100)
    p.add_argument('--seed',type=int,default=202611201101)
    args=p.parse_args()
    root=args.upstream.resolve();args.output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(root));sys.path.insert(0,str(root/'spvt'))
    from model.model import get_model
    from spvt.utils import dynamics
    torch.set_num_threads(2)
    trained_name=f'spvt_epoch{args.trained_epoch}'
    paths={'generator':root/'model/pretrained_ckpts/mlp_generator.pth',
        'pid_initializer':root/'model/pretrained_ckpts/pid_controller.pth',
        trained_name:args.trained_checkpoint or root/'spvt/logs/test/ckpts/checkpoint_epoch_100.pth'}
    sources={name:sha(root/name) for name in ['model/model.py','train.py','spvt/utils.py','spvt/loss.py','spvt/args.py']}
    protocol=dict(seed=args.seed,n=args.episodes,horizon=160,dt=.05,v=5.,wheelbase=2.5,
        initial='independent uniform CTE [-60,60] cm and heading [-7,7] degrees; registered benchmark distribution, not asserted original paper test draws',
        unsafe='any |CTE| >80cm; original SPVT training safety threshold; no added heading restriction',
        precision='safe for full rollout and |CTE|<=tolerance at every sample from 2 through 8 seconds',
        tolerances_cm=[5,10,20,40],latent_modes=['zero','iid','persistent'],latent_support=[-.8,.8],
        weights={k:dict(path=str(v),sha256=sha(v)) for k,v in paths.items()},sources=sources,
        checkpoint_selection=f'epoch{args.trained_epoch} fixed before testing; not best test checkpoint',
        trained_epoch=args.trained_epoch,
        provenance='existing server training artifact; not claimed author-published final checkpoint; training-run configuration not yet uniquely reconstructed',
        observations='original conditional generator, not real camera; no current-framework comparison claimed',
        evaluator_sha256=sha(__file__))
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rng=np.random.default_rng(protocol['seed'])
    initial=rng.uniform([-60.,-7.],[60.,7.],(args.episodes,2)).astype('float32')
    latent=dict(zero=np.zeros((args.episodes,160,10),np.float32),
        iid=rng.uniform(-.8,.8,(args.episodes,160,10)).astype('float32'),
        persistent=np.repeat(rng.uniform(-.8,.8,(args.episodes,1,10)).astype('float32'),160,axis=1))
    summary={}
    for name in ['pid_initializer',trained_name]:
        model=get_model().eval().requires_grad_(False)
        model.generator.load_state_dict(torch.load(paths['generator'],map_location='cpu')['generator_state_dict'])
        if name=='pid_initializer':model.controller.load_state_dict(torch.load(paths[name],map_location='cpu'))
        else:
            ckpt=torch.load(paths[name],map_location='cpu')
            if ckpt['epoch']!=args.trained_epoch:raise ValueError('not the registered final epoch')
            model.load_state_dict(ckpt['model_state_dict'],strict=True)
        # Verify the supposedly frozen generator really stayed frozen.
        original=torch.load(paths['generator'],map_location='cpu')['generator_state_dict']
        if any(not torch.equal(v,model.generator.state_dict()[k]) for k,v in original.items()):
            raise ValueError('generator changed in trained checkpoint')
        for mode,z in latent.items():
            state=torch.tensor(initial);states=[state.numpy().copy()];actions=[]
            with torch.no_grad():
                for t in range(160):
                    action=model(torch.tensor(z[:,t]),state).flatten()
                    cte,heading=dynamics(state[:,0],state[:,1],action)
                    state=torch.stack([cte,heading],1)
                    if not torch.isfinite(state).all():raise ValueError('nonfinite trajectory')
                    states.append(state.numpy().copy());actions.append(action.numpy())
            states=np.stack(states,1);actions=np.stack(actions,1)
            unsafe=(abs(states[:,:,0])>80).any(1)
            rmse=np.sqrt((states[:,1:,0]**2).mean(1));tail=abs(states[:,40:,0]).max(1)
            metric=dict(unsafe=unsafe,rmse_cm=rmse,tail_max_cm=tail)
            for tolerance in protocol['tolerances_cm']:metric['strict_%dcm'%tolerance]=(~unsafe)&(tail<=tolerance)
            key=name+'__'+mode
            np.savez_compressed(args.output/(key+'.npz'),states_cm_degrees=states,actions_normalized=actions,**metric)
            summary[key]={k:float(np.mean(v)) for k,v in metric.items()}
            print(json.dumps(dict(arm=key,metrics=summary[key])),flush=True)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))


if __name__=='__main__':main()
