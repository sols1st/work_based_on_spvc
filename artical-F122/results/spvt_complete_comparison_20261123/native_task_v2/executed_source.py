"""Freeze native SPVT checkpoints by development rollouts, then test unchanged.

No native source, dynamics, generator or controller parameter is modified.
Our old 128-D decoder is explicitly a component diagnostic, not RGB DINO.
"""
import argparse, importlib.util, json, sys, time
from pathlib import Path
import numpy as np
import torch

from .benefit_native_spvt import StateDecoder, target
from .proxy_architecture_compare import sha, write


def metrics(states, actions):
    e=abs(states[:,:,0]);unsafe=(e>80).any(1)
    tail=np.maximum.accumulate(e[:,::-1],axis=1)[:,::-1]
    settled=tail<=5
    settled[unsafe]=False
    first=np.where(settled.any(1),settled.argmax(1)*.05,8.)
    crossings=e>80
    failure_time=np.where(unsafe,crossings.argmax(1)*.05,8.)
    return dict(unsafe=unsafe,first_unsafe_capped_seconds=failure_time,
        rmse_cm=np.sqrt((states[:,1:,0]**2).mean(1)),
        strict_5cm=(~unsafe)&(e[:,40:].max(1)<=5),
        strict_10cm=(~unsafe)&(e[:,40:].max(1)<=10),
        settling_5cm_seconds=first,
        mean_action_change_degrees=abs(np.diff(actions,axis=1)).mean(1)*30)


def summarize(m):
    return {k:float(v.mean()) for k,v in m.items()}


def load_native(module, root, path, device):
    model=module.get_model()
    model.generator.load_state_dict(torch.load(root/'model/pretrained_ckpts/mlp_generator.pth',map_location='cpu',weights_only=False)['generator_state_dict'])
    if path.name=='pid_controller.pth':
        model.controller.load_state_dict(torch.load(path,map_location='cpu',weights_only=False))
    else:
        saved=torch.load(path,map_location='cpu',weights_only=False)
        expected={k:v.clone() for k,v in model.generator.state_dict().items()}
        model.load_state_dict(saved['model_state_dict'])
        assert all(torch.equal(v,model.generator.state_dict()[k]) for k,v in expected.items())
    return model.to(device).eval().requires_grad_(False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream',type=Path,default=Path('/root/SPVT'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--development',type=int,default=256)
    p.add_argument('--test',type=int,default=2000)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    root=a.upstream.resolve();sys.path[:0]=[str(root),str(root/'spvt')]
    from model import model as module
    from spvt.utils import dynamics
    candidates=sorted((root/'spvt/logs/test/ckpts').glob('checkpoint_epoch_*.pth'),key=lambda f:int(f.stem.split('_')[-1]))
    pid=root/'model/pretrained_ckpts/pid_controller.pth'
    own=Path('results/benefit_native_spvt_20261122')
    own_models=sorted(own.glob('current_lowdim_*.pt'))
    protocol=dict(task='native SPVT plant and native 128-D generator, no baseline source edits',
        generator=str(root/'model/pretrained_ckpts/mlp_generator.pth'),
        native_sources={name:sha(root/name) for name in ['model/model.py','train.py','spvt/utils.py','spvt/loss.py','spvt/args.py']},
        candidates={str(f):sha(f) for f in candidates+[pid]},
        own_components={str(f):sha(f) for f in own_models},
        development_n=a.development,test_n=a.test,dev_seed=202611231001,test_seed=202611231002,
        initial='uniform [-60,60] cm x [-7,7] degrees',horizon=160,dt=.05,
        speed_mps=5,wheelbase_m=2.5,latent='native uniform [-.8,.8]^10',
        selection='minimum unsafe fraction, then mean lateral RMSE; development IID mode only; no test selection',
        baseline_scope='existing original-code training checkpoints, not author-published paper-final weights',
        own_scope='old 128D semantic-decoder component diagnostic; not full RGB DINO',
        script_sha256=sha(__file__))
    protocol['component_source_sha256']=sha(Path(__file__).with_name('benefit_native_spvt.py'))
    write(a.output/'protocol.json',protocol)
    (a.output/'executed_source.py').write_bytes(Path(__file__).read_bytes())

    def scenarios(n,seed):
        rng=np.random.default_rng(seed)
        return rng.uniform([-60,-7],[60,7],(n,2)).astype('float32'),rng.uniform(-.8,.8,(n,160,10)).astype('float32')

    @torch.inference_mode()
    def run(model,initial,z,mode,decoder=False):
        state=torch.tensor(initial,device=a.device);states=[state.cpu().numpy()];actions=[]
        latent=torch.tensor(z,device=a.device)
        for t in range(160):
            zz=latent[:,t if mode=='iid' else 0] if mode!='zero' else torch.zeros_like(latent[:,0])
            if decoder:
                y=gen(zz,state);u=target(model(y))
            else:u=model(zz,state).flatten()
            e,h=dynamics(state[:,0],state[:,1],u)
            state=torch.stack([e,h],1);states.append(state.cpu().numpy());actions.append(u.cpu().numpy())
        states=np.stack(states,1);actions=np.stack(actions,1)
        return dict(states_cm_degrees=states,actions_normalized=actions,**metrics(states,actions))

    dev_x,dev_z=scenarios(a.development,protocol['dev_seed']);scores=[]
    started=time.time()
    for path in candidates:
        m=load_native(module,root,path,a.device)
        row=summarize({k:v for k,v in run(m,dev_x,dev_z,'iid').items() if k not in ['states_cm_degrees','actions_normalized']})
        scores.append(dict(path=str(path),sha256=sha(path),metrics=row))
        print(json.dumps(dict(stage='development',checkpoint=path.name,metrics=row)),flush=True)
    selected=min(scores,key=lambda r:(r['metrics']['unsafe'],r['metrics']['rmse_cm']))
    write(a.output/'development_selection.json',dict(selected=selected,candidates=scores,criterion=protocol['selection']))
    test_x,test_z=scenarios(a.test,protocol['test_seed']);np.savez_compressed(a.output/'test_scenarios.npz',initial=test_x,z=test_z)
    chosen=dict(spvt_dev_selected=Path(selected['path']),spvt_epoch100=root/'spvt/logs/test/ckpts/checkpoint_epoch_100.pth',pid_initializer=pid)
    models={key:load_native(module,root,path,a.device) for key,path in chosen.items()}
    gen=models['pid_initializer'].generator
    for path in own_models:
        m=StateDecoder().to(a.device);m.load_state_dict(torch.load(path,map_location=a.device,weights_only=False)['state_dict'])
        models[path.stem]=m.eval().requires_grad_(False)
    result={}
    for name,m in models.items():
        for mode in ['iid','persistent','zero']:
            r=run(m,test_x,test_z,mode,name.startswith('current_lowdim'))
            np.savez_compressed(a.output/(name+'__'+mode+'.npz'),**r)
            row=summarize({k:v for k,v in r.items() if k not in ['states_cm_degrees','actions_normalized']})
            result[name+'__'+mode]=row
            print(json.dumps(dict(stage='test',arm=name,mode=mode,metrics=row)),flush=True)
    assert all(sha(root/name)==digest for name,digest in protocol['native_sources'].items())
    write(a.output/'summary.json',dict(arms=result,selected_native=selected,elapsed_seconds=time.time()-started))


if __name__=='__main__':main()
