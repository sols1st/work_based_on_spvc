"""Paired task-precision study on SPVT's actual generator and vehicle model."""
import argparse,copy,importlib.util,json,math,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .proxy_architecture_compare import sha,write

SEEDS=(2026112211,2026112212,2026112213)
ROOT=Path('/root/SPVT')


class StateDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(128,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,2))
    def forward(self,y):
        return self.net(y)*y.new_tensor([60.,7.])


def target(x):
    # Continuous-time poles -2,-3: e'=5h, h'=2u.
    # u= -0.6e -2.5h, clipped at original +/-30 degree steering limit.
    angle=-(.6*(x[:,0]/100)+2.5*torch.deg2rad(x[:,1]))
    return (angle/(math.pi/6)).clamp(-1,1)


def dynamics(x,action):
    he=torch.deg2rad(x[:,1])
    next_he=he+5*torch.tan(action*math.pi/6)*.05/2.5
    next_he=(next_he+math.pi)%(2*math.pi)-math.pi
    return torch.stack([x[:,0]+25*torch.sin(he),torch.rad2deg(next_he)],1)


def run(args):
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    device=args.device
    spec=importlib.util.spec_from_file_location('native_spvt_models',ROOT/'model/model.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    gen=module.MLPGenerator().to(device).eval()
    gen.load_state_dict(torch.load(ROOT/'model/pretrained_ckpts/mlp_generator.pth',map_location=device,weights_only=False)['generator_state_dict'])
    gen.requires_grad_(False)
    paths={k:ROOT/v for k,v in dict(generator='model/pretrained_ckpts/mlp_generator.pth',pid='model/pretrained_ckpts/pid_controller.pth',
                                   trained='spvt/logs/test/ckpts/checkpoint_epoch_100.pth').items()}
    protocol=dict(seeds=SEEDS,n_training=24000,n_development=4000,updates=3000,batch=128,lr=.001,dt=.05,horizon=160,speed_mps=5,wheelbase_m=2.5,
        initial='uniform CTE [-60,60] cm, heading [-7,7] degrees',latent='uniform [-.8,.8]^10, iid each step or persistent across160 steps',
        training_domain='CTE [-75,75]cm, heading [-10,10]degrees',train_seed=202611229201,development_seed=202611229202,
        calibration_seed=202611229203,test_seed=202611229204,n_calibration=499,n_test=2000,
        target='u rad=clip(-0.6*(CTE/100)-2.5*heading_rad,+/-pi/6); poles -2,-3 of linearized bicycle',
        cp='max_t |actual_steering_deg - reference_steering_deg(current_true_state)|; calibration190? rank=ceil((499+1)*.95)=475',
        selection='development loss every100 updates; no test selection',
        model_sources={str(p):sha(p) for p in paths.values()},source_sha256=sha(__file__),
        observations='native SPVT generator128 outputs at evolving true states, same generator and latent law for every controller',
        methods={'spvt_epoch100':'native historical trained checkpoint','spvt_pid':'native released PID initializer',
                 'same_target_e2e':'native E2ENetworkBN architecture, retrained from random init on same samples and target as current',
                 'current_lowdim':'128->64 ReLU->32 ReLU->2 physical state decoder, then fixed reference feedback; h(x)=normalized identity'},
        objective='current: normalized state MSE + reference action MSE; e2e: reference action MSE; identical data batches and updates')
    protocol['cp']=protocol['cp'].replace('calibration190? ','')
    write(out/'protocol.json',protocol)
    def draw(n,seed):
        rng=np.random.default_rng(seed)
        x=rng.uniform([-75,-10],[75,10],(n,2)).astype('float32')
        z=rng.uniform(-.8,.8,(n,10)).astype('float32')
        return torch.tensor(x,device=device),torch.tensor(z,device=device)
    x,z=draw(24000,protocol['train_seed']);dx,dz=draw(4000,protocol['development_seed'])
    with torch.no_grad():
        obs=torch.cat([gen(z[i:i+256],x[i:i+256]) for i in range(0,len(x),256)])
        devobs=torch.cat([gen(dz[i:i+256],dx[i:i+256]) for i in range(0,len(dx),256)])
        targets=target(x);devtargets=target(dx)
    models={}
    history={}
    for seed in SEEDS:
        for kind in ('same_target_e2e','current_lowdim'):
            torch.manual_seed(seed)
            m=(StateDecoder() if kind=='current_lowdim' else module.E2ENetworkBN()).to(device)
            opt=torch.optim.Adam(m.parameters(),lr=.001)
            rng=np.random.default_rng(seed);best=None;hist=[]
            def loss(m,ob,states,labels):
                if kind=='current_lowdim':
                    estimated=m(ob)
                    return (((estimated-states)/states.new_tensor([60,7]))**2).mean()+((target(estimated)-labels)**2).mean()
                return ((m(ob).flatten()-labels)**2).mean()
            for step in range(3001):
                if step%100==0:
                    m.eval()
                    with torch.no_grad():dev=float(loss(m,devobs,dx,devtargets))
                    hist.append([step,dev])
                    if best is None or dev<best[0]:best=(dev,step,copy.deepcopy(m.state_dict()))
                if step==3000:break
                m.train();ids=rng.integers(0,len(x),128)
                ell=loss(m,obs[ids],x[ids],targets[ids]);opt.zero_grad();ell.backward();opt.step()
            m.load_state_dict(best[2]);m.eval().requires_grad_(False)
            key=kind+'_'+str(seed);models[key]=m
            torch.save(dict(kind=kind,seed=seed,state_dict=m.state_dict(),selected_step=best[1]),out/(key+'.pt'))
            history[key]=dict(development=hist,selected_step=best[1],parameters=sum(p.numel() for p in m.parameters()))
            print('trained',key,best[:2],flush=True)
    for name in ('spvt_pid','spvt_epoch100'):
        combined=module.get_model()
        if name=='spvt_pid':
            combined.controller.load_state_dict(torch.load(paths['pid'],map_location='cpu',weights_only=False))
        else:
            combined.load_state_dict(torch.load(paths['trained'],map_location='cpu',weights_only=False)['model_state_dict'])
            for k,v in gen.cpu().state_dict().items():
                assert torch.equal(v,combined.generator.state_dict()[k])
            gen.to(device)
        models[name]=combined.controller.to(device).eval().requires_grad_(False)
    write(out/'history.json',history)
    thresholds={};results={}
    for phase,n,seed in [('calibration',499,protocol['calibration_seed']),('test',2000,protocol['test_seed'])]:
        rng=np.random.default_rng(seed)
        initial=rng.uniform([-60,-7],[60,7],(n,2)).astype('float32')
        basez=rng.uniform(-.8,.8,(n,160,10)).astype('float32')
        np.savez_compressed(out/(phase+'_scenarios.npz'),initial=initial,z=basez)
        for mode in ('iid','persistent'):
            for key,m in models.items():
                state=torch.tensor(initial,device=device);states=[state.cpu().numpy()]
                errors=[];actions=[];decode_error=[]
                with torch.no_grad():
                    for t in range(160):
                        latent=torch.tensor(basez[:,t if mode=='iid' else 0],device=device)
                        y=gen(latent,state)
                        if key.startswith('current_lowdim'):
                            estimate=m(y);action=target(estimate);decode_error.append((estimate-state).cpu().numpy())
                        else:action=m(y).flatten()
                        errors.append(((action-target(state))*30).cpu().numpy())
                        actions.append(action.cpu().numpy())
                        state=dynamics(state,action);states.append(state.cpu().numpy())
                states=np.stack(states,1);errors=np.stack(errors,1);actions=np.stack(actions,1)
                score=abs(errors).max(1)
                dest=out/(mode+'__'+key);dest.mkdir(exist_ok=True)
                data=dict(states_cm_degrees=states,actions=actions,action_error_deg=errors,scores=score)
                if decode_error:data['state_error_cm_degrees']=np.stack(decode_error,1)
                np.savez_compressed(dest/(phase+'.npz'),**data)
                name=mode+'__'+key
                if phase=='calibration':
                    thresholds[name]=dict(q_deg=float(np.sort(score)[474]),rank=475,n=499)
                else:
                    unsafe=(abs(states[:,:,0])>80).any(1)
                    row=dict(rmse_cm=float(np.sqrt((states[:,1:,0]**2).mean(1)).mean()),unsafe=int(unsafe.sum()),
                        action_mae_deg=float(abs(errors).mean()),q_deg=thresholds[name]['q_deg'],covered=int((score<=thresholds[name]['q_deg']).sum()),n=n)
                    for tol in [5,10,20,40]:
                        row[f'precision_{tol}cm']=int(((abs(states[:,40:,0]).max(1)<=tol)&~unsafe).sum())
                    results[name]=row
                    print(phase,name,row,flush=True)
        if phase=='calibration':write(out/'frozen_thresholds.json',thresholds)
    write(out/'summary.json',results)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--device',default='cuda:1')
    a=p.parse_args();torch.set_num_threads(1);run(a)
