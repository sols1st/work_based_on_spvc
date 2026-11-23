"""Frozen SPVT/current-controller precision over nine initial-state domains."""
import argparse,importlib.util,math
from pathlib import Path
import numpy as np
import torch
from .benefit_native_spvt import ROOT,SEEDS,StateDecoder,dynamics,target
from .proxy_architecture_compare import sha,write

FROZEN=Path('results/benefit_native_spvt_20261122')


def main(out,device):
    out.mkdir(parents=True,exist_ok=False)
    spec=importlib.util.spec_from_file_location('native_models',ROOT/'model/model.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    g=module.MLPGenerator().to(device).eval().requires_grad_(False)
    generator=ROOT/'model/pretrained_ckpts/mlp_generator.pth'
    g.load_state_dict(torch.load(generator,map_location=device,weights_only=False)['generator_state_dict'])
    models={};files=[generator]
    for seed in SEEDS:
        for kind in ['same_target_e2e','current_lowdim']:
            key=f'{kind}_{seed}';p=FROZEN/(key+'.pt');files.append(p)
            m=StateDecoder() if kind=='current_lowdim' else module.E2ENetworkBN()
            m.load_state_dict(torch.load(p,map_location='cpu',weights_only=False)['state_dict'])
            models[key]=m.to(device).eval().requires_grad_(False)
    for key,name in [('spvt_pid','model/pretrained_ckpts/pid_controller.pth'),('spvt_epoch100','spvt/logs/test/ckpts/checkpoint_epoch_100.pth')]:
        p=ROOT/name;files.append(p)
        combined=module.get_model()
        saved=torch.load(p,map_location='cpu',weights_only=False)
        if key=='spvt_pid':combined.controller.load_state_dict(saved)
        else:
            combined.load_state_dict(saved['model_state_dict'])
            original=torch.load(generator,map_location='cpu',weights_only=False)['generator_state_dict']
            assert all(torch.equal(v,combined.generator.state_dict()[k]) for k,v in original.items())
        models[key]=combined.controller.to(device).eval().requires_grad_(False)
    conditions=[(e,h) for e in [20,40,60] for h in [2,4,7]]
    write(out/'protocol.json',dict(conditions=conditions,units=['cm','degree'],seeds=dict(calibration=202611239101,test=202611239102),
        n_calibration=499,n_test=1000,horizon=160,dt=.05,latent='iid uniform[-.8,.8]^10',
        goals='stay inside +/-5,10,20cm from t=2s through8s; all-time |CTE|<=80cm',
        frozen_source={str(p):sha(p) for p in files},source_sha256=sha(__file__),
        training='All models frozen before new initial-state scenarios; no new training',
        pairing='Same normalized initial draws and latent sequences reused across all9 conditions and all models'))
    thresholds={};summary={}
    for phase,n,seed in [('calibration',499,202611239101),('test',1000,202611239102)]:
        rng=np.random.default_rng(seed)
        unit=rng.uniform(-1,1,(n,2)).astype('float32');z=rng.uniform(-.8,.8,(n,160,10)).astype('float32')
        np.savez_compressed(out/(phase+'_draws.npz'),initial_unit=unit,z=z)
        for e,h in conditions:
            initial=unit*np.array([e,h],dtype='float32')
            for name,model in models.items():
                x=torch.tensor(initial,device=device);states=[x.cpu().numpy()];scores=torch.zeros(n,device=device)
                action_sum=torch.zeros(n,device=device)
                with torch.no_grad():
                    for t in range(160):
                        obs=g(torch.tensor(z[:,t],device=device),x)
                        action=target(model(obs)) if name.startswith('current_lowdim') else model(obs).flatten()
                        error=abs(action-target(x))*30
                        scores=torch.maximum(scores,error);action_sum+=error
                        x=dynamics(x,action);states.append(x.cpu().numpy())
                states=np.stack(states,1);score=scores.cpu().numpy()
                key=f'e{e}_h{h}__{name}';dest=out/key;dest.mkdir(exist_ok=True)
                unsafe=(abs(states[:,:,0])>80).any(1)
                rmse=np.sqrt((states[:,1:,0]**2).mean(1))
                tail=abs(states[:,40:,0]).max(1)
                suffix=np.maximum.accumulate(abs(states[:,::-1,0]),axis=1)[:,::-1]
                reached=suffix<=5;settle=np.where(reached.any(1),reached.argmax(1)*.05,8.)
                np.savez_compressed(dest/(phase+'.npz'),scores=score,rmse_cm=rmse,unsafe=unsafe,tail_max_cm=tail,
                    settling5_seconds=settle,action_mae_deg=action_sum.cpu().numpy()/160,
                    sample_states_cm_degrees=states[:16])
                if phase=='calibration':thresholds[key]=dict(q_deg=float(np.sort(score)[474]),rank=475,n=n)
                else:
                    q=thresholds[key]['q_deg']
                    summary[key]=dict(initial_halfwidth_cm=e,heading_halfwidth_deg=h,model=name,rmse_cm=float(rmse.mean()),
                        q_deg=q,covered=int((score<=q).sum()),unsafe=int(unsafe.sum()),n=n,
                        precision5=int(((tail<=5)&~unsafe).sum()),precision10=int(((tail<=10)&~unsafe).sum()),
                        precision20=int(((tail<=20)&~unsafe).sum()),settling5_seconds=float(settle.mean()))
            print(phase,e,h,flush=True)
        if phase=='calibration':write(out/'frozen_thresholds.json',thresholds)
    write(out/'summary.json',summary)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--device',default='cuda:1')
    a=p.parse_args();torch.set_num_threads(1);main(a.output,a.device)
