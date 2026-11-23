"""New paired scenario CP calibration for matched AEBS image conditions."""
import argparse, math, time
from pathlib import Path
import numpy as np
import torch
from .aebs_intuitive_evaluate import read,write,sha,episodes,load_banks,load_actors,predictions,collect,summary,static_metrics

BASE=Path('results/spvc_intuitive_retrained_20261120_v1')
SHIFT=Path('results/spvc_shift_closed_loop_20261120_v1')


def main(out):
    out.mkdir(parents=True,exist_ok=False)
    nominal=read(BASE/'protocol.json')
    shifted=read(SHIFT/'protocol.json')
    models=[dict(m,condition='nominal') for m in nominal['models'] if m['budget']==24000 and m['mode'] in ('gray32_mlp','gray32_spatial')]
    models += [m for m in shifted['models'] if m['mode'] in ('gray32_mlp','gray32_spatial')]
    for m in models:
        if sha(m['model_path'])!=m.get('model_sha256',m.get('checkpoint_sha256',sha(m['model_path']))):
            raise ValueError('changed checkpoint')
    protocol=dict(calibration_seed=202611229101,test_seed=202611229102,n_calibration=499,n_test=2000,alpha=.05,
                  models=models,source_sha256=sha(__file__),original_protocol_sha256=sha(BASE/'protocol.json'),
                  shift_protocol_sha256=sha(SHIFT/'protocol.json'),actors=['original_ppo','current_robust'],
                  condition='Each nominal or shifted image law separately calibrated; same random initial scenes across all arms',
                  outcome='Original AEBS goal d in [5,6]m and v<=0.5m/s; 400 steps at0.05s',
                  image_source='Common held-out 3229-image nearest-camera library, identical gray32 pixels')
    write(out/'protocol.json',protocol)
    actors=load_actors();bank=load_banks()['common']
    thresholds={};results={};static={}
    for phase,n,seed in [('calibration',499,protocol['calibration_seed']),('test',2000,protocol['test_seed'])]:
        initial,appearance=episodes(n,seed)
        np.savez_compressed(out/(phase+'_scenarios.npz'),initial=initial,appearance=appearance)
        for m in models:
            pred=predictions(m,bank)
            key=f"{m['mode']}_{m['train_seed']}_{m['condition']}"
            static[key]=static_metrics(bank,pred)
            for actor,policy in actors.items():
                name=key+'_'+actor
                data=collect(policy,bank,pred,initial,appearance)
                dest=out/name;dest.mkdir(exist_ok=True)
                np.savez_compressed(dest/(phase+'.npz'),**data)
                if phase=='calibration':
                    rank=math.ceil((n+1)*.95);q=float(np.sort(data['scores'])[rank-1])
                    thresholds[name]=dict(q_m=q,rank=rank,n=n,calibration_sha256=sha(dest/(phase+'.npz')))
                else:
                    results[name]=dict(summary(data,thresholds[name]['q_m']),mode=m['mode'],train_seed=m['train_seed'],condition=m['condition'],actor=actor,
                        test_sha256=sha(dest/(phase+'.npz')))
            print(phase,key,flush=True)
        if phase=='calibration':
            write(out/'frozen_thresholds.json',thresholds)
    write(out/'summary.json',results);write(out/'static.json',static)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args()
    torch.set_num_threads(1);main(a.output)
