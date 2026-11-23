"""Normal-image AEBS task completion across six distance/speed domains."""
import argparse,math
from pathlib import Path
import numpy as np
import torch
from .aebs_intuitive_evaluate import read,write,sha,load_banks,load_actors,predictions,collect,summary

FROZEN=Path('results/spvc_intuitive_retrained_20261120_v1')


def main(out):
    out.mkdir(parents=True,exist_ok=False)
    source=read(FROZEN/'protocol.json')
    models=[m for m in source['models'] if m['budget']==24000 and m['mode'] in ('gray32_mlp','gray32_spatial')]
    conditions=[(d,v) for d in [10,12,14] for v in [1,2]]
    write(out/'protocol.json',dict(conditions=conditions,distance_ranges='[d,d+2]m',speed_ranges='[v,v+1]m/s',models=models,
        n_calibration=499,n_test=1000,seeds=dict(calibration=202611239201,test=202611239202),
        source_sha256=sha(__file__),original_protocol_sha256=sha(FROZEN/'protocol.json'),
        actors=['original_ppo','current_robust'],images='normal unmodified common3229gray32 library',
        task='Original goal d in[5,6]m and v<=.5m/s; 400 steps0.05s',training='frozen models; no new training',
        cp='Each initial-state domain separately calibrated with trajectory maximum absolute measurement error'))
    actors=load_actors();bank=load_banks()['common'];qs={};results={}
    for phase,n,seed in [('calibration',499,202611239201),('test',1000,202611239202)]:
        rng=np.random.default_rng(seed);unit=rng.uniform(0,1,(n,2));looks=rng.integers(0,12,n)
        np.savez_compressed(out/(phase+'_draws.npz'),initial_unit=unit,appearance=looks)
        for d,v in conditions:
            initial=unit*np.array([2,1])+[d,v]
            for m in models:
                pred=predictions(m,bank)
                for actor,policy in actors.items():
                    name=f'd{d}_v{v}__{m["mode"]}_{m["train_seed"]}__{actor}'
                    data=collect(policy,bank,pred,initial,looks)
                    dest=out/name;dest.mkdir(exist_ok=True)
                    np.savez_compressed(dest/(phase+'.npz'),**data)
                    if phase=='calibration':qs[name]=dict(q_m=float(np.sort(data['scores'])[474]),n=n,rank=475)
                    else:
                        results[name]=dict(summary(data,qs[name]['q_m']),distance_start=d,speed_start=v,
                            model=m['mode'],train_seed=m['train_seed'],actor=actor)
            print(phase,d,v,flush=True)
        if phase=='calibration':write(out/'frozen_thresholds.json',qs)
    write(out/'summary.json',results)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args()
    torch.set_num_threads(1);main(a.output)
