"""Cross the pretrained SPVC actor and fixed robust actor with pixel-matched frontends."""
import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from stable_baselines3 import PPO

from .factorial_aebs_inputs import BUDGETS,CAL_SEED,TEST_SEED,TRAIN_SEEDS
from .matched_aebs_library import SCALE
from .matched_aebs_replay import collect,episodes
from ..vision_sls.conformal import cp_quantile
from ..vision_sls.deploy import VisionSLSController
from ..vision_sls.run import sha256


PPO_FILE=Path('Aebs/controller/best_model/best_model.zip')
LABELS=Path('Aebs/data/Downsampled.h5')


class OriginalActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('C',torch.ones(1))
        self.agent=PPO.load(PPO_FILE,device='cpu')
        with h5py.File(LABELS) as f:
            self.scale=float(f['y_train'][:].std())

    def forward_observation(self,y,v):
        # Turn the common semantic distance into the published actor's own units.
        inp=torch.cat([y*(SCALE/self.scale),v],dim=1).cpu().numpy()
        return torch.from_numpy(self.agent.predict(inp,deterministic=True)[0])


def read(path):
    return json.loads(path.read_text())


def save(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def run(args):
    root,out=args.root,args.output
    release=Path('results/aebs_robust_synthesis_20261117_release/package')
    library=Path('results/aebs_matched_replay_20261117/library')
    sources=[root/'protocol.json',library/'manifest.json',PPO_FILE,LABELS,release/'policy.pt',
             release/'config.json',release/'observation.pt',Path(__file__),
             Path('Aebs/experiments/matched_aebs_replay.py'),Path('Aebs/vision_sls/rollout.py')]
    token=dict(inputs={str(p):sha256(p) for p in sources},
               arms=['original_ppo','robust_actor'],library='common',modes=['gray32_mlp','gray32'],
               budgets=BUDGETS,train_seed=TRAIN_SEEDS[0],seeds=[CAL_SEED,TEST_SEED],
               n=[199,100],alpha=.05,primary='paired target success and dangerous terminations at fixed visual readout',
               scope='published PPO initialization versus current robust actor on identical saved camera and trajectory seeds; current barrier not transferred to original PPO')
    path=out/'protocol.json'
    if args.register:
        out.mkdir(exist_ok=False,parents=True)
        save(path,token)
        return
    if read(path)!=json.loads(json.dumps(token)):
        raise ValueError('cross-comparison differs from registration')
    if (out/'summary.csv').exists():
        raise ValueError('completed output exists')
    torch.set_num_threads(1)
    controllers={'original_ppo':OriginalActor(),'robust_actor':VisionSLSController(release).policy}
    with np.load(library/'predictions.npz') as old:
        table={k:old[k] for k in ['filename','weather','color','distance_m']}
    bank_index={s:i for i,s in enumerate(table['filename'])}
    ids=[bank_index[f] for f in read(library/'libraries.json')['common']]
    idx={s:i for i,s in enumerate(np.load(root/'ids.npy'))}
    rows=np.array([idx[f] for f in table['filename']])
    fits={}
    for n in BUDGETS:
        for mode in token['modes']:
            key=f'{mode}_{n}_{TRAIN_SEEDS[0]}'
            folder=root/'models'/key
            meta=read(folder/'fit.json')
            if meta['protocol_sha256']!=sha256(root/'protocol.json') or meta['prediction']!=sha256(folder/'prediction.npy'):
                raise ValueError('readout differs from target-domain study')
            fits[key]=meta['prediction']
    save(out/'checkpoints.json', fits)
    results=[]
    for key in fits:
        pred=np.load(root/'models'/key/'prediction.npy')[rows]
        outcomes={}
        for name,policy in controllers.items():
            target=out/(key+'__'+name)
            target.mkdir(exist_ok=True)
            for split,n,seed in [('calibration',199,CAL_SEED),('holdout',100,TEST_SEED)]:
                data=collect(policy,table,np.array(ids),pred,*episodes(n,seed))
                np.savez_compressed(target/(split+'.npz'),**data)
                if split=='calibration':
                    quantile=cp_quantile(data['scores'],.05)
                    save(target/'threshold.json',quantile)
                else:
                    counts={s:int((data['outcomes']==s).sum()) for s in ['success','unsafe','stopped_safe_outside_goal','out_of_domain','timeout']}
                    q=quantile['q_hat']
                    results.append(dict(frontend=key,actor=name,q_m=q,covered=int((data['scores']<=q).sum()),
                                        in_common_tube=int((data['scores']<=.45).sum()),n=n,**counts))
                    outcomes[name]=data['outcomes']
        a,b=outcomes.values()
        save(out/(key+'_paired.json'),dict(success_ppo_only=int(((a=='success')&(b!='success')).sum()),
             success_robust_only=int(((b=='success')&(a!='success')).sum()),
             unsafe_ppo_only=int(((a=='unsafe')&(b!='unsafe')).sum()),
             unsafe_robust_only=int(((b=='unsafe')&(a!='unsafe')).sum())))
    with (out/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--register',action='store_true')
    run(parser.parse_args())
