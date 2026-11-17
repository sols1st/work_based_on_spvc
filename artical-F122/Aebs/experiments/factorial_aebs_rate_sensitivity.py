"""At 400 images, vary only the gray MLP learning rate under identical optimization and replay."""
import argparse
import copy
import csv
from pathlib import Path
import numpy as np
import torch
from Combined_network.model import SubNet
from .factorial_aebs_inputs import BATCH,CAL_SEED,EPOCHS,LRATES,TEST_SEED,TRAIN_SEEDS
from .matched_aebs_library import SCALE,write
from .matched_aebs_replay import read,collect,episodes
from Aebs.vision_sls.conformal import cp_quantile
from Aebs.vision_sls.deploy import VisionSLSController
from Aebs.vision_sls.run import sha256


def run(a):
    torch.set_num_threads(1)
    a.output.mkdir(exist_ok=False,parents=True)
    write(a.output/'protocol.json',dict(parent=sha256(a.root/'protocol.json'),source=sha256(Path(__file__)),
         training_seeds=TRAIN_SEEDS,rates=LRATES,post_development='paired sensitivity prompted by 400-image development losses before main blinded outcomes',
         calibration_seed=CAL_SEED,holdout_seed=TEST_SEED,endpoint='95% trajectory CP radius and target outcomes'))
    rec=read(a.budgets/'protocol.json')['400']
    raw=np.load(a.root/'raw_gray.npy',mmap_mode='r')
    idx={v:i for i,v in enumerate(np.load(a.root/'ids.npy'))}
    split=np.array(rec['split'])
    x=torch.from_numpy(raw[[idx[f] for f in rec['files']]].copy()).float().mul_(2/255.).sub_(1.)
    y=torch.tensor(np.array(rec['distances'])/SCALE,dtype=torch.float32)[:,None]
    tx,ty=x[split=='train'],y[split=='train'];dx,dy=x[split=='validation'],y[split=='validation']
    with np.load(a.library/'predictions.npz') as f:
        table={k:f[k] for k in ['filename','weather','color','distance_m']}
    local={f:i for i,f in enumerate(table['filename'])}
    pixels=torch.from_numpy(raw[[idx[f] for f in table['filename']]].copy()).float().mul_(2/255.).sub_(1.)
    libs=read(a.library/'libraries.json')
    policy=VisionSLSController(a.package).policy
    rows=[]
    for seed in TRAIN_SEEDS:
        main=a.root/'models'/f'gray32_mlp_400_{seed}'
        summary=read(main/'fit.json')
        for rate in LRATES:
            torch.manual_seed(seed)
            net=SubNet([1024,256,64,1])
            order=torch.Generator().manual_seed(seed+1100)
            opt=torch.optim.Adam(net.parameters(),lr=rate)
            best=float('inf')
            for epoch in range(1,EPOCHS+1):
                net.train()
                for b in torch.randperm(len(tx),generator=order).split(BATCH):
                    loss=(net(tx[b])-ty[b]).square().mean()
                    opt.zero_grad(set_to_none=True);loss.backward();opt.step()
                net.eval()
                with torch.inference_mode(): loss=float((net(dx)-dy).square().mean())
                if loss<best:
                    best,be,bstate=loss,epoch,copy.deepcopy(net.state_dict())
            trial=next(r for r in summary['all_lr'] if r['lr']==rate)
            assert be==trial['epoch'] and best==trial['dev_mse']
            net.load_state_dict(bstate);net.eval()
            if rate==summary['rank_selection']['lr']:
                state=torch.load(main/'readout.pt',map_location='cpu',weights_only=True)
                assert all(torch.equal(t,state[k]) for k,t in bstate.items())
            with torch.inference_mode():
                predictions=torch.cat([net(v).flatten() for v in pixels.split(512)]).numpy()
            for name in ['strict_400','common']:
                ids=np.array([local[f] for f in libs[name]])
                folder=a.output/f'{seed}_{rate}_{name}';folder.mkdir()
                init,ap=episodes(199,CAL_SEED)
                c=collect(policy,table,ids,predictions,init,ap)
                np.savez_compressed(folder/'calibration.npz',**c)
                q=cp_quantile(c['scores'],.05)['q_hat']
                write(folder/'threshold.json',dict(q=q,rank=190,cal=sha256(folder/'calibration.npz')))
                init,ap=episodes(100,TEST_SEED)
                t=collect(policy,table,ids,predictions,init,ap)
                np.savez_compressed(folder/'holdout.npz',**t)
                rows.append(dict(seed=seed,lr=rate,library=name,q=q,covered=int((t['scores']<=q).sum()),
                    fixed=int((t['scores']<=.45).sum()),success=int((t['outcomes']=='success').sum()),
                    unsafe=int((t['outcomes']=='unsafe').sum()),dev_mse=best,epoch=be,
                    cal_queries=int(c['steps'].sum()),test_queries=int(t['steps'].sum())))
    with (a.output/'comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)


if __name__=='__main__':
    cli=argparse.ArgumentParser()
    cli.add_argument('root',type=Path);cli.add_argument('output',type=Path)
    cli.add_argument('--budgets',type=Path,default=Path('results/aebs_budget_scaling_20261110'))
    cli.add_argument('--library',type=Path,default=Path('results/aebs_matched_replay_20261117/library'))
    cli.add_argument('--package',type=Path,default=Path('results/aebs_robust_synthesis_20261117_release/package'))
    run(cli.parse_args())
