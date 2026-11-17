"""Matched target-domain training and development selection on frozen image inputs."""
import argparse
import copy
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from Combined_network.model import SubNet
from Aebs.vision_sls.model import ObservationProjection
from Aebs.vision_sls.run import sha256
from .factorial_aebs_inputs import BATCH, BUDGETS, EPOCHS, LRATES, MODES, TRAIN_SEEDS
from .matched_aebs_library import SCALE, write


def fit(spec):
    root, budgets, mode, n, seed = spec
    torch.set_num_threads(1)
    folder=root/'models'/f'{mode}_{n}_{seed}'
    if (folder/'fit.json').exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    rec=json.loads((budgets/'protocol.json').read_text())[str(n)]
    master={f:i for i,f in enumerate(np.load(root/'ids.npy').tolist())}
    rows=np.array([master[f] for f in rec['files']])
    split=np.array(rec['split'])
    raw=mode=='gray32_mlp'
    data=np.load(root/(('raw_gray' if raw else mode)+'.npy'), mmap_mode='r')
    x=torch.from_numpy(data[rows].copy()).float()
    if raw:
        x.mul_(2/255.).sub_(1.)
    y=torch.tensor(np.array(rec['distances'])/SCALE,dtype=torch.float32).reshape(-1,1)
    tr=np.flatnonzero(split=='train');dev=np.flatnonzero(split=='validation')
    x_train, y_train=x[tr],y[tr];x_dev,y_dev=x[dev],y[dev]
    start=time.perf_counter()
    trials=[]
    winner=None
    winner_info=None
    for rate in LRATES:
        torch.manual_seed(seed)
        model=SubNet([1024,256,64,1]) if raw else ObservationProjection()
        if not raw:
            model.feature_mean.copy_(x_train.mean(0,keepdim=True))
            model.feature_std.copy_(x_train.std(0,keepdim=True).clamp_min(.01))
        order=torch.Generator().manual_seed(seed+1100)
        opt=torch.optim.Adam(model.parameters(),lr=rate)
        best_loss=float('inf')
        for epoch in range(1,EPOCHS+1):
            model.train()
            for batch in torch.randperm(len(tr),generator=order).split(BATCH):
                loss=(model(x_train[batch])-y_train[batch]).square().mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            model.eval()
            with torch.inference_mode():
                loss=float((model(x_dev)-y_dev).square().mean())
            if loss < best_loss:
                best_loss, best_epoch = loss, epoch
                if winner_info is None or loss < winner_info['dev_mse']:
                    winner=copy.deepcopy(model.state_dict())
                    winner_info=dict(lr=rate,epoch=epoch,dev_mse=loss)
        trials.append(dict(lr=rate,epoch=best_epoch,dev_mse=best_loss))
    model.load_state_dict(winner)
    model.eval()
    all_output=[]
    with torch.inference_mode():
        for i in range(0,len(data),512):
            values=torch.from_numpy(np.array(data[i:i+512])).float()
            if raw:
                values.mul_(2/255.).sub_(1.)
            all_output.append(model(values).flatten().numpy())
    np.save(folder/'prediction.npy',np.concatenate(all_output))
    torch.save(winner,folder/'readout.pt')
    write(folder/'fit.json',dict(mode=mode,budget=n,seed=seed,rank_selection=winner_info,all_lr=trials,
          epochs=EPOCHS,batch=BATCH,n_train=len(tr),n_dev=len(dev),test=int((split=='test').sum()),
          gradients=int(len(LRATES)*EPOCHS*int(np.ceil(len(tr)/BATCH))),
          training_seconds=time.perf_counter()-start,head_params=sum(p.numel() for p in model.parameters()),
          prediction=sha256(folder/'prediction.npy'), model=sha256(folder/'readout.pt'),
          cache_sha256=sha256(root/'cache.json'),protocol_sha256=sha256(root/'protocol.json')))
    print('fit',folder.name,winner_info,flush=True)


def main(args):
    protocol=json.loads((args.root/'protocol.json').read_text())
    if protocol['bindings'][str(Path(__file__))]!=sha256(Path(__file__)):
        raise ValueError('training source changed after registration')
    cache=json.loads((args.root/'cache.json').read_text())
    for name,h in cache['files'].items():
        if sha256(args.root/name)!=h:
            raise ValueError('inconsistent feature cache')
    jobs=[(args.root,args.budgets,m,n,s) for n in BUDGETS for s in TRAIN_SEEDS for m in ('gray32_mlp',*MODES)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(fit,jobs))


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('root',type=Path)
    cli.add_argument('--budgets',type=Path,default=Path('results/aebs_budget_scaling_20261110'))
    cli.add_argument('--workers',type=int,default=6)
    main(cli.parse_args())
