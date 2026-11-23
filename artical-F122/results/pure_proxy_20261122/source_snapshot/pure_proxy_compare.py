"""Pure fully connected h(x), compared with archived Cx and residual-MLP arms."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from .proxy_architecture_compare import SEEDS, run, sha, write
from .proxy_depth_data import ARCHITECTURES

SOURCES = {
    'aebs': (Path('results/proxy_depth_data_20261122'), (2160,11338,14172), [100.]),
    'lane': (Path('results/proxy_depth_data_20261122'), (2160,6000,12000), [100.,180/np.pi]),
    'curve_braking': (Path('results/curve_proxy_depth_20261122'), (1920,6000,12000), [100.,180/np.pi,100.]),
}


class PureProxy(nn.Module):
    """C supplies only the input/output dimension at construction."""
    def __init__(self, C, arch):
        super().__init__()
        dimension = len(C)
        kind, width = arch.split('16', 1)
        if width not in ('', 'x16'):
            raise ValueError(f'Unsupported architecture {arch}')
        activation = {'tanh':nn.Tanh, 'relu':nn.ReLU, 'silu':nn.SiLU}[kind]
        layers = [nn.Linear(dimension,16), activation()]
        if width == 'x16':
            layers += [nn.Linear(16,16), activation()]
        layers += [nn.Linear(16,dimension)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def train(task, output):
    source, budgets, _ = SOURCES[task]
    for budget in budgets:
        old = source/str(budget)/task
        with np.load(old/'inputs.npz') as z:
            prepared = tuple(z[k] for k in ('x','visual','C','physical_scales','split'))
        scope = ('Pure h(x)=MLP(x) on exact archived frozen visual observations and splits; '
                 'original C used only for common evaluation/loss coordinate conversion; static experiment.')
        run(SimpleNamespace(output=output/str(budget),steps=1500),task,ARCHITECTURES,
            (*prepared,{str(old/'inputs.npz'):sha(old/'inputs.npz')}),
            scope=scope, model_factory=PureProxy,
            neural_description='h(x)=MLP(x), standard torch Linear initialization, no C parameters or Cx path in model')
        dest = output/str(budget)/task
        np.save(dest/'source_indices.npy',np.load(old/'source_indices.npy'))
        protocol = json.loads((dest/'protocol.json').read_text())
        protocol.pop('AEBS_state',None)
        protocol.update(model_source_sha256=sha(__file__), initialization='standard PyTorch Linear random initialization in all layers; no C pretraining',
                        physical_state_dimensions={'aebs':2,'lane':2,'curve_braking':5}[task],
                        proxy_dimensions=prepared[0].shape[1])
        write(dest/'protocol.json',protocol)


def audit_and_summarize(task, output):
    source,budgets,conversion = SOURCES[task]
    summary = {}
    reference_holdout = None
    previous_train = set()
    for budget in budgets:
        dest=output/str(budget)/task
        old=source/str(budget)/task
        z=np.load(dest/'inputs.npz')
        x,y,C,units,split=(z[k] for k in ('x','visual','C','physical_scales','split'))
        with np.load(old/'inputs.npz') as prior:
            for key in z.files:
                np.testing.assert_array_equal(z[key],prior[key])
        ids=np.load(dest/'source_indices.npy')
        train_rows=set(ids[split=='train'])
        assert len(train_rows)==budget and previous_train <= train_rows
        previous_train=train_rows
        holdout=ids[split!='train']
        if reference_holdout is not None:
            np.testing.assert_array_equal(reference_holdout,holdout)
        reference_holdout=holdout
        rows=json.loads((dest/'results.json').read_text())
        old_rows=json.loads((old/'results.json').read_text())
        assert len(rows)==21
        metrics={}
        for row in rows:
            arch,seed=row['architecture'],row['seed']
            key=arch if seed is None else f'{arch}_{seed}'
            prediction=np.load(dest/(key+'.npz'))['proxy']
            previous=next(r for r in old_rows if r['architecture']==arch and r['seed']==seed)
            if seed is not None:
                model=PureProxy(C,arch)
                model.load_state_dict(torch.load(dest/(key+'.pt'),map_location='cpu',weights_only=False)['state_dict'])
                assert not any('C' in k for k in model.state_dict())
                with torch.no_grad():
                    np.testing.assert_allclose(model(torch.from_numpy(x)).numpy(),prediction,atol=3e-6)
                assert row['extra']['batch_sha256']==previous['extra']['batch_sha256']
            else:
                np.testing.assert_array_equal(prediction,np.load(old/(key+'.npz'))['proxy'])
            normalized=(y-prediction)@np.linalg.inv(C).T
            e=abs(normalized[split=='test']*units)
            np.testing.assert_allclose(e.mean(0),row['metrics']['test']['mae_physical'],rtol=1e-5)
            np.testing.assert_allclose(np.mean(normalized[split=='development']**2),row['development_mse'],rtol=3e-5)
            arm='pure_'+arch if seed is not None else arch
            metrics.setdefault(arm,[]).append(dict(seed=seed,mae=(e.mean(0)*conversion).tolist(),
                p95=(np.quantile(e,.95,axis=0)*conversion).tolist(),development_mse=row['development_mse'],
                selected_step=row['extra'].get('selected_step')))
        for arch in ARCHITECTURES:
            metrics['residual_'+arch]=[dict(seed=r['seed'],mae=(np.array(r['metrics']['test']['mae_physical'])*conversion).tolist(),
                development_mse=r['development_mse']) for r in old_rows if r['architecture']==arch]
        aggregates={k:dict(mae=np.mean([r['mae'] for r in v],axis=0).tolist(),development_mse=float(np.mean([r['development_mse'] for r in v])),
                          runs=v) for k,v in metrics.items()}
        rank=sorted(aggregates,key=lambda k:aggregates[k]['development_mse'])
        pure_rank=[k for k in rank if k.startswith('pure_')]
        summary[budget]=dict(architectures=aggregates,development_rank=rank,selected_pure=pure_rank[0])
    write(output/(task+'_summary.json'),summary)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task',choices=list(SOURCES))
    parser.add_argument('output',type=Path)
    parser.add_argument('--audit-only',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1)
    if not args.audit_only:
        train(args.task,args.output)
    audit_and_summarize(args.task,args.output)
