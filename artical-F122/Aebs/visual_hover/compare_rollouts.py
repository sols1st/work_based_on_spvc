"""Render arm-specific hover trajectories, seal all calibration radii, then test."""
import argparse
from pathlib import Path

import numpy as np
import torch

from ..curve_braking.evaluate import digest, save
from ..curve_braking.models import Proxy
from ..curve_braking.vision import Encoder
from ..lane_keeping.cp import conformal_radius
from . import certificate, plant
from .compare_proxies import OUTPUT, SOURCE, REPO, WEIGHTS, read
from .evaluate import collect, metrics


def parameters(folder, frontend):
    name = folder.name.rsplit('_',1)[1]
    model = Proxy(name, frontend['basis'].shape[1])
    model.load_state_dict(torch.load(folder/'model.pt',map_location='cpu',weights_only=True))
    return model.eval().requires_grad_(False)


def stats(records, q, q_raw):
    raw = np.abs(records['raw']).max(axis=(1,2))
    return {**metrics(records['states']), 'covered':int(np.sum(records['scores']<=q)),
            'raw_covered':int(np.sum(raw<=q_raw)),
            'decoded_peak_mean':float(records['scores'].mean()), 'raw_peak_mean':float(raw.mean())}


def run(args):
    output,source=args.output,args.source
    protocol=read(output/'protocol.json')
    design=protocol['design']
    if (digest(source/'frontend.pt')!=protocol['source_files']['frontend.pt'] or
            digest(source/'evaluation/synthesis.json')!=protocol['source_files']['evaluation/synthesis.json'] or
            digest(args.weights)!=protocol['dino_weights'] or args.device!=protocol['encoder_device']):
        raise ValueError('frozen visual assets or inference device changed')
    if not all(digest(args.repo/name)==expected for name,expected in protocol['repo_sources'].items()):
        raise ValueError('frozen DINO implementation changed')
    frontend=torch.load(source/'frontend.pt',map_location='cpu',weights_only=True)
    encoder=Encoder(args.repo,args.weights,args.device)
    models=read(output/'all_models_frozen.json')
    def freeze_bindings():
        here=Path(__file__).resolve().parent
        files=[Path(__file__),here/'evaluate.py',here/'compare_proxies.py',here/'plant.py',
               here/'certificate.py',here/'perception.py',here.parent/'curve_braking/vision.py',
               here.parent/'curve_braking/models.py',here.parent/'lane_keeping/cp.py']
        return dict(protocol=digest(output/'protocol.json'),
                    models={arm:digest(output/arm/'model.pt') for arm in models},
                    source={p.relative_to(here.parent).as_posix():digest(p) for p in files})
    bindings=freeze_bindings()
    if bindings['models']!=models:
        raise ValueError('a frozen model changed')
    save(output/'evaluation_protocol.json',bindings)
    batch=protocol['image_batch']
    arrangement=[('calibration',protocol['calibration']),('holdout',protocol['holdout'])]
    for split,(n,seed) in arrangement:
        scenario=plant.scenarios(n,seed)
        if split=='holdout':
            hashes={arm:digest(output/arm/'contract.json') for arm in models}
            if read(output/'all_tubes_frozen.json')!=hashes:
                raise ValueError('contracts must all be frozen before blind testing')
        for arm in models:
            folder=output/arm
            model=parameters(folder,frontend)
            path=folder/f'{split}.npz'
            # An interrupted run can reuse a complete file bound to this frozen configuration.
            stamp=folder/f'{split}_provenance.json'
            provenance=dict(evaluation=digest(output/'evaluation_protocol.json'),split=split,seed=seed,n=n,
                            contract=digest(output/'all_tubes_frozen.json') if split=='holdout' else None)
            if path.exists() and stamp.exists() and read(stamp)=={**provenance,'records':digest(path)}:
                with np.load(path) as data:
                    records=dict(data)
            else:
                records=collect(scenario,design,encoder,frontend,model,batch,record_observations=True)
                np.savez_compressed(path,**records)
                save(stamp,{**provenance,'records':digest(path)})
            if split=='calibration':
                q,rank=conformal_radius(records['scores'],.05)
                q_raw,_=conformal_radius(np.abs(records['raw']).max(axis=(1,2)),.05)
                save(folder/'contract.json',dict(q=q,q_raw=q_raw,rank=rank,n=n,
                     definition='normalized decoded-position box; raw q is separate marginal diagnostic',
                     model=digest(folder/'model.pt'),calibration=digest(path)))
                audit=certificate.verify(design,q)
                audit['contract']=digest(folder/'contract.json')
                save(folder/'certificate.json',audit)
            else:
                contract=read(folder/'contract.json')
                save(folder/'blind.json',stats(records,contract['q'],contract['q_raw']))
            print(split,arm,flush=True)
        if split=='calibration':
            save(output/'all_tubes_frozen.json',{arm:digest(output/arm/'contract.json') for arm in models})
    if freeze_bindings()!=bindings:
        raise ValueError('frozen configuration changed during evaluation')


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--source',type=Path,default=SOURCE)
    cli.add_argument('--output',type=Path,default=OUTPUT)
    cli.add_argument('--repo',type=Path,default=REPO)
    cli.add_argument('--weights',type=Path,default=WEIGHTS)
    cli.add_argument('--device',default='cuda:0')
    args=cli.parse_args()
    torch.set_num_threads(2)
    run(args)
