"""Independent confirmation of the first frozen linear/neural pair and three box designs."""
import argparse
from pathlib import Path

import numpy as np
import torch

from ..conformal import trajectory_boxes as boxes
from ..curve_braking.evaluate import digest, save
from ..overhead_crane import cp_boxes,plant
from ..overhead_crane.camera import Camera
from ..overhead_crane.evaluate import collect
from .check import replay
from .check_boxes import global_wrapper
from .collect_boxes import OUT as SOURCE, BASE
from .evaluate import ImageDigest
from .train import ProxyObservation,arm_path,read


OUT=Path('results/crane_cp_boxes_confirmation_20261117_v1')
METHODS=('fixed_uniform','fixed_axis','state_bins')
FAMILIES=('linear','neural')
SEED=817731
SPLITS={'calibration':(499,819781),'holdout':(500,819782)}


def bind(args):
    p=read(SOURCE/'camera_protocol.json')
    models={}
    for family in FAMILIES:
        name=f'{SEED}_{family}'
        old=arm_path(BASE,family,SEED)
        if p['arms'][name]['checkpoint']!=digest(old/'frontend.pt'):
            raise ValueError('Changed observer')
        models[name]=dict(checkpoint=p['arms'][name]['checkpoint'],shapes=digest(SOURCE/name/'boxes.json'))
    if digest(args.weights)!=p['dino_weights'] or any(digest(args.repo/f)!=h for f,h in p['dino_source'].items()):
        raise ValueError('Frozen DINO asset mismatch')
    settings=dict(models=models,methods=list(METHODS),splits={k:list(v) for k,v in SPLITS.items()},
                  gains=p['gains'],batch=8,initial_scale=1,rank=475,alpha=.05,
                  previous_camera_protocol=digest(SOURCE/'camera_protocol.json'),
                  source=digest(Path(__file__)),score_implementation=digest(Path(boxes.__file__)),
                  certificate_implementation=digest(Path(cp_boxes.__file__)),
                  main_contrast='state_bins minus fixed_axis certified common initial scale')
    path=OUT/'protocol.json'
    OUT.mkdir(exist_ok=True)
    if path.exists():
        if read(path)!=settings:
            raise ValueError('Confirmation settings changed')
    elif args.phase=='calibration':
        save(path,settings)
    else:
        raise ValueError('Confirmation protocol missing')
    return settings


def contracts():
    return {f'{SEED}_{f}':digest(OUT/f'{SEED}_{f}'/'boxes.json') for f in FAMILIES}


@torch.inference_mode()
def run(args):
    protocol=bind(args)
    seal=OUT/'all_boxes_frozen.json'
    if args.phase=='holdout' and read(seal)!=contracts():
        raise ValueError('Seal all six confirmation tubes before using blind scenarios')
    if args.phase=='calibration' and seal.exists():
        raise ValueError('Confirmation tubes already frozen')
    camera=Camera(args.repo,args.weights,args.device)
    n,scene_seed=SPLITS[args.phase]
    scenario=plant.scenarios(n,scene_seed)
    all_rows={}
    for family in FAMILIES:
        name=f'{SEED}_{family}'
        arm=OUT/name
        arm.mkdir(exist_ok=True)
        shapes={k:read(SOURCE/name/'boxes.json')['boxes'][k]['shape'] for k in METHODS}
        model=ProxyObservation(family).to(args.device).eval()
        model.load_state_dict(torch.load(arm_path(BASE,family,SEED)/'frontend.pt',weights_only=True,map_location=args.device))
        capture=ImageDigest(camera,8,n)
        data=collect(scenario,protocol['gains'],capture,model,8)
        data['image_digests']=np.array([h.hexdigest() for h in capture.hashers])
        z=plant.observation(data['states'][:,:-1]).reshape(-1,4)
        h=model.nominal(torch.tensor(z,device=args.device,dtype=torch.float32)).cpu().numpy()
        data['latent_errors']=data['latent']-h.reshape(n,plant.HORIZON,4)
        data['latent_scores']=np.abs(data['latent_errors']).max((1,2))
        path=arm/(args.phase+'.npz')
        np.savez_compressed(path,**data)
        f=cp_boxes.features(data['states'][:,:-1])
        e=data['errors']
        replay(data,model.cpu(),n,scene_seed,protocol['gains'],True)
        rows={}
        for k,shape in shapes.items():
            if args.phase=='calibration':
                cp=boxes.calibrate(shape,f,e)
                if cp['rank']!=475:
                    raise ValueError('Unexpected order statistic')
                rows[k]=dict(**cp,shape=shape,audit=cp_boxes.audit(shape,cp['q']),
                             global_wrapper_audit=cp_boxes.audit(global_wrapper(shape),cp['q']))
            else:
                c=read(arm/'boxes.json')['boxes'][k]
                scores=boxes.trajectory_scores(shape,f,e)
                neg,pos=boxes.widths(shape,f)
                rows[k]=dict(covered=int((scores<=c['q']).sum()),
                             mean_volume=float(np.prod((neg+pos)*c['q'],axis=-1).mean()),
                             scale=c['audit']['certification_metric'],passed=c['audit']['passed'],
                             global_scale=c['global_wrapper_audit']['certification_metric'])
        if args.phase=='calibration':
            save(arm/'boxes.json',dict(boxes=rows,calibration=digest(path),protocol=digest(OUT/'protocol.json')))
        else:
            all_rows[name]=dict(boxes=rows,holdout=digest(path),**plant.metrics(data['states']))
        print(args.phase,name,n,flush=True)
    if args.phase=='calibration':
        save(seal,contracts())
    else:
        save(OUT/'summary.json',dict(arms=all_rows,protocol=digest(OUT/'protocol.json'),
                                     thresholds_seal=digest(seal),effective_frames=(499+500)*2*40))


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('phase',choices=['calibration','holdout'])
    cli.add_argument('--device',default='cuda:0')
    cli.add_argument('--repo',type=Path,default=Path('../external/dinov2'))
    cli.add_argument('--weights',type=Path,default=Path('../external/dinov2_weights/dinov2_vits14_pretrain.pth'))
    args=cli.parse_args()
    torch.set_num_threads(3)
    run(args)
