"""Register the common target-domain state-proxy offset before new resynthesis and CP."""
import argparse
import tempfile
from pathlib import Path

import numpy as np
import torch

from ..lane_keeping import core
from ..lane_keeping.cp import conformal_radius
from ..lane_keeping.learned_sbc import sha256
from ..lane_keeping.policy import load_actor
from ..lane_keeping.robust_trajectory_cp import write_json
from ..lane_contract_comparison.collect import parameters,scores
from .check import command
from .train import ARMS,BASE,CURVES,read,register


TARGET=Path('results/lane_paired_synthesis_target_20261117_v2')
DEV=Path('results/lane_fixed_five_curves_20261117_v1')


@torch.inference_mode()
def construct(root,development=DEV):
    torch.set_num_threads(1)
    source=development/'fixed/calibration.npz'
    frozen=development/'image_protocol.json'
    ip=read(frozen)
    spec=ip['actor_barrier']['fixed']
    assert spec['actor']==sha256(BASE/'actor.pt') and spec['sbc']==sha256(BASE/'sbc.pt')
    assert ip['observer']==sha256(BASE/'patch_projection.pt')
    data=np.load(source)
    assert str(data['protocol'])==sha256(frozen) and tuple(ip['kappas'])==CURVES
    actor=load_actor(BASE/'actor.pt').double()
    errors=data['errors']
    kappas=data['kappas']
    # A median within each rollout and then across rollouts weights each episode equally.
    means=np.median(errors,axis=1)
    centers={str(k):np.median(means[kappas==k],axis=0).tolist() for k in CURVES}
    shift=np.asarray([centers[str(k)] for k in kappas])[:,None,:]
    ideal=data['states'][:,:-1]/[core.E_SCALE,core.THETA_SCALE]
    corrected=data['observations']-shift
    visual=np.abs(corrected-ideal).max(axis=1)
    difference=np.abs(command(actor,corrected,kappas[:,None])-command(actor,ideal,kappas[:,None])).max(axis=1)
    scale=dict(visual=np.quantile(visual,.95,axis=0).tolist(),action=float(np.quantile(difference,.95)))
    design={}
    for arm,values in scores((visual,difference),scale).items():
        q,rank=conformal_radius(values,.05)
        design[arm]=dict(q=q,rank=rank,**parameters(arm,q,scale))
    # Development data set the offset, scales and design widths. Only a fresh,
    # post-training calibration on each final policy carries a coverage claim.
    root.mkdir(parents=True,exist_ok=False)
    offset=dict(centers=centers,scale=scale,raw_reference='g(x)=(e/E_SCALE,theta/THETA_SCALE)',
                observation='original DINO output z; online actor input z-b(kappa) and exact known curvature',
                corrected_state_proxy='g(x)+b(kappa) in original visual output space',
                fit='coordinatewise median per trajectory then median among trajectories with the fixed kappa',
                counts={str(k):int((kappas==k).sum()) for k in CURVES},
                source=str(source),source_sha256=sha256(source),source_protocol_sha256=sha256(frozen))
    write_json(root/'bias.json',offset)
    with tempfile.TemporaryDirectory() as tmp:
        original=Path(tmp)/'spec'
        register(original)
        protocol=read(original/'protocol.json')
    protocol.update(version='target-development/common known-curvature centering',
                    raw_camera_and_state_proxy='z minus (g(x)+b_kappa); commands pi(z-b_kappa,kappa)',
                    bias_sha256=sha256(root/'bias.json'),development_scale=scale,
                    development_contracts=design,
                    development_scores='counterfactual evaluation of shared base actor on saved development states; design only',
                    development_policy_differs_from_new_closed_loops=True)
    protocol['inputs_sha256'].update({str(p):sha256(p) for p in [source,frozen,Path(__file__)]})
    protocol['seeds'].update(calibration=202611178311,holdout=202611178312)
    write_json(root/'protocol.json',protocol)
    print('registered',root,'centers',centers,'design contracts',design,flush=True)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root',type=Path,default=TARGET)
    cli.add_argument('--development',type=Path,default=DEV)
    args=cli.parse_args()
    construct(args.root,args.development)
