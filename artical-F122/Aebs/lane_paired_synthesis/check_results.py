"""Independently reconstruct actions, transitions, CP ranks and sampled image hashes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ..lane_keeping import core
from ..lane_keeping.learned_sbc import sha256
from ..lane_keeping.policy import load_actor
from ..lane_keeping.renderer import RoadRenderer
from .evaluate import ARMS, REFERENCE, read


def check(root, package, mode):
    proto=read(root/'image_protocol.json')
    lock=read(root/'all_tubes_frozen.json')
    report=read(root/'blind_summary.json')
    assert lock['protocol']==sha256(root/'image_protocol.json')
    assert lock['contracts']==report['contracts']
    target=mode=='target'
    nets=(['fixed_raw','fixed_corrected',*ARMS] if target else (['fixed'] if mode=='fixed' else ARMS))
    old_scale=read(REFERENCE/'protocol.json')['development']['scale']
    bias=read(root/'bias.json') if target else None
    renderer=RoadRenderer(192,192)
    past={}
    stats={'calibration':0,'holdout':0,'image_digests_checked':0,'contracts':0}
    for name in nets:
        f=root/name
        netdir=package if name.startswith('fixed') else f
        model=load_actor(netdir/'actor.pt').double().eval().requires_grad_(False)
        group_scales=bias['scale'] if target and name!='fixed_raw' else old_scale
        for phase in ('calibration','holdout'):
            with np.load(f/(phase+'.npz')) as z:
                assert str(z['protocol'])==lock['protocol']
                states,actions,raw,ks=z['states'],z['actions'],z['observations'],z['kappas']
                n,T=actions.shape
                assert T==80 and n==(199 if phase=='calibration' else 100)
                stats[phase]+=n
                if phase in past:
                    np.testing.assert_array_equal(states[:,0],past[phase][0])
                    np.testing.assert_array_equal(ks,past[phase][1])
                past[phase]=(states[:,0].copy(),ks.copy())
                scale=[core.E_SCALE,core.THETA_SCALE]
                center=states[:,:-1]/scale
                y=raw
                raw_res=raw-center
                np.testing.assert_allclose(z['errors'],raw_res,rtol=0,atol=2e-13)
                if target and name!='fixed_raw':
                    shift=np.array([bias['centers'][str(k)] for k in ks])
                    y=raw-shift[:,None,:]
                    with np.load(f/(phase+'_corrected.npz')) as cz:
                        assert str(cz['raw_sha256'])==sha256(f/(phase+'.npz'))
                        np.testing.assert_allclose(y,cz['corrected_observations'],rtol=0,atol=1e-14)
                        np.testing.assert_allclose(y-center,cz['centered_errors'],rtol=0,atol=2e-14)
                stacked_k=np.broadcast_to(ks[:,None,None]/core.KAPPA_SCALE,(n,T,1))
                def action(obs):
                    x=torch.tensor(np.concatenate([obs,stacked_k],axis=2).reshape(-1,3),dtype=torch.float64)
                    return torch.cat([model(xx) for xx in x.split(512)]).numpy().reshape(n,T)
                actual=action(y)
                np.testing.assert_allclose(actual,actions,rtol=0,atol=1e-12)
                ref=action(center)
                e,theta=np.moveaxis(states[:,:-1],-1,0)
                next_e=e+core.SPEED*core.DT*np.sin(theta)
                next_theta=theta+core.SPEED*core.DT*np.tan(core.PHI_MAX*actions)/core.WHEELBASE-core.SPEED*core.DT*ks[:,None]
                np.testing.assert_allclose(np.stack([next_e,next_theta],axis=2),states[:,1:],rtol=0,atol=2e-13)
                # Re-render all 80 frames of one independently indexed episode in every split.
                h=hashlib.sha256()
                for state in states[-1,:-1]:
                    h.update(renderer.render(state,ks[-1]).tobytes())
                assert h.hexdigest()==str(z['image_sha256'][-1])
                stats['image_digests_checked']+=1
                eyes=np.abs(y-center).max(axis=1)
                fitted=(eyes/group_scales['visual']).max(1)
                joint=np.maximum(.5*fitted,np.abs(actual-ref).max(1)/group_scales['action'])
                scored={'uniform':eyes.max(1),'diagonal':fitted,'joint':joint}
                for arm in ARMS if name.startswith('fixed') else [name]:
                    if name=='fixed':
                        p=root/arm/'final_contract.json'; key=arm
                    elif name.startswith('fixed_'):
                        p=f/arm/'final_contract.json'; key=name+'/'+arm
                    else:
                        p=f/'final_contract.json'; key=arm
                    assert lock['contracts'][key]==sha256(p)
                    cp=read(p); assert cp['actor_sha256']==sha256(netdir/'actor.pt')
                    assert cp['sbc_sha256']==sha256(netdir/'sbc.pt')
                    row=scored[arm]
                    if phase=='calibration':
                        assert cp['calibration']==sha256(f/(phase+'.npz')) and cp['rank']==190
                        assert abs(np.sort(row)[189]-cp['q'])<1e-12
                        stats['contracts']+=1
                    else:
                        assert (row<=cp['q']).sum()==report['arms'][key]['all']['covered']
                        mae=np.abs(states[:,1:,0]).mean()
                        assert abs(mae-report['arms'][key]['all']['mean_abs_e_m'])<1e-13
    print(json.dumps(stats))


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('mode',choices=['fixed','trained','target'])
    cli.add_argument('root',type=Path)
    cli.add_argument('--package',type=Path,default=REFERENCE/'inputs/package')
    args=cli.parse_args()
    torch.set_num_threads(2)
    check(args.root,args.package,args.mode)
