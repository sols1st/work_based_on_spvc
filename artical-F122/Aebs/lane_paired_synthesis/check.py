"""Recheck final per-arm hashes, all closed-loop transitions, CP ranks and witnesses."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..lane_keeping import core
from ..lane_keeping.learned_sbc import load_sbc_checkpoint, sha256
from ..lane_keeping.policy import load_actor


def load(path):
    return json.loads(path.read_text())


def command(actor, values, kappas):
    values=np.asarray(values)
    z=np.column_stack((values.reshape(-1,2),np.broadcast_to(kappas,values.shape[:-1]).ravel()/core.KAPPA_SCALE))
    return actor(torch.from_numpy(z).double()).flatten().numpy().reshape(values.shape[:-1])


def check_witnesses(actor,barrier,fit,rows):
    metric_scale=np.array([core.E_SCALE,core.THETA_SCALE])
    for row in rows:
        for part in row['curvatures'].values():
            if part['status']!='witness':continue
            w=part['witness']
            x,e=np.asarray(w['x']),np.asarray(w['error'])
            assert np.all(abs(x)<=1) and np.all(abs(e)<=np.asarray(fit['width'])+1e-14)
            obs=float(command(actor,x+e,w['kappa']))
            nominal=float(command(actor,x,w['kappa']))
            if fit['action'] is not None:
                assert abs(obs-nominal)<=fit['action']+1e-12
            assert abs(w['additive_action'])<=row['normalized_margin']
            nxt=core.dynamics(x*metric_scale,(obs+w['additive_action'])*core.PHI_MAX,w['kappa'])/metric_scale
            np.testing.assert_allclose(nxt,w['next_state'],atol=1e-12)
            before,after=barrier(torch.tensor(np.stack([x,nxt]))).flatten().numpy()
            np.testing.assert_allclose([before,after],[w['before'],w['after']],rtol=0,atol=1e-10)
            assert before<=20 and after>before


@torch.inference_mode()
def check(root):
    torch.set_num_threads(1)
    design=load(root/'protocol.json')
    io=load(root/'image_protocol.json')
    calibration_locks=load(root/'all_tubes_frozen.json')
    blind=load(root/'blind_summary.json')
    centered='bias_sha256' in design
    token=sha256(root/'image_protocol.json')
    assert calibration_locks['protocol']==token==blind.get('protocol',blind.get('image_protocol'))
    if centered:
        assert sha256(root/'bias.json')==design['bias_sha256']==io['bias_sha256']
        centers=load(root/'bias.json')['centers']
    scale=design['development_scale']
    metric_scale=np.array([core.E_SCALE,core.THETA_SCALE])
    pairs={}
    for arm in design['arms']:
        out=root/arm
        fit=load(out/'final_contract.json')
        frozen=load(out/'frozen.json')
        audit=load(out/'audit.json')
        history=load(out/'training.json')
        assert calibration_locks['contracts'][arm]==sha256(out/'final_contract.json')==audit['final_contract']
        assert (sha256(out/'actor.pt')==fit['actor_sha256']==frozen['actor_sha256'] and
                sha256(out/'sbc.pt')==fit['sbc_sha256']==frozen['sbc_sha256'])
        assert io.get('actor_barrier',io.get('models'))[arm]==dict(actor=fit['actor_sha256'],sbc=fit['sbc_sha256'])
        assert sha256(out/'calibration.npz')==fit['calibration'] and fit['protocol']==token
        assert audit['protocol']==sha256(root/'protocol.json')==frozen['protocol']
        assert audit['frozen']==sha256(out/'frozen.json') and frozen['training_history']==sha256(out/'training.json')
        assert len(history)==design['training']['cegis_rounds']
        assert all(r['completed_actor_steps']==design['training']['actor_steps_per_round'] and
                   r['completed_barrier_steps']==design['training']['barrier_steps_per_round'] for r in history)
        actor=load_actor(out/'actor.pt').double().requires_grad_(False)
        barrier=load_sbc_checkpoint(out/'sbc.pt')[0].double().requires_grad_(False)
        assert sha256(out/'holdout.npz')==blind.get('holdouts',blind.get('holdout_sha256'))[arm]
        for split in ('calibration','holdout'):
            data=np.load(out/(split+'.npz'))
            state,observed,k=data['states'],data['observations'],data['kappas'][:,None]
            assert state.shape==(io['counts'][split],81,2)
            if split not in pairs:
                pairs[split]=(state[:,0],k)
            for a,b in zip(pairs[split],(state[:,0],k)):
                np.testing.assert_array_equal(a,b)
            current=state[:,:-1]
            shift=np.asarray([centers[str(float(v))] for v in k[:,0]])[:,None,:] if centered else 0.
            action=command(actor, observed-shift,k)
            np.testing.assert_allclose(action,data['actions'],rtol=0,atol=2e-12)
            steps=core.dynamics_batch(current.reshape(-1,2), action.ravel()*core.PHI_MAX,
                                      np.broadcast_to(k,action.shape).ravel())
            np.testing.assert_allclose(steps.reshape(current.shape),state[:,1:],rtol=0,atol=2e-12)
            err=observed-current/metric_scale
            np.testing.assert_allclose(err,data['errors'],rtol=0,atol=1e-13)
            if centered:
                wrapper=np.load(out/(split+'_corrected.npz'))
                assert str(wrapper['raw_sha256'])==sha256(out/(split+'.npz'))
                np.testing.assert_allclose(wrapper['corrected_observations'],observed-shift,atol=1e-13)
                np.testing.assert_allclose(wrapper['centered_errors'],err-shift,atol=1e-13)
            visual=np.abs(err-shift)
            act_delta=np.abs(action-command(actor,current/metric_scale,k))
            if arm=='uniform':
                scores=visual.max(axis=(1,2))
            elif arm=='diagonal':
                scores=(visual/scale['visual']).max(axis=(1,2))
            else:
                scores=np.maximum(.5*(visual/scale['visual']).max(axis=(1,2)),act_delta.max(1)/scale['action'])
            if split=='calibration':
                rank=int(np.ceil((len(scores)+1)*(1-design['alpha'])))
                assert rank==fit['rank']
                np.testing.assert_allclose(fit['q'],np.sort(scores)[rank-1],rtol=0,atol=2e-12)
            else:
                mask=(visual<=np.array(fit['width'])).all(axis=(1,2))
                if fit['action'] is not None:
                    mask &= (act_delta<=fit['action']).all(1)
                np.testing.assert_array_equal(scores<=fit['q'],mask)
                record=blind['arms'][arm]['all']
                assert mask.sum()==record['covered']
                assert (np.abs(state/metric_scale)>1).any(axis=(1,2)).sum()==record['unsafe']
                np.testing.assert_allclose(np.abs(state[:,1:,0]).mean(),record['mean_abs_e_m'])
        check_witnesses(actor,barrier,fit,audit['rows'])
        print(arm,'inputs, trajectory scores, outcomes and barrier witnesses verified',flush=True)
    if centered:
        modeldir={key:root/key for key in design['arms']}
        base=Path('results/lane_robust_synthesis_20261116_v2/package')
        if not (base/'actor.pt').is_file():
            base=Path('results/lane_contract_comparison_20261117_v1/inputs/package')
        modeldir['fixed']=base
        for mode in ['fixed_raw','fixed_corrected']:
            folder=root/mode
            data={split:np.load(folder/(split+'.npz')) for split in ['calibration','holdout']}
            net=load_actor(modeldir['fixed']/'actor.pt').double()
            b=load_sbc_checkpoint(modeldir['fixed']/'sbc.pt')[0].double()
            for arm in design['arms']:
                fit=load(folder/arm/'final_contract.json')
                assert fit['actor_sha256']==sha256(modeldir['fixed']/'actor.pt') and fit['sbc_sha256']==sha256(modeldir['fixed']/'sbc.pt')
                for split,z in data.items():
                    ks=z['kappas'][:,None]
                    shift=np.asarray([centers[str(float(v))] for v in ks[:,0]])[:,None,:] if mode=='fixed_corrected' else 0.
                    u=command(net,z['observations']-shift,ks)
                    np.testing.assert_allclose(z['actions'],u,rtol=0,atol=2e-12)
                    np.testing.assert_array_equal(z['states'][:,0],pairs[split][0])
                    np.testing.assert_array_equal(ks,pairs[split][1])
                    if split=='holdout':
                        e=z['errors']-shift
                        valid=(abs(e)<=fit['width']).all((1,2))
                        if fit['action'] is not None:
                            nom=command(net,z['states'][:,:-1]/metric_scale,ks)
                            valid &= (abs(u-nom)<=fit['action']).all(1)
                        assert valid.sum()==blind['arms'][mode+'/'+arm]['all']['covered']
                supp=load(folder/arm/'supplemental_audit.json')
                assert supp['contract']==sha256(folder/arm/'final_contract.json')
                check_witnesses(net,b,fit,supp['all']['rows'])
        study=root/'common_box_cross'
        spec=load(study/'protocol.json')
        assert sha256(root/'fixed_corrected/uniform/final_contract.json')==spec['cutoff_source']
        for path in study.glob('*__*.json'):
            entry=load(path)
            a,b=entry['actor'],entry['sbc']
            ap=modeldir[a]/'actor.pt'
            bp=modeldir[b]/'sbc.pt'
            assert spec['models'][a]['actor']==sha256(ap) and spec['models'][b]['sbc']==sha256(bp)
            assert entry['protocol']==sha256(study/'protocol.json')
            check_witnesses(load_actor(ap).double(),load_sbc_checkpoint(bp)[0].double(),
                            dict(width=spec['box'],action=None),entry['audit']['rows'])
        assert len(list(study.glob('*__*.json')))==10
        print('Both fixed-policy controls and all ten crossed actor/barrier pairs verified.')


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('root',type=Path)
    check(cli.parse_args().root)
