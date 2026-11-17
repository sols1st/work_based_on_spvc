"""Independently count input aliases, bound identities, exact replay costs and paired effects."""
import argparse
import csv
import hashlib
import itertools
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from .factorial_aebs_inputs import BUDGETS, CAL_SEED, MODES, TEST_SEED, TRAIN_SEEDS, pixels
from .matched_aebs_library import SCALE, write
from .matched_aebs_replay import episodes,read
from Aebs.vision_sls.run import sha256


def hash_inputs(item):
    return [hashlib.sha256(x.tobytes()).hexdigest() for x in pixels(item)]


def pixel_audit(args):
    master=json.loads((args.budgets/'protocol.json').read_text())
    files=master['24000']['files']
    libs=read(args.library/'libraries.json')
    with ThreadPoolExecutor(6) as pool:
        hashes=dict(zip(files,pool.map(hash_inputs,(args.images / f for f in files))))
    with np.load(args.library/'image_hashes.npz') as z:
        if any(hashes[f][0]!=h for f,h in zip(z['filename'],z['gray_sha256'])):
            raise ValueError('audited source images changed')
    arr=np.array([hashes[f] for f in files])
    np.savez_compressed(args.root/'pixel_hashes.npz',filename=np.array(files),modes=MODES,sha256=arr)
    rows=[]
    for n in BUDGETS:
        rec=master[str(n)]
        for j,m in enumerate(MODES):
            train={hashes[f][j] for f,s in zip(rec['files'],rec['split']) if s=='train'}
            dev={hashes[f][j] for f,s in zip(rec['files'],rec['split']) if s=='validation'}
            for lib in (f'strict_{n}','common'):
                held=[hashes[f][j] for f in libs[lib]]
                rows.append(dict(budget=n,mode=m,library=lib,frames=len(held),
                                 train_dev_intersection=len(train & dev),
                                 held_frames_alias_train_dev=sum(h in train|dev for h in held),
                                 unique_held=len(set(held))))
    write(args.root/'pixel_audit.json',dict(rows=rows,hashes=sha256(args.root/'pixel_hashes.npz'),
          source=sha256(Path(__file__)),decoded_for_audit=len(files)))
    print('maximum held alias',max(r['held_frames_alias_train_dev'] for r in rows),flush=True)


def table(path,rows):
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)


def report(args):
    root=args.root; run=root/'replay'
    pix=read(root/'pixel_audit.json')['rows']
    if any(r['held_frames_alias_train_dev']>0 for r in pix):
        raise ValueError('model-selection alias with evaluation images')
    rows=read(run/'summary.json')
    rseed=read(root/'protocol.json')
    st=read(run/'frozen.json')
    cache=read(root/'cache.json')
    assert cache['protocol_sha256']==st['protocol']==sha256(root/'protocol.json')
    assert set(st['models'])=={f"{r['frontend']}_{r['budget']}_{r['seed']}" for r in rows.values()}
    assert len(rows)==len(BUDGETS)*len(TRAIN_SEEDS)*(len(MODES)+1)*2
    parent=read(args.release/'development_certificate.json')
    for path,h in [(args.release/'package/policy.pt',st['actor']),(args.release/'package/sbc.pt',st['barrier']),
                   (args.release/'development_certificate.json',st['parent']),(args.library/'manifest.json',st['library'])]:
        assert sha256(path)==h
    for path,h in rseed['bindings'].items():
        assert sha256(Path(path))==h
    totals=dict(calibration=0,holdout=0)
    terminal=[]
    for key,r in rows.items():
        folder=run/key;fit=read(root/'models'/f"{r['frontend']}_{r['budget']}_{r['seed']}"/'fit.json')
        assert r['contract']==sha256(folder/'contract.json')
        assert read(folder/'contract.json')['source_observation_sha256']==fit['model']
        ar=read(folder/'audit.json')
        if ar['status']=='pass':
            assert ar['q']<=ar['outer_q']==.45 and ar['parent_cells']==parent['checked_cells']
        else:
            assert ar['status']=='witness' and ar['witness']['drift']>0
        for phase,n,seed in [('calibration',199,CAL_SEED),('holdout',100,TEST_SEED)]:
            with np.load(folder/(phase+'.npz')) as z:
                initial,appearance=episodes(n,seed)
                np.testing.assert_array_equal(z['initial'],initial)
                np.testing.assert_array_equal(z['appearance'],appearance)
                totals[phase]+=int(z['steps'].sum())
                scores=z['scores']
                outcomes=z['outcomes']
                labels=['success','unsafe','stopped_safe_outside_goal','out_of_domain','timeout']
                counts={label:int((outcomes==label).sum()) for label in labels}
                assert sum(counts.values())==n
                terminal.append(dict(arm=key,phase=phase,**counts))
                if phase=='calibration':
                    assert r['q']==float(np.sort(scores)[189])
                else:
                    assert int((scores<=r['q']).sum())==r['covered']
                    assert int((scores<=.45).sum())==r['in_fixed_tube']
                    assert int((z['outcomes']=='unsafe').sum())==r['unsafe']
        assert counts['success']==r['success'] and counts['stopped_safe_outside_goal']==r['early_stop']
    modes=['gray32_mlp',*MODES]
    grouped=[];effects=[]
    for lib_type,n in itertools.product(['common','strict'],BUDGETS):
        lib='common' if lib_type=='common' else f'strict_{n}'
        for m in modes:
            sel=[rows[f'{lib}__{m}_{n}_{s}'] for s in TRAIN_SEEDS]
            group=dict(library=lib,budget=n,mode=m)
            for col in ['q','frame_mae','covered','success','unsafe','early_stop','in_fixed_tube',
                        'mean_peak','calibration_queries','holdout_queries']:
                v=np.array([r[col] for r in sel])
                for label,func in [('median',np.median),('min',np.min),('max',np.max)]:
                    group[col+'_'+label]=float(func(v))
            group['passed_seeds']=sum(r['audit']=='pass' for r in sel)
            grouped.append(group)
        for reference,candidate in [('gray32_mlp','gray32'),('gray32','rgb32'),('gray640','rgb640'),
                                    ('gray32','gray640'),('rgb32','rgb640')]:
            for s in TRAIN_SEEDS:
                a=rows[f'{lib}__{reference}_{n}_{s}'];b=rows[f'{lib}__{candidate}_{n}_{s}']
                effects.append(dict(library=lib,budget=n,seed=s,reference=reference,candidate=candidate,
                    q_change_m=b['q']-a['q'],q_reduction_percent=100*(1-b['q']/a['q']),
                    frame_mae_change_m=b['frame_mae']-a['frame_mae'],
                    successes_change=b['success']-a['success'],unsafe_change=b['unsafe']-a['unsafe'],
                    mean_peak_change=b['mean_peak']-a['mean_peak']))
    table(run/'terminal_counts.csv',terminal)
    table(run/'seed_summary.csv',grouped)
    table(run/'paired_effects.csv',effects)
    write(run/'audit_costs.json',dict(arms=len(rows),trajectories_cal=199*len(rows),
         trajectories_test=100*len(rows),unique_paired_scenarios=[199,100],active_frame_queries=totals,
         unique_source_images=24000,extra_independent_live_frames=0,cache_encodings=cache['cached_features'],
         extra_decodes_for_hash=24000,fit_runs=2*len(st['models']),selected_models=len(st['models']),
         audit_source=sha256(Path(__file__)),seed_table=sha256(run/'seed_summary.csv'),
         contrast_table=sha256(run/'paired_effects.csv')))


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('phase',choices=['pixels','report']);cli.add_argument('root',type=Path)
    cli.add_argument('--library',type=Path,default=Path('results/aebs_matched_replay_20261117/library'))
    cli.add_argument('--budgets',type=Path,default=Path('results/aebs_budget_scaling_20261110'))
    cli.add_argument('--images',type=Path,default=Path('/root/carla_data_spvc_expanded/full01'))
    cli.add_argument('--release',type=Path,default=Path('results/aebs_robust_synthesis_20261117_release'))
    a=cli.parse_args()
    (pixel_audit if a.phase=='pixels' else report)(a)
