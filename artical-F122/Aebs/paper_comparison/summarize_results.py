"""Summarize the registered five-task controlled backend experiment.

Never counts diagnostic/pre-fix files or interprets inconclusive as unsafe.
Plots compare finite-horizon certification, not physical success probabilities.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


NAMES=('aebs','lane','braking','hover','crane')


def read(path):
    return json.loads(path.read_text())


def source_identity(metadata):
    return {key:metadata[key]['sha256'] for key in
            ('actor','controller','barrier','contract','error_source')
            if key in metadata and isinstance(metadata[key],dict)}


def summarize(base, include_spvc=True):
    rows=[];sources={};progress={};every_step={}
    for task in NAMES:
        folder=base/task
        paths=[folder/'current.json',folder/'spvt_sampled.json',folder/'spvt_global.json']
        if task in ('aebs','lane','braking'):
            paths.append(folder/'spvt_global_strengthened.json')
        if include_spvc:
            paths.extend(folder/('spvc_'+model+'.json') for model in ('observation','additive'))
            paths.append(folder/'spvc_stochastic_inheritance.json')
        for path in paths:
            sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        current, sampled, global_result=map(read,paths[:3])
        meta=current['protocol']['metadata']
        for other in [sampled['metadata'],global_result['metadata']]:
            if source_identity(meta)!=source_identity(other):
                raise ValueError('Different frozen model/contract identities: '+task)
            for key in ('error_lo','error_hi','noise_lo','noise_hi'):
                np.testing.assert_array_equal(meta[key],other[key],err_msg=task+' '+key)
        population=sampled['contexts']
        if len(population)!=1 or population[0]['context']!='sampled_iid':
            raise ValueError('Expected the registered iid population')
        last=population[0]['summary'][-1]
        whole=global_result['contexts']
        row=dict(task=task,horizon=meta['horizon'],dt=meta['dt'],
                 current_passed=current['passed'],current_seconds=current['elapsed_seconds'],
                 spvt_complete_certified=last['certified'],spvt_n=last['n_initial_states'],
                 spvt_sampled_seconds=sampled['total_wall_seconds'],
                 spvt_hoeffding_lower=last['hoeffding_lower'],
                 spvt_global_passed=all(c['status']=='certified' for c in whole),
                 spvt_global_prefix=min(c['certified_steps'] for c in whole),
                 spvt_global_seconds=global_result['total_wall_seconds'],
                 actual_unsafe_witnesses=sum(r['status']=='counterexample' for r in population[0]['results']))
        if task in ('aebs','lane','braking'):
            strengthened=read(folder/'spvt_global_strengthened.json')
            row['spvt_strengthened_seconds']=sum(c['wall_seconds'] for c in strengthened['contexts'])
            row['spvt_strengthened_geometric_fractions']=[c['certified_geometric_volume_fraction'] for c in strengthened['contexts']]
        if include_spvc:
            inheritance=read(folder/'spvc_stochastic_inheritance.json')
            if inheritance['identity_mismatches']:
                raise ValueError('Stochastic inheritance has identity mismatch: '+task)
            if inheritance['source_current_result']['sha256']!=sources[str(folder/'current.json')]:
                raise ValueError('Stochastic inheritance references another current audit: '+task)
            row['strong_same_support_stochastic_status']=inheritance['status']
            row['stochastic_integration_is_not_the_final_method_verdict']=True
            for model in ('observation','additive'):
                record=read(folder/('spvc_'+model+'.json'))
                row['spvc_'+model+'_status']=record['nonincrease_status']
                row['spvc_'+model+'_seconds']=record['elapsed_seconds']
                row['spvc_'+model+'_significant_witnesses']=len(record['positive_drift_witnesses'])
        rows.append(row)
        progress[task]=population[0]['summary']
        prefix=np.asarray([r['certified_steps'] for r in population[0]['results']])
        every_step[task]=[dict(horizon=k,certified=int(np.sum(prefix>=k)),
                              certified_fraction=float(np.mean(prefix>=k)))
                         for k in range(1,int(meta['horizon'])+1)]
    return dict(scope='controlled backend components; not native full-pipeline paper reproduction',
                fixed_visual_frontend_and_controller=True,new_visual_CP=False,
                rows=rows,progress=progress,every_step=every_step,source_sha256=sources)


def plot(report, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,name,title in zip(axes,('aebs','lane'),('AEBS: 400 control steps','Lane keeping: 80 control steps')):
        progress=report['every_step'][name]
        x=[p['horizon'] for p in progress]
        y=[100*p['certified_fraction'] for p in progress]
        ax.plot(x,[100]*len(x),color='#197b68',label='Robust barrier (whole initial domain)',lw=2)
        ax.step(x,y,where='post',color='#bb6e21',label='SPVT-style reachability (2,000 starts)',lw=2)
        ax.set(title=title,xlabel='Verification horizon (steps)',ylabel='Certified initial samples (%)',ylim=(-3,105))
        ax.grid(alpha=.2);ax.legend(loc='center right',fontsize=8)
    fig.suptitle('Shared frozen controller and error support; unverified does not mean unsafe',fontsize=12)
    fig.savefig(directory/'certified_horizon.png',dpi=180)
    fig.savefig(directory/'certified_horizon.svg')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=Path('results/paper_backend_comparison_20261118_v1'))
    p.add_argument('--figures',type=Path)
    p.add_argument('--without-spvc',action='store_true')
    a=p.parse_args()
    report=summarize(a.results,not a.without_spvc)
    target=a.results/('summary_without_spvc.json' if a.without_spvc else 'summary.json')
    target.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    keys=list(dict.fromkeys(k for row in report['rows'] for k in row))
    with target.with_suffix('.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(report['rows'])
    if a.figures:plot(report,a.figures)
    print(json.dumps(report['rows'],indent=2,ensure_ascii=False))


if __name__=='__main__':main()
