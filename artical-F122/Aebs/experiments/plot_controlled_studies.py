"""Paper panels for the matched camera comparison and nonlinear lane experiments."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    return json.loads(path.read_text())


def main(args):
    results=args.results
    root=results/'lane_paired_synthesis_target_20261117_v2'
    fair=results/'aebs_factorial_20261117/replay/seed_summary.csv'
    with fair.open() as f:
        rows=list(csv.DictReader(f))
    plt.rcParams.update({'font.size':9,'svg.fonttype':'none','axes.axisbelow':True})
    fig, axes=plt.subplots(1,3,figsize=(11,3.5),layout='constrained')
    ax=axes[0]
    for mode,label,color,marker in [('gray32_mlp','SPVC-architecture MLP','#38536a','s'),
                                   ('gray32','Pretrained DINO readout','#06735c','o')]:
        cells=sorted((r for r in rows if r['library']=='common' and r['mode']==mode),key=lambda r:int(r['budget']))
        med,low,high=[[float(r[k]) for r in cells] for k in ['q_median','q_min','q_max']]
        xx=np.arange(len(cells))
        ax.errorbar(xx,med,yerr=np.array([np.subtract(med,low),np.subtract(high,med)]),label=label,
                    color=color,marker=marker,capsize=4,lw=1.35)
    ax.set(xticks=xx,xticklabels=['400','10,000','24,000'],xlabel='Target-domain image budget',
           ylabel='Trajectory radius (m)',title='(a) Same gray pixels and tuning')
    ax.legend(frameon=False,fontsize=8)
    ax=axes[1]
    keys=['uniform','diagonal','joint']
    labels=['Uniform','Scaled','Visual + action']
    for y,key in enumerate(keys):
        rows=read(root/'fixed_corrected'/key/'supplemental_audit.json')['all']['rows']
        for i,row in enumerate(rows):
            status=row['status'] if 'status' in row else row['all']
            glyph,color={'pass':('o','#06735c'),'witness':('x','#b04938'),
                         'unresolved':('s','#aaa'),'timeout':('s','#aaa')}[status]
            ax.plot(i,y,marker=glyph,color=color,ms=8,mew=1.8)
    ax.set(yticks=range(3),yticklabels=labels,xticks=range(len(rows)),
           xticklabels=[f"{r['steering_rad']:.4g}" for r in rows],xlabel='Extra steering error (rad)',
           ylim=(2.7,-.6),title='(b) Fixed actor and barrier')
    ax.tick_params(axis='x',labelsize=7)
    ax.plot([],[],marker='o',color='#06735c',ls='none',label='Pass')
    ax.plot([],[],marker='x',color='#b04938',ls='none',label='Witness')
    ax.legend(loc='lower center',frameon=False,ncol=2,fontsize=8)
    ax=axes[2]
    experiments=[results/'lane_paired_synthesis_20261117_v1',root]
    for i,(folder,label,color) in enumerate(zip(experiments,['Original proxy','Corrected proxy'],['#38536a','#06735c'])):
        blind=read(folder/'blind_summary.json')['arms']
        values=[100*blind[k]['higher_curvature']['mean_abs_e_m'] for k in keys]
        bars=ax.bar(np.arange(3)+(i-.5)*.35,values,.35,color=color,label=label)
        ax.bar_label(bars,fmt='%.1f',fontsize=8,padding=2)
    ax.set(xticks=range(3),xticklabels=labels,ylabel='High-curvature mean |offset| (cm)',
           xlabel='Training contract',title='(c) Retrained controllers')
    ax.legend(frameon=False,fontsize=8)
    for a in axes:
        a.spines[['top','right']].set_visible(False)
        a.grid(axis='y',alpha=.18)
    args.output.mkdir(exist_ok=True,parents=True)
    for extension in ['png','svg']:
        fig.savefig(args.output/f'controlled_studies.{extension}',dpi=220)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=Path('results'))
    p.add_argument('--output',type=Path,default=Path('../docs/assets/spvc_benefit_20261117'))
    main(p.parse_args())
