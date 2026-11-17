"""Plot all three paired optimizer seeds without treating replay trajectories as training seeds."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

FRONTS=[('gray32_mlp','Gray 32, SPVC head','#303a4b','s'),
        ('gray32','Gray 32, DINO','#007a5c','o'),
        ('rgb32','RGB 32, DINO','#568f1b','^'),
        ('gray640','Gray 640, DINO','#c66b24','v'),
        ('rgb640','RGB 640, DINO','#8c4c8f','D')]
BUDGETS=[400,10000,24000]


def plot(args):
    with args.main.open() as f: rows=list(csv.DictReader(f))
    plt.rcParams.update({'font.size':9,'svg.fonttype':'none','font.family':'DejaVu Sans'})
    fig,axes=plt.subplots(1,2,figsize=(8.6,3.7),sharey=True,layout='constrained')
    for mode_idx,(mode,name,color,marker) in enumerate(FRONTS):
        for ax,prefix in zip(axes,['common','strict']):
            xs=[];medians=[]
            for i,budget in enumerate(BUDGETS):
                library='common' if prefix=='common' else f'strict_{budget}'
                q=np.array([float(r['q']) for r in rows if r['library']==library and
                            int(r['budget'])==budget and r['frontend']==mode])
                assert len(q)==3
                x=i+(mode_idx-2)*.085
                xs.append(x);medians.append(np.median(q))
                ax.plot([x,x],[q.min(),q.max()],lw=1.3,color=color)
            ax.plot(xs,medians,ls='none',marker=marker,ms=5,color=color,label=name)
    axes[0].set_ylabel('95% trajectory error radius (m)')
    for ax,title in zip(axes,['(a) Common held-image library','(b) Strict total-package library']):
        ax.axhline(.45,color='#995738',ls='--',lw=1,zorder=0)
        ax.set(yscale='log',xticks=range(3),xticklabels=['400','10,000','24,000'],
               xlabel='Target-domain images in package',title=title,xlim=(-.36,2.36))
        ax.grid(axis='y',which='major',lw=.5,alpha=.4)
        ax.spines[['top','right']].set_visible(False)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=5,loc='outside lower center',fontsize=8,frameon=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=190)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('main',type=Path);p.add_argument('output',type=Path)
    plot(p.parse_args())
