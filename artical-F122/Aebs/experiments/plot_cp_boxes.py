"""Paper figures for the three matched trajectory CP experiments."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from ..conformal.trajectory_boxes import METHODS, widths
from .high_dim_cp_boxes import OUTPUTS

LABELS = ['Uniform', 'Per axis', 'Signed, per axis', 'State bins', 'Signed state bins',
          'Interpolated', 'Two-feature grid', 'Log-linear']
TASKS = ['braking', 'hover', 'crane']
TITLES = ['5D braking', '6D hover', '8D crane']


def comparison(results, out):
    with (results/'high_dim_cp_boxes_20261117/summary.csv').open() as f:
        info = {(r['task'],r['method']):r for r in csv.DictReader(f)}
    fig, axes = plt.subplots(1,2, figsize=(8,4), layout='constrained', sharey=True)
    for side, (axis, field, title) in enumerate(zip(axes,['volume_ratio','metric_delta_pct'],
                                           ['(a) Mean error-box volume', '(b) Certified physical margin'])):
        raw=np.array([[float(info[(t,k)][field]) for t in TASKS] for k in METHODS])
        if side == 0:
            colors = -np.log2(raw)
            color_limit = np.max(np.abs(colors))
        else:
            colors = raw
            color_limit = np.max(np.abs(raw))
        axis.imshow(colors, cmap='RdBu', norm=TwoSlopeNorm(0,-color_limit,color_limit), aspect='auto')
        for i,j in np.ndindex(raw.shape):
            label = (f'{raw[i,j]:.2f}×' if side==0 else f'{raw[i,j]:+.1f}%')
            axis.text(j,i,label,ha='center',va='center',fontsize=9,
                      color='white' if abs(colors[i,j])/color_limit>.62 else '#15232d')
        axis.axhline(2.5,color='white',lw=2)
        axis.set_xticks(range(3),TITLES)
        axis.set_yticks(range(8),LABELS)
        axis.tick_params(length=0)
        axis.set_title(title)
        axis.set_xlabel('Ratio to uniform (smaller is tighter)' if side==0
                        else 'Change from uniform (larger is better)')
        for s in axis.spines.values():
            s.set_visible(False)
    export(fig,out/'box_comparison')


def brake_example(results, out):
    root = results/(OUTPUTS['braking']+'_cp_boxes_20261117_v1')
    arm = root/'20261117501_linear'
    boxes = json.loads((arm/'boxes.json').read_text())['boxes']
    with np.load(arm/'holdout.npz') as data:
        d = data['states'][:,:-1,2]
        error = 2*data['errors'][...,2]
    fig, ax = plt.subplots(figsize=(6.2,3.05),layout='constrained')
    ax.scatter(d.ravel(),error.ravel(),s=5,alpha=.2,c='#667077',linewidths=0,rasterized=True,
               label='Blind image errors')
    # Add both sides of discontinuities, so a step is drawn at the declared boundary.
    cuts = boxes['state_signed']['shape']['xcuts']
    grid = np.unique(np.r_[np.linspace(1.92,6.15,600),cuts,np.nextafter(cuts,-np.inf)])
    for key,label,color,ls in [('fixed_uniform','Uniform','#485171','--'),
                               ('state_signed','Signed distance bins','#a74a17','-')]:
        row = boxes[key]
        n,p = widths(row['shape'],np.column_stack([grid,0*grid]))
        ax.plot(grid,2*row['q']*p[:,2],ls,color=color,lw=1.6,label=label)
        ax.plot(grid,-2*row['q']*n[:,2],ls,color=color,lw=1.6)
    ax.axhline(0,lw=.7,color='#777')
    ax.set(xlabel='True obstacle distance (m)',ylabel='Camera distance error (m)')
    ax.spines[['top','right']].set_visible(False)
    ax.legend(loc='upper left',fontsize=8,ncol=3,frameon=False)
    ax.margins(y=.22)
    export(fig,out/'braking_box_example')


def export(fig,root):
    root.parent.mkdir(exist_ok=True,parents=True)
    for ext in ('.png','.svg'):
        fig.savefig(root.with_suffix(ext),dpi=210)
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, default=Path('results'))
    p.add_argument('--out',type=Path,default=Path('../docs/assets/high_dim_cp_boxes_20261117'))
    a = p.parse_args()
    plt.rcParams.update({'font.size':9,'axes.titlesize':11,'svg.fonttype':'none'})
    comparison(a.results,a.out)
    brake_example(a.results,a.out)
