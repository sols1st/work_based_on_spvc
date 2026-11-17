"""Plot paired high-curvature tracking and equal-budget five-curve verification."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def read(path):
    return json.loads(path.read_text())


def make_figure(old,new,path):
    fig,(tracking,cert)=plt.subplots(1,2,figsize=(10.7,4.15),layout='constrained',
                                     gridspec_kw={'width_ratios':[1,1.08]})
    names=['Uniform','Scaled','Visual + action']
    tags=['uniform','diagonal','joint']
    for pos,(study,label,marker,color) in enumerate([(old,'Original proxy','o','#4b6485'),
                                                     (new,'Curvature-adjusted proxy','s','#097457')]):
        report=read(study/'blind_summary.json')['arms']
        values=[100*report[t]['higher_curvature']['mean_abs_e_m'] for t in tags]
        tracking.plot(np.arange(3),values,marker=marker,color=color,label=label)
        for i,v in enumerate(values):
            tracking.annotate(f'{v:.2f}',(i,v),xytext=(0,-14 if pos==0 and i==2 else 7),textcoords='offset points',
                              ha='center',color=color,fontsize=8)
    tracking.set(xticks=np.arange(3),xticklabels=names,xlim=(-.3,2.3),
                 ylabel='Mean absolute lateral error (cm)',
                 title='(a) New controllers at |curvature| = 0.02 / m')
    tracking.grid(axis='y',color='#e3e6e9')
    tracking.legend(fontsize=8,loc='best',frameon=False)
    datasets=[('Old / uniform',new/'fixed_corrected/uniform/supplemental_audit.json'),
              ('Old / scaled',new/'fixed_corrected/diagonal/supplemental_audit.json'),
              ('Old / visual + action',new/'fixed_corrected/joint/supplemental_audit.json'),
              ('Retrained / uniform',new/'uniform/audit.json'),('Retrained / scaled',new/'diagonal/audit.json'),
              ('Retrained / visual + action',new/'joint/audit.json')]
    markers={'pass':('o','#097457'),'witness':('x','#a34530'),
             'unresolved':('s','#748397'),'timeout':('s','#748397')}
    for y,(lab,file) in enumerate(datasets):
        record=read(file)
        rows=record.get('all',record)['rows']
        for i,r in enumerate(rows):
            marker,color=markers[r.get('five_curvatures',r.get('status'))]
            cert.plot(i,y,marker=marker,color=color,markersize=6)
    cert.set(xticks=np.arange(5),xticklabels=['0','.045','.0675','.0855','.09'],yticks=np.arange(6),
             yticklabels=[lab for lab,_ in datasets],ylim=(6.4,-.55),xlim=(-.5,4.5),
             xlabel='Additional steering error (rad)',title='(b) Curvature-adjusted proxy, all five curves')
    cert.legend(handles=[Line2D([],[],ls='',marker=m,color=c,label=lab) for (m,c),lab in
                         zip([markers['pass'],markers['witness'],markers['unresolved']],['Verified','Witness','Unresolved'])],
                loc='lower center',fontsize=8,ncol=3,frameon=False)
    for ax in (tracking,cert):ax.spines[['top','right']].set_visible(False)
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path, dpi=200,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('v1',type=Path)
    cli.add_argument('v2',type=Path)
    cli.add_argument('output',type=Path)
    a=cli.parse_args()
    make_figure(a.v1,a.v2,a.output)
