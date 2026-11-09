from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('/Users/johann/project/work_based_on_spvc/artical-F122/results/lane_trajectory_cp_20261107_v1')
cal=np.load(root/'calibration.npz');test=np.load(root/'holdout.npz')
q=json.loads((root/'calibration.json').read_text())['radius']
fig,axes=plt.subplots(1,2,figsize=(11,4.2),gridspec_kw={'width_ratios':[1.35,1]},layout='constrained')
for data,label,color in [(cal,'Calibration: 199 runs','#2563eb'),(test,'Holdout: 100 runs','#0d9488')]:
    scores=np.sort(data['scores'])
    axes[0].step(scores,np.arange(1,len(scores)+1)/len(scores),where='post',label=label,color=color,lw=2)
axes[0].axvline(.08,color='#737373',ls='--',label='Previous radius: 0.08')
axes[0].axvline(q,color='#dc2626',ls='--',label=f'Trajectory CP: {q:.5f}')
axes[0].axhline(.95,color='#bbbbbb',ls=':',lw=1)
axes[0].set(xlabel='Maximum normalized error over one 80-step run',ylabel='Fraction of complete runs covered',ylim=(0,1.03),title='One calibration score per trajectory')
axes[0].legend(loc='lower right',fontsize=8,frameon=False)
counts=[int((test['scores']<=r).sum()) for r in [.08,q]]
axes[1].bar([0,1],counts,color=['#a3a3a3','#0d9488'],width=.55)
for i,n in enumerate(counts):axes[1].text(i,n+2,f'{n}/100',ha='center',fontsize=12,fontweight='bold')
axes[1].set(xticks=[0,1],xticklabels=['Previous box\n0.08','Trajectory CP box\n0.09009'],ylim=(0,110),ylabel='Fully covered holdout trajectories',title='Independent holdout (same frozen policy)')
axes[1].text(.5,.03,'100/100 runs stayed inside the lane.\nCoverage and safety are different events.',transform=axes[1].transAxes,ha='center',fontsize=9)
for a in axes:a.spines[['top','right']].set_visible(False)
fig.savefig(root/'trajectory_coverage.png',dpi=180)
fig.savefig(root/'trajectory_coverage.pdf')
print(root/'trajectory_coverage.png')
