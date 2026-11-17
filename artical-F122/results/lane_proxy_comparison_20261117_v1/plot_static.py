import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).parent
out=root.parents[2]/'docs/assets/lane_proxy_comparison_20261117'
# cwd-independent workspace path: results -> artical-F122 -> workspace.
out=root.parents[2]/'docs/assets/lane_proxy_comparison_20261117';out.mkdir(parents=True,exist_ok=True)
seeds=[202611171,202611172,202611173]
d={s:json.loads((root/f'train_{s}/report.json').read_text()) for s in seeds}
fig,axs=plt.subplots(2,2,figsize=(10,7),layout='constrained')
metrics=[('Lateral MAE (mm)',lambda z:1000*z['metrics']['test']['decoded_mae_physical'][0]),
         ('Heading MAE (mrad)',lambda z:1000*z['metrics']['test']['decoded_mae_physical'][1]),
         ('Decoded MSE (x 1e-3)',lambda z:1000*z['metrics']['test']['decoded_mse']),
         ('Non-affine residual max (x 1e-3)',lambda z:1000*z['geometry']['best_affine_fit_max_abs'])]
for ax,(label,f) in zip(axs.flat,metrics):
    x=np.arange(1,4)
    a=np.array([f(d[s]['linear']) for s in seeds]);b=np.array([f(d[s]['neural']) for s in seeds])
    for i in range(3):ax.plot([x[i]-.09,x[i]+.09],[a[i],b[i]],color='.7',lw=1.5)
    ax.scatter(x-.09,a,label='Trainable Cx',c='#1769aa',s=55,zorder=3)
    ax.scatter(x+.09,b,label='Neural h(x)',c='#d97919',s=55,marker='s',zorder=3)
    ax.set_xticks(x,['Seed 1','Seed 2','Seed 3']);ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
    ax.spines[['top','right']].set_visible(False)
axs[0,0].legend(frameon=False,fontsize=9)
fig.suptitle('Lane proxy comparison: matched data, batches, loss and update budget',fontsize=13)
fig.savefig(out/'static_paired.png',dpi=180)
for kind in ['linear','neural']:
 print(kind,{n:float(np.mean([fn(d[s][kind]) for s in seeds])) for n,fn in metrics})
print('MSE relative reductions',[100*(d[s]['linear']['metrics']['test']['decoded_mse']-d[s]['neural']['metrics']['test']['decoded_mse'])/d[s]['linear']['metrics']['test']['decoded_mse'] for s in seeds])
