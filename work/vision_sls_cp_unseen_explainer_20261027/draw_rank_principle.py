"""Illustrative rank argument; this figure contains no experimental data."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt,font_manager
from matplotlib.patches import FancyBboxPatch

root=Path('/Users/johann/project/work_based_on_spvc')
available=[f for f in font_manager.fontManager.ttflist if Path(f.fname).is_file()]
font=next(f for name in ['PingFang HK','Arial Unicode MS','Hiragino Sans GB'] for f in available if f.name==name)
plt.rcParams.update({'font.family':[font.name,'DejaVu Sans'],'font.size':12,'axes.unicode_minus':False})
fig,ax=plt.subplots(figsize=(12,4.8),layout='constrained')
ax.set(xlim=(-.15,10.15),ylim=(-.35,3.45));ax.axis('off')
ax.text(5,3.16,'原理示意：设想将19条校准运行与1条新运行一起排序',ha='center',va='center',fontsize=16)
ax.text(5,2.74,'每一格表示：新运行可能排在第几位',ha='center',va='center',fontsize=13,color='#555')
for i in range(20):
    col=i%10;row=i//10;y=1.50-row*1.05
    risk=i==19;color='#b64c41' if risk else '#217f85';fill='#fceae6' if risk else '#e3f0ef'
    ax.add_patch(FancyBboxPatch((col+.05,y),.85,.80,boxstyle='round,pad=0.02,rounding_size=0.04',facecolor=fill,edgecolor=color,lw=1.7))
    ax.text(col+.475,y+.53,str(i+1),ha='center',va='center',fontsize=17,color=color,fontweight='bold')
    ax.text(col+.475,y+.20,'超界' if risk else '覆盖',ha='center',va='center',fontsize=11,color=color)
ax.text(4.5,.10,'19个位置可以被覆盖：19/20 = 95%',ha='center',color='#217f85',fontsize=14,fontweight='bold')
ax.text(5,-.25,'固定模型、数据条件成立、无并列分数时各位置机会相同；实际算界只用19条校准运行。',ha='center',fontsize=10,color='#555')
path=root/'docs/assets/vision_sls_cp/cp_rank_principle_20_positions.png'
fig.savefig(path,dpi=180,facecolor='white');plt.close(fig)
print(path)
