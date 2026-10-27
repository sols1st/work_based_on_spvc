"""Consolidate verified tables and render explanatory figures; no control runs."""
from pathlib import Path
import hashlib
import json
import re
import sys
sys.dont_write_bytecode=True
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, Rectangle

ROOT=Path('/Users/johann/project/work_based_on_spvc')
BASE=Path('/Users/johann/Documents/Codex/2026-10-22/jian/outputs')
WORK=ROOT/'work/vision_sls_cp_consolidation_20261026'
ASSET=ROOT/'docs/assets/vision_sls_cp'
ASSET.mkdir(parents=True,exist_ok=True)
REPORT=ROOT/'docs/current/VISION-SLS_CP完整说明与可行性分析.md'
LEDGER=ROOT/'docs/current/VISION-SLS_CP全部尝试结果总账.md'
LEDGER_ARCHIVE=ROOT/'docs/archive/2026-10-26_VISION-SLS_CP阶段文档/project_docs/current/VISION-SLS_CP全部尝试结果总账.md'

# Read source tables verbatim, avoiding numeric transcription and rounding changes.
source=LEDGER if LEDGER.exists() else LEDGER_ARCHIVE
tables=re.findall(r'(?m)^\|[^\n]*\n(?:\|[^\n]*\n)+',source.read_text())
headers={
 'FIT_TABLE':'| 尺度模型 |', 'CAL_TABLE':'| 方法 | 场景 | 校准轨迹数 |',
 'OVERVIEW_TABLE':'| 编号 | 场景 | 尝试内容 |',
 'RETRO_TABLE':'| 编号 | 场景 | 范数 |',
 'METRIC_TABLE':'| 编号 | 场景 | 任务成功 |',
 'PAIRED_TABLE':'| 编号 | 任务成功：原→新 |',
 'PLAN_TABLE':'| 编号 | 场景 | 外层检查数 |',
 'BATCH_TABLE':'| 批次 | 场景 | 用途 |',
 'EVIDENCE_TABLE':'| 编号 | 主要原始证据 |',
}
text=REPORT.read_text()
for token,header in headers.items():
    matches=[t for t in tables if t.startswith(header)]
    assert len(matches)==1,(token,len(matches))
    text=text.replace('@@'+token+'@@',matches[0].rstrip())
assert '@@' not in text
REPORT.write_text(text)

data_path=BASE/'vision_sls_全部尝试结果汇总_20261026/全部尝试_结构化记录与来源.json'
data=json.loads(data_path.read_text())
assert len(data['experiment_records'])==22
assert len(data['image_batches'])==19
assert data['unique_visual_trajectories']==990 and data['image_observations']==32800
for path,expected in data['source_json_sha256'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,path
for batch in data['image_batches']:
    assert hashlib.sha256(Path(batch['trace_path']).read_bytes()).hexdigest()==batch['trace_sha256']

available=[f for f in font_manager.fontManager.ttflist if Path(f.fname).is_file()]
font=next((f for name in ['PingFang HK','Arial Unicode MS','Hiragino Sans GB'] for f in available if f.name==name),None)
assert font,'No available Chinese font'
font_manager.fontManager.addfont(font.fname)
plt.rcParams.update({'font.family':[font.name,'DejaVu Sans'],'font.size':12,
                     'axes.unicode_minus':False,'figure.facecolor':'white',
                     'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':170})
blue='#2671b8';orange='#d47424';green='#258c72';red='#bd3e4f'

fig,ax=plt.subplots(figsize=(9.2,6.4),layout='constrained')
b=.08
ax.add_patch(Circle((0,0),np.sqrt(2)*b,fill=False,ls='--',ec=green,lw=2.5,label='外包球：半径 √2 × 0.08'))
ax.add_patch(Rectangle((-b,-b),2*b,2*b,fc=blue,alpha=.12,ec=blue,lw=2.5))
ax.plot([-b,b,b,-b,-b],[-b,-b,b,b,-b],c=blue,lw=2.5,label='无穷范数盒：每个分量绝对值 ≤ 0.08')
ax.add_patch(Circle((0,0),b,fill=False,ec=orange,lw=2.5,label='同半径二范数球：总长度 ≤ 0.08'))
ax.scatter([.06],[.06],c=red,s=65,zorder=4)
ax.annotate('(0.06, 0.06)\n在盒内，在同半径球外',(.06,.06),xytext=(-.122,.118),
            arrowprops={'arrowstyle':'->','color':red},color=red,fontsize=12)
ax.axhline(0,c='#aaa',lw=.7);ax.axvline(0,c='#aaa',lw=.7)
ax.set(xlim=(-.13,.13),ylim=(-.13,.16),xlabel='第一个误差分量',ylabel='第二个误差分量',
       title='教学图：相同数值 0.08，可以定义不同的误差集合')
ax.set_aspect('equal');ax.legend(loc='lower center',fontsize=10,framealpha=.95)
fig.savefig(ASSET/'norm_geometry.png');plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
s=np.linspace(0,2,100);h=.04+.02*s
axes[0].plot(s,h,c=blue,lw=2.5,label='拟合尺度 h(s) = 0.04 + 0.02s')
axes[0].plot(s,1.25*h,c=orange,lw=2.5,label='CP 半宽 = 1.25 × h(s)')
axes[0].scatter([0,1,2],[.032,.072,.080],c=green,s=55,zorder=4,label='一条教学校准轨迹的三个误差')
axes[0].scatter([1.2],[.08],c=red,zorder=5,s=55)
axes[0].annotate('s=1.2 → 半宽 0.08',(1.2,.08),xytext=(.45,.105),fontsize=11,
                 arrowprops={'arrowstyle':'->','color':red},color=red)
axes[0].set(xlabel='状态 s',ylabel='无穷范数误差 / 半宽',title='1. 学空间趋势，再整体乘倍率',ylim=(.02,.115))
axes[0].legend(fontsize=9,loc='lower right');axes[0].grid(alpha=.15)
scores=np.array([.55,.60,.65,.70,.75,.80,.85,.90,.92,.94,.96,.98,1,1.02,1.05,1.10,1.15,1.20,1.25])
axes[1].bar(np.arange(1,20),scores,color=blue,alpha=.7)
axes[1].bar([19],[1.25],color=orange)
axes[1].axhline(1.25,c=orange,ls='--');axes[1].text(1,1.30,'n=19，目标95% → k=19 → q=1.25',fontsize=11)
axes[1].set(xlabel='排序后轨迹编号',ylabel='每条轨迹的最大归一化分数',title='2. 每条运行只贡献一个排序分数',ylim=(0,1.45),xticks=[1,5,10,15,19])
fig.suptitle('完整手算例子（教学数据，不是服务器实验结果）',fontsize=15)
fig.savefig(ASSET/'cp_worked_example.png');plt.close(fig)

# Recompute the focused D07 plot directly from the frozen model and two raw NPZ files.
doc=BASE/'vision_sls_建议原方案CP验证_20261026/quad'
sys.path.insert(0,str(ROOT/'work/vision_sls_doc_cp_20261026'))
from doc_cp_features import box_bound,positive_scale,envelope
model=json.loads((doc/'doc_cp_model.json').read_text())
metrics={};curves={}
for role,folder in [('参考策略','baseline'),('CP 新策略','doc_cp_convexified_iter50000_outer100_rf0p001')]:
    path=doc/folder/'diagnostic_traces.npz';a=np.load(path)
    n,ny,T=a['residuals'].shape
    states=a['observation_states'].transpose(0,2,1).reshape(-1,10)
    bound=box_bound(states,model).reshape(n,T)
    residual=np.max(np.abs(a['residuals']),axis=1)
    ratio=(residual/bound).max(axis=1)
    stage8=np.abs(a['states'][:,8,:-1]).max(axis=1)
    curves[role]=(ratio,stage8)
    metrics[role]={'trajectory_coverage':int((ratio<=1+1e-8).sum()),
                   'max_absolute_stage_state_8':float(stage8.max()),
                   'stage_state_8_violations':int((stage8>2.001).sum()),
                   'scale_floor_observations':int((positive_scale(states,model)<=.001+1e-12).sum()),
                   'unclipped_envelope_min':float(envelope(states,model).min()),
                   'trace_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
assert metrics['参考策略']['trajectory_coverage']==37
assert metrics['CP 新策略']['trajectory_coverage']==0
assert metrics['CP 新策略']['stage_state_8_violations']==50
assert metrics['CP 新策略']['scale_floor_observations']==1420
assert abs(metrics['CP 新策略']['max_absolute_stage_state_8']-2.05458941711)<1e-9
fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
for role,color in [('参考策略',blue),('CP 新策略',orange)]:
    ratio,stage8=curves[role]
    axes[0].plot(np.arange(1,51),np.sort(ratio),label=role,c=color,lw=2.3)
    axes[1].scatter(np.arange(1,51),stage8,label=role,c=color,s=24,alpha=.8)
axes[0].axhline(1,c=red,ls='--',label='全程覆盖阈值 = 1')
axes[0].set(yscale='log',xlabel='按各自最大比值排序的运行编号',ylabel='每条轨迹最大残差 / CP 盒半宽',title='新策略 50 条全部至少有一步超界')
axes[1].axhline(2,c=red,ls='--',label='阶段状态允许上限 = 2')
axes[1].set(xlabel='配对运行编号',ylabel='阶段状态分量 8 的最大绝对值',title='新策略 50 条全部违反该状态约束')
for ax in axes:ax.legend(fontsize=10);ax.grid(alpha=.15)
fig.suptitle('D07 实际数据：零碰撞仍不等于满足全部安全要求',fontsize=15)
fig.savefig(ASSET/'d07_diagnosis.png');plt.close(fig)

audit={'kind':'documentation consolidation; no new control experiments',
       'report':str(REPORT),'report_sha256':hashlib.sha256(REPORT.read_bytes()).hexdigest(),
       'verified_source_json_files':len(data['source_json_sha256']),
       'verified_unique_image_batches':len(data['image_batches']),
       'trajectories':990,'observations':32800,'experiment_records':22,
       'table_sources_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
       'figures':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(ASSET.glob('*.png'))],
       'D07_independent_plot_recalculation':metrics,'font':font.fname}
(WORK/'content_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:audit[k] for k in ['verified_source_json_files','verified_unique_image_batches','experiment_records','trajectories','observations']},ensure_ascii=False))
