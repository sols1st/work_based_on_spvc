"""Re-evaluate fixed-reference-policy CP coverage for a focused explanation.

No new rollouts, controller optimization, fitting, calibration changes or tuning.
"""
from pathlib import Path
import ast
import hashlib
import json
import math
import sys
sys.dont_write_bytecode=True
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt,font_manager

ROOT=Path('/Users/johann/project/work_based_on_spvc')
BASE=Path('/Users/johann/Documents/Codex/2026-10-22/jian/outputs')
WORK=ROOT/'work/vision_sls_cp_unseen_explainer_20261027'
ASSET=ROOT/'docs/assets/vision_sls_cp'
sys.path.insert(0,str(ROOT/'work/vision_sls_doc_cp_20261026'))
from doc_cp_features import positive_scale,box_bound

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def metrics(r,b):
    c=r<=b+1e-8
    return {'points_covered':int(c.sum()),'points':int(c.size),
            'point_coverage':float(c.mean()),'trajectories_covered':int(c.all(axis=1).sum()),
            'trajectories':len(c),'trajectory_coverage':float(c.all(axis=1).mean())}

result={'date':'2026-10-27','scope':'Frozen reference-policy original data re-evaluation; no new experiments',
        'comparison':'Same frozen scale h and same test residuals, before and after multiplying q',
        'scenes':{}}
for scene,expected_before,expected_after in [('parking',13,41),('quad',4,29)]:
    folder=BASE/'vision_sls_建议原方案CP验证_20261026'/scene
    mpath=folder/'doc_cp_model.json';model=json.loads(mpath.read_text())
    protocol=json.loads((folder/'protocol.json').read_text());split=protocol['evaluation']['axis_stress_rollouts']
    source=ROOT/'VISION-SLS/dyn'/('unicycle_of_perception.py' if scene=='parking' else 'quad_10d_of_perception.py')
    nodes=[n for n in ast.walk(ast.parse(source.read_text())) if isinstance(n,ast.Assign) and
      any(isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name) and t.value.id=='self' and t.attr=='C' for t in n.targets)]
    assert len(nodes)==1
    C=np.asarray(ast.literal_eval(nodes[0].value.args[0]),float)
    offset=np.array([3.,0,0,0]) if scene=='parking' else np.zeros(10)
    phases={};source_records={}
    for phase in ['calibration','baseline']:
        path=folder/phase/'diagnostic_traces.npz';d=np.load(path)
        n,ny,T=d['residuals'].shape
        x=d['observation_states'];reconstructed=d['observations']-np.einsum('ab,nbt->nat',C,x+offset[None,:,None])
        error=float(np.max(np.abs(reconstructed-d['residuals'])))
        assert error<1e-10
        xx=x.transpose(0,2,1).reshape(-1,d['states'].shape[1])
        r=np.max(np.abs(reconstructed),axis=1)
        h=positive_scale(xx,model).reshape(n,T);b=box_bound(xx,model).reshape(n,T)
        assert np.allclose(b,model['cp_multiplier']*h,rtol=1e-12,atol=1e-12)
        phases[phase]=(r,h,b)
        source_records[phase]={'path':str(path),'sha256':sha(path),'residual_reconstruction_max_error':error}
    r,h,b=phases['calibration'];n=len(r)
    k=math.ceil((n+1)*(1-model['alpha']));scores=(r/h).max(axis=1)
    q=float(np.sort(scores)[k-1]);assert abs(q-model['cp_multiplier'])<1e-12
    assert source_records['calibration']['sha256']==model['calibration_traces_sha256']
    test_r,test_h,test_b=phases['baseline']
    test={}
    for group,s in [('random',slice(split,None)),('stress',slice(None,split)),('mixed',slice(None))]:
        test[group]={'before':metrics(test_r[s],test_h[s]),'after':metrics(test_r[s],test_b[s])}
    assert test['random']['before']['trajectories_covered']==expected_before
    assert test['random']['after']['trajectories_covered']==expected_after
    comparison=json.loads((folder/'doc_cp_comparison.json').read_text())
    assert test['mixed']['after']['trajectories_covered']==comparison['methods']['baseline']['doc_box_trajectories_covered']
    misses=[]
    for i,t in np.argwhere(test_r[split:]>test_b[split:]+1e-8):
        j=int(i)+split;t=int(t)
        misses.append({'random_subset_run_1_based':int(i)+1,'saved_rollout_index_0_based':j,
                       'observation_index_0_based':t,'observation_number_1_based':t+1,
                       'r':float(test_r[j,t]),'h':float(test_h[j,t]),'b_cp':float(test_b[j,t]),
                       'r_over_h':float(test_r[j,t]/test_h[j,t]),'r_over_b_cp':float(test_r[j,t]/test_b[j,t])})
    assert len(misses)==1
    training=BASE/protocol['training']['source_task']/'calibration/diagnostic_traces.npz'
    assert sha(training)==model['training_traces_sha256']
    assert len({sha(training),source_records['calibration']['sha256'],source_records['baseline']['sha256']})==3
    result['scenes'][scene]={'training_trajectories':model['training_trajectories'],
       'training_points':model['training_points'],'training_trace_path':str(training),'training_trace_sha256':sha(training),
       'calibration_trajectories':n,'rank_k':k,'multiplier':q,'alpha':model['alpha'],
       'calibration_before':metrics(r,h),'calibration_after':metrics(r,b),
       'test':test,'random_test_uncovered_observations':misses,'source_traces':source_records,
       'calibration_sorted_last10':np.sort(scores)[-10:].tolist(),
       'model_path':str(mpath),'model_sha256':sha(mpath),'protocol_path':str(folder/'protocol.json'),
       'protocol_sha256':sha(folder/'protocol.json')}

fonts=[f for f in font_manager.fontManager.ttflist if Path(f.fname).is_file()]
font=next(f for name in ['PingFang HK','Arial Unicode MS','Hiragino Sans GB'] for f in fonts if f.name==name)
plt.rcParams.update({'font.family':[font.name,'DejaVu Sans'],'font.size':12,'axes.unicode_minus':False,
                     'figure.facecolor':'white','axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(10.8,5),layout='constrained')
for ax,scene,name in zip(axes,['parking','quad'],['停车场','无人机']):
    d=result['scenes'][scene]['test']['random']
    values=[100*d[s]['trajectory_coverage'] for s in ['before','after']]
    ax.bar([0,1],values,color=['#8c9dac','#217f85'],width=.58)
    for j,s in enumerate(['before','after']):
        a=d[s]
        ax.text(j,values[j]+2,f"{a['trajectories_covered']}/{a['trajectories']}\n{values[j]:.2f}%",ha='center',va='bottom')
    ax.axhline(95,c='#bc4f42',ls='--',lw=1.5,label='预设覆盖目标 95%')
    ax.set(xticks=[0,1],xticklabels=['校准前的尺度 h','CP 校准后的 q × h'],ylim=(0,116),
           ylabel='独立随机测试的整轨迹覆盖率（%）',title=name)
    ax.set_yticks([0,25,50,75,95]);ax.grid(axis='y',alpha=.12);ax.set_axisbelow(True)
    ax.legend(loc='upper left',fontsize=9)
fig.suptitle('控制器与测试数据保持相同：只比较 CP 校准前后的误差界',fontsize=15)
fig.supxlabel('图中是有限测试比例；不能只凭它证明真实覆盖概率超过95%。',fontsize=10,color='#555')
path=ASSET/'unseen_random_coverage_before_after.png';fig.savefig(path,dpi=180);plt.close(fig)
result['figure']={'path':str(path),'sha256':sha(path)}
(WORK/'固定策略CP覆盖_原始数据复核.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({s:result['scenes'][s]['test']['random'] for s in result['scenes']},ensure_ascii=False))
