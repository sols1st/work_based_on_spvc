"""Source-bound charts for the framework comparison presentation."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('results')
BLUE = '#2574a9'
GRAY = '#7b8189'
GOLD = '#b18320'
SLATE = '#545c6a'
CONDITIONS = {
    'nominal':'原始画面', 'brightness_0p7':'亮度×0.7', 'brightness_1p3':'亮度×1.3',
    'contrast_0p7':'对比度×0.7', 'contrast_1p3':'对比度×1.3', 'blur_0p5':'模糊0.5', 'blur_1p0':'模糊1.0',
}


def main(output):
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'Arial Unicode MS', 'axes.unicode_minus':False,
                         'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    sources = {}

    def read(name):
        path = ROOT/name
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    aebs = read('benefit_cp_aebs_20261122/aggregate.json')
    lane = read('benefit_native_spvt_20261122/aggregate.json')
    starts = read('general_aebs_precision_20261123/aggregate.json')
    domains = read('general_spvt_precision_20261123/aggregate.json')
    rgb = read('aebs_native_cross_20261117_v1/full_release_bank/report.json')
    tube = read('aebs_native_cross_20261117_v1/fixed_tube_2000/report.json')
    plotted = {}

    def av(condition, mode, actor='current_robust'):
        return next(r for r in aebs if r['condition']==condition and r['mode']==mode and r['actor']==actor)

    def sv(kind):
        return next(r['metrics'] for r in lane if r['mode']=='iid' and r['kind']==kind)

    def finish(fig, name, values):
        fig.savefig(output/(name+'.png'), dpi=180)
        plt.close(fig)
        plotted[name] = values

    def bars(ax, labels, values, colors, decimals=2, suffix=''):
        result = ax.bar(np.arange(len(values)), values, color=colors, width=.6)
        ax.set_xticks(np.arange(len(values)), labels)
        ax.set_ylim(0, max(max(values),1)*1.23)
        ax.set_axisbelow(True)
        ax.grid(axis='y', alpha=.2)
        ax.bar_label(result, labels=[f'{v:.{decimals}f}{suffix}' for v in values], padding=4, fontsize=10)

    # RGB full deployment. Training provenance differs across these two checkpoints.
    fig, axes = plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    labels = ['我的RGB前端\n＋我的鲁棒控制器', '原发布灰度头\n＋我的鲁棒控制器']
    q = [rgb['rgb_dino']['q']['max_truth_m'], rgb['rgb_native']['q']['max_truth_m']]
    success = [rgb['rgb_dino']['outcomes'].get('success',0)/500*100,
               rgb['rgb_native']['outcomes'].get('success',0)/500*100]
    inside = [tube['arms'][key]['in_tube']/2000*100 for key in ['rgb_dino','rgb_native']]
    for ax, vals, title, unit, dec in zip(
        axes, [q,success,inside],
        ['整轨测距CP半宽','任务完成率：500条','满足固定±0.45 m前提：2000条'],
        ['半宽（m），越小越紧','完成率（%）','整条轨迹在界内（%）'], [4,1,2]):
        bars(ax,labels,vals,[BLUE,GRAY],dec)
        ax.set(title=title,ylabel=unit)
        if vals is success or vals is inside:
            ax.set_ylim(0,112)
            ax.set_yticks([0,25,50,75,100])
    fig.suptitle('前端交叉实验：两边共用我的鲁棒控制器与屏障；原灰度头直接迁移')
    finish(fig,'chart_rgb_deployment',dict(q_m=q,success_percent=success,in_fixed_tube_percent=inside))

    # Gray ablation: each panel has its own units and all seven conditions.
    fig, axes = plt.subplots(1,2,figsize=(11,5.2),layout='constrained',sharey=True)
    index=np.arange(7)
    gray_values={}
    for ax,metric,title,unit,dec in zip(axes,['static_mae_cm','q_m'],
        ['同一批图片上的平均测距误差','各自闭环的轨迹CP半宽'],
        ['平均绝对误差（cm），越小越准','误差半宽（m），越小越紧'],[2,3]):
        current=[av(c,'gray32_spatial')[metric] for c in CONDITIONS]
        baseline=[av(c,'gray32_mlp')[metric] for c in CONDITIONS]
        for offset,values,label,color in [(-.19,current,'我的DINO灰度消融',BLUE),(.19,baseline,'SPVC结构MLP',GRAY)]:
            rects=ax.barh(index+offset,values,.34,label=label,color=color)
            ax.bar_label(rects,labels=[f'{v:.{dec}f}' for v in values],padding=3,fontsize=9)
        ax.set_xlim(0,max(max(current),max(baseline))*1.23)
        ax.set(title=title,xlabel=unit,yticks=index,yticklabels=list(CONDITIONS.values()))
        ax.set_axisbelow(True);ax.grid(axis='x',alpha=.2)
        gray_values[metric]=dict(current=current,baseline=baseline)
    axes[0].invert_yaxis()
    axes[0].legend(loc='upper right',fontsize=9)
    fig.suptitle('灰度受控消融：源图信息相同；本图对应32×32灰度版本')
    finish(fig,'chart_gray_errors',gray_values)

    # Same original estimator, two frozen controllers.
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
    index=np.arange(7);values={}
    for offset,actor,label,color in [(-.18,'current_robust','我的鲁棒控制器',BLUE),
                                    (.18,'original_ppo','SPVC发布PPO',GRAY)]:
        v=[av(c,'gray32_mlp',actor)['success']/20 for c in CONDITIONS]
        rects=ax.bar(index+offset,v,.34,label=label,color=color)
        ax.bar_label(rects,labels=[f'{x:.2f}' for x in v],padding=3,fontsize=9)
        values[actor]=v
    ax.set(xticks=index,xticklabels=list(CONDITIONS.values()),ylim=(0,116),
           ylabel='任务完成率（%），每种子2000条',title='相同原MLP前端下，两种控制器的任务完成情况')
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(loc='lower left')
    finish(fig,'chart_controller_completion',values)

    # Time tradeoff under nominal image conditions.
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
    pairs=[(d,v) for d in [10,12,14] for v in [1,2]]
    times={}
    for actor,label,color in [('current_robust','我的鲁棒控制器',BLUE),('original_ppo','SPVC发布PPO',GRAY)]:
        vals=[next(r['metrics']['capped_time_s'] for r in starts
                   if r['group']==[d,v,'gray32_mlp',actor]) for d,v in pairs]
        ax.plot(range(6),vals,marker='o',label=label,color=color)
        for i,v in enumerate(vals):ax.annotate(f'{v:.2f}',(i,v),xytext=(0,7),textcoords='offset points',ha='center')
        times[actor]=vals
    ax.set(xticks=range(6),xticklabels=[f'{d}–{d+2} m\n{v}–{v+1} m/s' for d,v in pairs],
           ylim=(0,11),ylabel='达标时间（s），越小越快',
           title='正常图像六个起步域：两控制器均100%完成，执行时间存在取舍')
    ax.grid(axis='y',alpha=.2);ax.legend()
    finish(fig,'chart_aebs_time',times)

    # Four checkpoints, same reference scenario pool.
    kinds=['current_lowdim','spvt_pid','spvt_epoch100','same_target_e2e']
    labels=['我的低维\n反馈适配','SPVT发布\nPID初始化','SPVT历史\n第100轮','原结构同目标\n监督重训']
    colors=[BLUE,GRAY,SLATE,GOLD]
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    lane_values={}
    for ax,key,title,unit,dec in zip(axes,
        ['rmse_cm','precision_5cm','settle5_capped_seconds'],
        ['全程横向跟踪误差','第2–8秒持续保持±5 cm','进入并保持±5 cm的时间'],
        ['RMSE（cm），越小越准','达标率（%），越高越好','时间（s），越小越快'],[2,1,3]):
        v=[sv(k)[key]/20 if key=='precision_5cm' else sv(k)[key] for k in kinds]
        bars(ax,labels,v,colors,dec)
        ax.set(title=title,ylabel=unit)
        if key=='precision_5cm':
            ax.set_ylim(0,112)
            ax.set_yticks([0,25,50,75,100])
        lane_values[key]=v
    fig.suptitle('SPVT原观测接口下的任务精度：同一2000个初态，分别执行闭环')
    finish(fig,'chart_lane_metrics',lane_values)

    # Full grid. Show same-target comparator to make the attribution visible.
    fig,axes=plt.subplots(1,4,figsize=(12,3.7),layout='constrained')
    grids={}
    for ax,kind,label in zip(axes,kinds,labels):
        grid=np.array([[next(r['metrics']['precision5'] for r in domains if r['group']==[e,h,kind])/10
                        for h in [2,4,7]] for e in [20,40,60]])
        im=ax.imshow(grid,vmin=0,vmax=100,cmap='YlGnBu')
        ax.set(xticks=range(3),xticklabels=['±2°','±4°','±7°'],yticks=range(3),
               yticklabels=['±20 cm','±40 cm','±60 cm'],xlabel='初始航向范围',title=label.replace('\n',''))
        for i,j in np.ndindex(grid.shape):
            ax.text(j,i,f'{grid[i,j]:.1f}%',ha='center',va='center',
                    color='white' if grid[i,j]>65 else '#222222',fontsize=10)
        grids[kind]=grid.tolist()
    axes[0].set_ylabel('初始横偏范围')
    fig.colorbar(im,ax=axes,label='持续±5 cm达标率（%）',shrink=.8)
    fig.suptitle('九个初始状态域：每格1000个配对场景，新训练组取三种子均值')
    finish(fig,'chart_lane_domains',grids)

    record=dict(source_sha256=sources,plot_values=plotted,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_scope='Existing archived experiments; models and scenarios specified in report captions.')
    (output/'comparison_charts_manifest.json').write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    main(args.output)
