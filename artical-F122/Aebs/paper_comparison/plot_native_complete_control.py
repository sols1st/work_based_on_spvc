"""Render direct plots from the registered native-pipeline comparison."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    data=json.loads((a.results/'summary.json').read_text())['arms']
    plt.rcParams.update({'font.family':'Arial Unicode MS','axes.unicode_minus':False,
                         'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    actors=['native_seed171','native_seed172','native_seed173','original_ppo','current_robust']
    labels=['原SPVC综合策略 171','原SPVC综合策略 172','原SPVC综合策略 173',
            '原发布PPO初始化','我的鲁棒控制器']
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
    values={}
    for ax,bank,title in zip(axes,['common','full24k'],['独立留出图库：3229张','固定部署图库：24k张（含训练图）']):
        matrix=np.array([[100*data[bank+'__'+front+'__'+actor]['outcomes']['success']/2000
                          for front in ['native_m','dino_m']] for actor in actors])
        ax.imshow(matrix,vmin=0,vmax=100,cmap='Blues',aspect='auto')
        ax.set_xticks([0,1],['原发布灰度前端','我的RGB DINO前端'])
        ax.set_yticks(range(len(labels)),labels);ax.set_title(title)
        for i in range(5):
            for j in range(2):
                ax.text(j,i,f'{matrix[i,j]:.0f}%\n{int(matrix[i,j]*20)}/2000',ha='center',va='center',
                        color='white' if matrix[i,j]>50 else '#263445')
        values[bank]=matrix.tolist()
    fig.suptitle('完整原SPVC策略与我方策略：谁能完成停车任务？',fontsize=15)
    fig.savefig(a.output/'complete_pipeline_matrix.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.3),constrained_layout=True)
    banks=['common','full24k'];banklabels=['留出图库','固定24k部署图库']
    q=[data[b+'__dino_m__current_robust']['q_m'] for b in banks]
    bars=axes[0].bar([0,1],q,color=['#2574a9','#56a399'],width=.5)
    axes[0].bar_label(bars,labels=[f'{v:.3f} m' for v in q],padding=5)
    axes[0].axhline(.45,color='#b54d3b',linestyle='--',label='原鲁棒屏障允许 ±0.45 m')
    axes[0].set_xticks([0,1],banklabels);axes[0].set_ylim(0,1.5)
    axes[0].set_ylabel('我方轨迹CP半宽（m）');axes[0].set_title('完成任务，也要检查是否满足证书前提')
    axes[0].legend(loc='upper right',fontsize=9)
    x=np.arange(2)
    times={}
    for i,(actor,label,color) in enumerate([('original_ppo','RGB DINO＋原发布PPO','#7b8189'),
                                           ('current_robust','RGB DINO＋我的控制器','#2574a9')]):
        t=[data[b+'__dino_m__'+actor]['time_to_goal_capped_seconds']['mean'] for b in banks]
        assert all(data[b+'__dino_m__'+actor]['outcomes']['success']==2000 for b in banks)
        br=axes[1].bar(x+(i-.5)*.34,t,width=.34,label=label,color=color)
        axes[1].bar_label(br,labels=[f'{v:.2f}s' for v in t],padding=4)
        times[actor]=t
    axes[1].set_xticks(x,banklabels);axes[1].set_ylim(0,12)
    axes[1].set_ylabel('成功到达目标的平均时间（s）');axes[1].set_title('两组均完成2000/2000，但速度有取舍')
    axes[1].legend(fontsize=9)
    fig.savefig(a.output/'cp_scope_and_time.png',dpi=180);plt.close(fig)
    manifest=dict(source_sha256=hashlib.sha256((a.results/'summary.json').read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  success_percent=values,own_q_m=q,successful_goal_time_s=times)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
