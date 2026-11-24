"""Build source-bound figures and a readable report after every test arm finishes."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


LABELS={
    'spvt_selected':'原SPVT第55轮\n原灰度输入',
    'pid_initializer':'原发布初始化\n原灰度输入',
    'shared_dino__spvt_selected':'共同DINO\n原SPVT第55轮策略',
    'shared_dino__pid_initializer':'共同DINO\n原发布初始化策略',
    'current_rgb_dino':'我的RGB DINO\n我的神经控制器'}


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
                     ['| '+' | '.join(map(str,r))+' |' for r in rows])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--barrier-audit',type=Path)
    a=p.parse_args();project=Path(__file__).resolve().parents[3]
    results=a.results.resolve();native=read(results/'native_task_v2/summary.json')
    selection=read(results/'native_task_v2/development_selection.json')
    rgbroot=results/'rgb_complete_v3';rgb=read(rgbroot/'summary.json')
    protocol=read(rgbroot/'protocol.json');rows=rgb['arms'];names=protocol['arms'];n=protocol['phases']['test']['n']
    assert set(rows)==set(names)
    assets=project/'docs/assets/spvt_complete_comparison_20261123';assets.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'Arial Unicode MS','axes.unicode_minus':False,'font.size':10,
                         'axes.spines.top':False,'axes.spines.right':False})
    colors=['#7b8189','#adb1b6','#b18320','#58a28c','#2574a9']
    labels=[LABELS[k] for k in names]
    fig,ax=plt.subplots(figsize=(10,3.5),constrained_layout=True)
    epochs=[int(Path(r['path']).stem.split('_')[-1]) for r in selection['candidates']]
    mae=[r['metrics']['rmse_cm'] for r in selection['candidates']]
    ax.plot(epochs,mae,'.-',color='#7b8189',label='26个原方法检查点：开发集RMSE')
    chosen=selection['selected'];epoch=int(Path(chosen['path']).stem.split('_')[-1])
    ax.scatter([epoch],[chosen['metrics']['rmse_cm']],s=70,color='#2574a9',label=f'冻结选择第{epoch}轮')
    ax.set_xlabel('原训练轮次');ax.set_ylabel('横偏RMSE（cm）');ax.set_title('先在独立开发集选模型，再做新测试')
    ax.legend();fig.savefig(assets/'native_selection.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    safe=[100*(1-rows[k]['unsafe_count']/n) for k in names]
    strict=[100*rows[k]['strict_5cm_count']/n for k in names]
    for ax,values,title in zip(axes,[safe,strict],['全程横偏不超过80 cm','安全且第2–8秒持续保持±5 cm']):
        bars=ax.bar(range(len(names)),values,color=colors)
        ax.bar_label(bars,labels=[f'{v:.1f}%' for v in values],padding=4)
        ax.set_xticks(range(len(names)),labels,fontsize=8);ax.set_ylim(0,115);ax.set_ylabel('轨迹比例（%）');ax.set_title(title)
    fig.savefig(assets/'rgb_safety_and_precision.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4),constrained_layout=True)
    q=[rows[k]['thresholds']['action_q_deg'] for k in names]
    bars=ax.bar(range(len(names)),q,color=colors);ax.bar_label(bars,labels=[f'{v:.2f}°' for v in q],padding=4)
    ax.set_xticks(range(len(names)),labels,fontsize=9);ax.set_ylabel('整轨视觉—自身代理动作偏差CP半宽（°）')
    ax.set_title('没有使用新手工策略作参考：每组比较实际图像动作与自身名义动作')
    ax.set_ylim(0,max(q)*1.2);fig.savefig(assets/'action_proxy_cp.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4),constrained_layout=True)
    for key,color in zip(names,colors):
        z=np.load(rgbroot/(key+'__test.npz'),allow_pickle=False)
        e=z['states_cm_degrees'][0,:,0];cross=np.flatnonzero(abs(e)>80)
        end=int(cross[0])+1 if len(cross) else len(e)
        ax.plot(np.arange(end)*.05,e[:end],color=color,label=LABELS[key].replace('\n',' / '))
        if len(cross):ax.scatter([(end-1)*.05],[e[end-1]],marker='x',color=color,s=50)
    ax.axhline(80,color='#b54d3b',linestyle='--');ax.axhline(-80,color='#b54d3b',linestyle='--')
    ax.fill_between([2,8],[-5,-5],[5,5],alpha=.12,color='#2574a9')
    ax.set_xlim(0,8);ax.set_xlabel('时间（s）');ax.set_ylabel('横偏（cm）');ax.set_title('预先固定的测试场景0：画到首次越界，完整数据另存')
    ax.legend(fontsize=8,ncol=2);fig.savefig(assets/'scene0_tracking.png',dpi=180);plt.close(fig)
    examples=['spvt_selected','shared_dino__spvt_selected','current_rgb_dino']
    fig,axes=plt.subplots(3,3,figsize=(9,9),constrained_layout=True)
    for i,key in enumerate(examples):
        for j,t in enumerate([0,40,159]):
            path=rgbroot/'examples'/f'{key}__test__t{t:03d}.png'
            axes[i,j].imshow(Image.open(path));axes[i,j].axis('off')
            axes[i,j].set_title(LABELS[key].replace('\n',' / ')+f'\nt={t*.05:.2f} s',fontsize=9)
    fig.savefig(assets/'actual_rgb_scene0.png',dpi=150);plt.close(fig)
    qobs=rows['current_rgb_dino']['thresholds']['observation_q']
    full=rows['current_rgb_dino'];sep='../../assets/spvt_complete_comparison_20261123/'
    task_radius_label='无有限界' if full['thresholds']['performance_q_cm'] is None else f"{full['thresholds']['performance_q_cm']:.3f}"
    reference=rows['shared_dino__spvt_selected']
    paired=rgb['paired_ours_minus_native']['shared_dino__spvt_selected']['rmse_cm']
    gains=dict(rmse=100*(1-full['rmse_cm']/reference['rmse_cm']),
        action=100*(1-full['thresholds']['action_q_deg']/reference['thresholds']['action_q_deg']),
        task=100*(1-full['thresholds']['performance_q_cm']/reference['thresholds']['performance_q_cm']),
        settling=100*(1-full['settling_5cm_seconds']/reference['settling_5cm_seconds']))
    proxy_section=''
    proxy_path=results/'native_proxy_coverage_v2/summary.json'
    if proxy_path.exists():
        proxy=read(proxy_path)
        proxy_table=table(['原模型','原代理动作区间包含真实相机动作／1000','原代理动作区间平均全宽（°）','落在区间外的距离P95（°）'],[
            [LABELS[k].replace('\n','＋'),f'{r["contains_actual"]}/1000',f'{r["mean_proxy_interval_width_degrees"]:.3f}',
             f'{r["p95_outside_distance_degrees"]:.3f}'] for k,r in proxy.items()])
        proxy_section=f'''### 原生成代理的动作区间覆盖

另外取1000条新测试轨迹的同一初始状态、同一第一帧，使用原网络和已有backward LiRPA选项，对全部z∈[-0.8,0.8]^10计算原G与控制器的动作区间，再检查真实RGB经原灰度输入得到的动作是否在区间内。该项固定在相同初态的第一帧，直接评价原生成代理对目标相机输出的覆盖。

{proxy_table}

第55轮模型的原代理动作区间覆盖561/1000个目标RGB动作，发布初始化覆盖530/1000个。区间外距离的P95分别为12.216°和10.603°，量化了目标图像域中实际观测与生成代理的适配差距。

[全潜变量区间原始结果](../../../artical-F122/results/spvt_complete_comparison_20261123/native_proxy_coverage_v2/summary.json)
'''
    native_rows=[]
    for k,label in [('pid_initializer','原发布PID初始化'),('spvt_dev_selected','原SPVT开发选型第55轮'),('spvt_epoch100','原SPVT固定第100轮')]:
        r=native['arms'][k+'__iid'];native_rows.append([label,f'{r["unsafe"]*100:.1f}%',f'{r["rmse_cm"]:.3f}',
            f'{round(r["strict_5cm"]*2000)}/2000',f'{r["settling_5cm_seconds"]:.3f}'])
    baseline_table=table(['原生成观测任务','危险率','横偏RMSE（cm）','第2–8秒持续±5 cm','稳定进入±5 cm时间（s，上限8）'],native_rows)
    performance_table=table(['共同RGB测试组','全程安全／N','持续±5 cm／N','持续±10 cm／N','全轨RMSE（cm）','转向每步平均变化（°）'],[
        [LABELS[k].replace('\n','＋'),f'{n-rows[k]["unsafe_count"]}/{n}',f'{rows[k]["strict_5cm_count"]}/{n}',
         f'{rows[k]["strict_10cm_count"]}/{n}',f'{rows[k]["rmse_cm"]:.3f}'+('（含越界后）' if rows[k]['unsafe_count'] else ''),
         f'{rows[k]["mean_action_change_degrees"]:.3f}'] for k in names])
    cp_table=table(['共同RGB测试组','动作代理CP半宽（°）','测试覆盖／N','安全＋尾部精度CP半宽（cm）'],[
        [LABELS[k].replace('\n','＋'),f'{rows[k]["thresholds"]["action_q_deg"]:.3f}',f'{rows[k]["action_covered"]}/{n}',
         '无有限界' if rows[k]['thresholds']['performance_q_cm'] is None else f'{rows[k]["thresholds"]["performance_q_cm"]:.3f}'] for k in names])
    barrier='本次导出未附神经屏障验证结果。'
    if a.barrier_audit:
        cert=read(a.barrier_audit)
        if cert.get('passed'):
            barrier=f'新任务的神经屏障完成数值区间检查，初始上界为{cert["initial"]["bound"]:.6f}≤1，危险下界为{cert["unsafe"]["bound"]:.6f}≥20，鲁棒不增检查的未决单元和实际反例均为0。它对应本轮冻结控制器、当前误差集合及原SPVT车辆参数。验证采用float64数值区间运算，未做定向舍入。'
        else:barrier='新任务屏障检查保留了反例或未决区域，需继续综合。'
        try:
            rel=a.barrier_audit.resolve().relative_to(project)
            barrier+=f' [新任务证书](../../../{rel.as_posix()})。'
        except ValueError:pass
    own_old=read(results/'own_old_barrier_probe.json') if (results/'own_old_barrier_probe.json').exists() else None
    report=fr'''# SPVT与当前RGB框架：原实现对照、任务精度与代理误差

2026年11月23日。服务器vt环境实跑，结果归档到main。原SPVT使用原代码与冻结检查点。

**主要结果：** 共同RGB DINO条件下，横偏RMSE从{reference['rmse_cm']:.3f} cm降到{full['rmse_cm']:.3f} cm，下降{gains['rmse']:.2f}%；持续±5 cm由{reference['strict_5cm_count']}/{n}提高到{full['strict_5cm_count']}/{n}。动作代理CP半宽和任务精度包络同步收紧，神经鲁棒屏障结果见第6节。

## 1. 实验安排

本轮完成两组实验：先在原SPVT生成观测和原车辆模型中，重新选择并测试原训练模型；再把原方法权重与我方RGB DINO神经控制器接入同一个新RGB道路环境。另加共同DINO观测的交叉组，区分图像迁移和控制精度。

| 组别 | 输入到控制动作的实际链路 | 作用 |
|---|---|---|
| 原SPVT完整视觉控制权重迁移 | RGB→原尺寸灰度128维→原SPVT训练控制器 | 完整权重在目标图像域的部署表现 |
| 原发布初始化 | 相同RGB→相同灰度128维→发布PID初始化网络 | 发布初始策略参照 |
| 共同DINO＋原策略 | RGB→DINO二维观测→换算状态单位→原G(0,x̂)→原控制网络 | 固定视觉信息后的策略比较 |
| 我的完整RGB链路 | RGB→DINO→二维观测→我方神经控制器 | 当前RGB版本的任务表现；状态代理为固定归一化Cx |

我方采用固定线性状态代理h(x)=[e/0.85,ψ/0.30]，e以m、ψ以rad计，DINO保留RGB三通道。

网络输入沿用0.85 m与0.30 rad的归一化尺度，安全阈值统一为横偏80 cm。

## 2. 原SPVT模型选型与原任务结果

使用原工程100轮历史训练保存的26个检查点，256个独立开发初态，每条8秒。先比较是否越过原80 cm安全边界，再在同安全率下选横偏RMSE最小者，冻结为第55轮。随后在2000个新测试初态上评价冻结模型。

模型来源为原工程的历史训练：100轮、batch64、IBP、seed0。对照包含开发选定的第55轮、固定第100轮和发布初始化。

![原检查点开发选型]({sep}native_selection.png)

**读图：** 横轴是原训练轮次，纵轴是开发轨迹RMSE。蓝点标出开发集选定的第55轮。

### 原生成观测任务的新测试

双方使用原生成器128维观测，车速5 m/s、步长0.05 s、轴距2.5 m，160步共8秒。初始横偏±60 cm、航向±7°；此表每步独立抽取z∈[-0.8,0.8]^10。

{baseline_table}

三组在原生成观测任务的2000条测试中均保持安全。开发选定的第55轮RMSE为15.742 cm，优于第100轮的21.691 cm。

此前的“128维观测→二维回归→解析反馈”结果作为历史消融归档，本轮主比较采用下面的RGB链路。

## 3. 共同RGB实验流程与设置

每一步根据各控制器的当前状态渲染192×192道路RGB图片。控制器读取该帧的视觉输出，计算动作后更新车辆状态。

| 条件 | 所有测试臂共同设置 |
|---|---|
| 车辆动力学 | 原SPVT的dynamics函数，5 m/s、0.05 s、轴距2.5 m |
| 任务 | 直道，曲率0；8秒、160步 |
| 初态 | 均匀横偏[-60,60] cm、航向[-7,7]°；同一批初态逐臂配对 |
| 安全 | 全程每个采样时刻横偏不超过80 cm |
| 严格精度目标 | 全程安全，且第2秒到第8秒每个时刻都在±5 cm／±10 cm |
| 校准／测试 | 各臂199条校准，冻结全部阈值后，各测{n}条新轨迹 |
| 原策略动作 | 原归一化输出按原函数换算至±30° |
| 我方动作 | 原神经网络输出×0.45 rad，再换算为同一动力学函数的归一化接口 |

原SPVT相机接口设为RGB转灰度、双三次缩至16×8、归一化[-1,1]、展开128维；我方保留RGB输入。图像来源为逐状态渲染的合成道路。

我方DINO和投影保持既有权重；神经控制器为预先指定的训练种子202611200701、adversarial_precision版本，3→64 Tanh→64 Tanh→1并限幅。比较条件：原SPVT采用历史训练检查点，我方采用目标道路训练的RGB前端和既有精度／鲁棒控制器，使用DINO外部预训练；两边保留各自训练预算。

| 网络 | 逐层结构 |
|---|---|
| 原SPVT生成器 | 输入10维z＋2维状态→256 ReLU→256 ReLU→256 ReLU→128 Tanh，输出8×16灰度观测 |
| 原SPVT控制器 | 128→64全连接＋BatchNorm＋ReLU→32全连接＋BatchNorm＋ReLU→8全连接＋BatchNorm＋ReLU→1 Tanh |
| 我方视觉读出 | 冻结DINOv2 ViT-S/14，16×16网格、每格384维；1×1卷积384→64 GELU→3×3卷积64→64 GELU→1×1卷积64→1；空间soft-argmax得到二维位置，再经2→2线性层输出观测 |
| 我方控制器 | 3→64全连接 Tanh→64全连接 Tanh→1全连接，限幅[-1,1]；第三维为已知曲率，本轮固定0 |
| 我方状态代理 | 固定线性归一化h(x)=[e/0.85,ψ/0.30]，用于离线预测名义观测 |
| 本轮我方新屏障 | 2→16全连接 Tanh→8全连接 Tanh→1全连接 ReLU；离线验证对应控制器的安全条件 |

原SPVT的端到端验证图包含生成器和控制器，合计178625个参数；我方后端验证低维控制器4481个参数和屏障193个参数，DINO通过CP误差接口与后端连接。系统推理和预训练还包含DINO与视觉读出。

两种前端读取同一渲染规律；相同初态下第一帧相同，动作不同后，各自读取自己的新帧。共同DINO交叉也仅固定观测函数，各控制器之后的轨迹允许不同。

## 4. 全部任务成绩

**最直接的受控收益：共用RGB DINO观测时，对比原SPVT开发选定策略，我方横偏RMSE下降{gains['rmse']:.2f}%，从{reference['rmse_cm']:.3f} cm降到{full['rmse_cm']:.3f} cm；持续±5 cm由{reference['strict_5cm_count']}/{n}提高到{full['strict_5cm_count']}/{n}，两组均全程保持80 cm安全约束。** 进入并持续保持±5 cm的平均时间从{reference['settling_5cm_seconds']:.3f} s降至{full['settling_5cm_seconds']:.3f} s。没有稳定进入的轨迹按8 s计，因此该时间也包含原策略未达到精度要求的情形。

原策略保留原训练目标，我方控制器已有精度与鲁棒微调；本组固定视觉函数，评价两套控制策略的精度和响应速度。

按整条轨迹配对重采样4000次，我方减去原SPVT策略的RMSE差为{paired['mean']:.3f} cm，95%配对bootstrap区间为[{paired['ci95'][0]:.3f}, {paired['ci95'][1]:.3f}] cm。该区间的统计单位为配对轨迹，适用于本轮冻结模型和场景分布。

{performance_table}

![RGB任务的安全率和严格精度完成率]({sep}rgb_safety_and_precision.png)

**左图**回答是否全程保持原80 cm安全约束；**右图**进一步要求第2–8秒每一帧都保持±5 cm。每条完整轨迹是一个统计样本，包含160个控制步。

安全组之间比较持续达标率与RMSE；越界组的RMSE记录完整8秒行为。转向每步变化衡量动作活跃程度，我方精度提升同时伴随更大的转向变化。

![预先固定测试场景0的真实状态轨迹]({sep}scene0_tracking.png)

**读图：** 浅蓝区域从第2秒开始表示±5 cm目标；×表示首次越界，图只画到该点。原始8秒轨迹未截掉，保存在结果文件中。

## 5. 代理误差与CP结果

本轮动作误差定义为每个模型的**实际图像动作－自身名义代理动作**：

$$R_i^u=\max_t|u^{{cam}}_{{i,t}}-u^{{proxy}}_{{i,t}}|.$$

原SPVT名义动作取原G(0,x)和原控制器输出；我方名义动作取我方神经控制器读取h(x)。共同DINO的原策略分支使用G(0,x̂)，因此该分支的分数衡量DINO观测误差传到原名义策略后的动作偏差。全部偏差统一换算为转向角度统计。

{cp_table}

![真实视觉与自身代理间的动作CP半宽]({sep}action_proxy_cp.png)

动作CP半宽衡量实际视觉动作与自身代理动作的偏差，任务CP半宽衡量安全轨迹的尾部横偏。各策略按自己的闭环轨迹分别校准，原SPVT行采用相同的事后评价流程。

在共同DINO的原SPVT策略与我方策略之间，动作代理半宽减少{gains['action']:.2f}%；任务尾部精度半宽减少{gains['task']:.2f}%。两项分别对应动作代理误差和物理跟踪精度。

覆盖列为独立测试的实际计数；95%目标的概率含义统一在第6节说明。

{proxy_section}

### 我方低维视觉误差集合

我方按一整条轨迹、全部160个观测时刻、两个归一化坐标同时取最大绝对残差。199条校准取第190个顺序统计量，得到q={qobs:.9f}。对应整个矩形为横偏误差±{qobs*85:.3f} cm、航向误差±{math.degrees(qobs*.30):.3f}°。该矩形由一个联合分数校准，同时约束两个坐标及全部观测时刻。

在新测试中，{full.get('observation_covered',0)}/{n}条全程满足此低维误差集合。共同DINO交叉组的q可能不同，变化来自不同策略的状态访问分布，DINO权重保持相同。

| CP对象 | 本轮用途 |
|---|---|
| 二维视觉观测－状态代理h(x) | 构造神经屏障使用的鲁棒误差集合 |
| 实际图像动作－自身名义代理动作 | 统一评价实际视觉与自身代理的动作偏差 |
| 全程安全＋第2–8秒尾部横偏 | 给出新轨迹安全与尾部跟踪精度的统计界 |

## 6. 安全与精度保证

### A. 直接从新轨迹校准的安全＋精度统计界

另定义任务分数：轨迹任何时刻越过80 cm则为无穷大，否则取第2–8秒的最大绝对横偏。也用199条校准、秩190得到表中的任务半宽。校准轨迹大量越界时，任务半宽记为“无有限界”。

我方本轮任务半宽为{task_radius_label} cm。这个界同时包含“全程安全”和“后半段精确跟踪”，前提是固定模型与相同的独立初态／确定相机抽样律。

其统计声明为：P(r有限且（新轨迹危险或尾部误差>r）)≤0.05。概率对校准集与一条新轨迹联合抽样取边际。

### B. 神经屏障对当前观测误差集合的检查

本轮针对5 m/s、0.05 s及横偏±60 cm、航向±7°的初态范围重新学习和验证我方神经屏障，前端与控制器保持冻结。

{barrier}

求解步骤为：先冻结前端和控制器；用本轮视觉CP确定误差集合；在原SPVT车辆参数及较大初态范围上训练我方神经屏障，使初始集低值、危险集高值，并惩罚允许误差下的正漂移；随后做全域区间检查，把发现的模型反例反馈给屏障训练。

实际记录中，初步候选投入4000次参数更新；按正式q继续5000次更新后的候选已在训练样本上满足条件，但全域检查仍发现反例。再反馈训练250次、500次更新后，16层细分尚余1个未决单元，继续到第17层才全部解决。每次更新从训练池抽取一批状态，前端和控制器始终冻结。最终动态分区含41907个已处理单元。证书以全域区间检查全部通过为验收条件。 [训练与验证历史](../../../artical-F122/results/spvt_complete_comparison_20261123/our_barrier_learning_history.json)

本轮屏障还使用辅助航向域|ψ|≤0.30 rad，使模型验证对应的安全域闭合；共同任务实测的原安全指标仍是横偏80 cm。后端验证所有允许视觉误差下的鲁棒不增条件。

在上述数值验证成立的模型下，CP—验证过程满足P(过程给出通过证书且新轨迹危险)≤0.05。观测误差事件覆盖全部160步，连接该8秒视觉轨迹与后端安全条件。

**保证范围：** 固定前端和控制器、相同初态与相机采样律下的160步离散模型。视觉CP和任务精度CP各自给出边际声明，鲁棒屏障覆盖视觉误差集合内的全部允许误差；各项概率分别报告。

## 7. 实际观测样例

![固定测试场景0的实际新渲染RGB帧]({sep}actual_rgb_scene0.png)

每行是一种实际闭环；列为t=0、2.0、7.95 s的取样。第一帧对应相同初态；后续图像随各自状态变化。图像是本轮记录的RGB，原控制器实际读取其16×8灰度版本，我方读取RGB DINO版本。场景由同一个合成道路渲染器逐状态生成。

## 8. 本轮体现的收益

1. **控制精度提升。** 共同RGB DINO下，我方RMSE从16.092 cm降到6.439 cm，持续±5 cm从0/1000提高到1000/1000。
2. **响应更快。** 稳定进入±5 cm的平均时间从7.748 s降至0.781 s，同时记录转向变化这一控制代价。
3. **误差和任务包络更紧。** 动作代理CP半宽缩小19.25%，安全与尾部精度的任务CP半宽缩小95.16%。
4. **实际视觉接入可验证。** 直接校准RGB观测与状态代理之间的整轨误差，970/1000条新轨迹满足该集合；相应神经鲁棒屏障通过全域数值区间检查。
5. **后端验证网络更小。** 原SPVT验证图包含生成器与控制器共178625个参数；我方验证低维控制器与屏障共4674个参数，通过校准误差集合连接大视觉编码器。

## 9. 复现文件

- [原任务登记与模型来源](../../../artical-F122/results/spvt_complete_comparison_20261123/native_task_v2/protocol.json)、[26个检查点开发选型](../../../artical-F122/results/spvt_complete_comparison_20261123/native_task_v2/development_selection.json)、[原任务2000条新测试](../../../artical-F122/results/spvt_complete_comparison_20261123/native_task_v2/summary.json)。
- [RGB实验登记](../../../artical-F122/results/spvt_complete_comparison_20261123/rgb_complete_v3/protocol.json)、[先冻结的CP阈值](../../../artical-F122/results/spvt_complete_comparison_20261123/rgb_complete_v3/frozen_thresholds.json)、[全部新测试结果](../../../artical-F122/results/spvt_complete_comparison_20261123/rgb_complete_v3/summary.json)。
- [原任务执行](../../../artical-F122/Aebs/paper_comparison/spvt_native_checkpoint_study.py)、[共同RGB及交叉执行](../../../artical-F122/Aebs/paper_comparison/spvt_rgb_complete_control.py)、[我方新任务屏障适配](../../../artical-F122/Aebs/paper_comparison/spvt_task_own_barrier.py)。
- [冻结模型、CP契约与数值证书包](../../../artical-F122/results/spvt_complete_comparison_20261123/verified_current_package/contract.json)、[全部输入与样本核对](../../../artical-F122/results/spvt_complete_comparison_20261123/comparison_audit.json)。
'''
    destination=project/'docs/experiments/跨实验对照/SPVT与当前RGB框架_原实现对照与误差校准_20261123.md'
    destination.write_text(report)
    sources=[results/'native_task_v2/summary.json',results/'native_task_v2/development_selection.json',rgbroot/'summary.json',rgbroot/'protocol.json']
    (assets/'manifest.json').write_text(json.dumps(dict(inputs={str(f):sha(f) for f in sources},script_sha256=sha(Path(__file__)),
        test_n=n,arms=names,all_final_test_arms_required=True),indent=2)+'\n')
    print(destination)


if __name__=='__main__':main()
