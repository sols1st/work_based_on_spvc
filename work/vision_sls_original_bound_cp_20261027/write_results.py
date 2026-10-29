"""Add this comparison to the existing two reports, keeping old results distinct."""
from pathlib import Path
import json
import hashlib
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt,font_manager

ROOT=Path('/Users/johann/project/work_based_on_spvc')
WORK=Path(__file__).resolve().parent
RESULT=WORK/'原界与原界加CP_完整对照.json'
d=json.loads(RESULT.read_text());scenes=d['scenes']
GUIDE=ROOT/'docs/current/VISION-SLS_CP未见随机样本覆盖_通俗详解.md'
MAIN=ROOT/'docs/current/VISION-SLS_CP完整说明与可行性分析.md'
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])
def pct(x):return f'{100*x:.2f}%'
def count(m):return f"{m['trajectories_covered']}/{m['trajectories']}（{pct(m['trajectory_coverage'])}）"
def points(m):return f"{m['points_covered']}/{m['points']}（{pct(m['point_coverage'])}）"
names={'parking':'停车场','quad':'无人机'}
primary=[];secondary=[];stress=[];width=[];context=[];cal=[];entries=[]
for scene,s in scenes.items():
    box=s['definitions']['box'];ball=s['definitions']['l2'];r=box['test']['random'];r2=ball['test']['random']
    primary.append([names[scene],f"{box['q']:.9f}",count(r['before']),count(r['after']),points(r['before']),points(r['after'])])
    secondary.append([names[scene],f"{ball['q']:.9f}",count(r2['before']),count(r2['after']),points(r2['before']),points(r2['after'])])
    width.append([names[scene],f"{r['mean_original_component_width']:.6f}",f"{r['mean_calibrated_component_width']:.6f}",f"{box['q']:.2f} 倍"])
    cal.append([names[scene],box['calibration_before']['trajectories'],box['calibration_rank'],count(box['calibration_before']),count(box['calibration_after'])])
    for definition,label in [('box','每分量盒'),('l2','原实现二范数')]:
        a=s['definitions'][definition]
        stress.append([names[scene],label,count(a['test']['stress']['before']),count(a['test']['stress']['after']),count(a['test']['mixed']['before']),count(a['test']['mixed']['after'])])
    h=s['previous_new_fitted_scale_context_box']['random']
    context.append([names[scene],count(r['after']),f"{r['mean_calibrated_component_width']:.6f}",count(h['after']),f"{h['mean_calibrated_h']:.6f}"])
for id,scene,definition,label in [('O01','parking','box','作者原界＋CP，分量盒'),('O02','quad','box','作者原界＋CP，分量盒'),('O03','parking','l2','作者原界＋CP，原实现二范数'),('O04','quad','l2','作者原界＋CP，原实现二范数')]:
    a=scenes[scene]['definitions'][definition];g=a['test']['random']
    entries.append([id,names[scene],label,'既有数据统计对照',f"q={a['q']:.6f}；随机轨迹 {g['before']['trajectories_covered']}/{g['before']['trajectories']} → {g['after']['trajectories_covered']}/{g['after']['trajectories']}；未重规划"])

fonts=[f for f in font_manager.fontManager.ttflist if Path(f.fname).is_file()]
font=next(f for name in ['PingFang HK','Arial Unicode MS','Hiragino Sans GB'] for f in fonts if f.name==name)
plt.rcParams.update({'font.family':[font.name,'DejaVu Sans'],'font.size':11,'axes.unicode_minus':False,
                    'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(11,7.4),layout='constrained')
for col,scene in enumerate(['parking','quad']):
    a=scenes[scene]['definitions']['box'];r=a['test']['random'];ax=axes[0,col]
    for j,phase in enumerate(['before','after']):
        m=r[phase];value=100*m['trajectory_coverage']
        ax.bar(j,value,width=.58,color=['#8b9eab','#217f85'][j])
        ax.text(j,value+2,f"{m['trajectories_covered']}/{m['trajectories']}\n{value:.2f}%",ha='center',va='bottom')
    ax.axhline(95,c='#b64d42',ls='--',label='预设目标95%')
    ax.set(xticks=[0,1],xticklabels=['作者原界','原界 × CP倍率'],ylim=(0,118),ylabel='随机测试的整轨迹覆盖率（%）',title=names[scene]+'：按每个分量是否超界判断')
    ax.legend(loc='upper left',fontsize=9);ax.grid(axis='y',alpha=.12);ax.set_axisbelow(True)
    ax=axes[1,col];values=[r['mean_original_component_width'],r['mean_calibrated_component_width']]
    ax.bar([0,1],values,width=.58,color=['#8b9eab','#217f85'])
    for j,value in enumerate(values):ax.text(j,value+max(values)*.04,f'{value:.4f}',ha='center')
    ax.set(xticks=[0,1],xticklabels=['作者原界','原界 × CP倍率'],ylim=(0,max(values)*1.27),ylabel='平均允许误差半宽（观测空间单位）',title=f"同一状态的每个半宽扩大到 {a['q']:.2f} 倍")
    ax.grid(axis='y',alpha=.12);ax.set_axisbelow(True)
fig.suptitle('作者原参数不变，只用独立校准数据算一个 CP 倍率',fontsize=15)
fig.supxlabel('同一批图像轨迹的回顾对照；图中比例不是未来覆盖保证，也不是新的控制运行成绩。',fontsize=10,color='#555')
FIG=ROOT/'docs/assets/vision_sls_cp/original_bound_cp_comparison.png'
fig.savefig(FIG,dpi=180,facecolor='white');plt.close(fig)

section=rf'''<a id="original-bound-cp"></a>
### 6.6 补做的公平对照：作者原界 → 作者原界加入 CP（2026 年 10 月 27 日）

这次直接回答：**保持作者已经保存的误差函数不变，只在外面乘上 CP 算出的倍率，会改善多少覆盖？**

#### 6.6.1 固定了哪些东西，改了什么

| 项目 | 本次处理 |
|---|---|
| 作者的原多项式系数、常数底线 | 保持原样，不重新拟合 |
| 原界随状态变化的形状 | 保持原样 |
| 无人机各分量不同的半宽 | 原样保留，五个分量统一乘同一倍率 |
| 视觉网络、控制器、轨迹、实际误差 | 全部固定，使用之前同一批记录 |
| 校准数据 | 停车场 40 条、无人机 60 条，仍用原来的随机校准批次 |
| 测试数据 | 同一批停车场 42 条随机＋8 条压力，无人机 30 条随机＋20 条压力 |
| 唯一调整 | 分别按既定判定规则，从校准数据排序算出 q，再将原界整体乘 q |

这里的“原界”确实来自作者保存参数。程序核对了参数文件与最初复现记录的文件指纹，并按原公式在全部状态上独立重算，与轨迹中保存的原界输出比较；最大差异小于 3×10⁻¹⁶。图像对应状态的延迟和停车场坐标平移也重新核对了。

因此，本次不是重新造一个较弱的初步函数作为比较对象。

#### 6.6.2 如何对原界算 CP 倍率

停车场的原界给三个观测分量相同的半宽；无人机给五个分量各自的半宽。每个分量先计算：

$$\text{{这个分量需要的倍率}}=\frac{{\text{{该分量误差的绝对值}}}}{{\text{{该分量的作者原半宽}}}}.$$

同一次观测取各分量中最大的倍率，再对一条完整运行中的全部观测取最大。接下来与前面完全相同：对校准运行分数排序，停车场取第 39 小，无人机取第 58 小。

最后，每个原半宽都乘上同一个 q。原界的空间形状及各分量之间的比例都保留了。

#### 6.6.3 主结果：前后都按“每个分量不超界”判断

{table(['场景','从校准数据算出的 q','原界完整覆盖','原界＋CP 完整覆盖','原界逐次观测覆盖','原界＋CP 逐次观测覆盖'],primary)}

**停车场在这批随机测试中，从 0/42 提高到 42/42。无人机从 0/30 提高到 28/30，即 93.33%，仍未达到本批实测至少 95% 的目标。** 两个场景都显著提高了覆盖，但不能把它们合并写成“均已通过 95% 验证”。

校准集本身的结果也单独保存：

{table(['场景','校准运行数','排序位置','原界完整覆盖','原界＋CP 完整覆盖'],cal)}

这里校准成绩来自参与算倍率的数据，不代替上面的测试成绩。

#### 6.6.4 提高覆盖的代价：允许误差的范围明显扩大

{table(['场景','随机测试上原平均半宽','乘 CP 后平均半宽','每个状态下的统一扩大倍数'],width)}

半宽表示一个分量允许的误差绝对值；把所有随机测试观测分量的半宽求平均，得到表中的平均值。它们属于网络观测空间，不能直接读成米。停车场的放大幅度很大，说明作者原界在本轮图像数据上不足以容纳大量真实误差；CP 通过明显扩大范围提高覆盖，并没有使原始视觉误差变小。

![作者原界加入 CP 的覆盖改善与范围扩大]({FIG})

因此评价这次增益时，要同时看“多覆盖了多少”和“范围扩大了多少”。本次没有用新界重新规划控制器，尚未检验如此扩大的范围是否仍让控制任务容易求解。

#### 6.6.5 无人机剩下两条为什么没覆盖

按随机测试子集的运行编号，第 18、19 条分别需要约 3.065544、3.048540 倍；从校准数据固定得到的倍率为 3.042165。所以两条各有一个观测超出最终范围，最终为 28/30。

它们相对最终范围分别超出约 0.77%、0.21%。超出不大，但仍应记为未完整覆盖；没有根据测试结果再把倍率改大。

93.33% 是这批有限测试的结果。它不能直接证明普通 CP 的总体排序结论错误，也不能被忽略而声称本批已达到 95%。

#### 6.6.6 原代码的另一种判定也补做，单独报告

前面的主表与通俗讲解一致，要求每个误差分量都在范围内。原控制代码还采用另一种判定：把各分量误差除以自己的半宽后，平方、相加、开平方，检查总长度是否不超过 1。

我们按这个二范数规则另外计算了 CP 倍率，前后始终保持该规则不变：

{table(['场景','该规则算出的 q','原界完整覆盖','原界＋CP 完整覆盖','原界逐次观测覆盖','原界＋CP 逐次观测覆盖'],secondary)}

这套判定下，两场景本批随机测试都变为完整覆盖。它与主表的集合形状、倍率不同，不能拿其中更高的结果替代主表。两套结果都保留。

#### 6.6.7 压力样本与混合结果也全部保留

{table(['场景','判定规则','原压力覆盖','加 CP 后压力覆盖','原混合覆盖','加 CP 后混合覆盖'],stress)}

压力样本不是按随机校准机制生成的，这些比例用于额外诊断，不混作同机制随机测试的理论覆盖结论。

#### 6.6.8 与前面的“新拟合 h＋CP”放在同一批数据上看

下表只使用相同的每分量判定，比较各自完成 CP 后的结果：

{table(['场景','作者原界＋CP 完整覆盖','作者原界＋CP 平均半宽','新拟合 h＋CP 完整覆盖','新拟合 h＋CP 平均半宽'],context)}

停车场作者原界＋CP 多覆盖一条随机运行，但平均范围更大；无人机新拟合 h＋CP 在这批数据中多覆盖一条，平均半宽也更小。平均半宽只是总体规模指标，不证明某个界在每个状态、每个分量上都比另一个小。

所以，本次对照支持“给作者原界加 CP 也能明显改善该固定控制器数据上的覆盖”，没有显示它在覆盖与范围大小上统一优于重新拟合的方案。

#### 6.6.9 本次结果的适用范围与证据

这是对已有图像数据的回顾分析。校准与测试沿用原先分开的批次，倍率只由校准数据决定，没有按照测试结果选择倍率；但测试记录之前已经看过，因此本次不是新采集的盲测。

没有新增图像轨迹或新控制器运行。原有 990 条轨迹、32800 次观测不因本次重复统计而增加；这次结果也不能当作“换界后新控制器已安全”的结论。

协议、完整次数与源文件指纹见 [原界 CP 对照协议](<{WORK/'protocol.json'}>)、[完整对照结果](<{RESULT}>)。每条校准运行的分数和测试逐步比值见 [原始计算数组](<{WORK/'校准分数与测试逐步比值.npz'}>)，复算程序见 [compare_original_cp.py](<{WORK/'compare_original_cp.py'}>)。

'''

guide=GUIDE.read_text()
assert '<a id="original-bound-cp">' not in guide
guide=guide.replace('这张表回答的是“本轮新拟合函数 h，加入 CP 后改善多少”。若要评价“CP 相对作者已经保存的原误差界改善多少”，需要另做同一批数据、同一误差定义下的“作者原界→作者原界加 CP”对照，不能用这张表替代。',
 '上面的表回答的是“本轮新拟合函数 h，加入 CP 后改善多少”。下面第 6.6 节已经补做同一批数据、同一种判定规则下的“作者原界→作者原界加 CP”对照，两组结果分开列出。')
marker='<a id="misses"></a>'
assert marker in guide;guide=guide.replace(marker,section+'\n'+marker,1)
note='''新增对照（2026 年 10 月 27 日）：第 6.6 节已直接保留作者原误差参数，只加 CP。按每个分量是否超界判断，随机测试全程覆盖为停车场 **0/42→42/42**、无人机 **0/30→28/30（93.33%）**；另列原代码二范数规则。它与此前“新拟合 h→h＋CP”的成绩是两组对照。\n\n'''
guide=guide.replace('阅读顺序可以按表中四个问题来：',note+'阅读顺序可以按表中四个问题来：',1)
GUIDE.write_text(guide)

summary=f'''<a id="original-envelope-cp-results"></a>
### A.7 2026 年 10 月 27 日新增：作者原界保持不变，再加 CP

本次直接用作者已保存参数计算原界，在与文档版相同的 40/60 条随机校准和 50/50 条参考控制器测试记录上作对照。只用校准运行确定倍率，未拟合新的尺度、未重规划。O01—O04 为四项统计对照，连同此前 22 项，现行报告共收录 26 个汇总条目；独立图像批次和轨迹数不增加。

{table(['编号','场景','内容','阶段','结果'],entries)}

按分量盒判断：

{table(['场景','q','原界整轨迹覆盖','原界＋CP 整轨迹覆盖','原逐点覆盖','加 CP 后逐点覆盖'],primary)}

原代码二范数规则是另一个独立配对：

{table(['场景','q','原界整轨迹覆盖','原界＋CP 整轨迹覆盖','原逐点覆盖','加 CP 后逐点覆盖'],secondary)}

盒规则下平均分量半宽为：

{table(['场景','原平均半宽','加 CP 后平均半宽','统一扩大倍数'],width)}

主要结论是覆盖明显改善，但作者原界需要明显放大；盒规则下无人机随机测试仍只有 93.33%，未达到本批实测 95% 目标。二范数结果不能替代盒结果。均为既有测试记录的回顾对照，未新增盲测或证明新控制器安全。

作者参数与初次复现记录的哈希核对一致；原函数输出逐状态重算最大差异小于 3e−16；延迟、坐标、残差、原界形状及配对统计已核对。完整方法、全部压力样本、未覆盖轨迹和与新拟合尺度的比较见 [通俗详解第 6.6 节](<{GUIDE}#original-bound-cp>)，原始证据见 [完整对照 JSON](<{RESULT}>)、[预设协议](<{WORK/'protocol.json'}>)。

'''
main=MAIN.read_text();assert '<a id="original-envelope-cp-results">' not in main
main=main.replace('整理日期：2026 年 10 月 26 日。实验范围：截至本次整理已保存并核验的停车场、无人机仿真记录。本次整理没有新增控制实验。',
 '整理日期：2026 年 10 月 26 日；统计补充更新：2026 年 10 月 27 日。图像实验范围仍为此前已保存并核验的停车场、无人机仿真记录。10 月 27 日新增四项作者原界 CP 的既有数据对照，没有新增控制实验。',1)
top=f'新增结果（2026 年 10 月 27 日）：已补做“作者原界→作者原界加 CP”的四项配对统计，见 [附录 A.7](#original-envelope-cp-results)。按分量盒判断，随机轨迹覆盖为停车场 0/42→42/42、无人机 0/30→28/30；同时报告二范数结果与范围扩大代价。\n\n'
main=main.replace('## 阅读导航',top+'## 阅读导航',1)
main=main.replace('本文收录 22 个汇总条目，其中 E01—E04 展开为 12 个回顾参数点。',
 '截至 10 月 26 日的记录为 22 个汇总条目，其中 E01—E04 展开为 12 个回顾参数点。10 月 27 日增加 O01—O04 四项作者原界 CP 对照，合计 26 项，详见附录 A.7。',1)
main=main.replace('### A.1 22 个条目总览','### A.1 截至 10 月 26 日的 22 个条目总览',1)
anchor='<a id="appendix-b"></a>';assert anchor in main;main=main.replace(anchor,summary+'\n'+anchor,1)
MAIN.write_text(main)

index=ROOT/'docs/README.md';s=index.read_text()
s=s.replace('停车场和无人机全部 22 项尝试、12 个回顾参数点、19 个原始图像批次。',
 '停车场和无人机截至10月26日的22项尝试，以及10月27日新增的4项“作者原界＋CP”统计对照；12个回顾参数点、19个原始图像批次。')
index.write_text(s)
(WORK/'写入文档的原界对照段.md').write_text(section)
record={'date':'2026-10-27','source_results':str(RESULT),'source_results_sha256':hashlib.sha256(RESULT.read_bytes()).hexdigest(),
 'updated_reports':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [GUIDE,MAIN,index]],
 'figure':str(FIG),'figure_sha256':hashlib.sha256(FIG.read_bytes()).hexdigest(),'new_image_batches':0,'new_trajectories':0}
(WORK/'报告写入记录.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(record,ensure_ascii=False))
