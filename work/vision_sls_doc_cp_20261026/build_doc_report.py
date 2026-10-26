"""Write factual results for the advisor-document CP experiment."""
import json,shutil
from pathlib import Path

repo=Path('/Users/johann/project/work_based_on_spvc')
out=Path('/Users/johann/Documents/Codex/2026-10-22/jian/outputs/vision_sls_建议原方案CP验证_20261026')
data={s:json.loads((out/s/'doc_cp_comparison.json').read_text()) for s in ['parking','quad']}
def pct(x):return f'{x*100:.2f}%'
def control_table(scene):
    d=data[scene];base=d['methods']['baseline'];new=d['methods'].get('doc_cp')
    rows=[('任务成功数 / 50','author_successes'),('采样碰撞轨迹数 / 50','collision_rollouts'),
      ('实际状态、输入、碰撞联合违规轨迹数 / 50','full_state_input_collision_constraint_violations'),
      ('状态管束越界轨迹数 / 50','state_tube_violating_rollouts_tolerance_1e-6'),
      ('输入管束越界轨迹数 / 50','input_tube_violating_rollouts_tolerance_1e-6'),
      ('平均终点位置误差（米）','terminal_position_distance_mean_m'),
      ('最小采样障碍物净距（米）','minimum_sampled_obstacle_clearance_m'),
      ('文档 CP 盒逐时刻覆盖率','doc_box_point_coverage'),
      ('文档 CP 盒整轨迹覆盖数 / 50','doc_box_trajectories_covered'),
      ('文档 CP 盒随机子集整轨迹覆盖数','doc_box_random_trajectories_covered'),
      ('文档 CP 盒压力子集整轨迹覆盖数','doc_box_stress_trajectories_covered'),
      ('外包球整轨迹覆盖数 / 50','outer_ball_trajectories_covered'),
      ('反馈算子 Frobenius 范数','feedback_operator_frobenius_norm')]
    lines=['| 指标 | 原控制器数据 | 按文档计算界后重新规划 |','|---|---:|---:|']
    for label,key in rows:
        values=[]
        for m in [base,new]:
            val='未产生可测试控制器' if m is None else pct(m[key]) if 'point_coverage' in key else f'{m[key]:.6g}' if isinstance(m[key],float) else str(m[key])
            values.append(val)
        lines.append('| '+label+' | '+' | '.join(values)+' |')
    return '\n'.join(lines)
def scene_conclusion(scene):
    d=data[scene];m=d['methods'].get('doc_cp')
    if m is None:
        q=json.loads((out/scene/'qp_audit.json').read_text())
        last=q['records'][-1]
        return f"严格保持原规划器时，没有得到可测试的新控制器。最后一次局部 QP 状态为 `{last.get('solver_status')}`，最小 Hessian 特征值为 {last['min_eigenvalue']:.6g}。因此不能给出新控制器的成功率或碰撞率，更不能把它们写成零。该失败只定位当前误差模型及外包接口下的原规划流程，并不证明原非线性问题无解。"
    return f"已完成 50 条新控制器视觉闭环：任务成功 {m['author_successes']}/50，碰撞 {m['collision_rollouts']}/50，文档盒整轨迹覆盖 {m['doc_box_trajectories_covered']}/50，外包球整轨迹覆盖 {m['outer_ball_trajectories_covered']}/50。控制效果与误差覆盖需按下表分别判断。"

def failure_table():
    lines=['| 原规划器诊断 | 停车场 | 无人机 |','|---|---:|---:|']
    audits={s:json.loads((out/s/'qp_audit.json').read_text())['records'] for s in data}
    rows=[('局部 QP 调用数',*[str(len(audits[s])) for s in data]),
      ('已完成的外层收敛检查数',*[str(data[s]['plan_status']['convergence_checks']) for s in data]),
      ('最后一次 QP 返回状态',*[audits[s][-1].get('solver_status','未记录') for s in data]),
      ('最后一次 Hessian 最小特征值',*[f"{audits[s][-1]['min_eigenvalue']:.6g}" for s in data]),
      ('是否生成控制器',*['是' if data[s]['plan_status']['plan_exported'] else '否' for s in data])]
    for row in rows:lines.append('| '+' | '.join(row)+' |')
    return '\n'.join(lines)

p,q=[data[s]['model'] for s in ['parking','quad']]
new_trajectories=200+sum(50 for s in data if 'doc_cp' in data[s]['methods'])
followup=Path(__file__).parent/'numerical_followup_section.md'
if followup.exists():
    new_trajectories+=sum(50 for s in data for suffix in ['_convexified_iter50000_outer80','_convexified_iter50000_outer100_rf0p001','_convexified_iter100000_outer100_rf0p001_normalized']
        if (out/s/('doc_cp_comparison'+suffix+'.json')).exists() and 'doc_cp' in json.loads((out/s/('doc_cp_comparison'+suffix+'.json')).read_text())['methods'])
text=rf'''# 按导师建议计算 CP 误差界：原公式验证、控制效果与实现范围

**日期：2026 年 10 月 26 日。服务器：`root@10.11.154.191:20020`。全部为图像渲染与网络推理参与的闭环仿真。**

## 1. 这次真正验证了什么

为了回答“如果完全按照建议方案做，效果怎么样”，本轮将误差界计算改为建议文档的明确公式：**无穷范数残差、带松弛的凸包络拟合、归一化比值分数、乘法 CP 界**。不再使用上一版的二阶分位回归或加法偏移。

但“完全按照”仍需说明范围：建议文档定义了盒集合，没有提供完整的原生盒集合 SLS 求解器。本轮没有把盒当作同半径球，而是用严格包含盒子的球接入原控制器；这个接口可能增加保守性。因此准确名称为：**按建议公式计算的轨迹 CP 盒界＋外包球接入原 SLS 规划器**。它不是原生盒集合控制器的完整实现或验证。

停车场：{scene_conclusion('parking')}

无人机：{scene_conclusion('quad')}

在独立原策略随机测试子集上，文档盒界完整覆盖停车场 41/42（97.62%）、无人机 29/30（96.67%）；这是已有控制器数据上的覆盖表现，**不能当作重新规划后的新控制器效果**。

## 2. 与建议文档的对应关系

依据 [技术说明原文](/Users/johann/project/work_based_on_spvc/VISION_SLS_状态相关误差界计算.pdf) 第 2、3—5、7、8 页：

| 环节 | 本轮做法 | 对应或边界 |
|---|---|---|
| 固定视觉模型与 C | 保留原 DINOv2、投影网络和观测矩阵 | 对应建议的先固定模型 |
| 残差 | 原始向量取无穷范数 | 对应第 2 页 |
| 停车场包络 | 只使用二维位置，完整三阶，10 项 | 对应文档介绍的车辆函数族 |
| 无人机包络 | 五个位置／姿态特征，完整四阶，126 项 | 对应文档介绍的无人机函数族 |
| 输入归一化 | 训练均值／标准差标准化，标准差下限 0.05 | 文档建议归一化；具体下限为本轮设定 |
| 拟合 | 文档归一化目标、L2 系数正则、非负松弛、训练点非负约束 | 与第 7 页示例结构一致 |
| 超参数 | gamma=1e-4，mu=20 | 沿用文档示例，未通过本轮测试调参；文档没有声称它们普遍最优 |
| CP 尺度 | h0(x)=max(0.001,b0(x)) | 对应正尺度截断；0.001 为本轮设定 |
| CP 分数 | 每条轨迹的最大归一化无穷范数残差 | 保留比值结构，按文档末尾提示扩展到轨迹级 |
| CP 输出 | bCP(x)=q_alpha*h0(x) | 对应第 8 页乘法公式 |
| 目标 | alpha=0.05，整轨迹边际覆盖 | 本轮设定，不是原脚本安全验收阈值 |
| 误差集合 | 保留文档无穷范数盒用于覆盖核验 | 不把外包球覆盖冒充盒覆盖 |
| 控制接口 | 半径 sqrt(ny)*bCP(x) 的外包球 | 额外保守接口；不等同于原生盒 SLS |
| 控制求解 | 保持原求解器，不做 Hessian 凸化 | 避免混入上一版无人机数值适配 |

## 3. 具体计算链与输入输出

### 3.1 原始残差

输入状态—图像配对数据，用固定视觉网络得到低维观测，计算

$$\delta_{{i,t}}=y_{{i,t}}^{{\rm vis}}-Cx_{{i,t}}^{{\rm image}},\qquad r_{{i,t}}=\|\delta_{{i,t}}\|_\infty.$$

停车场使用原代码坐标平移 $C(x+[3,0,0,0]^\top)$；两场景均保留原图像延迟。每个误差必须和生成图像的状态对齐，不能直接与接收观测时的当前状态相减。

### 3.2 带松弛的包络拟合

用训练状态构造标准化后的完整多项式矩阵 M，求解

$$\min_{{\beta,\xi\ge0}}\ \frac1N\mathbf1^\top M\beta
+10^{{-4}}\|\beta\|_2^2+\frac{{20}}N\mathbf1^\top\xi,$$

$$M\beta+\xi\ge r,\qquad M\beta\ge0.$$

输出 $b_0(x)=m(x)^\top\beta$。由 CVXPY 与 OSQP 求解；另保存每个训练点的包络值和松弛量，并独立重算目标与约束。

| 训练结果 | 停车场 | 无人机 |
|---|---:|---:|
| 训练轨迹数 | {p['training_trajectories']} | {q['training_trajectories']} |
| 训练观测数 | {p['training_points']} | {q['training_points']} |
| 多项式项数 | {p['monomials']} | {q['monomials']} |
| 拟合状态 | {p['solver_status']} | {q['solver_status']} |
| 训练点未校准覆盖率 | {pct(p['training_point_coverage'])} | {pct(q['training_point_coverage'])} |
| 平均松弛 | {p['training_slack_mean']:.8f} | {q['training_slack_mean']:.8f} |
| 最大松弛 | {p['training_slack_max']:.8f} | {q['training_slack_max']:.8f} |

训练覆盖不是最终覆盖保证。训练点非负也不是全域非负；正尺度截断仅避免分母为零，不能证明误差界在域外足够大。

### 3.3 独立轨迹校准

冻结 b0，另外采集原控制器随机轨迹。令

$$h_0(x)=\max\{{0.001,b_0(x)\}},\qquad
A_i=\max_t\frac{{r_{{i,t}}}}{{h_0(x_{{i,t}}^{{\rm image}})}}.$$

排序后取

$$k=\lceil(n_{{\rm cal}}+1)(1-0.05)\rceil,\quad q_\alpha=A_{{(k)}},\quad
\boxed{{b_{{\rm CP}}(x)=q_\alpha h_0(x)}}.$$

若 k 超出样本数，代码实现文档的无穷大约定；本次 k 均在样本范围内。$q_\alpha$ 是倍率，不是上一版的加法偏移量。

| 校准结果 | 停车场 | 无人机 |
|---|---:|---:|
| 独立随机校准轨迹数 | {p['calibration_trajectories']} | {q['calibration_trajectories']} |
| 种子 | {p['calibration_seed']} | {q['calibration_seed']} |
| 排序索引 k | {p['calibration_rank']} | {q['calibration_rank']} |
| 乘法倍率 q | {p['cp_multiplier']:.9f} | {q['cp_multiplier']:.9f} |
| 校准集逐点覆盖 | {pct(p['calibration_point_coverage'])} | {pct(q['calibration_point_coverage'])} |
| 校准集整轨迹覆盖 | {pct(p['calibration_trajectory_coverage'])} | {pct(q['calibration_trajectory_coverage'])} |
| 校准状态上的盒半宽范围 | {p['calibration_box_bound_min']:.6f}—{p['calibration_box_bound_max']:.6f} | {q['calibration_box_bound_min']:.6f}—{q['calibration_box_bound_max']:.6f} |

校准覆盖由顺序统计量构造决定，不是独立测试的泛化表现。轨迹内相关性由时间最大值处理，不把所有帧当作独立校准样本。

### 3.4 盒到球的接口与其代价

建议集合为

$$\mathcal B_\infty(x)=\{{\delta:\|\delta\|_\infty\le b_{{\rm CP}}(x)\}}.$$

利用 $\|\delta\|_2\le\sqrt{{n_y}}\|\delta\|_\infty$，用

$$\mathcal B_\infty(x)\subseteq
\mathcal B_2\!\left(\sqrt{{n_y}}b_{{\rm CP}}(x)\right)$$

接入原单位二范数扰动接口。因此控制模型实际使用

$$F(x)=\sqrt{{n_y}}b_{{\rm CP}}(x)I,\qquad \|e\|_2\le1.$$

停车场因子为 sqrt(3)，无人机为 sqrt(5)。这项转换不会把盒内误差漏到控制集合外，但可能扩大管束、改变反馈或使原规划器更难求解。它既不修复 CP 的新策略分布问题，也不提供非线性安全证明。原生盒支持函数应使用相应的一范数，不能仅把 F 改成 bI 就声称实现了原生盒控制。

## 4. 数据划分与公平性

训练复用此前参考控制器的原始残差，没有用新测试结果训练或调参。新校准、原方法新测试和可规划场景的新方法测试均另行采集。本轮实际新增 {new_trajectories} 条视觉轨迹。

| 用途 | 停车场 | 无人机 |
|---|---|---|
| 新 CP 校准 | 40 条，种子 2026102611，纯随机 | 60 条，种子 2026102612，纯随机 |
| 原方法新测试 | 50 条，种子 2026102613 | 50 条，种子 2026102614 |
| 新方法测试 | 规划成功才运行，同场景同种子与扰动 | 规划成功才运行，同场景同种子与扰动 |
| 测试组成 | 8 条轴向压力＋42 条随机 | 20 条轴向压力＋30 条随机 |

没有使用上一版测试种子；没有为了让结果变好继续改变 gamma、mu、多项式、倍率或 Hessian。若规划失败，保留失败，不随意补一个不同控制器来冒充该方案结果。

## 5. 停车场结果

{scene_conclusion('parking')}

{control_table('parking')}

原控制器列的“文档盒覆盖”是在原策略数据上离线评价同一个新盒界，不代表原控制器实际使用了新界。随机子集分母为 42，压力子集分母为 8。采样碰撞按原障碍物定义检查；最小净距为负代表采样侵入障碍物。

![停车场：文档方案诊断](parking/doc_cp_comparison.png)

## 6. 无人机结果

{scene_conclusion('quad')}

{control_table('quad')}

随机子集分母为 30，压力子集分母为 20。作者任务成功口径使用较宽终端位置区域；平均终点误差另行报告，不把原成功率等同于精确到达目标。

![无人机：文档方案诊断](quad/doc_cp_comparison.png)

## 7. 怎样理解这些结果

### 7.1 失败具体发生在哪里

{failure_table()}

停车场在多次局部 QP 后失败，无人机第一次局部 QP 就失败。停车场虽出现过非凸相关底层日志，最终状态是“达到最大迭代数”；不能把最终返回状态改写成“非凸拒绝”。局部 Hessian 的负特征值说明该局部二次目标不定，但仅凭它也不能推导整个非线性控制问题无可行解。

针对停车场的迭代上限，另作了一个单独记录的数值诊断：将 QP 最大迭代预算设为 50000，误差模型、校准倍率、目标与硬约束均不变，没有做 Hessian 投影。它仍在第 81 次 QP 调用处返回 `maximum iterations reached`，没有生成控制器。该诊断不属于“原参数完全不变”的主实验，日志与状态单独保存为 `plan_iter50000.log`、`plan_status_iter50000.json`、`qp_audit_iter50000.json`。

### 7.2 参考策略数据上的覆盖效果

停车场原策略随机子集为 41/42、压力子集为 3/8，混合后为 44/50；无人机随机子集为 29/30、压力子集为 8/20，混合后为 37/50。纯随机校准与轴向压力样本的分布不同，所以不应把混合测试的 88% 和 74% 直接解释为“同分布 CP 理论被否定”。同样，也不能因为一个有限随机测试集超过 95%，就声称每次固定校准后的真实覆盖概率已被证明超过 95%。

本次原策略下的盒界逐点覆盖分别为 99.53%、96.80%；更大的外包球整轨迹覆盖为 48/50、40/50。外包球覆盖较高符合包含关系，但不是文档原盒界的覆盖成绩。

无人机的训练点包络均为正，但在原策略压力测试中已有负的未截断预测值；在控制规划的初始 36 个状态点上，有 27 个点落到正尺度下限。该初始化轨迹上的未截断包络范围约为 -0.6559 至 1.7900，乘法盒界范围约为 0.001187 至 2.124764。这个检查表明：校准参考轨迹附近可用的函数，放入规划搜索区域后未必保持合理形状。它是重要诊断，尚未单独证明负预测或外包球是求解失败的唯一原因。

### 7.3 可以得出的结论及限制

第一，必须同时区分包络拟合、校准覆盖、原策略独立测试、新策略覆盖与控制效果。前三者正常不自动推出后两者成功；无法规划时也没有新策略闭环统计可供推断。

第二，这次与上一版同时改变了范数、拟合目标、函数族、加法／乘法校准与控制集合大小，而且测试种子不同。它回答的是“更贴近文档的一整条方案在本轮条件下如何”，**不能将改善或退化单独归因于无穷范数或乘法校准**。

第三，文档中的 CP 保证仍要求固定模型与适用的交换性。原策略校准后再优化新策略，依赖校准结果并改变轨迹分布，这一条件问题仍然存在。轴向压力数据另作诊断，不和纯随机校准数据混为同分布样本。

第四，Hessian 非凸属于当前局部控制 QP 的数值／模型组合问题；本轮的凸包络拟合已成功，不能把两者混淆。规划器返回 success 也不自动代表严格收敛或完整非线性约束得到证明，另记录外层收敛检查及保存计划的一致性诊断。

第五，即使全部测试没有碰撞，也只能描述这些初始条件与采样扰动。本轮没有硬件测试，没有覆盖论文所有 15 组初始条件，也没有建立全状态确定性安全证书。

## 8. 独立核验与复现入口

程序独立重算了训练目标与松弛约束、CP 轨迹分数及倍率；对已完成的测试，从原始观测和状态重建残差，从四个 SLS 响应矩阵重建实际输入，并用独立 NumPy RK4 重算每一步动力学。还核对原／新扰动相同、保存的控制球半径确实等于 sqrt(ny) 倍盒界，以及模型构造没有读取旧误差参数。

每个场景目录包含 `protocol.json`、`doc_envelope.json`、`training_fit.npz`、`doc_cp_model.json`、`doc_cp_scores.npz`、`plan_status.json`、`qp_audit.json`、`doc_cp_comparison.json`、原始轨迹、回放与执行脚本。规划失败时没有新控制器轨迹文件是预期结果，不是数据丢失。源码哈希、模型来源与环境见 `provenance.json`。

```text
/root/work_based_on_spvc/vision_sls_doc_cp_parking_20261026_v1
/root/work_based_on_spvc/vision_sls_doc_cp_quad_20261026_v1
```

计算链为 `fit_doc_envelope.py → calibrate_doc_cp.py → plan_doc_cp.py → run_visual_rollout.py（规划成功时）→ analyze_doc_cp.py`。建议文档没有指定唯一训练集、epsilon、所有求解细节或部署策略校准方法，因此本轮记录了这些具体选择，没有把一次具体实现的结果扩大为该方向的普遍结论。
'''
if followup.exists():
    line=text.index('\n')
    text=text[:line]+'\n\n> **后续推进已完成：** 第 1—8 节保留原参数对照组的结果；第 9 节补充局部数值适配及无人机 100 条新闭环测试。加强正则版本虽通过原收敛判据，但实际阶段状态约束和整轨迹覆盖仍失败。\n'+text[line:]
    text+='\n'+followup.read_text()
all_attempts=repo/'work/vision_sls_all_attempts_20261026/all_attempts_summary.md'
if all_attempts.exists():text+='\n'+all_attempts.read_text().replace('## 11. 全部尝试结果总览','## 10. 全部尝试结果总览',1)
report=out/'VISION-SLS_按建议原公式CP_详细实验报告.md';report.write_text(text)
print(report);print('Characters',len(text))
summary=['## 10. 后续补充：按建议原公式计算 CP 的新实验','',
 '本节在前述加法变体之后完成。误差界改为无穷范数、文档凸包络拟合、比值轨迹分数与乘法界；控制阶段明确使用包含该盒集合的外包球，保持原求解器，未做 Hessian 凸化。它不是原生盒集合 SLS 的完整验证。','',
 '停车场：'+scene_conclusion('parking'),'', '无人机：'+scene_conclusion('quad'),'',
 '### 10.1 停车场：原策略测试 8 条压力＋42 条随机','',control_table('parking'),'',
 '### 10.2 无人机：原策略测试 20 条压力＋30 条随机','',control_table('quad'),'',
 f'完整的数据、实现范围、倍率、规划状态与限制见[按建议原公式的详细实验报告]({report})。',
 '', '前述“建议原方案尚未做完整验证”等表述记录的是加法变体阶段的进度；本轮已补做文档误差界公式及外包球接入的测试，原生盒控制器仍未实现。新结果仍不能直接提升为新策略概率安全证明。','']
if followup.exists():
    summary+=['### 10.3 继续推进后的数值适配与闭环结果','',
      '误差界模型、校准数据及倍率保持不变，继续尝试局部 Hessian 凸化、更强正则和数值目标缩放。停车场所有配置仍未生成控制器。无人机两个配置各完成 50 条新图像轨迹；其中加强正则的版本通过了原收敛判据，但实际阶段状态约束违规为 50/50，文档盒界整轨迹覆盖为 0/50。',
      '', '| 无人机，50 条配对测试 | 原控制器 | 文档界＋加强局部正则 |','|---|---:|---:|',
      '| 作者任务成功 | 50 | 50 |','| 采样碰撞 | 0 | 0 |','| 实际阶段状态约束违规 | 0 | 50 |',
      '| 文档盒界整轨迹覆盖 | 37 | 0 |','| 外包球整轨迹覆盖 | 40 | 0 |',
      '| 平均终点目标距离（米） | 0.174228 | 0.163598 |','',
      '实际越界发生在模型索引 8 的状态分量：允许绝对值不超过 2，最大超出约 0.054589。作者任务成功口径没有包含所有阶段约束，因此不能将 50/50 成功和 0 碰撞汇报为安全验证通过。新策略下 81.14% 的观测落到误差尺度下限，整轨迹覆盖失败。',
      '', '当前尚未完成建议所提醒的整个使用状态域的非负性／误差充足性验证。按其示例拟合并作正尺度截断，不等于已经满足完整区域条件。新的结果进一步说明需要验证模型使用域及新策略分布，不能继续仅靠修改求解器参数来宣称安全。',
      '',f'各数值版本、完整指标与失败诊断见[新报告第 9 节]({report})。','']
(Path(__file__).parent/'latest_doc_experiment_section.md').write_text('\n'.join(summary))
