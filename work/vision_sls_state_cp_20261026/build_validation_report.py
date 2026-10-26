"""Build a Chinese evidence-based report from completed server experiments."""
import json,shutil
from pathlib import Path

repo=Path('/Users/johann/project/work_based_on_spvc')
output=Path('/Users/johann/Documents/Codex/2026-10-22/jian/outputs/vision_sls_状态相关CP验证_20261026')
data={s:json.loads((output/s/'state_cp_comparison.json').read_text()) for s in ['parking','quad']}
audit={s:json.loads((output/s/'calibration_and_extrapolation_audit.json').read_text()) for s in data}
comparison=(Path(__file__).parent/'建议文档与当前实现对照_正文.md').read_text()
def pct(x):return f'{100*x:.2f}%'
def table(s):
    b,c=[data[s]['methods'][m] for m in ['baseline','state_cp']]
    rows=[('作者口径任务成功（50 条）','author_successes'),('采样碰撞轨迹数','collision_rollouts'),
          ('实际状态／输入／碰撞联合违规轨迹数','full_state_input_collision_constraint_violations'),
          ('状态管束越界轨迹数，容差 1e-6','state_tube_violating_rollouts_tolerance_1e-6'),
          ('输入管束越界轨迹数，容差 1e-6','input_tube_violating_rollouts_tolerance_1e-6'),
          ('终点位置误差均值（米）','terminal_position_distance_mean_m'),
          ('最小采样障碍物净距（米）','minimum_sampled_obstacle_clearance_m'),
          ('新状态相关界的逐时刻覆盖率','new_state_bound_point_coverage'),
          ('新状态相关界完整轨迹覆盖（50 条）','new_state_bound_trajectories_covered'),
          ('其中随机扰动轨迹完整覆盖','new_state_bound_random_trajectories_covered'),
          ('其中轴向压力轨迹完整覆盖','new_state_bound_stress_trajectories_covered'),
          ('反馈算子 Frobenius 范数','feedback_operator_frobenius_norm')]
    lines=['| 指标 | 原控制器 | 新状态相关 CP 控制器 |','|---|---:|---:|']
    for label,key in rows:
        values=[pct(x[key]) if 'point_coverage' in key else f'{x[key]:.6g}' if isinstance(x[key],float) else str(x[key]) for x in [b,c]]
        lines.append('| '+label+' | '+' | '.join(values)+' |')
    return '\n'.join(lines)

parking=data['parking'];quad=data['quad']
pm=parking['state_cp_model'];qm=quad['state_cp_model']
pc=parking['methods']['state_cp'];qc=quad['methods']['state_cp']
text=rf'''# 用 CP 计算 VISION-SLS 感知误差界：状态相关版本的实测验证与导师汇报稿

**验证日期：2026 年 10 月 26 日。** 本次在 `root@10.11.154.191:20020` 实际运行新实验，包含状态相关误差模型拟合、独立轨迹校准、重新规划，以及原控制器和新控制器的配对视觉闭环仿真。全部结果为仿真结果。

## 1. 给导师的直接回答

老师，CP 确实可以从感知残差计算误差集合，也可以做成依赖状态的函数。我已经把这条链实际实现，并在停车场和无人机两个场景做了新的独立校准与配对测试。

先说明与建议文档的关系：**本次运行的是参考其思路设计的变体，并非原样复现。建议文档使用无穷范数、比值分数与乘法校准；当前实现使用二范数、差值分数与加法校准，拟合方法和多项式也有调整。** 逐项差异及其影响见 3.4 节。因此下面的失败结果不能等同于建议文档原始方案已经验证失败。

但是，新验证结果要求我们收紧之前“可行”的表述。**能计算 CP 界、能把界送进控制器，不等于新控制器已获得原实现相同或更强的安全保障。** 此前常数 CP 版本在已测条件下控制表现较好；这次状态相关版本不能直接继承那个结论。

停车场的新状态相关版本，任务成功为 {pc['author_successes']}/50，采样碰撞 {pc['collision_rollouts']}/50，实际联合约束违规 {pc['full_state_input_collision_constraint_violations']}/50；新控制器的完整轨迹误差覆盖为 {pc['new_state_bound_trajectories_covered']}/50。

无人机直接替换后，原求解器首先因局部 QP 非凸而失败。保留失败记录后，我另加局部 Hessian 凸化来检查后续行为。这个适配版本虽然输出了计划并完成了仿真，但只有 {qc['author_successes']}/50 次作者口径任务成功，出现 {qc['collision_rollouts']}/50 条碰撞轨迹，完整轨迹误差覆盖为 {qc['new_state_bound_trajectories_covered']}/50；同扰动的原控制器为 50/50 次任务成功、0 次碰撞。

因此，目前更准确的结论是：**CP 误差界计算与接口接入已经验证；“状态相关 CP 直接替换后仍保持原安全表现”这项主张没有通过本次验证。无人机当前实现明确失败，需要改进后重新测试。** 这不能据此推导出 CP 方法本身不可行，它说明当前误差模型、策略变化和数值求解的组合还没有解决好。

| 本次独立新测试 | 原控制器任务成功 | 新控制器任务成功 | 新控制器碰撞 | 新控制器整轨迹覆盖 |
|---|---:|---:|---:|---:|
| 停车场 | 50/50 | 50/50 | 0/50 | 0/50 |
| 无人机，含局部凸化适配 | 50/50 | 1/50 | 49/50 | 0/50 |

## 2. 这次验证相较前一版增加了什么

此前报告的最终闭环数字来自常数球：停车场半径约 1.896320，无人机约 0.141924；两场景当时均完成 50/50 次任务，且无采样碰撞。但新控制器完整轨迹覆盖分别只有 9/50、31/50。那些是 10 月 25 日实验，不是本次状态相关实验。

本次新增的是一个真正随状态变化的误差函数：先用原始残差训练一个新的预测器，再用另一批新轨迹作 CP 校准，随后重新规划并实际测试。**没有读取或缩放作者旧多项式误差包络。** 原视觉网络与投影网络没有重训，状态观测矩阵 C 沿用原实现。

| 数据用途 | 停车场 | 无人机 |
|---|---|---|
| 新预测器训练 | 此前原控制器的 40 条校准轨迹，1200 个观测 | 此前原控制器的 100 条校准轨迹，3500 个观测 |
| 本次独立 CP 校准 | 新采集 40 条纯随机轨迹 | 新采集 60 条纯随机轨迹 |
| 校准种子 | 2026102601 | 2026102602 |
| 本次原控制器测试 | 50 条 | 50 条 |
| 本次新控制器测试 | 同扰动的 50 条 | 同扰动的 50 条 |
| 测试种子 | 2026102603 | 2026102604 |
| 测试组成 | 8 条轴向压力＋42 条随机 | 20 条轴向压力＋30 条随机 |
| 每条轨迹观测数 | 30 | 35 |

新视觉闭环轨迹总数为 300 条：停车场 40＋50＋50＝140，无人机 60＋50＋50＝160。旧训练轨迹另外复用，未计入这 300 条。训练、校准与测试用途分开；测试结果没有用于调节系数或 CP 偏移量。无人机数值适配在其新控制器闭环测试前决定并记录。

## 3. 每一步输入什么、得到什么

### 3.1 先明确要约束的是哪一个误差

输入是真实渲染图像、固定视觉网络及真实图像来源状态。网络输出低维观测 $y^{{\rm vis}}$。需要估计的原始残差是

$$r_t=y_t^{{\rm vis}}-Cx_t^{{\rm image}}.$$

停车场实现有坐标平移，实际重算使用 $C(x_t^{{\rm image}}+[3,0,0,0]^\top)$。这个平移也用于反馈创新项，不能遗漏。停车场残差维度为 3，无人机为 5。

代码保留原视觉延迟：第 0 次与第 1 次观测均对应初始状态，后续观测对应上一时刻状态。独立核验明确重建了 `[0,0,1,…,T-2]` 的图像状态索引，避免把延迟误当成网络误差。

这里的 $r_t$ 是观测空间误差，不是位置误差的米制数值。后文误差界数值不能直接读成“多少米”。本次取标量目标 $a_t=\|r_t\|_2$，让一个球同时包住所有观测分量。

### 3.2 从训练数据拟合一个新的状态相关预测器

输入为训练状态及对应的原始误差范数。停车场使用

$$\phi(x)=[p_x,p_y,\sin\theta,\cos\theta],$$

无人机使用

$$\phi(x)=[p_x,p_y,p_z,x_6,x_7],$$

其中 $x_6,x_7$ 指从 0 开始计数的原动力学两个姿态状态。加入姿态是为了允许“同一位置、不同朝向或姿态”产生不同视觉误差；这不是所有可能特征中最优的选择。

先按训练均值与标准差标准化，标准差最小取 0.05；再构造总次数不超过 2 的完整多项式：常数、所有一次项、所有二次和交叉项。因此停车场有 15 个系数，无人机有 21 个系数。

令 $\psi(x)$ 为上述多项式基，训练目标是

$$\min_\beta\ \frac1N\sum_j\rho_{{0.9}}\!\left(\frac{{a_j}}{{s}}-\psi(x_j)^\top\beta\right)
+0.001\sum_{{k>0}}|\beta_k|,$$

$$\rho_\tau(u)=\begin{{cases}}\tau u,&u\ge0,\\(\tau-1)u,&u<0.\end{{cases}}$$

这里 $s$ 是训练误差范数的中位数，只作目标归一化；截距不正则化。用 SciPy HiGHS 线性规划求解，输出

$$g_\theta(x)=s\psi(x)^\top\beta.$$

直观上，0.9 分位回归对“误差被低估”的惩罚更大，使预测器倾向于描述较大的误差尺度。**0.9 是训练损失参数，不是闭环安全概率，也不是最终 95% 覆盖保证。** 预测器是否拟合好主要影响集合大小与外推行为；CP 的边际覆盖还需要独立校准及交换性条件。

### 3.3 冻结预测器，再用新轨迹计算 CP 修正量

输入为全新的原控制器随机轨迹。保持视觉网络、$C$、$g_\theta$ 和原参考控制器不变，给每条轨迹一个分数：

$$S_i=\max_{{t=0,\ldots,T-1}}\bigl(\|r_{{i,t}}\|_2-g_\theta(x_{{i,t}}^{{\rm image}})\bigr).$$

一条轨迹只提供一个最大分数。这样保留轨迹内时间相关性，不把相邻图像错误地当成大量独立校准样本。排序后取

$$k=\lceil(n_{{\rm cal}}+1)(1-\alpha)\rceil,\qquad Q_\alpha=S_{{(k)}},\qquad\alpha=0.05.$$

这里的 95% 是本次 CP 候选方案额外设定的整轨迹覆盖目标，不是原论文脚本自带的“成功率必须达到 95%”验收指标。

输出的新误差界为

$$\boxed{{b_{{\rm CP}}(x)=\max\{{0.001,g_\theta(x)+Q_\alpha\}}}}.$$

| 校准量 | 停车场 | 无人机 |
|---|---:|---:|
| 轨迹样本数 | {pm['calibration_trajectories']} | {qm['calibration_trajectories']} |
| 有限样本排序索引 k | {pm['rank_k']} | {qm['rank_k']} |
| CP 加法偏移量 Q | {pm['cp_offset']:.9f} | {qm['cp_offset']:.9f} |
| 校准集完整轨迹覆盖 | {pct(pm['calibration_bound_trajectory_coverage'])} | {pct(qm['calibration_bound_trajectory_coverage'])} |
| 校准状态上的界范围 | {pm['calibration_bound_min']:.6f}—{pm['calibration_bound_max']:.6f} | {qm['calibration_bound_min']:.6f}—{qm['calibration_bound_max']:.6f} |

校准集覆盖不是独立测试成绩。比如停车场取从小到大第 39 个顺序统计量，覆盖 39/40 本来就是构造的结果，不能再据此声称新控制器已达到 97.5%。本次 k 均不超过样本数；一般情况下若 k 超过样本数，标准有限样本构造应取无穷大界，不能随意改成样本最大值后仍声称原覆盖目标。

### 3.4 是否按照建议文档实现：逐项差异、原因与验证范围

{comparison}

### 3.5 为什么这个公式在适用条件下有 CP 理论支撑

当训练模型固定，校准轨迹和一条新轨迹可交换时，有限样本排序给出

$$\mathbb P\{{S_{{\rm new}}\le Q_\alpha\}}\ge1-\alpha.$$

而 $S_{{\rm new}}\le Q_\alpha$ 意味着该轨迹每个观测时刻都有

$$\|r_t\|_2\le g_\theta(x_t^{{\rm image}})+Q_\alpha\le b_{{\rm CP}}(x_t^{{\rm image}}).$$

所以这是整条有限时域轨迹的边际覆盖事件。概率同时平均了校准数据与新轨迹的随机性；它不是每个固定状态的条件覆盖，也不是保证每次固定校准之后真实覆盖恰好为 95%。CP 的有限样本秩和交换性条件可参见 [Angelopoulos 与 Bates 的 CP 教程](https://arxiv.org/html/2107.07511v6)。

问题在于：**校准由原策略产生，而使用校准界重新优化出的新策略依赖校准结果，还会改变后续状态与图像。** 因此不能直接把同分布参考策略下的结论转移给新策略。轴向压力样本也不与纯随机校准样本同分布，必须单列。即使误差覆盖成立，要推出闭环安全，仍需检查动力学近似、优化可行性、管束与障碍物约束等其余条件。

原论文鲁棒控制推导所需的是适用域内误差集合及相应控制条件成立，原代码的数据拟合与非线性近似也不自动构成全域确定性证明。因此本次比较的是有明确范围的原实现表现与新实现表现，不能把原实现或 CP 替换版本任意一方提升为已经证明普遍安全。

### 3.6 F(x)e 怎样接入控制

输入为新误差界函数，输出为观测模型中的扰动矩阵：

$$F(x)=b_{{\rm CP}}(x)I,\qquad y=Cx+F(x)e,\qquad\|e\|_2\le1.$$

如果实际残差满足 $\|r\|_2\le b_{{\rm CP}}(x)$，取 $e=r/b_{{\rm CP}}(x)$ 就能表示它；如果不覆盖，则 $\|e\|_2>1$，原控制器按单位球设计的推理就不能直接套用。

规划器通过 CasADi 对实际观测函数求导，在零观测扰动处得到 $C$ 与随状态变化的 $F$；随后更新 SLS 响应、反馈和管束。界的形状会影响路线、控制增益与可行性，不只是把图上的阴影改大或改小。测试时界用于诊断和鲁棒控制设计，没有用真实残差在线修正控制输入，也没有给控制器补充真实状态反馈。

0.001 仅为正值下限：避免负数或零半径造成表达退化。**它不是经验证的全域误差上界，不能补足外推安全性。**

## 4. 停车场：完整新测试结果

{table('parking')}

表中“新界覆盖”两列使用的是同一个新界函数，分别在原策略和新策略生成的数据上评价，目的是查看策略变化的影响；不是说原控制器实际用上了新界。

随机子集分母为 42，压力子集分母为 8。任务成功与实际约束检查保留原场景口径；管束越界另外统计。即使某个方法 50 次均无碰撞，也只能描述这些离散采样轨迹。

这里有一个容易误读的结果：停车场不是“差一点才达到 95% 整轨迹覆盖”。新控制器 1500 个观测中有 381 个误差未被覆盖，每条轨迹至少一次未覆盖，所以完整轨迹覆盖为零。第 10—15 个观测索引（从 0 开始）处，逐步覆盖均为零；它表现为一段共同经过的路线持续低估，而非仅有少数偶然异常点。

同时，新反馈算子范数仅为原来的约 {100*pc['feedback_operator_frobenius_norm']/parking['methods']['baseline']['feedback_operator_frobenius_norm']:.3f}%。这支持“当前控制对视觉修正的依赖明显减弱”的解释，但未单独做零反馈消融，不能直接说控制成功完全不依赖视觉。当前试验的初始条件固定、过程扰动较小，名义路线和较宽管束可能使控制仍然成功；这个现象不能弥补误差覆盖条件的失败。

![停车场：路线、整轨迹覆盖与沿途误差界](parking/state_cp_comparison.png)

## 5. 无人机：直接替换失败，数值适配后的闭环也失败

### 5.1 严格保持原数值求解器时发生了什么

新函数通过了数值／符号一致性和观测 Jacobian 检查，但第一次局部 QP 被 OSQP 拒绝，真实返回状态为 `problem non convex`。其 Hessian 最小特征值约为 $-2.41989\times10^5$，最大特征值约为 $8.71153\times10^6$。

上层通用日志把该情况写成 QP 不可行；底层审计表明首先遇到的是局部目标非凸。**不能据此断言原非线性控制问题在数学上无可行解。** 严格替换失败日志保存在 `quad/strict_replacement_failure/`。

### 5.2 为继续诊断，加了哪一项适配

仅对局部 QP 的对称 Hessian 做特征值投影。当最小特征值小于 -1e-7 时，把特征值下限设为 1e-6，随后交给原 QP 求解器。原误差函数、CP 偏移量、动力学、硬约束、时域与测试种子没有修改。

这个修改改变了 SCP 的局部求解过程，因此下面结果应称为“**状态相关 CP＋局部 Hessian 凸化**”，不能称为完全不改求解器的纯替换结果。总共记录 335 次 QP 调用，其中 329 次应用了该投影。

适配后输出有限计划并不意味着已经获得有效鲁棒轨迹。独立检查发现：

| 规划诊断 | 原无人机计划 | 新适配计划 |
|---|---:|---:|
| 单步非线性名义动力学最大绝对缺陷 | 7.12e-12 | 0.056047 |
| 收紧后阶段线性约束最大残差 | 5.68e-7 | 0.000982 |
| 按保存管束半宽外接圆膨胀的最小名义 XY 净距（米） | -7.59e-8 | -0.022478 |
| 在最终轨迹重新线性化后的最大 SLS 恒等式残差 | 9.84e-7 | 0.021131 |

正的约束残差代表该检查中存在超出；负的膨胀净距代表该外接圆检查未通过，它是额外的保守几何诊断，不能单独等同于实际碰撞。动力学缺陷跨多个状态分量取最大值，单位不统一，不能直接读成位置偏差多少米。SLS 响应可能对应上一轮局部模型，因此重新线性化残差属于一致性诊断。

还需注意原源码 `solver/SCP_OF_SLS.py` 的行为：达到最大 SCP 迭代数时也可以设置 `success=True`。本次未额外记录完整外层收敛轨迹，所以不能仅凭返回标志推断它达到了严格收敛。实际计划与闭环检查比这个标志更有说服力。

### 5.3 与原控制器配对的 50 条视觉闭环

{table('quad')}

随机子集分母为 30，压力子集分母为 20。作者无人机“任务成功”使用较宽的终端位置范围，并非“距指定目标 0.25 米以内”；本报告另给终点目标距离。碰撞按原五个 XY 圆柱障碍物在离散轨迹采样点检查。

该版本已经比原基线明显退化，不应继续当成可靠替代控制器使用。仿真完成和程序没有抛异常，只能说明执行链跑完。

![无人机：失败路线、覆盖与误差界变化](quad/state_cp_comparison.png)

## 6. 失败原因能说到什么程度

### 6.1 有直接数据支持的事实

无人机新控制器下，{pct(qc['feature_outside_training_coordinate_ranges_fraction'])} 的图像来源状态，其至少一个特征超出了训练数据各坐标的最小／最大范围。这个范围只是简单矩形诊断，不代表严格训练分布支持集。

新预测器在这些新策略样本上的最小值约为 {audit['quad']['policies']['state_cp']['predictor_min']:.6f}，而训练样本上的预测器为正。加上 CP 偏移量后，1750 个观测中有 {audit['quad']['policies']['state_cp']['floor_points']} 个被截到 0.001 下限，约占 {pct(audit['quad']['policies']['state_cp']['floor_point_fraction'])}；这些下限点全部未覆盖实际误差。

原策略测试数据上的新界逐点覆盖仍为 {pct(quad['methods']['baseline']['new_state_bound_point_coverage'])}，完整轨迹为 45/50，其中随机子集为 28/30。换成新控制器之后，逐点覆盖降为 {pct(qc['new_state_bound_point_coverage'])}，完整轨迹为 0/50。新策略的反馈算子范数也由约 0.215 增至 4.710。

停车场原策略随机测试子集的完整覆盖为 40/42，换新策略后为 0/42。其未覆盖点没有落到 0.001 下限，且全部仍位于训练特征的逐坐标最小／最大矩形范围内。因此停车场失败不能简单解释成“把下限设大就好了”，也说明“各特征没有超范围”不足以确认联合状态或图像分布没有变化。

### 6.2 这些事实支持的解释及其边界

一个合理解释是：训练仅覆盖原策略附近区域，新二次预测器在新区域外推不可靠；规划对这个状态相关函数敏感，界变小会影响反馈和管束；新策略偏离后，又产生更大的视觉残差。与此同时，新适配计划本身已有动力学一致性等数值问题，不能将所有退化单独归因于分布变化。

这组测试没有做完整因果消融，不能断言“只要换某个多项式项就能修复”，也不能把碰撞后的所有大残差都归为碰撞前的原因。可确认的是：**当前组合方案不满足替代原实现的要求，而且 CP 在原策略上的校准不能掩盖这些问题。**

## 7. 如何确认不是统计或回放脚本算错

对两场景的两种测试策略，独立从保存的原始数组重新计算：

1. 图像对应状态索引与坐标平移，然后重建全部原始视觉残差。
2. 从保存的四个 SLS 响应矩阵恢复反馈算子 $K=\Phi_{{uy}}-\Phi_{{ux}}\Phi_{{xx}}^{{-1}}\Phi_{{xy}}$，再用历史创新项重建每一步实际控制输入。
3. 独立写出 NumPy 动力学，用原步长 RK4 重算每个状态转移。
4. 按原约束、障碍物、管束和终端条件重算指标，并与仿真汇总核对。
5. 重算新界，确认与闭环逐时记录的界一致；重算整条轨迹分数与有限样本顺序统计量。
6. 检查同场景原／新方法的过程扰动数组完全相同，全部保存数组为有限数。
7. 在模型构造检查中禁止读取旧误差参数；检验新函数不是常数、数值与 CasADi 表达一致，以及 $\partial y/\partial e=b(x)I$。

对应误差与核验布尔值见各场景 `state_cp_comparison.json`、`calibration_and_extrapolation_audit.json`、`model_checks.json`；全部断言已通过。由此可以支持“计算记录与本次实现一致”，不能支持“当前实现已经安全”。

## 8. 下一步应怎样推进

本次验证已经完成；以下为后续研究建议，没有把它们写成已完成的修复。

第一，先处理**误差模型有效区域与数值规划质量**。不能只靠一个很小的正值下限防止外推，需明确状态域、约束模型在域外的行为，并对动力学缺陷、管束和收紧约束设置实际验收条件。当前无人机计划应拒绝进入可靠控制流程。

第二，区分“冻结控制器后的独立验证”和“用校准结果再优化控制器”。可以用训练／调参数据完成候选策略，再保留完全独立的最终轨迹做验证；但如果校准后又改界并重新优化策略，就再次改变了数据分布。反复采样、重规划直至看起来覆盖，并不自动恢复 CP 理论条件。

第三，若要给新策略可用的概率安全结论，需要明确选择一种能够支持策略自适应或分布变化的理论路线，或者构造在规定状态域与动力学条件下适用的误差集合及控制证书。单纯增加校准轨迹数量，不能解决所有策略变化问题。

第四，新方案确定后，应重新预先划分训练、校准、测试数据，保留压力样本和随机样本的区分，同时报告误差覆盖与实际控制表现。不要用这次已看过的测试结果继续调参后，再把同一批数据称作独立测试。

## 9. 汇报时建议采用的最终表述

> 我已经确认，CP 可以计算感知误差界，并已实现状态相关版本及控制接口。当前测试的是参考建议设计的加法轨迹 CP 变体，并非文档中的无穷范数比值／乘法方案。此前常数版本在两场景中取得较好的仿真控制表现，但新策略整轨迹覆盖不足。这次变体测试中，停车场控制表现较好但覆盖失败，无人机当前实现明确失败。它不能说明建议文档原方案已经失败；该原方案仍缺完整独立闭环验证。主要待解决的是新策略与校准分布的关系、误差模型的域外行为，以及非线性规划的数值有效性。因此可以汇报“计算与接入可行，可靠闭环替代尚未验证成功”，不能汇报“加上 CP 后已获得同等或更强安全保障”。

## 附录：复核入口与文件

本地每个场景文件夹包含：

- `protocol.json`：预先保存的数据划分、特征、损失、CP 分数及种子。
- `state_cp_model.json`：预测器、标准化系数、CP 偏移量与校准信息。
- `state_cp_scores.npz`：全部校准轨迹分数与逐点误差。
- `baseline/diagnostic_traces.npz`、`state_cp/diagnostic_traces.npz`：实际轨迹、视觉输出、输入和扰动。
- `baseline/first_rollout.mp4`、`state_cp/first_rollout.mp4`：首条可视化回放。
- `state_cp_plan/`、`baseline_plan_checks/`：新计划及新旧计划一致性检查。
- `state_cp_comparison.json`：最终配对结果及独立重算误差。
- `calibration_and_extrapolation_audit.json`：校准排序复核及域外行为诊断。
- `provenance.json`：环境、源码版本与脚本／数据哈希。
- `quad/strict_replacement_failure/`：原求解器直接替换失败证据；`quad/solver_adaptation.json`：适配记录。

脚本副本在本报告同目录 `scripts/`，工作副本在项目 `work/vision_sls_state_cp_20261026/`。为避免意外覆盖原实验，本次服务器输出单独位于：

```text
/root/work_based_on_spvc/vision_sls_state_cp_parking_20261026_v1
/root/work_based_on_spvc/vision_sls_state_cp_quad_20261026_v1
```

可按 `fit_state_predictor.py → calibrate_state_cp.py → plan_state_cp.py → run_visual_rollout.py → compare_state_cp.py → audit_calibration.py` 追踪每个环节。重新采集图像还依赖对应场景原源码、已安装环境、DINO 骨干及原投影网络；`protocol.json` 记录训练数据来源。数值适配通过 `--convexify` 显式启用，原始失败保留，不能将其覆盖成无条件成功。

此前完整的常数 CP 阶段汇报稿保留为 [此前汇报稿_常数CP实验阶段_保留.md](此前汇报稿_常数CP实验阶段_保留.md)。其中“下一步状态相关验证”等描述是当时进度，当前结论以本报告为准。
'''

latest_section=repo/'work/vision_sls_doc_cp_20261026/latest_doc_experiment_section.md'
if latest_section.exists():
    latest=latest_section.read_text()
    first_newline=text.index('\n')
    notice='\n\n> **后续验证已补充：** 本文第 1—9 节记录加法 CP 变体阶段；第 10 节记录随后按建议文档原公式计算误差界、通过外包球接入原规划器的新实验；第 11 节汇总全部成功、失败和诊断尝试。请结合后两节判断最新进度，不能把不同阶段混为同一次试验。\n'
    text=text[:first_newline]+notice+text[first_newline:]
    text=text.replace('#### 3.4.7 如果要更严格按照建议实现，应怎样定义待验证版本','#### 3.4.7 与建议更接近的方案定义（后续试验见第 10 节）')
    text=text.replace('下面是与建议更接近的**待做方案说明**，不是本次已运行的新结果：','下面记录加法变体之后提出的方案定义。后续已经实施文档误差界公式及外包球接口的试验，结果见第 10 节；原生盒集合控制器仍未实现：')
    text=text.replace('该原方案仍缺完整独立闭环验证。主要待解决的是','后续原公式与外包接口试验见第 10 节，原生盒集合控制器仍未实现。主要待解决的是')
    before,appendix=text.split('## 附录：复核入口与文件',1)
    text=before+latest+'\n## 附录：复核入口与文件'+appendix
all_attempts=repo/'work/vision_sls_all_attempts_20261026/all_attempts_summary.md'
if all_attempts.exists():
    before,appendix=text.split('## 附录：复核入口与文件',1)
    text=before+all_attempts.read_text()+'\n## 附录：复核入口与文件'+appendix
report=output/'VISION-SLS_状态相关CP_服务器验证与导师汇报.md'
report.write_text(text)
canonical=repo/'docs/current/VISION-SLS_CP误差界可行性_导师汇报稿.md'
backup=output/'此前汇报稿_常数CP实验阶段_保留.md'
if not backup.exists():shutil.copy2(canonical,backup)
# Canonical file uses absolute artifact links, while exported report remains portable.
canonical_text=text.replace('(parking/',f'({output}/parking/').replace('(quad/',f'({output}/quad/')
canonical_text=canonical_text.replace('(scripts/',f'({output}/scripts/')
canonical_text=canonical_text.replace('(此前汇报稿_常数CP实验阶段_保留.md)',f'({backup})')
canonical.write_text(canonical_text)
export=Path('/Users/johann/Documents/Codex/2026-10-22/jian/outputs/vision_sls_CP可行性_导师汇报_20261026/关于用CP计算VISION-SLS感知误差界的可行性_导师汇报稿.md')
shutil.copy2(canonical,export)
print(report)
print('Characters:',len(text))
