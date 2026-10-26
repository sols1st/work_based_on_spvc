"""Publish a Chinese explanation using only completed experiment outputs."""
import json
from pathlib import Path

root=Path(__file__).resolve().parent
out=root/'results'
r=json.loads((out/'direct_cp_comparison.json').read_text())
b,c=r['methods']['baseline'],r['methods']['direct_cp']
cal=r['calibration']
rows=[('停车成功次数','author_successes','d'),
      ('碰撞次数','collision_rollouts','d'),
      ('状态、终端、输入或碰撞约束违反次数','full_state_input_collision_constraint_violations','d'),
      ('最小离散采样障碍物净距 / m','minimum_sampled_obstacle_clearance_m','.6f'),
      ('平均终点位置误差 / m','terminal_position_distance_mean_m','.6f'),
      ('最大终点位置误差 / m','terminal_position_distance_max_m','.6f'),
      ('状态管束越界轨迹数','state_tube_violating_rollouts_tolerance_1e-6','d'),
      ('输入管束越界轨迹数','input_tube_violating_rollouts_tolerance_1e-6','d'),
      ('自身误差界覆盖的完整轨迹数','own_bound_l2_trajectories_covered','d'),
      ('自身误差界覆盖的随机完整轨迹数 / 42','own_bound_l2_random_trajectories_covered','d'),
      ('反馈矩阵 Frobenius 范数','feedback_operator_frobenius_norm','.6g')]
table='| 指标（各 50 次） | 原代码 | 直接 CP |\n|---|---:|---:|\n'
table+=''.join(f'| {label} | {b[key]:{fmt}} | {c[key]:{fmt}} |\n' for label,key,fmt in rows)
good=(c['author_successes']>=b['author_successes'] and
      c['collision_rollouts']<=b['collision_rollouts'] and
      c['full_state_input_collision_constraint_violations']<=b['full_state_input_collision_constraint_violations'])
conclusion=('这组配对试验中，直接 CP 在停车成功、碰撞和实际约束违反指标上达到或优于原代码。'
            if good else '这组配对试验中，直接 CP 尚未在停车成功、碰撞和实际约束违反指标上全部达到原代码。')
text=fr'''# 停车场：直接 CP 替换误差界的闭环对照

{conclusion}

实验运行在 `10.11.154.191:20020`，使用默认初始条件 0。本文结论针对该初始条件的已完成对照，不代替其余 14 个初始条件的测试。

## 1. 直接替换了什么

原方法使用位置相关误差界 $b(x)=\max\{{q_\beta(x),b_{{\min}}\}}$。本实验直接从新校准数据计算一个 CP 半径，使用

$$F_{{\rm CP}}=R_{{\rm CP}}I_3,\qquad y=Cx+F_{{\rm CP}}e,\quad \|e\|_2\le1.$$

CP 模型构造时绕过原多项式参数加载；检查程序禁止 `numpy.load`，仍能成功构造该模型。原来的 $C$、车辆动力学、DINOv2 ViT-g/14 权重、低维表示网络、图像预处理、OpenGL 渲染、SLS 求解器、代价、时域及约束保持原设置。新半径会重新计算控制器和轨迹。

因此这里没有“原误差界乘一个校准系数”的操作。采用常数半径也意味着不再保留原误差界对位置的依赖。

## 2. 校准的输入、计算和输出

使用原控制器生成 **40 条新随机扰动轨迹**，种子 `2026102503`；每条 30 步，时域 4.5 秒。测试集未参与半径选择。

停车场渲染使用坐标平移 $o=[3,0,0,0]^T$。对每张图像，记录真实生成状态和视觉网络输出，计算

$$r_{{i,t}}=y_{{i,t}}-C(x^{{\rm image}}_{{i,t}}+o),\qquad s_i=\max_{{t=1,\ldots,30}}\|r_{{i,t}}\|_2.$$

原控制代码有一步观测延迟：第一步图像来自 $x_0$，后续第 $t$ 步图像来自 $x_{{t-2}}$。记录残差时按该实际图像状态对齐。$r$ 是三维表示误差；这里的数值不是车辆位置误差的米数。

取 $\alpha=0.05$，使用有限样本修正的顺序统计量：

$$k=\lceil(40+1)(1-0.05)\rceil=39,\qquad
R_{{\rm CP}}=s_{{(39)}}={cal['radius']:.12g}.$$

其中 $s_{{(39)}}$ 表示从小到大排列后的第 39 个轨迹分数。输出为一个标量半径和 $3\times3$ 对角矩阵。完整轨迹作为校准单位，未将同一轨迹的 30 个相关时刻当成 30 条独立轨迹。

## 3. 如何比较

重新规划耗时约 5.1 秒。独立检查得到：名义轨迹离散动力学最大残差 {r['direct_cp_plan_checks']['max_nonlinear_nominal_dynamics_defect']:.3g}；收紧后路径约束最大残差 {r['direct_cp_plan_checks']['max_tightened_stage_constraint_residual']:.6f}，终端约束最大残差 {r['direct_cp_plan_checks']['max_tightened_terminal_constraint_residual']:.6f}（负数表示保留裕量）；以位置管束角点半径膨胀后的名义障碍物最小净距 {r['direct_cp_plan_checks']['minimum_tube_inflated_nominal_obstacle_clearance_m']:.6f} m。

直接 CP 控制器测试 50 次，种子 `42`。前 8 次为原代码的四个状态分量正负轴向扰动；后 42 次为原随机单位二范数扰动。原方法复用此前保存的同种子 50 次数据；已逐元素检查两组过程扰动完全一致，并验证原始核心代码指纹一致。

初始状态为 $[-2.1,-1.75,\pi/2,0]^T$，目标位置为 $[0.25,0.5]^T$，状态包含位置、朝向和速度。停车成功沿用作者代码：终点位置进入指定矩形且没有采样碰撞。实际约束检查另计路径状态、终端状态及输入约束。

{table}

自身误差界的逐时刻二范数覆盖率：原方法 **{100*b['own_bound_l2_point_coverage']:.2f}%**，直接 CP **{100*c['own_bound_l2_point_coverage']:.2f}%**。这与原程序另存的逐分量最大绝对值覆盖率不是同一口径；两组在此统一使用与实际并行求解分支对应的二范数。

管束越界与物理约束违反是两种指标：实际轨迹偏离规划管束，并不必然越过停车场边界或撞上障碍物。

### 控制器为什么会变化

原方法反馈矩阵范数为 {b['feedback_operator_frobenius_norm']:.6g}，直接 CP 为 {c['feedback_operator_frobenius_norm']:.6g}，约为原来的 {c['feedback_operator_frobenius_norm']/b['feedback_operator_frobenius_norm']:.3%}。常数 CP 半径较大，使求解器显著减小视觉反馈增益；在这组已知初始状态、较小过程扰动的试验中，新控制器主要依靠名义计划运行。视觉网络没有变得更准确。解释安全结果时，应同时看到这种反馈减弱以及位置相关信息的丢失。

## 4. CP 的统计含义与本实验结论

本次预设的 95% 是**参考控制器分布下的完整轨迹覆盖目标**，不是原论文或原代码要求的停车成功率门槛。它依赖校准轨迹与未来参考轨迹的可交换性。

将同一半径应用于原控制器的独立测试轨迹，覆盖 **{b['direct_cp_radius_trajectories_covered']}/50** 条完整轨迹，其中随机部分 **{b['direct_cp_radius_random_trajectories_covered']}/42**。重新规划后直接 CP 控制器的完整轨迹覆盖为 **{c['own_bound_l2_trajectories_covered']}/50**，其中随机部分 **{c['own_bound_l2_random_trajectories_covered']}/42**。轴向压力测试不属于随机校准分布；新控制器也会改变视觉访问状态分布，故以上两类覆盖均单独实测。

新控制器的完整轨迹覆盖为 **{100*c['own_bound_l2_trajectories_covered']/50:.0f}%**，没有达到此次预设的 95% 整轨迹覆盖目标。逐时刻覆盖率高于 95% 并不意味着整条轨迹以 95% 的概率都不越界。此处的主要问题是重新规划改变了数据分布，校准时参考控制器的 CP 结论不能直接转移给新控制器。

与原代码相同层级的判断应主要看前表的停车成功、碰撞及实际约束指标。{conclusion} 对所有状态的逐点误差包络与非线性余项等更强条件，本实验没有额外建立；原实现的相同限制继续存在。

## 5. 可复查材料

- `direct_cp_comparison.json`：完整指标、协议及规划检查。
- `direct_cp_calibration.json`、`direct_cp_scores.npz`：CP 半径及 40 条校准分数。
- `baseline/diagnostic_traces.npz`、`direct_cp/diagnostic_traces.npz`：50 对逐步状态、输入、观测、残差、扰动和管束。
- `direct_cp_plan/`：重新求得的计划和独立检查。
- `direct_cp/first_rollout.mp4`：直接 CP 第一条闭环视频。
- `provenance.json`、`scripts/`：核心代码指纹、环境与复现实验脚本。

碰撞净距和碰撞次数沿用代码中的离散位置采样与圆形障碍物检查。
'''
(out/'停车场_直接CP实验报告.md').write_text(text)
print('Chinese report written.')
