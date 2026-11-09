# 车道保持：80 步轨迹级 CP 与 SBC

2026-11-07。已冻结原 PPO、patch-DINO、名义模拟器分布，采集 199 条校准和 100 条独立留出轨迹。每条 80 步，分数为所有时刻、两个归一化观测分量的最大绝对误差。

## 结果

- 校准半径：0.09009315804748166，第 190/199 个分数；alpha=0.05。
- 留出完整覆盖：95/100；旧 0.08 半径在同一批轨迹上为 82/100。
- 留出帧覆盖：99.9375%；留出实际越界：0/100。
- 同一策略与 SBC 在新半径下：仿射界、普通 IBP 均通过，三个曲率零未解决区间。
- 模型权重均未更新；27 项软件回归通过。

这是注册名义模拟器分布下的整条轨迹边缘覆盖 CP，不是实车、无限时域、任意条件分布或固定校准集的条件 95% 保证。还剩 5 条留出轨迹超包络，它们实测安全，但不能把它们的安全归因于包络内 SBC 论证。

## 文件

- [冻结协议](protocol.json)：样本数、初态/曲率采样规律、环境、模型和源码散列。
- [校准结果](calibration.json)、calibration.npz：199 个轨迹分数及完整记录。
- [留出与总报告](report.json)、holdout.npz：100 条轨迹的独立评估。
- [主审计](audit/certificate.json)、[IBP 复核](audit_ibp/certificate.json)。
- [连接检查](bridge.json)：模型、半径、任务一致性及概率解释。
- [测试](tests.txt)、[实验图](trajectory_coverage.png)、[PDF 图](trajectory_coverage.pdf)。
- actor.pt、sbc.pt、patch_projection.pt：沿用冻结的 11 月 6 日模型。
- source_snapshot、environment.json、manifest.json：复现源码、环境和包内文件校验。
- [完整步骤与理论解释](../../../docs/current/车道保持_轨迹级CP校准与SBC衔接_20261107.md)。

数组采用 episodes × time × coordinates；states 含初态和 80 个后继，errors/observations/actions 含 80 步。calibration/ 与 holdout/ 下的逐批文件用于恢复采集，不能将批次或单帧误认为独立校准单位。

## 复现

从 artical-F122 目录运行，输出路径须不存在：

~~~bash
../.venv-vision-sls/bin/python -m Aebs.lane_keeping.trajectory_cp \
  --package results/lane_keeping_20261106_aebs_method_sbc_v1 \
  --output-dir results/lane_trajectory_cp_reproduce \
  --dino-repo ../external/dinov2 \
  --dino-weights ../external/dinov2_weights/dinov2_vits14_pretrain.pth
~~~

默认注册 199 条校准、100 条留出、80 步和 alpha=0.05。恢复中断采集需加 --resume，且所有冻结配置及模型必须一致。先写入校准阈值，才开始留出集采集。

~~~bash
../.venv-vision-sls/bin/python -m Aebs.lane_keeping.verify_learned \
  --actor results/lane_trajectory_cp_20261107_v1/actor.pt \
  --sbc results/lane_trajectory_cp_20261107_v1/sbc.pt \
  --radius 0.09009315804748166 --grid 40 --depth 16 --error-splits 4 \
  --output-dir results/lane_trajectory_cp_audit_reproduce
~~~

普通 IBP 复核使用 --method ibp --grid 50 --depth 18 --error-splits 16，并使用另一输出目录。训练或改变闭环后不能直接沿用本包的轨迹分布保证。

## 准确的概率表述

CP 控制的是随机校准集与新轨迹的边缘失覆盖概率。若接受事件 A 依赖校准半径，而 U 是新运行越界，则基于可靠的确定性审计可连接为 Pr(A 且 U)≤0.05；不能自动改写为 Pr(U | A)≤0.05。当前区间审计还是数值证据，并非严格向外舍入的浮点形式证明。完整解释见文档第 4—5 节。
