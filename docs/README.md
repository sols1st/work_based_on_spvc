# 项目文档导航

更新：2026-11-09。本页只保留当前主线入口；11/05–11/06 的设计稿、早期探针和被取代的汇总已移到[归档](archive/2026-11-09_current_cleanup/README.md)。

## 推荐阅读顺序

1. [当前文档入口](current/README.md)：当前主线、最新结论和归档说明。
2. [可视化网站](../site/index.html)：总览、实验配置/步骤/图片/结果包入口。
3. [当前方案与验证结果](current/当前方案与验证结果.md)：四个实验的统一状态、保证和边界。
4. [AEBS RGB 修复后完整实验](current/AEBS真实RGB图像_完整实验_20261108.md)：旧单位错误撤回；轨迹 CP 0.418359 m + 新神经 SBC 通过。
5. [AEBS 前端对比](current/AEBS前端对比_cGAN_灰度_真实RGB_20261108.md)：cGAN/灰度/真实 RGB 同测试集对比。
6. [AEBS 状态相关 CP](../artical-F122/results/real_aebs_state_cp_20261109_units_v2/README.md)：参考 10/29 的 Bernstein b0(d) 变体；平均半径 0.3217 m，SBC 41,070 格通过。
7. [车道状态相关 CP（已撤回）](../artical-F122/results/lane_state_cp_20261109_v1/AUDIT.md)：代码/报告不一致、留出分数误用校准残差；不可引用。
8. [AEBS 与车道逐步流程](current/AEBS与车道保持实验_逐步流程与保证_20261108.md)：动力学、数据流、输入图像。
9. [车道轨迹级 CP 与 SBC](current/车道保持_轨迹级CP校准与SBC衔接_20261107.md)：2D latent、199/100 轨迹、证书连接。
10. [停车场神经 SBC 专题](current/停车场神经SBC_构造证明与实验_20261103.md)。
11. [移动机器人三维视觉实验](current/移动机器人_三维视觉实验_20261108.md)：图像任务跑通，完整 SBC 未通过。
12. [VISION-SLS/SPVC 完整技术主文档](current/VISION-SLS_SPVC_SBC当前框架_步骤原理实验与安全保证_20261029.md)：很长，按需查。
13. [VISION-SLS/SPVC 工作汇报](current/VISION-SLS_SPVC_SBC_20261030.md)：导师汇报版，部分结果已被 11/09 修复更新。
14. [论文与代码实验对照](references/论文实验与迁移分析_20261107.md)：论文/代码维数与迁移分析。

## 当前文档职责

| 文件 | 用途 | 维护原则 |
|---|---|---|
| [current/README.md](current/README.md) | 当前入口与最新结论 | 只列当前主线；历史移归档 |
| [当前方案与验证结果](current/当前方案与验证结果.md) | 统一结果与状态总览 | 新结果先更新这里 |
| [AEBS RGB 修复后完整实验](current/AEBS真实RGB图像_完整实验_20261108.md) | AEBS 真实 RGB 主结果 | 单位、回放、轨迹 CP、新 SBC 以这里为准 |
| [AEBS 前端对比](current/AEBS前端对比_cGAN_灰度_真实RGB_20261108.md) | 三前端误差/CP 对比 | cGAN 旧检查点要标非同预算 |
| [AEBS 状态相关 CP](../artical-F122/results/real_aebs_state_cp_20261109_units_v2/README.md) | 参考 10/29 的 Bernstein 变体 | 平均更紧但留出覆盖 92/100；不替换常量包 |
| [车道状态相关 CP v2（修复）](../artical-F122/results/lane_state_cp_20261109_repaired_v2/README.md) | 99/100 分离拟合/校准；无 cap | 13,000 区间 0/0；旧数据回归 93/100；非新盲测 |
| [车道状态相关 CP v1（已撤回）](../artical-F122/results/lane_state_cp_20261109_v1/AUDIT.md) | 代码/报告不一致 | **withdrawn / unverified**；不引用 |
| [AEBS 与车道逐步流程](current/AEBS与车道保持实验_逐步流程与保证_20261108.md) | 两实验动力学/数据流/图像 | 证书与行为证据分开 |
| [车道轨迹级 CP 与 SBC](current/车道保持_轨迹级CP校准与SBC衔接_20261107.md) | 车道统计校准与证书 | 区分单帧/轨迹、边缘/条件 |
| [车道 AEBS 同类训练流程](current/车道保持_AEBS同类训练流程与神经SBC_20261106.md) | 车道 SBC 方法与消融 | 区分学习式/几何式 |
| [停车场神经 SBC 专题](current/停车场神经SBC_构造证明与实验_20261103.md) | 停车场证明域 | 保留精确假设与证据 |
| [移动机器人三维视觉实验](current/移动机器人_三维视觉实验_20261108.md) | 三维视觉闭环 | 明确证书未通过 |
| [VISION-SLS/SPVC 完整技术主文档](current/VISION-SLS_SPVC_SBC当前框架_步骤原理实验与安全保证_20261029.md) | 完整技术参考 | 不替代当前结果总览 |
| [VISION-SLS/SPVC 工作汇报](current/VISION-SLS_SPVC_SBC_20261030.md) | 导师汇报 | 旧数字以最新修复为准 |

## 模型与实施证据

| 实验 | 运行说明 | 结果依据 | 对应控制结构 |
|---|---|---|---|
| AEBS RGB 2026-11-09（修复） | [发布包](../artical-F122/results/real_aebs_repair_20261109_release/README.md) | [验证报告](../artical-F122/results/real_aebs_repair_20261109_release/verification_report.json) | PPO 直接输出；轨迹 CP + 新非负 SBC |
| AEBS 2026-10-29（旧版） | [包说明](../artical-F122/results/vision_sls_spvc_sbc_pass_20261029/README.md) | [汇总](../artical-F122/results/vision_sls_spvc_sbc_pass_20261029/summary.json) | 旧灰度/固定图片回放；不与 11/09 RGB 混用 |
| 停车场 2026-11-03 | [包说明](../artical-F122/results/vision_sls_parking_20261103_neural_sbc_certificate_v1/README.md) | [证书审计](../artical-F122/results/vision_sls_parking_20261103_neural_sbc_certificate_v1/certificate.json)、[测试记录](../artical-F122/results/vision_sls_parking_20261103_neural_sbc_certificate_v1/validation.json) | 参考反馈与 PPO 建议＋在线后备检查；增广可达域神经 SBC |

代码入口与复现命令以对应包和完整步骤文档为准。文档整理没有改写冻结证据。历史候选包中的“未通过”继续针对它自身的闭环，不因最新包通过而改名或覆盖。

## 方法学习

下表是方法资料。归档报告里的实验数字和“下一步”属于原阶段，不作为最新状态。

| 学习内容 | 入口 |
| --- | --- |
| VISION-SLS 原论文与代码 | [论文与代码逐步详解](references/VISION-SLS_论文与代码逐步详解.md) |
| CP 为什么用、怎样排序与校准 | [CP 通俗详解（截至 10 月 27 日）](archive/2026-10-29_主线更新前文档/current/VISION-SLS_CP未见随机样本覆盖_通俗详解.md) |
| CP 的历史完整尝试与可行性分析 | [CP 综合报告（截至 10 月 27 日）](archive/2026-10-29_主线更新前文档/current/VISION-SLS_CP完整说明与可行性分析.md) |
| PPO 的输入、输出和训练职责 | [PPO 学习笔记](archive/2026-10-29_主线更新前文档/current/ppo.md) |
| Model-Free NBC 方法与源码 | [论文与代码详解](references/Model-Free_NBC_论文与代码详解.md) |
| BarrierNet 与 Opt-ODENet | [QP 部分对比](references/BarrierNet与Opt-ODENet_QP部分详细对比.md) |

## 论文与参考原件

| 原件 | 用途 |
| --- | --- |
| [原 SPVC 论文](references/720_file_Paper.pdf) | 原 SBC 框架依据 |
| [VISION-SLS 论文](<../VISION-SLS- Safe Perception-Based Control from Learned Visual Representations via System Level Synthesis.pdf>) | 视觉低维观测与 SLS 原设计 |
| [状态相关误差界说明](../VISION_SLS_状态相关误差界计算.pdf) | 包络拟合与 CP 衔接 |
| [CP-NCBF 论文](references/2503.17395v2.pdf) | CP 校准与神经屏障方法参考 |
| [硬约束训练论文](<references/Training with hard constraints- Learning neural certificates and controllers for sdes.pdf>) | 神经证书与控制器训练参考 |
| [潜在空间安全论文](<references/Safety Certification in the Latent space using Control Barrier Functions and.pdf>) | 历史世界模型分支参考 |

其余参考论文与解读保留在 `references/`；论文结论不等于本项目已实现的结果。

## 历史资料与时间线

| 阶段 | 内容与当时结论 | 入口 |
|---|---|---|
| 2026-11-02 停车场迁移 | 历史 PPO 图像 12/15，SBC 未通过 | [迁移原文](archive/2026-11-04_停车场阶段记录/停车场迁移实验_20261102.md) |
| 2026-11-03 规划反馈 | 参考、延迟观测器与残差 PPO 改善任务表现；此时尚无后备证书 | [规划反馈原文](archive/2026-11-04_停车场阶段记录/停车场规划反馈优化_20261103.md) |
| 2026-11-03 SBC 失败诊断 | 冻结旧控制器存在 CP 盒内超速反例 | [求解与反例原文](archive/2026-11-04_停车场阶段记录/停车场新控制器_SBC求解与反例_20261103.md) |
| 2026-11-03 受保护闭环 | 新增后备后的神经 SBC；完整条件见当前专题 | [最新专题](current/停车场神经SBC_构造证明与实验_20261103.md) |
| 2026-10-29 及更早 | 旧主线、CP 尝试、双路径、世界模型、QP 等 | [全部归档](archive/README.md) |

[本次归档说明](archive/2026-11-04_停车场阶段记录/README.md)记录旧路径、新路径和整理前快照。历史正文保留当时的“当前”“下一步”，不能作为今天的状态摘要。

## 研究草稿与维护约定

[根目录比较草稿](../比较.md)和[论文改进思路](references/论文改进思路.md)属于研究想法，尚未实施的比较不写进已完成结果。

当前状态统一从本页及总览进入；方法与结果变化时同步更新主文档/证明专题的范围说明，导师版只保留必要结论。冻结模型包保存其生成时的文档快照和散列，后续编辑使用 `docs/current/`，不修改包内快照来替换历史证据。早期全文归档，旧链接通过跳转页继续可用。

最新修复：[车道状态相关 v2](../artical-F122/results/lane_state_cp_20261109_repaired_v2/README.md)：99/100 分离拟合与校准，无 cap，实际状态集合 SBC 数值通过；旧数据回归不冒充新增盲测，旧 v1 继续撤回。
