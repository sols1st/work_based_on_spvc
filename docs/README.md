# 项目文档导航

更新：2026-11-04。本页统一管理阅读入口。AEBS 和停车场分别报告；文件名中的日期标识原始版本，最新补充日期和范围看各文件开头。

## 推荐阅读顺序

1. [当前方案与验证结果](current/当前方案与验证结果.md)：先了解两个实验、当前结果和保证边界。
2. [工作汇报](current/VISION-SLS_SPVC_SBC_20261030.md)：适合向导师介绍；第 1—4 节为 AEBS，第 5 节为扩展设计，第 6 节为停车场最新实测结果。
3. [完整步骤文档](current/VISION-SLS_SPVC_SBC当前框架_步骤原理实验与安全保证_20261029.md)：每步输入、处理、输出与原理；停车场完整流程在第 17 章，新旧区别在 17.14，保证在 17.15。
4. [停车场神经 SBC 专题](current/停车场神经SBC_构造证明与实验_20261103.md)：集中核查新受保护闭环的网络、不变域、CP 连接和实验。

## 每份当前文档的职责

| 文件 | 用途 | 维护原则 |
|---|---|---|
| 当前方案与验证结果 | 统一结果与状态总览 | 新结果先更新这里；不堆积完整历史训练日志 |
| VISION-SLS_SPVC_SBC_20261030 | 导师汇报与扩展设计 | 区分已经完成和后续设计；标题日期不表示只含旧结果 |
| VISION-SLS_SPVC_SBC当前框架_步骤原理实验与安全保证_20261029 | 完整技术主文档 | 保存完整流程、输入输出、理论和跨版本对比 |
| 停车场神经SBC_构造证明与实验_20261103 | 最新停车场证明专题 | 保留精确假设、公式及直接证据，供独立查阅 |

主文档与专题中有必要的重叠：前者面向完整流程，后者集中讲证明。它们引用同一冻结结果包，不代表两组不同实验。`current/` 中其余三份早期停车场文件只是归档跳转入口，完整正文已移至阶段归档。

## 模型与实施证据

| 实验 | 运行说明 | 结果依据 | 对应控制结构 |
|---|---|---|---|
| AEBS 2026-10-29 | [包说明](../artical-F122/results/vision_sls_spvc_sbc_pass_20261029/README.md) | [汇总](../artical-F122/results/vision_sls_spvc_sbc_pass_20261029/summary.json) | PPO 直接输出；包络/固定图片回放数值验证 |
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
| [VISION-SLS 论文](../VISION-SLS-%20Safe%20Perception-Based%20Control%20from%20Learned%20Visual%20Representations%20via%20System%20Level%20Synthesis.pdf) | 视觉低维观测与 SLS 原设计 |
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
