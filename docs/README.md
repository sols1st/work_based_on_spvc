# 项目文档导航

整理日期：2026-10-29。最新主线集中到一份总览；过去的实验与计划保留在历史归档，避免多篇“当前”报告相互混淆。

## 当前主线

| 阅读目的 | 文档 |
| --- | --- |
| 了解最新框架、验证结果与结论范围 | [当前方案与验证结果](current/当前方案与验证结果.md) |
| 使用已通过的模型、复查连续域条件 | [模型包说明](../artical-F122/results/vision_sls_spvc_sbc_pass_20261029/README.md) |
| 核对原始结果和模型绑定 | [结果汇总](../artical-F122/results/vision_sls_spvc_sbc_pass_20261029/summary.json) |
| 安装、训练接口和部署细节 | [VISION-SLS 实现说明](../artical-F122/Aebs/vision_sls/README.md) |

当前场景为 AEBS，采用 DINO 与低维观测、PPO、非负 SBC。已有结果是限定模型下的数值区间检查通过；CP 包络条件、固定回放范围及浮点验证限制均见最新总览和模型包。停车场、双路径 R0/R1/R2、Transformer 和 CP-NCBF 的历史结果不与它合并引用。

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

## 历史资料

| 归档 | 内容 |
| --- | --- |
| [2026-10-29 主线更新前文档](archive/2026-10-29_主线更新前文档/README.md) | 16 篇旧状态/学习文档、两份旧 PDF、整理前入口与原始快照 |
| [2026-10-26 VISION-SLS CP 阶段文档](archive/2026-10-26_VISION-SLS_CP阶段文档/README.md) | 更早的阶段报告、重复导出与备份 |
| [2026-10-09 历史方案与已完成阶段](archive/2026-10-09_历史方案与已完成阶段/README.md) | 旧计划、汇报、采集/回传手册和 QP 记录 |
| [全部历史归档](archive/README.md) | 分类索引与早期散篇 |

本次只整理文档：实验代码、数据、模型和验证结果位置保持原状。归档目录提供[路径对照](archive/2026-10-29_主线更新前文档/归档清单.json)和原始文件快照；移动后的文档链接已重新定位。
