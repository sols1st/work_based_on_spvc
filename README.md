# SPVC 改进项目

文档整理日期：2026-11-09。项目将原 cGAN 视觉前端替换为 DINO 与 VISION-SLS 风格低维观测，研究 PPO 控制和非负 SBC。当前保留 **AEBS**、**停车场** 与 **车道保持** 三组结果，安全结论按各自闭环和验证范围引用。车道保持现直接复用 AEBS 的屏障网络进行训练，两个 SBC 种子通过，发布候选经两种区间方法复核；上一轮几何证书保留为对照。结论限误差包络和注册模型。

## 从这里开始

| 阅读目的 | 入口 |
|---|---|
| 当前文档入口（先看这个） | [docs/current/README.md](docs/current/README.md)：主线、最新结论、归档说明 |
| 看可视化网站 | [site/index.html](site/index.html)：总览、四个实验、图片库、数据/报告入口；本地服务器命令见 [site/README.md](site/README.md) |
| 仓库太大/怎么清理 | [GIT_CLEANUP.md](GIT_CLEANUP.md)：`.git` 体积、已跟踪大文件、`.gitignore` 与取消跟踪/历史清理建议 |
| 先看当前做到哪一步、有什么保证 | [当前方案与验证结果](docs/current/当前方案与验证结果.md) |
| 看 AEBS 真实 RGB 修复后的主结果 | [AEBS RGB 完整实验](docs/current/AEBS真实RGB图像_完整实验_20261108.md)：0.418359 m 轨迹 CP、新神经 SBC、292,024 区间通过 |
| 看 cGAN / 灰度 / 真实 RGB 前端对比 | [AEBS 前端对比](docs/current/AEBS前端对比_cGAN_灰度_真实RGB_20261108.md)：同一 test 集误差/CP 半径 |
| 看 AEBS 与车道的动力学、逐步流程、输入图像 | [AEBS 与车道逐步流程](docs/current/AEBS与车道保持实验_逐步流程与保证_20261108.md) |
| 看车道轨迹级 CP 与 SBC | [车道轨迹 CP](docs/current/车道保持_轨迹级CP校准与SBC衔接_20261107.md) |
| 重点核查停车场 SBC | [构造、证明与实验](docs/current/停车场神经SBC_构造证明与实验_20261103.md) |
| 看移动机器人三维视觉实验 | [移动机器人](docs/current/移动机器人_三维视觉实验_20261108.md)：图像跑通，完整 SBC 未通过 |
| 给导师介绍方法与实验 | [工作汇报](docs/current/VISION-SLS_SPVC_SBC_20261030.md)；完整技术见[主文档](docs/current/VISION-SLS_SPVC_SBC当前框架_步骤原理实验与安全保证_20261029.md) |
| 查论文/代码的状态维、观测维与实验设计 | [论文与代码实验对照](docs/references/论文实验与迁移分析_20261107.md) |
| 查历史设计稿/早期探针 | [2026-11-09 归档](docs/archive/2026-11-09_current_cleanup/README.md) |

## 当前实验

| 场景 | 已完成结果 | 适用范围与运行入口 |
|---|---|---|
| AEBS RGB（2026-11-09 修复） | 轨迹 CP 0.4183585684 m；新神经 SBC 292,024 区间 0 见证/0 未解决；12 组回放 4,800/4,800 成功 | 冻结图片回放分布；旧 0.2307 m tube 通过结论已撤回。[修复包](artical-F122/results/real_aebs_repair_20261109_release/README.md) |
| AEBS 旧版（2026-10-29） | 10/29 灰度/固定图片包仍作为历史证据 | 不能与 11/09 RGB 包混用。[旧包](artical-F122/results/vision_sls_spvc_sbc_pass_20261029/README.md) |
| AEBS 状态相关 CP（对照） | Bernstein b0(d)：平均 0.3217 m、最大 0.720 m；SBC 41,070 格 0/0 | 留出覆盖 92/100，低于常量版；不替换发布包。[对照包](artical-F122/results/real_aebs_state_cp_20261109_units_v2/README.md) |
| 停车场（2026-11-03） | 新增后备控制后的结构化神经 SBC，完成可达不变域论证与连续初态盒子检查；30 次图像仿真全部成功 | 初态各分量 ±0.02、30 步、零过程扰动模型；依赖在线后备检查，不是原无过滤器 SPVC 的全域证书。[模型包](artical-F122/results/vision_sls_parking_20261103_neural_sbc_certificate_v1/README.md) |
| 车道保持（2026-11-07） | 原 PPO＋AEBS 同类训练式 SBC；轨迹 CP 半径 0.0900931580 下双界审计通过 | 199 条轨迹校准、100 条留出；95/100 全程覆盖、100/100 无越界。限注册名义分布与 80 步，CP 为边缘覆盖。[结果与范围](artical-F122/results/lane_trajectory_cp_20261107_v1/README.md) |
| 车道状态相关 CP（已撤回） | 代码/报告不一致、留出分数误用校准残差 | **withdrawn / unverified**；不引用。[审计](artical-F122/results/lane_state_cp_20261109_v1/AUDIT.md) |
| 车道状态相关修复 v2 | 99 拟合＋100 校准，无 cap；实际集合 SBC 13,000 区间 0/0 | 原留出回归 93/100，非新盲测；旧 v1 继续撤回。[修复包](artical-F122/results/lane_state_cp_20261109_repaired_v2/README.md) |
| 移动机器人（2026-11-08） | 三维视觉闭环 24/24 安全到达；轨迹 CP 已算 | CP 盒内有 28/32 实际反例，完整 SBC 未通过。[专题](docs/current/移动机器人_三维视觉实验_20261108.md) |

性能测试、模型安全条件和 CP 概率前提分别报告；没有宣称真实车辆或完整浮点程序的无条件安全保证。详细条件见[统一总览](docs/current/当前方案与验证结果.md)。

## 目录分工

- `docs/current/`：当前总览、导师汇报、完整步骤和最新证明专题；早期报告原路径仅保留归档跳转。
- `docs/archive/`：按阶段保存旧实验、失败记录和整理前快照。
- `docs/references/`：论文原件与方法学习资料，不作为当前实验状态。
- `docs/assets/`：文档图片资源。
- `artical-F122/`：实验代码；`results/` 保存冻结模型、配置和验证证据。

本次文档整理的[归档与路径对照](docs/archive/2026-11-04_停车场阶段记录/README.md)可用于追溯。实验代码、模型和原始结果未因整理而移动或改写。
