# 论文原件与阅读笔记

从[项目文档总入口](../README.md)可回到实验报告。下表将论文原文、项目内的理论解读和具体实验分开，便于写论文时找到对应的出处。

## 本项目的三条方法来源

| 问题 | 原文 | 项目内对应材料 |
|---|---|---|
| 原 SPVC 怎样用视觉代理、控制器和随机屏障建立安全保证？ | [作者署名版 PDF](<Provably Probabilistic Safe Controller Synthesis for Vision-Based Neural Network Control Systems.pdf>)、[旧报告引用的匿名稿 PDF](720_file_Paper.pdf) | [原随机屏障与当前鲁棒屏障的数学条件](../framework/SBC_CBC_CBF_CP概念与安全保证详解_20261116.md) |
| VISION-SLS 怎样从图像得到低维观测，怎样训练状态代理 C？ | [VISION-SLS 原论文](<VISION-SLS- Safe Perception-Based Control from Learned Visual Representations via System Level Synthesis.pdf>) | [低维观测与可观测性训练](VISION-SLS原文解读_低维观测与联合训练及动力学可观测性_20261113.md)、[原论文与代码逐步详解](VISION-SLS_论文与代码逐步详解.md)、[状态相关误差界计算说明](VISION_SLS_状态相关误差界计算.pdf) |
| CP 如何用于神经屏障研究？本项目校准的是什么？ | [CP-NCBF 原论文 v2](2503.17395v2.pdf) | [视觉残差、轨迹 CP、PPO 与鲁棒屏障的连接](../framework/CP如何连接视觉代理_PPO与神经SBC_详细说明_20261116.md) |

## 相关方法与实验设计

| 主题 | 材料 |
|---|---|
| 视觉控制验证 | [半概率安全保证论文](<Learning Vision-Based Neural Network Controllers with Semi-Probabilistic.pdf>)、[潜在空间的屏障验证论文](<Safety Certification in the Latent space using Control Barrier Functions and.pdf>) |
| 强化学习与神经屏障 | [Model-Free NBC 原论文](Model-Free_Safe_Reinforcement_Learning_Through_Neural_Barrier_Certificate.pdf)、[论文与代码详解](Model-Free_NBC_论文与代码详解.md)、[随机微分方程的硬约束训练论文](<Training with hard constraints- Learning neural certificates and controllers for sdes.pdf>) |
| 可微优化与安全控制 | [BarrierNet](BarrierNet_Differentiable_Control_Barrier_Functions_for_Learning_of_Safe_Robot_Control.pdf)、[Opt-ODENet](Opt-ODENet.pdf)、[两篇工作中 QP 的详细对比](BarrierNet与Opt-ODENet_QP部分详细对比.md) |
| 研究设计 | [论文、状态维数与可迁移实验对照](论文实验与迁移分析_20261107.md)、[早期论文改进思路](论文改进思路.md)。已实际完成的实验另见[实验目录](../experiments/实验报告导读.md)。 |
