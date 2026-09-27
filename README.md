# SafePVC 论文改进项目

> **研究主线更新（2026-09-22）：** 当前主线为“语义转换器 + PPO名义控制器 + SBC约束 + 可微QP安全层”。采用BarrierNet式安全层位置和Opt-ODENet式简化QP，先固定现有网络检查QP可行性，再逐阶段联合学习。详见 [`新方案_语义PPO_SBC可微QP路线.md`](新方案_语义PPO_SBC可微QP路线.md)。旧AEBS实验保留为基线，scenario-LP计划停止。

## 当前文档

建议按以下顺序阅读：

1. [`新方案_语义PPO_SBC可微QP路线.md`](新方案_语义PPO_SBC可微QP路线.md)：**当前唯一主计划**，包含两篇QP论文对比、数学接口、训练阶段、实验和止损条件。
2. [`完整实验汇报.md`](完整实验汇报.md)：旧路线已完成工作的完整汇报，作为AEBS基线和历史结果。
3. [`改进实验记录.md`](改进实验记录.md)：旧路线实验命令和逐轮结果记录。
4. [`实验通俗进展.md`](实验通俗进展.md)：旧实验的通俗术语解释。
5. [`论文改进思路.md`](论文改进思路.md)：最初的论文改进需求，作为来源保留。
6. [`720_file_Paper.pdf`](720_file_Paper.pdf)：原SafePVC论文。
7. [`BarrierNet_Differentiable_Control_Barrier_Functions_for_Learning_of_Safe_Robot_Control.pdf`](BarrierNet_Differentiable_Control_Barrier_Functions_for_Learning_of_Safe_Robot_Control.pdf)：可微HOCBF-QP参考。
8. [`Opt-ODENet.pdf`](Opt-ODENet.pdf)：简化CBF-QP与轨迹联合训练参考。

## 代码

- [`artical-F122/`](artical-F122/)：实验代码与 `results/mvp` 结果目录。
- [`artical-F122/Aebs/semantic_spvc/`](artical-F122/Aebs/semantic_spvc/)：原SPVC的受控最小替换版；只用语义转换器替代 `cGAN + state_net`，原PPO、SBC、噪声、网格、IBP和阈值保持不变。
- [`artical-F122/Aebs/dqp/`](artical-F122/Aebs/dqp/)：计划新增的SBC-QP实现位置；尚未开始编码。
- [`external/Certified-Reach-Avoid-via-Neural-Synthesis/`](external/Certified-Reach-Avoid-via-Neural-Synthesis/)：上一阶段参考代码，保留但不作为当前主实现。

## 旧路线已完成结论

- standalone PPO v2 在现有四种测试模式中均达到 100% success、0% unsafe、0% timeout。
- 多版 neural SBC 尚未证成，不再继续盲目调参。
- teacher → standalone PPO 固定网格安全转移诊断已通过：teacher/student 均为 4483/4483 sampled-pass。
- 近距离高速区加密检查发现 standalone PPO v2 有 19/18646 个局部反例，teacher 为 0 个。
- Saturation repair 已通过：四种 rollout 全部 100% success、0% unsafe/timeout，局部加密 violations=0，retention MAE=0.005445。
- 16段 semantic split-IBP 已完整证成局部 `6–8m × 2–3m/s` 动作条件：certified=46.37%、物理不可恢复=53.63%、unresolved=0。
- 首次全域 split-IBP 已运行：certified=85.47%、物理不可恢复=14.52%、unresolved=0.01%。
- 修正后全域验证仍有0.01%未决；新最差格在 `v≈0.65m/s` 离散制动步数切换点，确实需要最大制动，不再降低验证阈值。
- result 13 已学会低速边界且四类rollout全通过，但遗忘了原高速临界带：高速反例0→17，全域unresolved 627→1831，因此不采用。
- result 14 联合回放已成功保住两组边界：四类rollout和高速局部回归全通过；全域只剩248个未决cell、0.003440%面积。
- 64段semantic IBP继续改善：未决cell 248→135，未决面积0.003440%→0.001873%，最差动作界缺口0.061253→0.032828。
- 最终256段semantic split、max depth 9全域验证已完全通过：`local_action_condition_verified`、3624 evaluated、2002 certified、0 unresolved。
- 最终面积分解为85.53%可恢复且已证成、14.47%物理不可恢复、0%未决；85.53%不是安全概率。
- 控制器与确定性条件验证已经收口，这些结果作为新QP路线的基线。
- 固定网格或语义采样通过只能作为 go/no-go 依据，不能直接写成连续状态空间安全证明。

## 当前新路线状态

- 已完成BarrierNet与Opt-ODENet的QP层对比和适配分析。
- 已确定正确结构：PPO提供名义动作，SBC并行提供QP约束，QP输出实际动作。
- 尚未编写QP代码；下一步只完成固定网络下的SBC约束、一维解析QP、可微QP和可行性rollout。
- QP固定网络验证通过以前，不联合训练PPO/SBC，不进行参数扫描。
