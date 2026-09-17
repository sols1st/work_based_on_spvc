# SafePVC 论文改进项目

## 当前文档

建议按以下顺序阅读：

1. [`完整实验汇报.md`](完整实验汇报.md)：面向导师汇报的完整工作、实验效果、最终证明和局限总结。
2. [`改进实验记录.md`](改进实验记录.md)：实验计划、进度台账、运行命令和逐轮结果记录。
3. [`可投稿创新方向调研报告.md`](可投稿创新方向调研报告.md)：文献调研、创新性判断和推荐论文路线。
4. [`论文改进思路.md`](论文改进思路.md)：最初提出的完整改进要求，作为需求来源保留。
5. [`720_file_Paper.pdf`](720_file_Paper.pdf)：原论文。

## 代码

- [`artical-F122/`](artical-F122/)：实验代码与 `results/mvp` 结果目录。
- 新实验进度只更新到 `改进实验记录.md`，不再新建零散计划文档。

## 历史资料

`archive/` 保存仍有参考价值但已经不再代表当前计划的旧材料：

- `改进实验总结.md`：早期导师汇报稿，尚未包含 standalone PPO 和安全转移检查。
- `后续实验改进方案_深度调查.md`：早期针对 neural SBC/LP 路线的详细调查。

## 当前结论

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
- 控制器与确定性条件验证已收口；后续只需整理论文结果和单独补轨迹级概率风险口径。
- 固定网格或语义采样通过只能作为 go/no-go 依据，不能直接写成连续状态空间安全证明。
