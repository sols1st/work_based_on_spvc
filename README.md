# SafePVC 论文改进项目

文档已按“当前工作、历史归档、参考资料”整理。根目录只保留本入口，避免多个旧计划同时出现。

**当前进度总结：** [DINO双路径方案：进度、结果与问题](docs/current/DINO双路径方案_当前进度与问题总结.md)。已完成24k数据、R2同距离外观对齐及适配R2的新PPO。R2验证相对latent误差由R1的8.695%降至7.633%，P95由0.6477降至0.5948；R2-PPO的q与12种train外观nearest回放均400/400成功，train精确配对图像独有unsafe由R1-PPO的3例降为0。尚无当前模型的SBC或安全保证。

**2026-10-05当前计划：** [基于现有24,000张数据的后续实验详细计划](docs/current/现有24000张数据_后续实验详细计划.md)。从数据验收、26/9/9距离区间划分、RGB DINO缓存开始，优先完成latent对齐对照，再推进PPO及误差契约/SBC；未实现入口已明确标记。

## 当前主要看这八份

2026-09-29下一步运行入口：[`DINO_PPO_SBC改进计划.md`](docs/current/DINO_PPO_SBC改进计划.md)。包含按距离分组的图像库隔离、用户执行命令、结果回传格式，以及表示对齐、新数据、SBC复核的分阶段计划。新实验尚待用户执行。

1. [`docs/current/DINO_Latent新方案计划与记录.md`](docs/current/DINO_Latent新方案计划与记录.md)
   **最新主方向。**按建议文档使用冻结DINO、32维safety latent、新latent PPO、SBC和可微QP；当前实施版明确不使用CP。

2. [`docs/current/当前实验效果_QP作用与后续实验路线汇报.md`](docs/current/当前实验效果_QP作用与后续实验路线汇报.md)
   **主文档。**当前结论、QP作用、trajectory-level CP结果、失败原因、未解决问题和下一步计划都维护在这里。

3. [`docs/current/QP_逐步实验记录.md`](docs/current/QP_逐步实验记录.md)
   **实验流水账。**保存每一步代码修改、服务器命令、输出数字和成功/失败判断。

4. [`docs/current/完整实验汇报.md`](docs/current/完整实验汇报.md)
   **汇报底稿。**用于向导师完整介绍从早期SBC、standalone PPO到语义PPO/SBC/QP/CP的全部工作。

5. [`docs/current/代码文件说明.md`](docs/current/代码文件说明.md)
   **代码索引。**说明主要代码入口、每个文件的用途及结果目录对应关系。

6. [`docs/current/hb.md`](docs/current/hb.md)
   **口头汇报稿。**可直接照着讲，后半部分含术语解释、常见追问和不能夸大的表述。

7. [`docs/current/整体框架与实验效果.md`](docs/current/整体框架与实验效果.md)
   **导师阅读版技术说明。**分别说明不带CP和带CP版本的独立框架图、原理、实验效果、关系与结论边界。

8. [`docs/current/CP-NCBF论文方法与整体框架详解.md`](docs/current/CP-NCBF论文方法与整体框架详解.md)
   **参考方法解读。**说明CP-NCBF原论文的方法、假设及其与当前实现的区别。

## 目录结构

```text
docs/current/       当前仍维护的8份文档
docs/archive/       已被主文档取代的旧计划、旧日志和分散解释稿
docs/references/    原论文、参考论文、最初改进需求和论文对比笔记
artical-F122/       实验代码与结果
figures/            汇报和论文架构图
external/           外部参考代码
```

`docs/archive/` 中的文件没有删除，只是不再作为当前结论入口。历史数字若与当前主文档不同，以当前主文档及对应 `metrics.json` 为准。

## 当前状态一句话

旧“距离语义→PPO→SBC/QP”版本作为基线保留。最新主方向是“冻结DINOv2→32维safety latent→新latent PPO→SBC→可微QP”。混合latent PPO已在 `qψ` 和真图像最近邻两条路径上均达到400/400成功、0%不安全；配套SBC在原SPVC 100×100网格上为 `0/10000` 下降违反，原代码计算的安全到达概率下界为95.977%。当前不使用CP；QP和在线DINO图像闭环仍是后续工作。
