# SafePVC 论文改进项目

文档已按“当前工作、历史归档、参考资料”整理。根目录只保留本入口，避免多个旧计划同时出现。

## 当前主要看这八份

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

旧“距离语义→PPO→SBC/QP”版本作为基线保留。最新主方向是“冻结DINOv2→32维safety latent→新latent PPO→SBC→可微QP”。DINO、projection、`qψ(d)` 和33维PPO已经跑通。两个20万步PPO对照在各自训练latent上都达到400/400成功，但交叉使用 `qψ` 与真图像latent时分别出现提前停车和100%不安全，说明两条latent路径尚不一致。当前不使用CP，历史CP代码和结果不参与主流程。
