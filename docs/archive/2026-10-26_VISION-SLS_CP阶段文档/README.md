# VISION-SLS CP 阶段文档归档

整理日期：2026 年 10 月 26 日。这里用于追溯历史，不是日常阅读入口。

**现在只需阅读 [CP 加入 VISION-SLS 统一主报告](</Users/johann/project/work_based_on_spvc/docs/current/VISION-SLS_CP完整说明与可行性分析.md>)。** 主报告已经整合方法来由、符号、手算例子、代码链路、全部 22 项尝试、12 个回顾参数点、19 个图像批次及可行性分析。

本次移入 16 份重复/阶段说明：3 份原 current CP 文档，10 份复现/CP 阶段导出稿，以及 3 份逐步详解导出副本、备份和修订说明。逐步详解的现行参考版本仍在 docs/references，未删除独有内容。

没有删除或移动任何原始 JSON、NPZ、模型、图片、视频、日志或实验程序。归档可读稿增加历史提示，并把相对路径和相互引用迁移到有效位置；正文的实验数字与结论没有改写。

如需逐字恢复整理前原文，使用 [原文恢复压缩包](<归档前原文_可恢复.zip>)，按 [归档清单](<归档清单.json>) 中的 original_path 放回。压缩包每个原文的 SHA256 已核对；恢复时注意不要覆盖后来新建的文件。

历史稿中“当前”“下一步”“最新”只对当时有效。一些早期结论已经被后续状态相关实验限定，不能单独据此宣称当前 CP 安全替换已经完成。

## 原位置与可读归档对照

| 原位置 | 归档文件 |
|---|---|
| docs/current/VISION-SLS_CP误差界可行性.md | [VISION-SLS_CP误差界可行性.md](</Users/johann/project/work_based_on_spvc/docs/archive/2026-10-26_VISION-SLS_CP阶段文档/project_docs/current/VISION-SLS_CP误差界可行性.md>) |
| docs/current/VISION-SLS_CP误差界可行性_导师汇报稿.md | [VISION-SLS_CP误差界可行性_导师汇报稿.md](</Users/johann/project/work_based_on_spvc/docs/archive/2026-10-26_VISION-SLS_CP阶段文档/project_docs/current/VISION-SLS_CP误差界可行性_导师汇报稿.md>) |
| docs/current/VISION-SLS_CP全部尝试结果总账.md | [VISION-SLS_CP全部尝试结果总账.md](</Users/johann/project/work_based_on_spvc/docs/archive/2026-10-26_VISION-SLS_CP阶段文档/project_docs/current/VISION-SLS_CP全部尝试结果总账.md>) |
| outputs/vision_sls_10server_20261024/实验运行报告.md | [实验运行报告.md](</Users/johann/project/work_based_on_spvc/docs/archive/2026-10-26_VISION-SLS_CP阶段文档/exports/vision_sls_10server_20261024/实验运行报告.md>) |
| outputs/vision_sls_CP可行性_导师汇报_20261026/关于用CP计算VISION-SLS感知误差界的可行性_导师汇报稿.md | [关于用CP计算VISION-SLS感知误差界的可行性_导师汇报稿.md](</Users/johann/project/work_based_on_spvc/docs/archive/2026-10-26_VISION-SLS_CP阶段文档/exports/vision_sls_CP可行性_导师汇报_20261026/关于用CP计算VISION-SLS感知误差界的可行性_导师汇报稿.md>) |
| outputs/vision_sls_parking_direct_cp_20261025/停车场_直接CP实验报告.md | [停车场_直接CP实验报告.md](</Users/johann/project/work_based_on_spvc/docs/archive/2026-10-26_VISION-SLS_CP阶段文档/exports/vision_sls_parking_direct_cp_20261025/停车场_直接CP实验报告.md>) |
| outputs/vision_sls_quad_10server_20261024/无人机实验运行报告.md | [无人机实验运行报告.md](</Users/johann/project/work_based_on_spvc/docs/archive/2026-10-26_VISION-SLS_CP阶段文档/exports/vision_sls_quad_10server_20261024/无人机实验运行报告.md>) |
| outputs/vision_sls_全部尝试结果汇总_20261026/VISION-SLS_CP全部尝试_完整结果总账.md | [VISION-SLS_CP全部尝试_完整结果总账.md](</Users/johann/project/work_based_on_spvc/docs/archive/2026-10-26_VISION-SLS_CP阶段文档/exports/vision_sls_全部尝试结果汇总_20261026/VISION-SLS_CP全部尝试_完整结果总账.md>) |
| outputs/vision_sls_复现与CP替换_综合报告_20261025/VISION-SLS_实验复现与直接CP替换_详细报告.md | [VISION-SLS_实验复现与直接CP替换_详细报告.md](</Users/johann/project/work_based_on_spvc/docs/archive/2026-10-26_VISION-SLS_CP阶段文档/exports/vision_sls_复现与CP替换_综合报告_20261025/VISION-SLS_实验复现与直接CP替换_详细报告.md>) |
| outputs/vision_sls_建议原方案CP验证_20261026/VISION-SLS_按建议原公式CP_详细实验报告.md | [VISION-SLS_按建议原公式CP_详细实验报告.md](</Users/johann/project/work_based_on_spvc/docs/archive/2026-10-26_VISION-SLS_CP阶段文档/exports/vision_sls_建议原方案CP验证_20261026/VISION-SLS_按建议原公式CP_详细实验报告.md>) |
| outputs/vision_sls_状态相关CP验证_20261026/VISION-SLS_状态相关CP_服务器验证与导师汇报.md | [VISION-SLS_状态相关CP_服务器验证与导师汇报.md](</Users/johann/project/work_based_on_spvc/docs/archive/2026-10-26_VISION-SLS_CP阶段文档/exports/vision_sls_状态相关CP验证_20261026/VISION-SLS_状态相关CP_服务器验证与导师汇报.md>) |
| outputs/vision_sls_状态相关CP验证_20261026/此前汇报稿_常数CP实验阶段_保留.md | [此前汇报稿_常数CP实验阶段_保留.md](</Users/johann/project/work_based_on_spvc/docs/archive/2026-10-26_VISION-SLS_CP阶段文档/exports/vision_sls_状态相关CP验证_20261026/此前汇报稿_常数CP实验阶段_保留.md>) |
| outputs/vision_sls_覆盖率解释补充_20261026/CP_轨迹覆盖率与95%_从头解释.md | [CP_轨迹覆盖率与95%_从头解释.md](</Users/johann/project/work_based_on_spvc/docs/archive/2026-10-26_VISION-SLS_CP阶段文档/exports/vision_sls_覆盖率解释补充_20261026/CP_轨迹覆盖率与95%_从头解释.md>) |
| outputs/vision_sls_逐步详解_易读扩充_20261026/VISION-SLS_论文与代码逐步详解.md | [VISION-SLS_论文与代码逐步详解.md](</Users/johann/project/work_based_on_spvc/docs/archive/2026-10-26_VISION-SLS_CP阶段文档/exports/vision_sls_逐步详解_易读扩充_20261026/VISION-SLS_论文与代码逐步详解.md>) |
| outputs/vision_sls_逐步详解_易读扩充_20261026/修订前备份.md | [修订前备份.md](</Users/johann/project/work_based_on_spvc/docs/archive/2026-10-26_VISION-SLS_CP阶段文档/exports/vision_sls_逐步详解_易读扩充_20261026/修订前备份.md>) |
| outputs/vision_sls_逐步详解_易读扩充_20261026/本次补充说明.md | [本次补充说明.md](</Users/johann/project/work_based_on_spvc/docs/archive/2026-10-26_VISION-SLS_CP阶段文档/exports/vision_sls_逐步详解_易读扩充_20261026/本次补充说明.md>) |

## 保留在现行目录的其他专题

- 原论文与代码解读：用于查理论和原代码；其中 CP 历史片段以主报告为准。
- 图像到低维表示、停车场前端接 SPVC/PPO/SBC：包含其他独立实验，不属于本轮 SLS CP 的重复阶段稿。
- 原始实验目录：继续保存原位置，由主报告附录直接链接。

历史报告生成脚本保留以便追溯，但它们生成的是历史文稿，不是当前主报告。不要将重新生成的旧阶段稿作为现行入口。
