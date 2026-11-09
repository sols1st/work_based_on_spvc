# AEBS RGB 修复发布包（2026-11-09）

**新的训练式非负神经 SBC 已通过数值验证。** 保留原 PPO 和 RGB 前端，SBC 经两轮训练更新；不是直接沿用旧 SBC。

- 轨迹 CP 半径 **0.4183585683531419 m**；199 条校准、100 条独立留出；99/100 整条覆盖，100/100 行为成功。
- 连续域审计 **292,024** 个状态区间，深度 29 完成，**0 正漂移见证、0 未解决**。
- 初始上界 **0**，危险下界 **25.21363605977684**；Tanh 隐藏层、ReLU 输出的非负神经 SBC。
- 仿射 actor 界，观测 8 段、后继速度 64 段。没有缩小误差半径或安全任务。
- 12 组天气/颜色各 400 条，全库和 test 库分别 **4,800/4,800 成功**。

## 证据和范围

[验证报告](verification_report.json)、[轨迹 CP 报告](trajectory_cp_report.json)、[冻结协议](protocol.json)、[总摘要](summary.json)。本包的 actor、前端、SBC、误差合同与验证记录通过文件哈希绑定；`manifest.json` 列出包内全部文件（自身除外）。外部 DINO 主干权重与仓库仍需按前端检查点配置提供。

`trajectory_cp_report.json` 是 SBC 重验前产生的原始校准记录，其中 `certificate_passed=false` 当时正确；本包最终证书状态由 `verification_report.json` 与 `summary.json` 给出，未改写校准历史。

安全结论限定于停止 AEBS 动力学、给定初态、动作约束、米制观测包络和 float64 数值网络模型。点态不增保证包络内轨迹不进入危险集。CP 覆盖只针对已冻结图片库的注册回放分布；不包括新图像、实车、任意给定初态的条件覆盖或到达性。数值审计没有全面向外舍入，不是浮点程序形式化证明。

旧报告单位错误已在新合同修复；[旧 SBC 修正后的对照](old_sbc_corrected_results.json)仍为失败。较宽包络上的新 SBC 通过，由集合包含关系也适用于更小的静态 RGB/灰度误差集合，但不自动建立其他前端的新图像统计保证。

## 重验

在服务器 `/root/work_based_on_spvc/artical-F122`、`conda activate vt` 后运行，使用新输出目录：

```bash
python -m Aebs.vision_sls.verify_sbc_cells \
  --artifact-dir results/real_aebs_repair_20261109_release \
  --barrier-dir results/real_aebs_repair_20261109_release \
  --perception-contract results/real_aebs_repair_20261109_release/perception_contract.json \
  --actor-bound-method affine --actor-subdivisions 8 --successor-subdivisions 64 \
  --max-depth 40 --max-cells 4000000 \
  --output-dir results/real_aebs_release_reverify
```

[详细步骤与修复解释](../../../docs/current/AEBS真实RGB图像_完整实验_20261108.md)；[前端公平性对照](../../../docs/current/AEBS前端对比_cGAN_灰度_真实RGB_20261108.md)。
