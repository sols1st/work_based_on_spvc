# 车道前端收益证明：阶段一配对先导

固定 C 与联合 C 使用同一组新外部条件种子（校准 202611100101，留出 202611100102），但各自生成闭环轨迹。

| 指标 | 固定 C | 联合 C |
|---|---:|---:|
| q（归一化） | 0.0886902335 | 0.0864385599 |
| 校准覆盖 | 190/199 | 190/199 |
| 旧数据回归/新留出覆盖 | 96/100 | 95/100 |
| 留出覆盖率 95% 区间 | [0.901, 0.989] | [0.887, 0.984] |
| 留出 unsafe 轨迹 | 0 | 0 |
| 留出平均帧误差 | 0.022661 | 0.022016 |
| 留出最大帧误差 | 0.107082 | 0.110041 |
| 平均 |action| | 0.086861 | 0.083491 |
| 平均 |Δaction| | 0.118736 | 0.112698 |
| SBC 状态 | passed_numerical | passed_numerical |
| q 差 / 比 | — | -0.0022516737 / 0.974612 |
| 覆盖率差 | — | -0.010 |

配对初始条件一致：cal={'initial_states_equal': True, 'kappas_equal': True, 'episode_seeds_equal': True}，holdout={'initial_states_equal': True, 'kappas_equal': True, 'episode_seeds_equal': True}。

配对留出覆盖：both=92, fixed-only=4, joint-only=3, neither=1。

证书：
- 固定 C: {"passed": true, "checked_cells": 6199, "actual_witnesses": 0, "unresolved_cells": 0, "initial": {"passed": true, "unresolved": 0, "bound": 0.08582801041697322, "partition_area_error": 0.0, "history": [{"depth": 0, "unresolved": 0, "accepted": 256}]}, "unsafe": {"passed": true, "unresolved": 0, "bound": 23.653495792448606, "partition_area_error": 0.0, "history": [{"depth": 0, "unresolved": 0, "accepted": 1024}]}, "checks": [{"kappa": -0.01, "passed": true, "witnesses": 0, "unresolved": 0, "checked": 2042}, {"kappa": 0.0, "passed": true, "witnesses": 0, "unresolved": 0, "checked": 2070}, {"kappa": 0.01, "passed": true, "witnesses": 0, "unresolved": 0, "checked": 2087}]}
- 联合 C: {"passed": true, "checked_cells": 6135, "actual_witnesses": 0, "unresolved_cells": 0, "initial": {"passed": true, "unresolved": 0, "bound": 0.08582801041697322, "partition_area_error": 0.0, "history": [{"depth": 0, "unresolved": 0, "accepted": 256}]}, "unsafe": {"passed": true, "unresolved": 0, "bound": 23.653495792448606, "partition_area_error": 0.0, "history": [{"depth": 0, "unresolved": 0, "accepted": 1024}]}, "checks": [{"kappa": -0.01, "passed": true, "witnesses": 0, "unresolved": 0, "checked": 2026}, {"kappa": 0.0, "passed": true, "witnesses": 0, "unresolved": 0, "checked": 2051}, {"kappa": 0.01, "passed": true, "witnesses": 0, "unresolved": 0, "checked": 2058}]}

**边界：**这是一对检查点、一次新种子先导，不足以宣称总体显著优势；q 是归一化坐标，物理半宽是 `0.85q m` 和 `0.30q rad`。不能把 q 差直接归因于 C，也不能用留出覆盖 95/100 之外的读数替代条件 SBC。
