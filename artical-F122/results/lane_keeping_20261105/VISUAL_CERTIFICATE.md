# 视觉误差下的 SBC 证书：平底二次 SBC + 有界神经残差类

> **2026-11-07 更新：** 本文是 11 月 5 日的结构化证书分支，历史对照用；当前最新为原 PPO + AEBS 同类训练式神经 SBC + 轨迹级 CP 0.090093，见 `../lane_keeping_20261106_aebs_method_sbc_v1/` 与 `../lane_trajectory_cp_20261107_v1/`。


2026-11-05。**结论：在 patch-DINO 观测的 CP 包络（0.08 归一化，95% 校准/测试覆盖 ≥95%）下，`LQR + 0.5·clamp(B_flat(obs),0,1)·tanh(残差网络)` 这一类神经策略通过了视觉误差的自适应区间证书。**证书文件：`visual_sbc.json`；服务器细网格：`visual_sbc_server.json`。

**仍不是原始 PPO 的证书，也不是实车/连续曲率的证书。**

## 1. 证书对象

```
B(x)    = relu( x^T P x - c ),   c = 0.5
u(x_obs) = clip( u_LQR(x_obs, κ) + ε·clamp(B(x_obs),0,1)·tanh(r_θ(x_obs,κ)) )
ε = 0.5
```

- `B` 是平底二次 SBC：在 `x^T P x ≤ 0.5` 的原点邻域内 `B=0`，避免观测噪声在原点制造“B 从 0 变大”的假违例。
- 残差网络任意，只要输出是 `tanh`；证书不依赖其权重。
- `x_obs` 是 patch-DINO 的二维观测，带 CP 误差。

## 2. 证书内容

| 项目 | 值 |
|---|---|
| 观测 | 冻结 DINOv2 patch token（16×16）→ 空间 soft-argmax → `(ê,θ̂)` |
| 观测误差盒 | 归一化半径 0.08（patch CP 95% 半径 0.072，证书用 0.08 留余量） |
| CP 覆盖 | 校准 96.7%；测试 95.6%（半径 0.08） |
| 初态 | `|e|≤0.25 m`，`|θ|≤0.06 rad`；`B_quad≤0.99` → `B_flat≤0.49` |
| 认证集 `{B_flat≤0.49}` | `|e|≤0.3057 m`，`|θ|≤0.2616 rad`，在车道内 |
| 曲率 | `{-0.01, 0, 0.01}` 逐个验证 |
| 验证 | 自适应区间，160×160 起，最大深度 14 |
| 结果 | **0 unresolved**；19,150 严格格 + 8,442 平底格 |
| 独立复现 | 服务器 `vt`（Python 3.8 / torch 2.3.1）同样 0 unresolved；文件 `visual_sbc_server.json` |

**逻辑：**初态 `B_flat≤0.49`；对真实状态在认证集内、观测误差在 CP 盒内的所有组合，`B_flat(next)≤B_flat(current)`；因此 `B_flat` 不超过 0.49，状态留在认证集；认证集在车道内，所以不越界。平底区域 `B=0`，`B(next)=0` 也成立。

## 3. 验证方法

- 真实状态格子 × 观测误差盒（`state ± 0.08`）。
- LQR 动作区间是仿射精确；残差项区间 = `±0.5·clamp(B_flat_obs_max,0,1)`，覆盖任意 tanh 残差。
- 下一步状态用单调 `sin/tan` 端点；`B_flat` 区间由二次型盒界得到。
- 与认证集相交的格子检查 `max B_flat(next) ≤ min B_flat(current)`；否则一分为四，直到通过或深度 14。
- 数值区间（float64），不是形式化浮点证明。

## 3.1 前端对照：CLS vs patch

同一平底 SBC、同一 LQR+残差类、同一验证器下，**两个前端都能通过视觉证书**：

| 前端 | CP95 半径 | 视觉证书 | 认证集 | patch 图像闭环 |
|---|---:|---|---|---|
| DINO-CLS | 0.134 | 通过（`visual_cls_cp134_server.json`，160×160/深度 14，0 unresolved） | 相同：`|e|≤0.3057 m, |θ|≤0.2616 rad` | 6/6，mean `|e|` 0.029–0.041 m |
| DINO-patch | 0.072 | 通过（本文件，`visual_sbc.json`） | 相同 | **12/12**，mean `|e|` 0.022–0.037 m |

因此前端替换的正确主张是：**同覆盖率下 patch 把 CP 半径从 0.134 降到 0.072、把图像闭环从 6/6 提到 12/12**；不是"只有 patch 能过证书"。两个认证集相同，说明证书的可行性主要来自 SBC/控制器结构，前端的价值要用潜在误差和性能量化。

## 4. 训练后的平坦残差策略

`ppo_residual_flat/`：100k PPO，30/30 语义成功，mean `|e|` 0.0117–0.0129 m；patch-DINO 图像闭环 **12/12 成功**（6 个测试场景 ×2 回合），CP 覆盖 98.1–100%，mean `|e|` 0.022–0.037 m；随机光照/遮挡/模糊 9/9 成功。图像闭环是行为证据，不是这个证书的证明对象。

## 5. 复现

```bash
cd artical-F122
../.venv-vision-sls/bin/python -m Aebs.lane_keeping.certify_visual \
  --output results/lane_keeping_20261105/certificate/visual_sbc_repro.json \
  --radius-e .08 --radius-theta .08 --grid 160 --max-depth 14
```

## 6. 明确不覆盖

- **原始 PPO actor**：它的动作不在 `ε·B_flat` 缩放内。
- **实车图像/新场景**：CP 只覆盖 patch-DINO 在 `lane_keeping_v2` 名义渲染分布上的误差。
- **连续曲率**：只验证三个注册 `κ`。
- **形式化浮点**：数值区间，非形式证明。
- **目标到达**：只保证留在认证集，不保证按时到达。
- **前端主张**：主证书使用 patch；CLS 在 CP 0.134 下也通过同一视觉证书（见 3.1），不能写"只有新前端能过证书"。
