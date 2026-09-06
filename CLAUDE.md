# CLAUDE.md — SafePVC 论文改进项目

## 项目概述

目标：改进论文 *Provably Probabilistic Safe Controller Synthesis for Vision-Based Neural Network Control Systems*（SafePVC），
围绕"语义抽象 + 状态相关不确定性 + 分布鲁棒 SBC + 自适应反例驱动联合优化"重构方法。

- 原论文：`Provably Probabilistic Safe Controller Synthesis for Vision-Based Neural Network Control Systems.md`（另有 `720_file_Paper.pdf`）
- 源码：`artical-F122/`（**仅含 CARLA Emergency Braking 基准**，代码目录名 `Aebs`；论文提到的 X-Plane 基准代码未提供）
- 改进思路：`论文改进思路.md`（25 个阶段，已精简格式）
- **可行性分析与详细实施计划（本项目的核心文档）**：`改进方案可行性分析与实施计划.md`

## 仓库关键文件（源码结构）

| 文件 | 职责 |
|---|---|
| `artical-F122/Aebs/system/env.py` | 动力学 `d'=d-v·dt`，`v'=v-acc·dt`；状态/初始/不安全集；扰动盒 `noise_bounds` |
| `artical-F122/Aebs/system/estimate.py` | 用不同 latent `z` 统计真实扰动 Δs/Δv（**结果未接入验证**） |
| `artical-F122/Aebs/cGAN/train_mlp.py` | 训练"cGAN"生成器（**实为纯监督 MSE，无判别器**） |
| `artical-F122/Aebs/controller/StateEstimate_train.py` | 训练 `state_net`：图像(1024)→估计距离 `d̂`(1维) |
| `artical-F122/Aebs/controller/Controller_train.py` | PPO 训练控制器（**观测是真实状态 `[d_norm,v]`，非图像**） |
| `artical-F122/Combined_network/model.py` | `AebsEnd2EndNet = gen_net + state_net + controller_net`（VCLS） |
| `artical-F122/Aebs/VT/utils.py` | barrier `MLP`（tanh+softplus）、`martingale_loss`、`triangular` 噪声 |
| `artical-F122/Aebs/VT/train.py` | `VTLearner`：barrier `l_model` 与 controller `p_net` 训练 loss |
| `artical-F122/Aebs/VT/verify.py` | `VTVerifier`：均匀网格 + IBP(auto_LiRPA) + 局部 Lipschitz + 下降条件判定 |
| `artical-F122/Aebs/VT/loop.py` | 主循环：训 barrier → 验证 → 训 controller（Algorithm 1） |

状态空间 2 维 `[d_norm, v]`（`d∈[5,16]m`，`v∈[0,3]m/s`，`d_norm=d/std1`）；动作 `acc∈[-3,3]`；`dt=0.05`。
感知链路：`(z∈R^4, d) → gen_net → 图像(1024) → state_net → d̂(1维) → concat([d̂,v]) → controller_net → acc`。
**关键约束：现有生成器 `gen_net(z,d)` 的输出只依赖 `(z,d)`，不含 `v`，即合成图像物理上不编码速度。**

## 已确认的核心结论（本次调研的成果，勿重复推导）

改进思路指出的**三个理论缺口，在源码里都有直接证据**：

1. **视觉近似误差未进入证书**：`loop.py:112` 硬编码 `k_except_l=1.2`，`verify.py:412` 用 `K = 1.2 * delta`。
   Theorem 3.1 的 `K = τL_B(1+L_f√(1+(L_πL_g)²))` 未实现，`L_f/L_π/L_g` 从未估计。
2. **扰动被错误简化成 state-independent 固定分布**：`env.py:171-177` 扰动 = 手工指定全局均匀盒
   `(high-low)*0.01`；`estimate.py` 算出的随状态变化的 Δs/Δv 统计量**从未被 `env.py`/`verify.py` 引用**。
3. **有限样本估计无统计置信度**：`verify.py:319-349` 直接把离散 pmass 当真实概率，无 Hoeffding/Bernstein 裕量。

**额外发现的实现级问题**（改进时一并修复）：
- 训练用 `triangular()` 三角分布（`train.py:131`）vs 验证用均匀盒（`verify.py`），分布不一致。
- `verify.py:109` `refine_enabled=False` → "反例精化"实际关闭，`train('p')` 退化为整网格重训。
- `reach_prob` 不一致：learner 0.95 vs verifier 0.9（`loop.py:191/203`）。
- `square_l_output=True` 实际是 softplus（`utils.py:53-55`），命名误导。

## 改进方案概要（详见 `改进方案可行性分析与实施计划.md`）

核心模型：`s' = f(s,π(ξ)) + w`，`ξ∈C_ξ(s,z)`（conformal 语义集合），`w~Q(·|s)∈P(s)`（Wasserstein ambiguity set）。

六个阶段（MVP 合计约 9–14 周）：
- **阶段 1** 语义抽象 + split conformal 校准（把 `state_net` 升级为 `E_φ`，产出 `C_ξ`）★核心贡献
- **阶段 2** 语义控制器训练 + 不确定性增广
- **阶段 3** 状态相关扰动估计 + ambiguity set（替换 `env.py` 手工均匀盒）★核心贡献
- **阶段 4** 分布鲁棒 SBC（下降条件改 `sup_{ξ,Q} E_Q[B]`）★核心贡献
- **阶段 5** 自适应验证 + 联合反例 `(s*,ξ*,Q*,w*)` ★核心贡献
- **阶段 6** 理论整合（三层保证）+ 对照实验

**关键务实决策**（已定，勿重议）：
- 第一版语义 ξ 只取 `d`（或 `(d,c_v)`），`v` 继续直连控制器（因现有图像不含 v）。
- DRO 首选"区间/矩 ambiguity set"（支撑盒 + Hoeffding 均值置信区间），避免逐点 LP；训练 loss 可用 KL/χ² 球（可微）。
- 无限时域 conformal 只用 **marginal** 覆盖，采用"确定性 worst-case 感知契约"避免序列级 conformal 理论。

## 下一步（待办）

- [ ] 阶段 0：复现基线 `python -m Aebs.VT.loop`，记录 `max_reach_prob`/`hard_violations`/迭代数。
- [ ] 阶段 1：新建 `Aebs/semantic/`（encoder + conformal），改 `Combined_network/model.py`。
- 完整逐阶段验收标准见 `改进方案可行性分析与实施计划.md` §5。

## 约定

- 文档与交流用中文；代码注释沿用现有英文风格。
- 数学公式用 Markdown + LaTeX（`$...$` / `$$...$$`），与论文 md 一致。
- 涉及源码改动前，先读 `改进方案可行性分析与实施计划.md` 对应阶段的"改动点"，避免重复调研。
