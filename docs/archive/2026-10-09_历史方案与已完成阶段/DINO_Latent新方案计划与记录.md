# DINO Safety-Latent 无CP实施方案：框架、数据流与实验记录

> 2026-09-29补充：后续执行请看 [DINO_PPO_SBC改进计划.md](DINO_PPO_SBC改进计划.md)。下文95.977%仅为原SPVC实现报告值；此前区间实现问题尚未消除，且该结果只涉及qψ代理路径，不能视为已确认的图像闭环安全保证。历史400/400是已有图像库回放结果，新的图像库隔离实验待用户运行。

更新时间：2026-09-28

文档用途：本文是当前DINO方向的唯一主实施文档，用于后续写代码、跑实验和向导师汇报。当前版本**不使用CP，不构造latent不确定集合，不加入额外验证方法**。实施严格围绕 `semantic+latent方案.pdf` 的主干：冻结DINO、32维安全latent、新控制器、SBC和QP。

## 0. 当前结论和进度

当前方案不再自己训练视觉骨干，也不把DINO特征压缩成1维距离后强行复用旧PPO。图像先经过官方预训练且完全冻结的DINOv2-S/14，再经过小型projection生成32维safety semantic latent。新控制器将直接读取这32维latent和车速。

已完成：

- [x] 官方DINOv2代码和权重的离线部署；
- [x] DINO的22,056,576个参数全部冻结；
- [x] `384→128→32` safety projection训练；
- [x] 部署路径 `图像→32维latent` 接口；
- [x] 离线训练路径 `qψ(d)→32维latent` 接口；
- [x] 8张真实数据集图像的双路径smoke test。

控制与SBC进度：

- [x] 输入 `[32维latent,车速]` 的新PPO，已完成三个20万步对照；
- [x] 混合latent PPO在 `qψ` 与真图像最近邻路径上均400/400成功；
- [x] 与混合latent PPO配套的SBC，原SPVC网格 `0/10000` 违反；
- [ ] PPO和SBC一起输入可微QP层；
- [ ] `图像→DINO→latent→PPO→SBC-QP→下一状态` 的完整闭环。

当前的准确结论是：**DINO表示层、同时适应两条latent路径的33维PPO和原SPVC口径的SBC已经跑通。PPO在两条评估路径上均400/400成功；SBC为 `0/10000` 下降违反，原代码计算的安全到达概率下界为95.977%。可微QP和真正在线DINO图像闭环尚未完成。**

## 1. 完整架构

### 1.1 部署时的真实闭环

```text
当前摄像头图像 o_t
        │
        ▼
冻结官方 DINOv2-S/14
        │ 384维视觉特征
        ▼
Safety Projection Pρ: 384 → 128 → 32
        │ 32维 safety semantic latent z_t
        ├──与当前车速 v_t 拼接
        ▼
新 Latent PPO πθ(z_t,v_t)
        │ 名义动作 a_nom
        ▼
SBC 给出安全约束
        │
        ▼
可微 QP 尽量保留 a_nom 并满足约束
        │ 最终动作 a_QP
        ▼
AEBS 环境/车辆动力学
        │
        ▼
下一物理状态 x_(t+1)
```

这是论文中真正要实现和评估的系统。部署时不输入真实距离，不运行辅助距离decoder，也不运行 `qψ`。

### 1.2 离线训练SBC时的小模型路径

```text
物理状态 x=[d,v]
        ├── d → qψ(d) → 32维latent近似 → Latent PPO → 动作
        └── x → SBC Bφ(x) → 动力学与SBC下降条件
```

`qψ` 是小型、确定性的“物理距离到latent”替代模型。它让网格状态训练和SBC计算不需要每次渲染图像、再运行2200万参数的DINO。当前只把 `qψ(d)` 当作点估计，不在其周围构造CP集合。

## 2. 为什么不输出1维归一化距离

旧控制器输入是 `[距离,速度]`，所以让DINO输出距离可以最少改代码，但与建议文档的semantic+latent思路不一致：

1. DINO的384维通用语义被强行压成一个标量；
2. 控制器仍被限制为只能看人工指定的“距离”；
3. 换到车道、姿态、遮挡等复杂任务时很难扩展。

因此当前保留32维latent作为控制器的真实视觉输入。距离标签只用来训练辅助decoder，检查latent是否保留安全信息；辅助decoder不接入控制闭环。

## 3. 已实现的表示学习

### 3.1 冻结DINO

- 模型：Meta官方 `dinov2_vits14`；
- 固定官方代码提交：`2302b6bf46953431b969155307b9bed152754069`；
- 权重SHA-256：`b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`；
- 参数量：22,056,576；可训练DINO参数：0；
- 32×32灰度图复制为RGB，双三次放大到224×224，再做ImageNet归一化。

### 3.2 Safety Projection

```text
384 → ReLU(128) → 32
```

训练目标：

- safety sufficiency：辅助decoder从latent恢复归一化距离；
- nuisance invariance：同一图像经过亮度、对比度和轻微噪声后，latent尽量一致；
- state separation：距离相差较大的物理状态不能映射到几乎相同的latent。

```text
L = L_safety + 0.2 L_inv + 0.1 L_sep + 1e-4 L_reg
```

当前外观增强只是400张小图像上的简化实现，不等于已覆盖真实天气、遮挡和摄像机变化。

### 3.3 物理状态到latent的替代模型

```text
qψ: normalized distance → 64 → 64 → 32维latent
L_q = ||qψ(d_i)-Pρ(E_DINO(o_i))||²
```

`qψ` 用于生成便宜的物理网格训练输入，并让SBC在物理状态空间快速计算控制动作。它不出现在真实部署路径中。

## 4. 当前实验结果

数据为 `Aebs/data/Downsampled.h5` 中的400张32×32灰度图像，距离范围为5到16米。固定划分为240张建模数据和两组80张留出评估数据；当前无CP版不用任何一组数据计算覆盖半径。

| 指标 | 结果 | 含义 |
| --- | ---: | --- |
| DINO参数 | 22,056,576，全部冻结 | 不自己训练视觉骨干 |
| Projection参数 | 53,408 | 将384维特征变为32维latent |
| 辅助decoder参数 | 1,089 | 只检查latent的安全信息 |
| `qψ`参数 | 6,368 | 离线物理距离到latent近似 |
| 最佳projection epoch | 171 | 由建模数据内部验证选择 |
| 留出评估辅助距离MAE | 0.1951 m | latent包含距离信息 |
| 留出评估辅助距离RMSE | 0.2954 m | 只是表示层诊断 |
| 增强视图latent MSE | 0.02336 | 初步外观不变性指标 |
| `qψ`每坐标latent RMSE | 0.27191 | 替代模型存在近似误差 |

8张图像的smoke test输出：图像路径 `[8,32]`，物理替代路径 `[8,32]`，`runtime_output_is_distance=false`。这些结果只证明表示层可运行，还不是控制成功率或安全性结论。

### 4.1 无CP Latent PPO闭环对照

两个PPO都沿用原AEBS动力学、reward、动作范围、终局定义和PPO超参数，固定随机种子7，训练20万步。评估使用初始距离15至16米、速度2.5至3.0 m/s的20×20固定网格，共400回合。没有使用CP、SBC或QP。

| 训练时的latent | 评估输入 | success | unsafe | timeout | 提前停在目标外 |
| --- | --- | ---: | ---: | ---: | ---: |
| `qψ(d)` | `qψ(d)` | 100.0% | 0.0% | 0.0% | 0.0% |
| `qψ(d)` | 距离最近的真实图像latent | 9.5% | 0.0% | 0.0% | 90.5% |
| 距离最近的真实图像latent | `qψ(d)` | 0.0% | 100.0% | 0.0% | 0.0% |
| 距离最近的真实图像latent | 距离最近的真实图像latent | 100.0% | 0.0% | 0.0% | 0.0% |
| 每回合50% `qψ`、50%真图像最近邻 | `qψ(d)` | 100.0% | 0.0% | 0.0% | 0.0% |
| 每回合50% `qψ`、50%真图像最近邻 | 距离最近的真实图像latent | 100.0% | 0.0% | 0.0% | 0.0% |

结论：两个单来源PPO只适应自己的训练latent，证明了明显的train–deployment mismatch。不改网络结构，只在每个训练回合开始时等概率选择 `qψ` 或真图像最近邻latent后，同一个PPO在两条路径均达到400/400成功。这是当前选定的合格PPO checkpoint。

### 4.2 无CP Latent PPO + 原SPVC SBC结果

SBC仍使用原SPVC的二维物理状态 `x=[d,v]`、`[2,16,8,1]` 网络、原噪声、原损失、原100×100网格和原通过口径。网格路径通过 `qψ(d)` 生成32维latent，再调用上述混合PPO。使用同一AEBS物理域上已训练SBC作为初始值，然后针对新latent PPO重新训练10 epochs并完整验证。

| 指标 | 结果 |
| --- | ---: |
| 验证网格 | 100×100 = 10,000状态 |
| 下降违反 | **0/10,000** |
| 初始区SBC上界 | 0.984850 |
| 不安全区SBC下界 | 19.896048 |
| 全域SBC下界 | 0.192187 |
| 原SPVC概率下界 | **95.977%** |
| checkpoint是否对应当次验证 | true |

循环在得到零违反和超过90%目标概率后，于下一次PPO更新之前停止，因此不存在“验证的模型和最后保存模型不同”的问题。这个结论是**原SPVC实现口径下的结果**；它尚未覆盖新的在线图像、未见外观变化或QP层。

## 5. 模块表

| 模块 | 输入 | 输出 | 是否训练 | 用途 |
| --- | --- | --- | --- | --- |
| DINOv2-S/14 | 224×224 RGB图像 | 384维CLS特征 | 完全冻结 | 通用视觉特征 |
| Safety Projection `Pρ` | 384维特征 | 32维latent | 已训练 | 提取安全相关语义 |
| 辅助decoder | 32维latent | 归一化距离 | 已训练 | 仅用于训练和诊断 |
| `qψ` | 归一化真实距离 | 32维latent近似 | 已训练 | 离线网格训练/SBC计算 |
| Latent PPO | 32维latent+速度 | 名义动作 | 未训练 | 完成AEBS任务 |
| SBC | 物理状态 | 证书值 `B(x)` | 已训练并按原SPVC验证 | 形成安全下降约束 |
| 可微QP | PPO动作+SBC约束 | 最终动作 | 未接入 | 最小改动PPO动作并尽量满足SBC |
| AEBS环境 | 物理状态+动作 | 下一状态/终局 | 固定 | 闭环任务 |

## 6. 新PPO如何训练

旧PPO只接受 `[归一化距离,速度]` 两个输入，新PPO需要接受 `[z_1,...,z_32,速度]` 共33个输入，因此不能直接复用旧网络。最简实施顺序为：

1. 新建33维observation的latent PPO；
2. 在AEBS训练状态中，用 `qψ(d)` 生成32维latent；
3. 将 `[qψ(d),v]` 作为PPO observation，沿用原AEBS动力学、reward、动作界和终局判定；
4. 确认新PPO在 `qψ(d)` 输入下能完成任务；
5. 用数据集图像经DINO得到的真实latent做回放评估；
6. 只有PPO闭环跑通后，才进入SBC和QP。

这一阶段不加CP、不加latent球、不加额外验证器。

## 7. SBC和QP如何接入

SBC仍定义在AEBS物理状态 `x=[d,v]` 上，控制动作由latent PPO给出：

```text
a_nom = πθ(qψ(d),v)
Bφ(f(x,a_nom)) ≤ Bφ(x)-epsilon
```

当前使用 `qψ(d)` 的确定性latent，不扩展为CP集合。

PPO先给出名义动作 `a_nom`，SBC给出局部安全约束，QP求解：

```text
min 0.5||a-a_nom||² + 0.5 rho xi²
```

并同时满足动作范围和SBC约束。QP最终要作为可微层接在PPO后面，使用最终动作的任务损失和SBC相关损失反向传播到PPO。

## 8. 代码和结果位置

| 路径 | 作用 |
| --- | --- |
| `Aebs/dino_latent/backbone.py` | 加载冻结的官方DINO，处理AEBS灰度图并输出384维特征。 |
| `Aebs/dino_latent/models.py` | 定义32维projection、辅助decoder、`qψ`和latent controller网络。 |
| `Aebs/dino_latent/train_representation.py` | 提取冻结DINO特征，训练projection、decoder和 `qψ`。 |
| `Aebs/dino_latent/encoder.py` | 提供 `图像→32维latent` 和 `距离→32维latent` 接口。 |
| `Aebs/dino_latent/smoke_test.py` | 检查运行时输出形状、数值和DINO冻结状态。 |
| `Aebs/dino_latent/train_latent_ppo.py` | 用原AEBS口径训练33维输入PPO，并在400个固定起点上交叉评估 `qψ` latent与真图像latent。 |
| `Aebs/dino_latent/spvc_policy.py` | 把 `qψ(d)+速度→latent PPO` 包装成原VT学习器所需的策略接口。 |
| `Aebs/dino_latent/run_spvc.py` | 使用新latent PPO运行原SPVC的SBC训练和100×100网格验证。 |
| `tests/test_dino_latent.py` | 检查DINO预处理、projection、`qψ`和控制器接口。 |
| `results/dino_safety_latent_stage1/` | 当前表示模型、数据配置、数值指标和latent数组。 |
| `results/dino_latent_ppo_stage2_no_cp_threads1_20260928/` | 使用 `qψ(d)` 训练的PPO、配置和交叉评估。 |
| `results/dino_latent_ppo_stage2_image_20260928/` | 使用距离最近真图像latent训练的PPO、配置和交叉评估。 |
| `results/dino_latent_ppo_stage2_mixed_20260928/` | 同时适应两种latent输入的合格PPO checkpoint和400起点评估。 |
| `results/dino_latent_spvc_stage3_20260928/` | 与合格latent PPO配套的SBC、VT策略state dict和验证指标。 |

`Aebs/dino_latent/calibrate_latent.py` 和 `results/dino_latent_cp_stage2/` 仅作为历史探索保留，**当前无CP主流程不调用它们，不使用其结果作为训练或通过条件。**

## 9. 当前限制

1. 原图像只有32×32，放大到224×224不会创造新细节，DINO的能力可能没有充分发挥。
2. 400张图像主要随障碍物距离变化，尚不能证明latent能在多天气、多摄像机、多遮挡下保持不变。
3. `qψ(d)` 是对图像latent的近似，其每坐标RMSE为0.27191；交叉评估已证明这个差异会导致控制结果完全改变。
4. 当前不使用CP，所以不声称 `qψ` 近似误差有概率覆盖保证。
5. PPO与SBC已在原SPVC口径下跑通，但QP和真正在线图像闭环尚未完成；95.977%不能解释为所有未见环境中的实际安全率。

## 10. 后续最简计划

### 阶段A：新latent PPO

- [x] 实现33维observation的AEBS环境包装；
- [x] 分别使用 `qψ(d)` 和真图像最近邻latent训练PPO；
- [x] 沿用原AEBS的reward、动作界、初始状态和终局规则；
- [x] 自动输出success、unsafe、timeout和平均步数；
- [x] 在400个固定起点上交叉评估两种latent输入；
- [x] 通过每回合混合两种latent输入，得到两条评估路径均400/400成功的同一PPO。

阶段A已完成：混合latent PPO在两条评估路径上均400/400成功、0%不安全、0%超时。

### 阶段B：SBC

- [x] 将latent PPO动作接入原SPVC的SBC训练路径；
- [x] SBC仍使用原物理状态、原区域和原判定口径；
- [x] 得到 `0/10000` 下降违反、合格区域分离和95.977%原SPVC概率下界。

### 阶段C：可微QP

- [ ] QP同时接收latent PPO的名义动作和SBC约束；
- [ ] 先做一步前向与梯度smoke；
- [ ] 再将QP加到训练图中；
- [ ] 比较PPO与PPO+QP的success、unsafe、timeout、return、QP干预率和slack。

### 阶段D：完整闭环

- [ ] 运行 `真图像→冻结DINO→32维latent→PPO→QP→环境`；
- [ ] 与原SPVC和旧距离语义版做同口径比较；
- [ ] 只报告已实际完成的结果，不把表示smoke写成安全证明。

## 11. 论文方向的准确表述

如果后续阶段成功，方法可表述为：

1. 使用冻结的预训练DINO替代原SPVC自训练的cGAN视觉生成/估计链；
2. 通过安全充分性、外观不变性和状态分离学习32维safety-relevant latent；
3. 训练直接使用latent和车速的新PPO，而不是把latent再退化成1维距离；
4. SBC仍在可解释的物理状态上建立安全条件；
5. PPO和SBC通过可微QP层在同一控制链中联合学习。

当前已完成第1、2、3项和第4项的原SPVC口径实验：DINO latent能训练出同时适应两条输入路径的PPO，配套SBC达到零网格违反和95.977%概率下界。第5项QP联合学习尚未完成。CP不属于当前方案。
