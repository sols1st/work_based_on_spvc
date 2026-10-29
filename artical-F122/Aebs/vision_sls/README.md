# VISION-SLS 视觉前端替换 SPVC

**当前主线：按用户要求保留原 SPVC 的非负 SBC。** CP-NCBF 保留为独立对照实验，不作为默认后端，不将其有限回放统计界算作原 SBC 证书。标准训练入口仍使用原 `VTLearner` / `VT.utils.MLP` 的 SBC 与 actor 交替更新；标准部署不启用动作修正器。只有显式给评估命令传 `--certificate-dir` 才加载 CP-NCBF 分支。

**2026-10-29 已完成后续 SBC 适配和交替训练：** [通过模型包](../../results/vision_sls_spvc_sbc_pass_20261029/README.md)中的非负 SBC 已通过 CP 包络和固定图片回放两条连续域数值区间检查，分别覆盖 4,971,398 / 105,608 个检查区间，零真实反例、零未解决区间。模型包完整说明了输出激活、停止过程、条件期望的适用范围与浮点验证限制。原基线和下方历史记录保留，不覆盖。

新默认入口：`python -m Aebs.VT.loop`。它执行感知训练、PPO 初始化、原 SPVC 的 SBC/actor 交替训练、最终保存模型的复查和离线闭环评估。cGAN 历史入口保留为 `python -m Aebs.VT.loop --frontend cgan`，旧数据和模型不删除、不覆盖。

## 对照实验：CP 与离散 CP-NCBF（2026-10-28）

在下述已能到达目标的 VISION-SLS/SPVC 模型上，新增状态相关视觉误差包络、CP 校准、离散神经屏障和最小动作修正器。原视觉模型和 PPO 权重保持冻结。**固定 400 张图片库回放**中，512 条校准轨迹、1000 条独立随机初态测试轨迹和 441 个网格起点全部成功；冻结组合的轨迹分数校准分位数为 0。此回放分布下，400 步内、截至原终止事件的安全概率下界为 **99.1046%，置信水平 99%**。它不保证目标到达概率，不是原 SPVC 无限时域 SBC，也不覆盖新图像或实时 CARLA。方法、限制与命令见下方“CP / CP-NCBF 扩展”。

## 已修复提前停车（2026-10-28）

完整预算结果在 `results/vision_sls_spvc_goalfix_full_20261028/`：PPO 30000 步、101 轮 SBC/actor 更新、100×100 验证网格。更新前后，在原初始域 `[15,16] m × [2.5,3] m/s` 的 441 个网格点和 1000 个随机初态上，名义观测和留出图片回放均为 **1441/1441 成功**。任务成功条件、奖励和物理动力学均未放宽。详细过程见下方“故障定位与修复”。仍未取得形式安全证书。

## 对 AEBS 的适配

参考仓库 `VISION-SLS/expe/hardware/train_observability.py` 的 `loss_uni`：冻结 DINO 提取特征，再训练投影匹配归一化观测矩阵产生的 `Cx`。原实现还支持学习 C、可观测性损失及后续误差界和 SLS 控制。这里替换的是视觉前端，后端保留 SPVC 的 PPO/SBC，不移植停车场或四旋翼的 MPC。

AEBS 的物理坐标为 `x=[d/s, v]`，其中 `s` 是训练子集距离标准差。单帧训练图像只有距离标签，车速由原系统直接提供，因此固定单位范数 `C=[1,0]`，低维视觉观测是一个长度为 1 的向量。将它与实测车速拼为两维控制器输入。固定 C 防止可学习观测矩阵塌缩，避免把没有速度标签的静态图像当成车速传感器。组合观测 `[Cx,v]` 已能观察完整 AEBS 状态。对于未饱和的离散动力学，距离序列的可观测矩阵 `[C;CA]` 行列式为 `-dt/s`，也非零。

- 实际图像路径：32×32 灰度图 → 三通道/224×224/ImageNet 归一化 → 冻结 DINOv2-S/14 的 384 维 CLS → 128→128→1 GELU 投影 → 拼车速 → PPO actor。
- 离线验证路径：原 VT 网格 `[d/std1,v]` → 显式换算到训练尺度 → `Cx` → 同一 actor。
- 不再经过 cGAN、图像重建、旧 state_net 或 R2 的可学习 `q_psi`。
- 采用小型 DINO 和小投影是对 400 张 AEBS 小图及本机运行的适配；不加载停车场的 1536 维 DINO 模型或其 C/误差模型。
- 图像误差明确为 `e=projection(DINO(I))-Cx`；分别报告训练、验证、测试的误差。原 VT 保持名义路径；新增 CP-NCBF 扩展单独传播观测误差区间，因此原 VT 的名义结果仍不能作为真实图像闭环的安全概率证明。

## 安装与运行

在仓库根目录创建独立环境（不会改变已有 conda 环境）：

```bash
uv venv --python 3.11 .venv-vision-sls
uv pip install --python .venv-vision-sls/bin/python -r artical-F122/requirements-vision-sls.txt
uv pip install --python .venv-vision-sls/bin/python --no-deps ./artical-F122/auto_LiRPA
cd artical-F122
../.venv-vision-sls/bin/python -m Aebs.VT.loop --smoke --output-dir results/vision_sls_first_run
```

这里用普通安装而非 editable 安装 auto_LiRPA，以免仓库同名目录遮蔽包。需要已有 `external/dinov2/hubconf.py` 和 `external/dinov2_weights/dinov2_vits14_pretrain.pth`；可通过 `--dino-repo`、`--dino-weights` 指定位置。默认 CPU，也支持显式指定感知提取的 `--device cuda` 或 `--device mps`。

`--smoke` 使用 512 步 PPO、6×6 网格和一轮 VT（10 个 SBC epoch + 1 个 actor epoch），默认仍训练 250 个投影 epoch。它检查真实计算链路，不代表训练收敛或证书成立。

正式预算运行：

```bash
../.venv-vision-sls/bin/python -m Aebs.VT.loop \
  --feature-cache results/vision_sls_first_run/features.pt \
  --output-dir results/vision_sls_full_run
```

默认投影 250 epoch、PPO 30000 步、原 100×100 网格、VT 3600 秒/最大迭代索引 100。VT 的超时在迭代边界检查，单轮计算和最终复查可能超过该预算。可设置 `--epochs`、`--ppo-steps`、`--vt-seconds`、`--max-iteration-index` 和 `--actor-learning-rate`（默认 0.0003）。输出目录必须为空，避免覆盖之前的实验。缓存必须匹配数据、权重和预处理指纹。

## 数据与结果含义

按图像内容分组，60% 训练、20% 验证、20% 测试；重复图像不会跨组。归一化只使用训练集，投影按验证误差选最优模型。PPO 仅用训练集图像观测与名义 Cx 混合的 episode；测试图片不进入梯度训练或 actor 回退检查。固定留出图片库已用于故障诊断和前后对照，不能当成新采集的独立测试集。这是静态图片划分，不是对未见天气、相机或场景的泛化保证。

离线图像闭环按当前真实距离查找留出图片库中的最近图像，输入的是该图片产生的观测；真实距离仅用于离线取图，不进入部署控制器。结果单独记录最近邻距离偏差。此回放不是 CARLA 实时摄像机闭环。

产物：`observation.pt`（投影/归一化/C/来源）、`ppo_initial.zip`（VT 前初始化）、`policy.pt`（VT 后实际 actor）、`sbc.pt`、`features.pt`、`observations.npz`、`splits.npz`、`perception_metrics.json` 和 `metrics.json`。后者记录最终模型复查、参数是否更新、图像前向一致性、两条闭环结果。实际部署必须使用 `policy.pt`，不能误用 VT 前的 PPO。

## 图像部署接口

```python
import torch
from Aebs.vision_sls.deploy import VisionSLSController
controller = VisionSLSController('results/vision_sls_first_run', device='cpu')
# images: [B,32,32] 或 [B,1/3,H,W]，uint8 或 [0,1] float。
# speed: [B] 或 [B,1]，实测 m/s；无需传入真实距离。
action = controller(images, speed)
```

输出限制在 [-3,3]，正值表示制动。checkpoint 绑定投影的 SHA-256，防止加载不匹配的观测与 actor。移动实验文件后，可以给加载器显式传入 `dino_repo` 和 `dino_weights`。

## 验证

```bash
../.venv-vision-sls/bin/python -m pytest tests/test_vision_sls_spvc.py tests/test_minimal_latent_spvc.py -q
```

新增测试检查坐标换算、实际观测与名义状态分离、投影保存加载、以及创建区间验证器后 SBC 确实发生参数更新。公共 learner 取消了创建验证器时对训练参数的永久冻结，验证器修复了网格完全包含目标盒时的相交判断。名义 VT 算法和历史筛选阈值沿用现有实现；本次不宣称修复其所有理论或验证口径问题。

## 首轮实跑：历史失败基线（2026-10-28）

正式网格的结果目录：`results/vision_sls_spvc_20261028_final/`。预算为 400 张真实图像、250 个投影 epoch、8192 步 PPO、一轮原 SPVC（10 个 SBC epoch + 1 个 actor epoch）、100×100 网格，最后重新验证保存的模型。

- 80 张留出测试图像：观测距离 RMSE **0.16544 m**，MAE **0.12927 m**；验证集 RMSE 0.46765 m。
- SBC 和 actor 均实际更新；保存的 actor 从真实图像重新加载推理通过。
- 名义和留出图像回放各 25 条：均为 `stopped_safe_outside_goal`，即提前停车，**到达目标成功数为 0**。
- 最终原验证器记录 `0/10000` 硬违例，但其历史筛选条件实际仅检查 139 个状态；不可解释成全部状态的充分验证。
- 最终 SBC 边界：初始集上界 1.12744，不安全集下界 0.34559，全域下界 0.25084。区域分离不成立，**没有获得安全概率证书**，也未认证真实图像误差。
- 21 项相关测试通过。新增修复包括 SBC 参数被永久冻结、包含目标区域的网格被漏判为不相交；验证结果额外记录实际筛选后的检查数。

这证明替换后的完整软件链路可运行，不证明控制训练收敛或端到端安全。

已有模型无需重新训练，可通过已替换的标准回放入口评估：

```bash
../.venv-vision-sls/bin/python -m Aebs.system.combined \
  --artifact-dir results/vision_sls_spvc_goalfix_full_20261028
```

它重新从真实留出图像计算 DINO 观测并加载最终 actor，写入 `evaluation_reload.json`。旧回放可用 `--frontend cgan` 显式调用，代码保存在 `Aebs/system/combined_legacy.py`。


## 故障定位与修复

第一轮的“提前停车”不是视觉输入无法控制，而是 SBC 更新把已能完成任务的 PPO 破坏了。固定同一个观测模型、动力学和成功判定后：

| 阶段 | 名义初始网格 | 留出图片回放 | 典型行为 |
| --- | --- | --- | --- |
| 首轮 PPO，进入 SBC 之前 | 25/25 成功 | 25/25 成功 | 正常接近并制动 |
| 首轮 SBC actor 更新后，学习率 0.05 | 0/25 成功 | 0/25 成功 | 全程接近最大制动 3.0，提前停车 |
| 修复后，连续 5 轮更新 | 25/25 成功 | 25/25 成功 | 保持目标到达能力 |

独立学习率对照使用同一个已保存 SBC、同一个初始 actor、同一套网格和原损失。0.05 会复现提前停车，0.0003 和 0.0001 都保留名义/训练图片 25/25 成功。证据在 `results/vision_sls_goal_diagnosis_20261028/stage_comparison.json` 和 `actor_lr_ablation.json`。学习率对照只用训练图片做比较。

修复只作用于新 VISION-SLS 入口：

1. SBC 对 actor 的更新学习率改为 0.0003，原 cGAN 入口和公共 learner 的默认学习率保留。
2. 每个 actor 更新周期前后，在固定的 32 个训练初态上分别检查名义与训练图片闭环。若任一已成功轨迹失去成功，或增加不安全轨迹，恢复该次更新前的 actor 参数及 Adam 状态。它是有限训练样本上的功能回归检查，不是安全证书；不使用验证/测试图片。
3. 保存更新前的训练轨迹、每次更新的接受/拒绝记录、最终模型复查及逐轨迹终点，避免只报告最后一个汇总数字。
4. 批量回放与原 `ObservationEnv.step` 的奖励、逐步浮点存储、终止规则做了对照测试，扩大的评估没有改写成功条件。

完整预算结果：101 次 actor 更新全部被接受，actor 和 SBC 均实际改变，并非冻结控制器或一直回滚来取得成功。最终模型在 21×21 网格及种子 20261028 的 1000 个随机初态上，名义与留出图片两路均 1441/1441 成功。测试范围仍是原 SPVC 初始域，图片回放仍是固定 80 张留出图片的最近邻查询，不是实时 CARLA 或新天气泛化实验。

最终形式验证仍不成立：初始集上界 13.92847，高于不安全集下界 10.36498；原筛选器检查 53 个状态，记录 1 个硬违例（历史容差会将其标为 decrease satisfied）。因此不把“任务成功”写成“获得概率安全保证”。

新增产物：`actor_update_history.json`、`pre_vt_training_rollouts.json`、`expanded_benchmark.json`、`witness_trajectory.json`。25 项相关测试通过。

复现扩大测试（只评估冻结模型，不更新参数）：

```bash
../.venv-vision-sls/bin/python -m Aebs.vision_sls.benchmark \
  --artifact-dir results/vision_sls_spvc_goalfix_full_20261028
```

## CP / CP-NCBF 扩展

参考用户提供的 `VISION_SLS_状态相关误差界计算.pdf` 中式（31）的误差包络拟合与 CP 校准，以及 `docs/references/2503.17395v2.pdf`（CP-NCBF，2025-05-17 v2）第 3 节的三类约束分数、开发阶段鲁棒裕度调整、Beta 置信界与 CBF-QP。这里实现的是适配 AEBS 的离散版本，不直接套用连续时间 Lie 导数结论。

### 从视觉误差到动作修正

1. **视觉尺度**：物理单位残差 `r=|s*phi(I)-d|`。仅用原训练集 240 张图片，通过带松弛项的凸优化拟合正的三次 Bernstein 多项式 `b0(d)`；取式（31）的 `gamma=0`，松弛惩罚 20。正系数和非负基保证整个 `[5,16]m` 域内为正。
2. **感知 CP 参考**：原 80 张测试图片拆成 40 张校准、40 张审计。分数为 `r/b0(d)`，有限样本分位次序 `ceil((n+1)*(1-alpha))`，`alpha=.05`，得到 `q=2.522199` 和 `bCP=q*b0`。覆盖分别为 39/40、40/40。该库已用于前序诊断，采集可交换性未验证，因此它是参考误差包络，不能宣称独立新图像 95% 覆盖。对应 99% 置信下界也只有 84.5267%，不能把目标覆盖率当成固定校准集的置信下界。
3. **鲁棒传播**：观测区间 `[d-bCP(d),d+bCP(d)]/s` 经过冻结 PPO 的区间运算，得到动作区间。按原 `dt=.05` 动力学传播；距离下一步仍为 `d-.05*v`，速度由动作决定并裁剪到 `[0,3]`。不再把视觉误差人为加成独立的距离/速度过程噪声。
4. **离散 NCBF**：`h(d,v)=d-6-[g(v)-g(.5)]-.02`，其中 `g` 为正权重的凸单调 ReLU 网络。约束为初始集 `h>=0`、不安全集 `h<=-.02`、活动安全状态 `h(next)>=(1-.05)*h(now)`。按原成功或安全停车终止语义构造吸收终点。开发集可用于选择模型和调整裕度；冻结后才生成最终校准集。初始集最小值约 **6.13939**，不安全集最大值 **-0.02**，由结构单调性计算整个矩形区域的界。
5. **最小动作修正**：部署仅接收视觉估计和实测车速。先反解 `|hat_d-d|<=bCP(d)` 的距离区间；检查 `|bCP'|<1` 后，用二分求保守距离下的最小可行动作，再将 PPO 动作投影到可行制动区间。单一制动输入使标量最小二乘问题可用单调二分求解，不依赖通用 QP 求解器。数值裕度 `1e-5`，动作仍在 `[-3,3]`。无解时返回最大制动，并在 `policy.last_info` 明确报告 `feasible=False`；观测超出包络定义域另记为 `observation_in_contract_domain=False`，两者都不能视为安全已认证。

没有动作修正器时，即便区域分离成立，最终 8192 状态校准分位数仍为 **0.00495014 > 0**，703 个校准样本、653 个审计样本存在正违例。这说明单纯给误差套 CP 并不会自动修好闭环屏障条件。接入动作修正器后，新种子的 8192 校准状态与 8192 审计状态，正分数均为 0。随机状态结论仍然不能推成整条轨迹保证，区间计算也未做浮点外向舍入的形式验证。

### 轨迹级校准与适用范围

在上述模型与控制流程冻结后，先登记协议，再从初始域 `[15,16]m × [2.5,3]m/s` 均匀独立采样 512 个初态。每条轨迹最多 400 步，沿用原终止条件和 float32 状态存储。渲染器是全部 400 张固定图片按距离最近邻查询，含训练图片；其意义是固定回放世界的验证，不是未见图片测试。

轨迹分数取初始屏障和所有活动步离散屏障违例的最大值，并与 0 取最大；若出现 unsafe 或 out_of_domain 事件，分数至少为 1。安全终止后不再模拟后续运动。因此 **分数为 0 蕴含这个停止系统在该有限时域内未发生上述违例**，但不蕴含到达目标。采用预先固定的最大次序统计量：`n=512, alpha=1/513, rank=512`。实际最大分数为 0；在 IID 初态、冻结模型及固定回放世界假设下，Beta 下置信界为 `Beta.ppf(.01,512,1)=0.9910458562`。这是将同一有限样本方法应用于“整条轨迹作为一个样本”的扩展，并非论文随机状态定理自动给出的轨迹结论。

| 检查 | 样本数 | 到达目标 | 最大轨迹分数 | QP 无解步数 | 视觉误差超界步数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 冻结后校准，种子 9051 | 512 | 512 | 0 | 108 | 696 |
| 独立初态测试，种子 9052 | 1000 | 1000 | 0 | 272 | 1359 |
| 21×21 初始网格 | 441 | 441 | 0 | 127 | 580 |

每条轨迹都至少出现过一次视觉误差超界；上述 QP 无解步也均伴随观测超出包络定义域。闭环实际安全与点态误差包络成立是两件事。这里的轨迹统计界直接校准最终控制器在固定回放世界中的真实分数，没有用无效的点态覆盖乘积推算轨迹安全，也没有隐藏最大制动回退。要推广到新图像，需要新的独立轨迹/场景采集及匹配的最终校准；改图片库、控制器、误差包络或动力学后，不能沿用此统计界。

结果目录：

- `results/vision_sls_cp_ncbf_20261028/`：`perception_contract.json`、`ncbf.pt`、未加动作修正的 `report.json`、实际图像加载检查 `evaluation_reload.json`。
- `results/vision_sls_cp_ncbf_filtered_20261028/`：预登记 `protocol.json`、逐轨迹 `report.json`、`scores.npz` 和简要 `summary.json`。
- `results/vision_sls_certificate_audit_20261028/audit.json`：原验证器的区间调用、区域分离和噪声口径审计。旧路径的“decrease satisfied”并不等于完整证书成立；此次扩展也不把原 SBC 的失败改写为成功。

### 复现与使用

在 `artical-F122` 下执行，输出目录必须不存在：

```bash
# 训练包络和结构 NCBF；视觉和 PPO 均读取已有成功基线。
../.venv-vision-sls/bin/python -m Aebs.vision_sls.run_cp_ncbf \
  --artifact-dir results/vision_sls_spvc_goalfix_full_20261028 \
  --output-dir results/my_cp_ncbf

# 冻结组合，独立校准状态约束及完整回放轨迹。
../.venv-vision-sls/bin/python -m Aebs.vision_sls.calibrate_cp_filter \
  --artifact-dir results/vision_sls_spvc_goalfix_full_20261028 \
  --certificate-dir results/my_cp_ncbf \
  --output-dir results/my_cp_ncbf_filtered

# 标准评估入口：重新计算 80 张真实图片的 DINO 观测，再检查闭环。
../.venv-vision-sls/bin/python -m Aebs.system.combined \
  --artifact-dir results/vision_sls_spvc_goalfix_full_20261028 \
  --certificate-dir results/vision_sls_cp_ncbf_20261028
```

实际图像部署：

```python
from Aebs.vision_sls.deploy_cp import VisionSLSCPController
controller = VisionSLSCPController(
    'results/vision_sls_spvc_goalfix_full_20261028',
    'results/vision_sls_cp_ncbf_20261028', device='cpu')
action = controller(images, measured_speed)
status = controller.policy.last_info
```

加载器检查 NCBF、误差包络、视觉投影与 PPO 的文件哈希绑定。真实 80 张留出图片重新前向通过，图像直达动作与缓存观测达动作的最大差为 `5.96e-6`；名义与留出图片各 25/25 成功。这个 80 张库的加载检查不继承全部 400 张库的轨迹统计界。原 `evaluation_reload.json` 不被覆盖，新结果写到 `--certificate-dir`。

```bash
../.venv-vision-sls/bin/python -m pytest \
  tests/test_vision_sls_cp_ncbf.py tests/test_vision_sls_spvc.py \
  tests/test_minimal_latent_spvc.py -q
```

24 项测试通过，包括误差包络反解、最小制动可行性、区间动作下界、无解与不安全轨迹显式记录、模型绑定，以及原环境终止/奖励一致性。

## 保留原 SPVC SBC 的 CP 接口检查

按用户要求，主线保留非负随机屏障 `B`，不切换为有符号 NCBF，也不额外修改部署动作。保持的是 SBC 方法、网络结构与期望不增条件；更换视觉前端后，候选 SBC 权重可以继续训练，原证书不能直接继承。现有默认训练/部署一直保留此路线。

新增 `sbc_cp.py` 和 `audit_sbc_cp.py`，直接读取原 `sbc.pt`：`MLP([2,16,8,1], tanh, square_output=True)`。这里历史参数名虽叫 square_output，实际输出激活是 softplus。SBC 仍读取原物理状态归一化坐标，观测投影使用其训练集尺度；两者显式换算。所有区域与漂移检查均使用该同一个 SBC，不读取 `ncbf.pt`，不修改 PPO 动作。`perception_contract.json` 只是独立的感知误差包络，尽管本次复用的文件位于先前 CP-NCBF 实验目录中，它不含 NCBF 网络或动作过滤逻辑。

CP 包络经 PPO 和原 AEBS 动力学得到下一步状态区间，再对原 `B` 做区间传播。检查 `sup B(next)+epsilon<=B(current)` 是对“误差分布完全支持于该区间”时条件期望约束的充分条件。**普通边际 CP 覆盖并不满足该前提**，不能假设区间内均匀分布，也不能忽略区间外尾部。若以后采用 `(1-delta)*U+delta*M` 上界，需要另行建立每个相关状态下的条件尾概率上界 `delta`，及尾部屏障上界 `M>=U`；现有静态图片 CP 不提供这样的条件保证。该接口因此明确返回 `expectation_assumption_established=False`，不会伪造 SBC 安全概率。

只读实跑结果 `results/vision_sls_original_sbc_cp_20261028/report.json`：原视觉、PPO、SBC 文件哈希均未改变。初始集数值区间上界 13.9300，不安全集下界 9.1965，未通过原 `B(init)<=1`、`B(unsafe)>=20` 的 95% 目标检查。10204 个状态中按原 `B<=20` 且非不安全状态规则检查 7276 个，5500 个鲁棒充分条件未解决；其中 1180 个存在名义转移严格漂移违例。正区间上界不自动意味着真实违例，因此两者分别报告。没有沿用原验证器的窄带筛选或 0.1% 容差。

这里保留原 `epsilon=.1` 严格漂移条件，包括安全终点和停车状态，并显式记录 16 个可能离开建模距离域的后继状态。安全停车的吸收语义与原无限时域严格漂移定理需要明确衔接，不能为了通过而静默删除这些状态。这次检查不宣称取得新的 SBC 证书，也没有重新训练候选网络。

```bash
../.venv-vision-sls/bin/python -m Aebs.vision_sls.audit_sbc_cp \
  --artifact-dir results/vision_sls_spvc_goalfix_full_20261028 \
  --perception-contract results/vision_sls_cp_ncbf_20261028/perception_contract.json \
  --output-dir results/my_original_sbc_cp_audit

../.venv-vision-sls/bin/python -m pytest \
  tests/test_vision_sls_original_sbc.py tests/test_vision_sls_spvc.py -q
```

11 项测试通过，包括实际原 SBC 输出的区间包含性、原坐标尺度、停车状态严格漂移检查以及检查不修改权重。
