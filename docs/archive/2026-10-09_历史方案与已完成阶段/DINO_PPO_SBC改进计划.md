# DINO → PPO → SBC 改进计划与运行手册

> 2026-10-05：当前执行计划已更新为 [现有24,000张数据后续实验详细计划](现有24000张数据_后续实验详细计划.md)。先数据验收、固定划分、RGB特征缓存和latent数据对照；下文旧400张数据的PPO命令保留为历史记录，不是本轮新数据入口。

## 当前下一步：B组表示 → 新PPO → 准确配对诊断

用户已完成表示A/B：validation latent RMSE由0.256620降至0.093820，相对误差由0.451398降至0.196262；图像距离RMSE由0.524770 m降至0.419987 m。B组状态分离约束违反比例由14.76%增至33.68%，因此选择B作为控制候选，不宣称全部指标改善。

新增统一入口 `Aebs/dino_latent/run_aligned_ppo.py`。运行顺序：

1. 加载B_joint权重，核对原距离分组manifest和DINO权重哈希。
2. 只计算train/validation的图像latent。缓存使用原图像行号；test行仅占位，available_indices阻止调用占位值。
3. 生成绑定B权重和新缓存的派生manifest，距离组索引保持相同。
4. 从头训练33维PPO，20万步、seed=7，每回合50% qψ、50% train图像库。环境和qψ均使用B checkpoint中的新距离尺度，保持物理距离一致。
5. 仅评估qψ和train图像库的400起点闭环；跳过稀疏validation图像库闭环，不再把它当泛化门槛。
6. 固定新PPO，对train/validation每张图像自身距离和31个速度做准确配对的一步诊断。

```bash
cd /root/work_based_on_spvc/artical-F122
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/opt/miniconda3/bin/conda run --no-capture-output -n vt \
python -m Aebs.dino_latent.run_aligned_ppo
```

输出根目录：`results/dino_improvement/06_B_joint_ppo/`。PPO权重与回合指标在ppo/，一步诊断在matched/。发回从 `[B representation -> fresh PPO -> matched diagnostic]` 开始的全部摘要即可；需要详细分析时提供ppo/metrics.json、matched/metrics.json和matched/samples.json。

判断：先看新PPO在qψ/train回放能否保持任务能力，再看validation动作误差和单侧unsafe是否改善。前一轮准确配对参考值：validation MAE=0.163265、max_abs=2.374193、image_only_next_unsafe=1、both_next_unsafe=94。该参考来自旧表示，不是同一严格表示A/B下训练的PPO，因此这里只能作为推进诊断；若要归因于联合表示训练，之后还需用A_sequential表示训练同设置PPO作为对照。

不要求靠调参把动作误差变成零；qψ也不是安全动作真值。一步结果变好不能替代独立在线闭环。此次不训练SBC，也不接QP。B采用train距离尺度，旧VT状态归一化适配尚待后续处理，不能直接套旧run_spvc入口。

同一命令可跳过已经完整落盘且哈希匹配的阶段。中途失败留下非空子目录时会拒绝覆盖，请保留报错并指定新output-dir重跑，不自动删除已有结果。正式运行由用户执行，助手只做代码测试。

## 2026-09-30当前执行：集中改善两条latent的一致性

最新一步诊断：train动作MAE=0.183695，validation=0.163265；validation有一个图像路径独有的下一步unsafe，状态d=6.005025 m、v=0.6 m/s，图像制动1.254506、qψ制动2.033834。94个两条路径都unsafe的状态按现有动力学即使最大制动也不能避免下一步unsafe。这些结果只用于诊断，不作为正确动作监督。

当前暂停PPO和SBC调参，先运行表示层A/B。新增 `Aebs/dino_latent/align_representation.py`，默认一次执行两组：

| 项目 | A_sequential | B_joint |
| --- | --- | --- |
| DINO | 官方预训练、冻结 | 相同 |
| 图像划分 | 已登记距离组，仅train更新，validation选模型 | 相同 |
| 初始化 | 从头，seed=7 | 相同初始参数 |
| 第一阶段250轮 | Projection+辅助decoder，沿用原表示损失 | 在原损失上加MSE(z_image,qψ(d))及qψ辅助距离重建，同时更新qψ |
| 第二阶段400轮 | 固定选定Projection，拟合qψ | 同样固定选定Projection、继续拟合qψ |
| 第一阶段选模型 | validation图像距离归一化RMSE + 0.2×sqrt(不变性MSE) | 同一准则 |
| 第二阶段选模型 | validation latent MSE | 同一准则，包含第二阶段开始前的qψ候选 |

一致性权重预设1.0，qψ辅助重建权重1.0，其余表示损失保持既有系数。B是“联合训练+一致性+双路径辅助重建”的组合改动，不应写成仅一项损失的严格消融；B第一阶段额外更新了qψ，计算量也不完全相同。先验证这个最小可用组合，再决定是否需要拆分消融。

特征均值/标准差和距离归一化尺度均只从train计算。test没有特征提取、没有指标计算、没有模型选择；validation历史上已被检查，因此仍称开发对照。输出与旧PPO的latent坐标及距离尺度不兼容，暂时不要将新权重直接塞回旧PPO或原VT入口。

运行命令：

```bash
cd /root/work_based_on_spvc/artical-F122
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/opt/miniconda3/bin/conda run --no-capture-output -n vt \
python -m Aebs.dino_latent.align_representation
```

每25轮输出一次进度。结果写入 `results/dino_improvement/05_representation_alignment_ab/`：A_sequential和B_joint各有dino_safety_latent.pt及metrics.json；根目录有预登记config.json、汇总comparison.json。拒绝覆盖非空目录；复跑需显式指定新的output-dir。此入口不生成全数据latent缓存，避免此时打开test。

发回 `[DINO representation alignment A/B]` 开始的全部末尾摘要，或comparison.json。每组同时给出train、validation的：

- latent_rmse：每坐标误差；
- latent_centered_rms：图像latent跨样本变化尺度；
- relative_latent_rmse：前两者之比，排除仅整体缩小造成的假改善；
- residual_l2_p95：样本误差范数的95分位，只是描述统计，不是CP；
- image/physical_distance_mae_m与rmse_m：图像和qψ是否还保留距离信息；
- invariance_mse与far_pair_margin_violation_fraction：外观稳定性和距离不同的状态是否仍可区分。

不预先承诺B一定更好，不只凭latent误差下降自动选择模型。先要求validation相对误差下降，同时距离信息没有明显退化、latent没有坍缩；若指标冲突，结合全部结果决定。只有表示层选择完成后，才重新生成对应训练缓存并从头训练PPO，再处理SBC。此次只同步代码并检查接口，正式实验由用户运行。

更新：2026-09-29。当前使用冻结DINO、32维projection、物理到latent代理qψ、PPO及原SPVC的SBC。CP暂不使用。本轮代码由助手修改、同步和做接口检查；训练、评估由用户在服务器运行。

## 2026-09-30更新：当前先运行准确距离配对诊断

用户已完成旧PPO诊断及新PPO训练：新PPO在qψ/train回放上均100%成功，validation回放100% unsafe。只读检查发现validation最大图像距离仅14.221 m，评估起点却为15–16 m，最大查表错配1.779 m。因此原步骤2–3是稀疏图像库回放压力结果，不能直接解释为未见图像泛化失败。暂停步骤4及后续重训，保留所有历史结果。

新增 `Aebs/dino_latent/diagnose_matched_images.py`：加载已训练的新PPO，使用train、validation的每张图像自身标签距离，与0–3 m/s的31个速度配对。直接读取该图像行的缓存DINO latent，不做最近邻检索；同一物理状态分别输入图像latent和qψ，比较实际裁剪动作和原AEBS环境的一步结果。已处于终止状态的配对排除并计数。test不参与。

```bash
cd /root/work_based_on_spvc/artical-F122
/opt/miniconda3/bin/conda run --no-capture-output -n vt \
python -m Aebs.dino_latent.diagnose_matched_images
```

默认模型：`results/dino_improvement/02_train_only_ppo/latent_ppo.zip`。
默认结果：`results/dino_improvement/04_matched_image_diagnostic/`，包含metrics.json和逐配对samples.json。已有输出时拒绝覆盖，复跑请指定新的`--output-dir`。

发回从 `[DINO matched-image one-step diagnostic]` 到 `per-pair data:` 的全部摘要。重点看动作MAE、最大差、仅图像路径下一步unsafe数量、仅qψ路径unsafe数量和两者都unsafe数量。正动作在当前环境表示制动；`image_less_braking_fraction`只表示图像动作比qψ小，不表示已认定其制动不足。qψ不是正确动作标签；两种动作相同也可能都不安全。

本诊断不训练、不运行完整轨迹、不修改模型，也不给自动安全通过结论。一致性差提示检查表示/代理，一致性好不能单凭这一步判定所有闭环失败都是查表造成。缓存沿用原表示模型，因此仍是开发诊断。后续分阶段计划暂保留供参考，以本节为当前执行顺序。

## 1. 先看进度和下一步

- [x] 已有混合PPO：qψ路径和全部历史图像回放路径各400/400成功。
- [x] 已有SBC：原实现报告0/10000和95.977%；这只是原实现输出，不能称为已确认的端到端安全保证。
- [x] 新增按距离分组的PPO图像库划分、禁止训练访问留出图像的入口。
- [x] 新增固定旧PPO诊断、全新PPO训练及验证入口，保存逐回合结果和最近邻距离偏差。
- [ ] 用户运行步骤1–3，将末尾摘要发回。
- [ ] 根据验证结果，决定先修表示对齐还是继续封存模型测试。
- [ ] 全新数据/全流程表示重训及测试。
- [ ] 最终PPO的SBC重训、同组落盘模型复核。
- [ ] 若论文需要严格概率保证，完成已有区间实现问题的复核及图像误差到证书的衔接。

本轮最先回答的问题：当PPO只能使用一部分距离组的图像训练时，它在另外的距离组图像回放上还能不能完成任务？

## 2. 现有结论哪里需要改进

### 2.1 400/400代表什么

已有评估起点为d∈[15,16] m、v∈[2.5,3] m/s的20×20网格。其余状态域没有因此全部得到闭环评估。图像来自已有400张图像的最近邻回放，而非CARLA在线新图像；训练PPO时也访问了这批latent。这个结果证明流程可运行，不能据此宣称对新图像泛化。

### 2.2 旧PPO的“留出测试”不能重新变成独立测试

旧混合PPO已经访问全部图像latent。重新划分后对旧PPO评估，只能检查减少图像库后的行为变化。新的PPO必须从头训练，且只能从train图像库取图，才能检查PPO层面的图像隔离。

此外，本轮复用的projection和qψ早已在旧划分上训练，因此即使新PPO不访问test图像，这也不是全流程未见数据实验。输出明确标记 `representation_is_newly_heldout=False`。已查看过的历史结果仍属于开发数据，最终独立结论需要新封存的数据。

### 2.3 qψ和图像latent不相同

单来源训练曾出现交叉输入失败，混合训练改善了已有400回合的行为，但没有证明latent误差消失，也没有证明两个策略映射处处相同。因此qψ上的SBC结果不能直接转移为图像上的保证。

### 2.4 SBC原实现的问题仍然存在

此前区间审计发现旧上界调用与下角点普通值相同。复用原代码得到95.977%并不能消除这个问题。当前保留原口径作为可比较的历史实验结果；若要发表严格概率保证，必须修复或证明其计算正确，并覆盖图像路径近似误差。增加样本或把网格违反降到零，均不能替代这一步。此次没有擅自修改原验证器。

## 3. 本轮已实现的最小改进

新增 `Aebs/dino_latent/holdout_experiment.py`，提供prepare、evaluate、train三个子命令；底层使用现有AEBS环境和PPO超参数。

数据按距离分组：以数据最小距离为起点，每0.25 m为一组，固定seed=7，将约60%距离组分到train、20%到validation、20%到test。同一组的全部图像放在同一部分；实际图像数量以prepare输出为准。它是现有距离域内的留组实验，不代表新天气或距离域外泛化。

- prepare只生成固定manifest，包含索引、距离组、数据/表示权重/latent缓存SHA-256。
- evaluate固定已有PPO，分别回放qψ、train图像库、选定的validation或test图像库。
- train新建PPO，每回合50%使用qψ、50%使用train图像库；训练完成只评估validation，test需要单独命令。
- 所有结果目录拒绝覆盖；同一manifest的文件哈希改变会报错。
- 输出每个起点的终局、步数、回报，同时报告最近邻图像与真实状态的距离差均值和最大值。稀疏图像库会增大这一差值，必须一起解释。
- 所有网络输入仍为32维latent+速度。每20,000步打印进度，默认单CPU线程。

## 4. 服务器执行顺序

进入正确目录，并限制小MLP的CPU线程：

```bash
cd /root/work_based_on_spvc/artical-F122
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

### 步骤1：生成一次固定图像划分

```bash
/opt/miniconda3/bin/conda run --no-capture-output -n vt \
python -m Aebs.dino_latent.holdout_experiment prepare
```

输出 `results/dino_improvement/split.json`。此后不要为追求更好结果更换划分。重复运行提示已存在是正常保护；继续下一步即可。

### 步骤2：固定旧混合PPO，做开发诊断

```bash
/opt/miniconda3/bin/conda run --no-capture-output -n vt \
python -m Aebs.dino_latent.holdout_experiment evaluate \
  --output-dir results/dino_improvement/01_old_ppo_validation
```

默认加载 `results/dino_latent_ppo_stage2_mixed_20260928/latent_ppo.zip`。不更新任何模型。三个输入各评估400回合。此步骤不是“旧PPO未见图像测试”。

### 步骤3：从头训练图像库隔离的新混合PPO

```bash
/opt/miniconda3/bin/conda run --no-capture-output -n vt \
python -m Aebs.dino_latent.holdout_experiment train \
  --timesteps 200000 \
  --output-dir results/dino_improvement/02_train_only_ppo
```

训练会保存 `latent_ppo.zip`、`config.json`、`metrics.json`。SB3以2048步为一批收集数据，实际步数可能略高于200000，metrics记录实际值。该步骤从新PPO初始化开始，不继承看过全图像库的旧PPO参数。

**先运行到这里，把步骤2和3结果发回。不要马上用test挑选模型。**

### 步骤4：锁定候选模型后，单独打开test图像库

只有步骤3在validation上满足下面开发门槛、并决定锁定模型后再运行：

```bash
/opt/miniconda3/bin/conda run --no-capture-output -n vt \
python -m Aebs.dino_latent.holdout_experiment evaluate \
  --controller results/dino_improvement/02_train_only_ppo/latent_ppo.zip \
  --eval-split test \
  --output-dir results/dino_improvement/03_locked_ppo_test
```

这是PPO层面的test图像库评估，仍不代表表示层未见数据。若查看test后据此调参，后续结果必须标为开发性结果，不能反复把同一test当成首次独立测试。

## 5. 发回哪些结果，如何判断

每一步把最后 `[DINO PPO image-library holdout: development diagnostic]` 开始到 `metrics:` 的内容完整复制给我，包含：

```text
representation_is_newly_heldout=False
development_gate_pass=...
surrogate: success=... unsafe=... timeout=... stopped_safe=...
image_train: ...
image_validation: ...  # 步骤4会是image_test
各路径lookup_distance_error_m=...
checkpoint: ...
metrics: ...
```

也可以直接提供对应metrics.json。如果报错，发完整Traceback和执行命令；不必发每一轮训练日志。

固定开发门槛：qψ路径400/400成功且0 unsafe；validation图像路径至少95%成功且0 unsafe。该门槛是推进实验的工程标准，不是统计安全证书。提前停车、超时、越域必须分别报告，不能把它们算成功。

| 结果 | 下一步 |
| --- | --- |
| train和validation均好，qψ也好 | 锁定模型，再运行步骤4；准备新数据测试 |
| train好、validation差 | 检查距离查表误差及失败起点；再判断表示泛化问题 |
| qψ好、图像差 | 优先对齐图像latent与qψ，暂不增加SBC训练 |
| 图像好、qψ差 | 代理模型不适合当前控制器，先改善代理一致性 |
| 两条路径均差 | 检查新PPO学习曲线和固定设置，不能归因于SBC |

## 6. 后续表示训练改进：需要时再实施

此阶段尚未实现，避免在诊断前同时改变数据、表示和控制器。根据步骤3的失败类型决定是否推进。

使用同一距离组划分重新训练projection和qψ，train用于梯度更新，validation用于选择checkpoint。DINO持续冻结，辅助decoder继续只在训练中使用。加入确定性一致性损失：

`L_phys = mean(||P(DINO(o)) - qψ(d)||²)`。

总损失保留安全信息、外观不变性、状态分离项，再加入预先固定权重的L_phys。必须防止两个模型一起输出常量；同时检查辅助距离误差、latent变化幅度、qψ误差和控制结果。L_phys变小本身不代表成功。

最小对照只做两种：原损失、原损失+L_phys。使用同一分组和固定seed，不做多随机种子搜索。两者都应在相同严格训练划分上重新训练，避免把划分变化误认为算法增益。新的latent缓存生成后需要新的manifest并记录哈希，旧实验保持原样。完成表示训练后，新建PPO重训，重复验证门槛。

## 7. 新数据如何增加

首先采集“同一距离、不同外观”的配对图像，保留足够分辨率供DINO使用。记录距离、场景、天气/光照、相机位置及采集批次。训练与最终测试按场景/采集批次分开，不按连续视频帧随机拆分。

第一批新数据用于独立复核冻结候选模型；若用于继续训练，就另留一批不查看的最终测试。接下来做CARLA在线渲染回合，输入当前真实图像而非由真实距离选取历史图像。届时仍分别报告任务成功、unsafe、提前停车、超时和回报。

新增数据优先解决外观变化和图像泛化；它不能修复验证器实现，也不能自动把qψ证书转成图像证书。

## 8. SBC后续安排与成功口径

最终PPO和表示权重固定后再做SBC。先以原代码口径作可比较实验，记录权重哈希、实际受检区域大小、全网格数、违反数、区域界及同组落盘权重。原来的0/10000分母是全网格数，不能据此认为所有10000点都检查了下降条件。

先保留已有SBC warm start作为快速复现，再补从随机初始化的SBC实验。当前run_spvc入口默认加载旧SBC，因此“从头SBC训练”尚需补充入口支持，不能直接把默认命令当成从头训练。

原口径开发门槛可保持零报告违反、初始区上界低于不安全区下界、原计算概率下界≥90%，并核对保存前后同一策略。但通过后只能写“通过原实现判定”。严格保证另需复核此前的区间接口错误、噪声假设和图像到qψ的误差传递；这些未完成前不将95.977%写成已证明的视觉闭环安全率。

QP放在PPO与SBC候选模型锁定之后再接入。本轮不启动QP训练。

## 9. 实验记录填写格式

每次保留：运行日期、命令、seed、manifest哈希、输入模型哈希、输出目录、实际训练步数、运行时间、各路径回合数和终局统计、最近邻距离偏差、是否查看test、这一步解决和未解决的问题。新入口已保存上述主要机器可读字段及逐回合结果；表示层来源和test使用历史需在本计划中人工补充。

本轮状态：代码已准备；实验数值待用户运行后填写，不使用旧400/400数字填充新实验结果。

2026-09-29同步检查：上述代码及文档已同步至服务器 `/root/work_based_on_spvc/`；替换的旧文件保留 `.before_holdout_20260929` 备份。vt环境中6项接口/隔离测试通过，CLI帮助入口正常；检查了同组距离不跨集合、最近邻查表不能访问被排除的图像。没有运行prepare、训练或回合评估，尚无本轮实验结果。
