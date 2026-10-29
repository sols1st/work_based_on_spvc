# VISION-SLS + 原 SPVC 非负 SBC：通过的适配模型

2026-10-29。本模型保持 **DINO → 低维观测 → PPO** 的部署链路，离线使用非负 SBC 训练和验证。**没有部署 NCBF 或 QP 动作修正器**。视觉模型冻结，PPO 经 SBC 反馈小步更新；SBC 与策略交替改进后，完成了连续状态域数值区间检查。

## 已通过的检查

| 检查 | 要求 | 结果 |
| --- | --- | --- |
| SBC 非负 | 全域 `B>=0` | 由非负 ReLU 输出保证 |
| 初始区域 | `B<=1` | 数值区间上界 **0** |
| 不安全区域 | `B>=20`，对应原 95% 目标 | 数值区间下界 **24.9659319** |
| CP 观测包络内的 SBC 不增 | 覆盖所有相关连续状态与包络内观测 | **4,971,398** 个检查区间，零反例、零未解决区间 |
| 固定图片库实际回放的 SBC 不增 | 覆盖完整连续初态域与逐状态图片查询 | **105,608** 个检查区间，零反例、零未解决区间 |
| 新的随机状态检查 | 65,536 个新状态；60,719 个满足 SBC 活动条件 | 零正违例 |
| 目标到达 | 441 个网格初态 + 1000 个随机初态 | 名义、全部图片、留出图片各 **1441/1441 成功** |
| 软件回归 | 区间、梯度、加载、绑定、原环境语义 | **32 项测试通过** |

连续域检查不是仅在上述有限随机状态上测试：验证器从完整活动域 `d∈(6,16]m, v∈(1e-6,3]m/s` 开始划分，通过区间上、下界证明每个叶区间的条件；未解决区间继续细分，真实反例反馈训练。`B` 下界达到 20 的区间按非负超鞅到达阈值的停止论证处理，不采用历史验证器的窄带筛选或允许少量违例的容差。回放模式按图片距离的 Voronoi 分界划分，处理边界两侧的观测；不把最近邻图片变换假设成光滑函数。

## 保留了什么，适配了什么

保留原 SBC 的核心：**非负屏障、初始值不超过 1、不安全值至少 20、条件期望不增，以及 SBC/策略交替训练**。网络隐藏结构仍是 `[2,16,8,1]`、tanh 隐藏层。物理动力学、初始域、目标和不安全判定不变。

必要适配有两项，不能隐去：

1. **停止过程与下降条件**：车辆按原环境的目标到达或停车事件停止，终止后屏障值保持不变。验证条件是非负超鞅的不增条件。旧 `epsilon=.1` 若被当成每步必需下降量，在初始值最多 1、需要 106–117 步的无额外过程噪声轨迹上不相容；名义轨迹也属于 CP 观测包络。现在 `.0001` 仅作为训练裕度，不作为安全区零值状态必须满足的全域严格下降量。保留原 **95%** 区域目标，没有调低安全概率门槛。
2. **非负输出**：Softplus 对照仍有下降违例，最终使用 **ReLU 非负输出**，使安全区域可以精确取零。它仍是连续、Lipschitz 的非负 SBC；不采用有符号 NCBF。checkpoint 明确记录激活，禁止把同一组权重误加载为 Softplus。

非负超鞅的标准停止论证给出到达风险阈值的概率上界 `B(initial)/20`。这里检查的是实现该论证的模型条件，不将随机状态 CP 覆盖率当作整个轨迹的安全概率。背景可参照 [Prajna 等的随机屏障工作](https://web.mit.edu/~jadbabai/www/papers/TAC05-Barrier.pdf) 和 [离散时间神经屏障的无限时域停止论证](https://link.springer.com/chapter/10.1007/978-3-031-65630-9_20)。

## 结论范围

- **CP 包络检查**证明的是：若每次观测确实在给定状态相关包络内，则所有包络内动作所形成的后继都满足 SBC 条件；它不需要把区间人为当成均匀分布。现有复用图片 CP 不能证明每个状态下的条件支持集假设，不能据此宣布真实新图像 95% 安全。
- **固定回放检查**直接验证 400 张冻结图片对应的确定性最近邻输入，包括实际超出 CP 包络的图片；该模型的条件转移为已知的点质量分布。模型停止后吸收，连续域检查不依赖 400 步测试上限，但不代表实际车辆达到目标后继续运动的模型。
- **数值边界**：使用 float64 数值区间运算及经区间导数验证的单调性。未做向外舍入或 IEEE 浮点执行的形式证明。因此准确名称是“连续状态域数值区间检查通过”，不能改写为真实环境/硬件的完整形式证书。
- 上述三路 1441 次到达是额外的行为回归证据；SBC 约束证明安全，不单独证明目标可达。

## 原先为什么失败

原候选未满足区域分离，训练中下降损失会把 SBC 推向常数；旧正下降裕度与长轨迹/吸收终点冲突。只检查随机区域样本还漏掉了关键边界 `(6m,.5m/s)`。后续补齐完整边界与反例反馈，保护已通过的区域条件，并对 PPO 采用小步更新与任务回退检查。

开发集曾先通过，但 65,536 个新状态检查找到 30 个违例，初次连续区间验证又找到 158 个反例；这些反例被用于重新训练，未被算作通过。最终发布的是后续 r14 候选，完成了上表中的两条连续域检查。带有旧失败/未解决记录的实验目录均保留。

## 直接运行

在仓库的 `artical-F122` 目录执行：

```bash
../.venv-vision-sls/bin/python -m Aebs.system.combined \
  --artifact-dir results/vision_sls_spvc_sbc_pass_20261029
```

不要给这个命令传 CP-NCBF 的 `--certificate-dir`。本模型的 SBC 作用在离线训练与验证中，运行时 PPO 直接输出制动动作。

```python
from Aebs.vision_sls.deploy import VisionSLSController
controller = VisionSLSController('results/vision_sls_spvc_sbc_pass_20261029')
action = controller(images, measured_speed)
```

需要读取 SBC 时使用激活感知加载器：

```python
from Aebs.vision_sls.sbc_cp import load_sbc_checkpoint
barrier, metadata = load_sbc_checkpoint(
    'results/vision_sls_spvc_sbc_pass_20261029/sbc.pt')
# 输入为 [distance / metadata['barrier_distance_scale_m'], speed]。
```

模型包重新加载后，实际图片推理及回放验证均复查通过。图像模型仍使用本仓库已有 DINO 权重路径；模型包不重复打包 DINO 的大文件。

## 复查连续区域

输出目录必须不存在：

```bash
../.venv-vision-sls/bin/python -m Aebs.vision_sls.verify_sbc_cells \
  --artifact-dir results/vision_sls_spvc_sbc_pass_20261029 \
  --barrier-dir results/vision_sls_spvc_sbc_pass_20261029 \
  --perception-contract results/vision_sls_spvc_sbc_pass_20261029/perception_contract.json \
  --observation-model replay --max-cells 2000000 --max-depth 24 \
  --output-dir results/my_sbc_replay_check
```

CP 包络路径改为 `--observation-model tube --max-cells 5000000 --max-depth 28`，并使用另一个输出目录。该项本机运行约 6.6 分钟，达到计算上限而仍有未解决区间时必须判未通过。

## 文件与训练路径

- `policy.pt`、`observation.pt`：已配对的最终 PPO 和 VISION-SLS 观测模型。
- `sbc.pt`：包含结构、**ReLU 激活**、坐标尺度、策略/视觉哈希的非负 SBC checkpoint。
- `perception_contract.json`：参考 CP 包络。策略更新后重新绑定 actor 哈希，未伪装成新的独立图像校准。
- `summary.json`：结果、明确限制、文件哈希与软件版本。
- `verification/tube.json`、`verification/replay.json`：完整连续域检查记录；`reload_replay/` 是发布包重新加载后的复查。
- `verification/expanded_task_check.json`、`evaluation_reload.json`：目标到达及真实图片加载证据。
- `source_snapshot/`：发布时使用的相关代码快照。

本次可追溯训练链包括原成功基线 `vision_sls_spvc_goalfix_full_20261028`、SBC 候选 r9、策略更新 `vision_sls_sbc_actor_20261029_r1`，以及针对该策略的 SBC r11→r12→r13→r14。失败的其他轮次保留用于分析。开发输入已从临时目录保存到 `vision_sls_sbc_training_inputs_20261029_initial`；更新策略对应的反例输入在 `vision_sls_sbc_inputs_20261029_actor_r1`。这些都是开发数据，不能再声称独立最终校准数据。

相关可执行阶段是 `prepare_sbc_training`、`train_sbc_cp`、`refine_sbc_actor`、`verify_sbc_cells`、`package_sbc`。最后的打包步骤拒绝未通过、有未解决区间或模型哈希不匹配的结果。重新训练可能产生不同候选，不能跳过重新验证而复用本包的结论。
