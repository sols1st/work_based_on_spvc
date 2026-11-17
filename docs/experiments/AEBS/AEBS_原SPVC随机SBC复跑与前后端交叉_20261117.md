# AEBS 原生 SPVC 随机屏障复跑与前后端交叉实验

## 1. 同一个“原 SPVC”需要对应哪套权重

原论文表 1 的 CARLA 紧急制动基准给出的无限时域概率安全下界为，固定策略 **72.6%**，同时更新超时（OT），交替优化 **94.2%**。这三个数作为论文原报值保留。论文给出的作者仓库目前只有 main 分支。核对 `6279872ed7708cd103c0801d758d2a95c203ce27` 及全部 14 个可达提交，AEBS 权重只有生成器、状态估计器、初始 PPO 三件。生成器文件是发布的条件 MLP 生成路径，输入为距离与四维潜变量，输出 32×32 灰度图。原始 400 张 H5、这三件权重均与当前实验项目中的对应文件逐一哈希一致。上游未公布与表 1 绑定的最终 AEBS 策略与 SBC，旧 `info.log` 的五条记录均为 UNSAFE。服务器几个原始工程检出中，也未找到作者注释路径下的这对最终权重。[原论文表 1、定理 2.2](<../../references/Provably Probabilistic Safe Controller Synthesis for Vision-Based Neural Network Control Systems.pdf>) · [冻结的作者代码](https://github.com/QLong01/artical/tree/6279872ed7708cd103c0801d758d2a95c203ce27) · [前一版历史日志](https://github.com/QLong01/artical/blob/b628b61c62212e9a536b84ea31f210fa36fef39b/info.log) · [权重来源与本次实跑总表](../../../artical-F122/results/aebs_original_random_sbc_20261117/evidence.json)

为此直接从作者发布的三个原件重新综合随机屏障和控制器。下面的数字来自这次重训。此前用初始 PPO 或共用新鲁棒控制器的实验，回答的是另一个部署问题。

## 2. 原生实验怎样做

1. 从隔离的作者仓库加载 cGAN 条件生成路径、32×32 灰度估计网络和 PPO 策略。H5 的距离标准差作为归一化尺度。生成器和状态头保持冻结。训练屏障时用大小为 2–16–8–1 的 Tanh 网络，最后一层 Softplus 保证非负。策略更新时优化 `controller_net`。
2. 每次迭代使用作者的 100×100 状态网格、2,048 训练 batch。交替组运行原生 10 轮屏障更新与 1 轮策略更新。原代码的 `Loop.run` 以 0 为起点，执行到 100（含），共 101 次屏障验证。固定策略组使用相同代码，只跳过策略梯度更新。作者入口设置扰动比例为状态域宽度的 5%，训练抽三角噪声，验证使用原实现的均匀扰动格与概率质量计算。
3. 分别运行发布源码默认 Adam 学习率和论文正文的学习率，固定同一随机种子 202611171。模型实跑在 `vt` 中使用 CPU 四线程，原 CUDA 标签权重只作设备映射。论文原运行平台和未知随机种子没有复建。输入协议、每轮分数和最佳/末轮权重保存在各自新目录，计分和存权重均在本轮验证完成、下一次策略更新前。发布的原数据、原模型和作者脚本均保留。
4. 当上游验证器接受候选并给出可分离的区域界时，用作者的原概率分数 `1 − (初始上界 − 全域下界)/(危险下界 − 全域下界)`。按这一规则跟踪最高分，单列最后一轮当次分数。没有分离的组将状态如实记为无分数。

| 训练组 | 屏障 / 策略学习率 | 源验证器接受轮数 | 成功分离并计分 | 最高原分数 | 末轮当次分数 |
|---|---|---:|---:|---:|---:|
| 固定策略，源码默认 | 0.003 / 冻结 | 92/101 | 0 | 无 | 无 |
| 交替训练，源码默认 | 0.003 / 0.05 | 97/101 | 58 | **95.4759%**（第 93 轮） | **94.6483%** |
| 固定策略，正文学习率 | 0.001 / 冻结 | 101/101 | 0 | 无 | 无 |
| 交替训练，正文学习率 | 0.001 / 0.005 | 101/101 | 0 | 无 | 无 |

例如默认交替组最佳屏障的初始上界是 1.2526655，危险下界 19.7787819，全域下界 0.3748139，对应分数 0.9547592。默认固定组末轮的初始上界 10.1521、危险下界 9.5843，无法由这两个界算出正的安全分数。一次随机初始化不能说明固定策略或另一组学习率一般都会失败。[作者学习/验证入口](https://github.com/QLong01/artical/blob/6279872ed7708cd103c0801d758d2a95c203ce27/Aebs/VT/loop.py) · [原生训练参数](https://github.com/QLong01/artical/blob/6279872ed7708cd103c0801d758d2a95c203ce27/Aebs/VT/train.py) · [记录程序](../../../artical-F122/Aebs/experiments/run_upstream_native_aebs.py) · [默认交替逐轮结果](../../../artical-F122/results/aebs_original_random_sbc_20261117/alternating/rounds.jsonl)

**默认源码换种子的敏感性。** 观察第一种子后，预先确定两个额外种子，两个训练组逐对复跑，其余设置完全相同。下面列出各运行在候选轮次中的最高源码分数，三个种子全部保留。

| 随机种子 | 固定原 PPO | 交替训练 |
|---:|---:|---:|
| 202611171 | 无分数 | 95.4759% |
| 202611172 | 91.6185% | 90.5584% |
| 202611173 | 83.1985% | 90.9510% |

固定组在其中两个种子得到可计分的候选。一组配对里固定组的原分数超过交替组。加上前三段的学习率敏感性，合计 8 次完整原生运行、808 次网格验证，不能仅选最高的种子来表示一种方法的一般性能。[所有轮次、权重与证据](../../../artical-F122/results/aebs_original_random_sbc_20261117)

## 3. 为什么 95.4759% 要标作源码分数

论文定理 2.2 要求，安全状态中所有低于危险屏障阈值的点都满足严格的**期望**下降。源码先按屏障值筛选危险下界的 95% 至 100% 之间的窄带，再对候选网格中心检查，并允许硬违规数占原 10,000 个格点的比例不超过 0.001。该源码接受标记不足以单独推出定理条件。即便最佳轮在源码筛选出的点上记录 0 个硬违规，仍需检查窄带以外的点。[作者核验规则](https://github.com/QLong01/artical/blob/6279872ed7708cd103c0801d758d2a95c203ce27/Aebs/VT/verify.py)

对最佳第 93 轮冻结策略和屏障，我们在作者实际均匀后继扰动律上找到一个可复查的数值反例。车距 **11.80625 m**、速度 **3.0 m/s** 是域内非危险状态，原始屏障为 **5.99391**，明显低于危险集下界 **19.77878**。把扰动矩形划为 16×16 个等质量小格，用 Tanh 的单调性和各线性层的区间界逐格取后继屏障的下界，再按概率取平均，得到 **期望后继屏障至少 7.73609**，正漂移至少 **1.74218**。线性归一化不改变漂移符号。这是正期望漂移，不能用一次随机采样偶然上升来替代。

该例位于速度上边界。源码在名义动力学中先裁剪车速，再加扰动，部分后继速度会高于 3 m/s，神经屏障在这些点仍可求值。作为敏感性检查，若将加噪后的状态再裁剪回声明矩形，同一点的期望漂移下界仍为 **3.75647**。另一次搜索仅考虑全部后继均留在域内的点，未找到反例；采样搜索的阴性结果不构成证明。区间核算采用 float64，未做定向舍入。因此本次 **95.4759% 不作为已建立的无限时域保证**。这项反例针对本次重训权重和所述后继律，未检验论文未公开的最终权重。[期望反例明细](../../../artical-F122/results/aebs_original_random_sbc_20261117/alternating/drift_audit_domain.json) · [域内后继搜索](../../../artical-F122/results/aebs_original_random_sbc_20261117/alternating/drift_audit_interior_full.json) · [独立区间检查](../../../artical-F122/Aebs/experiments/audit_upstream_native_drift.py)

对另外四个有源码分数的最佳候选，也用全状态域搜索并对后继均匀分布逐格算期望下界。种子 172 的固定、交替组有正漂移下界 1.4796、0.9529；种子 173 的固定、交替组为 0.9616、0.6564。四个见证的前驱都是域内速度上界 3 m/s，采用加噪后裁剪的敏感性检查也均为正。由此五个得到概率分数的最佳候选都未通过这个独立的定理条件检查。[五组候选的数值证据](../../../artical-F122/results/aebs_original_random_sbc_20261117/evidence.json)

**原生成路径的有限回合诊断。** 在原 400 图构造出的名义生成路径上，另用同一批 2,000 个新初态比较原初始 PPO 和第 93 轮策略。这里参考原声明域和危险集。首次到危险集或离开声明状态域就停止，最多 400 步，离域后记为未知。无加性扰动时，原初始 PPO 的 2,000 条均先离域，原交替策略的 2,000 条均先到危险集。按验证器的 5% 均匀加性扰动，在两个臂上使用相同的逐轨迹扰动抽样，两者均为 2,000 条先离域。离域数不能作为已安全的轨迹数，名义无扰动样本也不能当作均匀随机律的反证。该测试显示为什么在此声明域上报告风险时必须说明离域处理。[原生成与随机后继停止轨迹](../../../artical-F122/results/aebs_original_random_sbc_20261117/stochastic_rollouts/summary.json)

## 4. 交叉切换实际前端和已训练控制器

从前一节**只按上游原分数**选中的第 93 轮取控制策略。前端切换到相机回放的原灰度状态头或当前 RGB DINO 输出。控制器切换到刚综合的原生 SPVC 策略或当前发布的鲁棒策略。四组均采用同一稠密 24,000 帧图库、相同 2,000 个起点及固定天气/车色，使用相同的制动动力学、最高 400 步停止判据。部署对照不另加上一节的随机后继噪声。观测转换为物理米后按各策略的尺度传入。本段测试把不同体系接在同一物理任务上衡量行为；只有当前鲁棒控制器具有 ±0.45 m 的相应父证书。

| 前端 | 控制器 | 目标成功 | 危险终止 | 全程物理测距误差 ≤0.45 m |
|---|---|---:|---:|---:|
| 原发布灰度状态头 | 原生交替重训 | 0/2,000 | 2,000/2,000 | 0/2,000 |
| 当前 DINO | 原生交替重训 | 0/2,000 | 2,000/2,000 | 1,985/2,000 |
| 原发布灰度状态头 | 当前鲁棒控制器 | 0/2,000 | 2,000/2,000 | 0/2,000 |
| 当前 DINO | 当前鲁棒控制器 | 2,000/2,000 | 0/2,000 | 1,985/2,000 |

在这四个冻结组合中，切换前端使两种控制器下的在管条数都从 0 增至 1,985。相机使用 DINO 时，控制器选择又使实际任务成功在 0 与 2,000 之间变化。因此准确读数与控制策略共同决定任务表现。该 24k 库包含当前视觉头见过的图，每条轨迹按其状态查询同外观的最近帧，概率单元为新抽初态/外观。原发布头与目标域适配的 DINO 在预训练、分辨率和目标数据量上有差异。等信息、等目标图像总包的前端对照需另列。本节 2,000 个场景与先前固定管测试相同，以便隔离控制器变动；不能与先前场景数相加。[交叉运行协议与实际结果](../../../artical-F122/results/aebs_original_random_sbc_20261117/camera_cross/summary.json) · [源码](../../../artical-F122/Aebs/experiments/native_trained_actor_camera_cross.py) · [已有同预算分析](AEBS同预算_双协议冻结回放与鲁棒证书_20261117.md)

## 5. 复跑

在服务器进入项目的 `artical-F122` 目录，激活 `vt`。下列命令把作者仓库检出到独立路径，并把输出写入新目录。脚本自行绑定作者代码的包路径，使用该版本的 auto_LiRPA。输出路径需尚不存在。历史发布的三个权重和 H5 已包含在该作者提交中。

```bash
git clone https://github.com/QLong01/artical.git /tmp/spvc-author-reproduction
git -C /tmp/spvc-author-reproduction checkout --detach 6279872ed7708cd103c0801d758d2a95c203ce27
root_src=/tmp/spvc-author-reproduction
run_dir=results/aebs_native_reproduction
export CUDA_VISIBLE_DEVICES=""
export OPENBLAS_NUM_THREADS=1
for arm in fixed alternating; do
  python Aebs/experiments/run_upstream_native_aebs.py "$root_src" "$run_dir/$arm" --arm "$arm"
  python Aebs/experiments/run_upstream_native_aebs.py "$root_src" "$run_dir/paper_$arm" --arm "$arm" --paper-rates
done
for seed in 202611172 202611173; do
  for arm in fixed alternating; do
    python Aebs/experiments/run_upstream_native_aebs.py "$root_src" "$run_dir/seed${seed}_${arm}" --arm "$arm" --seed "$seed"
    python Aebs/experiments/audit_upstream_native_drift.py "$root_src" "$run_dir/seed${seed}_${arm}" --full-domain
  done
done
python Aebs/experiments/audit_upstream_native_drift.py "$root_src" "$run_dir/alternating" --full-domain
python Aebs/experiments/audit_upstream_native_drift.py "$root_src" "$run_dir/alternating" --interior-full
python -m Aebs.experiments.native_trained_actor_camera_cross "$run_dir" results/aebs_native_cross_20261117_v1
python -m Aebs.experiments.native_stochastic_rollout_diagnostic "$run_dir"
```

原工程每轮在加载的模型上更新参数。此处日志中的 `[SAVED]` 在作者代码里只打印提示，实际模型保存行被注释，本次记录程序才在验证与下次策略更新之间保存权重。复算时检查 `best.pt` 对应第 93 轮，`final_verified.pt` 对应第 100 轮的验证时刻。每个训练目录都保留了逐轮数据与检查点哈希。[完整实验结果包](../../../artical-F122/results/aebs_original_random_sbc_20261117)
