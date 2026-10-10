# R2 DINO＋原版SPVC最小替换实验

更新日期：2026-10-09（增加最终checkpoint的同口径范围对照）。

## 第23步：新旧PPO精确图片配对评估（代码就绪，待用户运行）

目的：检查第22步为SBC交替更新的PPO是否破坏了原R2图像控制行为。不再训练SBC，不添加QP、过滤器、CP或新安全定义。

### 本次做什么、不做什么

- 固定第08步R2表示与DINO缓存。
- 旧控制器：`09_R2_ppo/latent_ppo.zip`。
- 新控制器：`22_R2_original_spvc_scope_compare/latent_ppo_spvc.pt`，直接加载保存的actor；不是把旧zip改名，也不回退使用旧权重。
- 在每张train/validation图像自身的距离标签上，使用相同的31个速度值，对比“旧图像、旧q、新图像、新q”四种动作及单步后继。
- 排除当前已经终止的状态，记录排除原因；未用最近邻检索，不编码/评估test，不进行CARLA在线采集。
- 动作使用确定性actor输出并按原范围`[-3,3]`限幅；不使用旧PPO的随机分布或价值网络解释更新后的actor。
- 后继沿用已有`one_step`及AEBS事件定义，不额外注入原VT均匀噪声。这是控制行为开发诊断，不是重跑第22步噪声积分或屏障检查。
- 单步结果不是整回合成功率，也不是安全证书。无单步退化仍需后续在线轨迹评估。

### 加载保护与输出内容

运行前核对R2、旧PPO、新actor、SBC和第22步对照文件的哈希，检查q权重没有被更新。以图片与q两种输入测试：旧actor适配与旧PPO的动作一致、新actor适配与直接前向一致，以及q路径物理距离单位转换一致。超出`1e-5`动作差会拒绝继续评估，避免把加载错误当成模型退化。

保存的指标包括：

1. 新旧各自图像/q动作MAE和最大差；
2. 同一图像输入下新旧策略的动作漂移；
3. 图像独有unsafe、q独有unsafe、两者共同unsafe；
4. **旧策略未unsafe、新策略变unsafe**的配对数，以及反向改善数；
5. 分train/validation、分天气颜色的汇总与最坏20个例子；
6. 每个实际检查样本的动作、后继和outcome编码，保存到NPZ。

即使新策略两条路径的动作完全一致，只要它们一起从安全变成unsafe，也会被记录为退化，不能靠一致性指标掩盖控制问题。

### 实现与测试状态

- [x] 新增`spvc_saved_actor.py`：加载第22步actor并做身份/动作一致性检查。
- [x] 新增`evaluate_spvc_pair.py`：固定数据的新旧四分支单步配对评估。
- [x] 新增`spvc_pair_summary.py`：动作差与配对unsafe变化统计。
- [x] 新增`test_spvc_pair.py`：3项轻量统计测试通过，2项torch数值测试因本机缺少torch而跳过；全部新增文件语法检查通过。
- [ ] 在服务器vt环境执行全部测试，确认没有skip或fail。
- [ ] 用户运行第23步并回传结果；未自动连接跳板、传输或运行实验。

### 23.1 本机测试跳板连接

连接路径：`本机 → ab@frp-put.com:13409 → root@10.11.154.191:20020`。

在本机终端执行：

```bash
ssh -J ab@frp-put.com:13409 -p 20020 root@10.11.154.191
```

成功后应到达原实验服务器的root终端。首次连接请核对两台服务器的主机指纹，按提示使用各自账号的认证方式；可能先后要求两次密码。不要把密码写进命令，不需要将本机私钥复制到跳板机。测试成功后执行`exit`返回本机，再传文件。

若提示`administratively prohibited`，通常需要管理员允许跳板SSH转发；若连接内网地址超时/拒绝，需确认跳板机确实能访问`10.11.154.191:20020`。这不是Python或实验代码问题；不要通过关闭主机密钥检查解决。

### 23.2 本机同步代码：经跳板直接到内网服务器

无需先把代码存到跳板机。以下命令只同步本步文件和加载适配器，不同步数据/模型，也不删除远端文件；已有同名代码会留下`.before_step23`备份。

```bash
cd /Users/johann/project/work_based_on_spvc/artical-F122

rsync -avR --backup --suffix=.before_step23 \
  -e 'ssh -J ab@frp-put.com:13409 -p 20020' \
  Aebs/dino_latent/spvc_saved_actor.py \
  Aebs/dino_latent/spvc_pair_summary.py \
  Aebs/dino_latent/evaluate_spvc_pair.py \
  Aebs/dino_latent/test_spvc_pair.py \
  Aebs/dino_latent/spvc_policy.py \
  root@10.11.154.191:/root/work_based_on_spvc/artical-F122/
```

该命令以服务器已有的完整R2代码和数据为基础；依赖此前的`diagnose_expanded_failures.py`、`diagnose_matched_images.py`等已有模块，不是空目录安装包。

### 23.3 登录内网服务器，先测试再运行

本机执行：

```bash
ssh -J ab@frp-put.com:13409 -p 20020 root@10.11.154.191
```

登录后执行：

```bash
cd /root/work_based_on_spvc/artical-F122

/opt/miniconda3/bin/conda run --no-capture-output -n vt \
  python Aebs/dino_latent/test_spvc_pair.py -v
```

预期5项测试均为`ok`。若有fail/error或torch测试被跳过，先处理环境或回传日志，不启动正式评估。

然后运行：

```bash
/opt/miniconda3/bin/conda run --no-capture-output -n vt \
  python -m Aebs.dino_latent.evaluate_spvc_pair \
  --spvc-dir results/dino_expanded_v1/22_R2_original_spvc_scope_compare \
  --output-dir results/dino_expanded_v1/23_R2_spvc_matched_pair
```

默认只做CPU推理，不需要重新提取DINO特征。输出目录若已存在会拒绝覆盖；重新运行请更换输出目录后缀，不要删除旧结果。会逐批打印进度和最后摘要。若SSH连接不稳定，可在服务器已有的tmux会话内执行上述命令。

### 23.4 给我看什么结果

回传最后完整的：

```text
[Step23 loader checks] ...
[R2 old vs SPVC-updated PPO: exact-image matched diagnostic]
...
```

并提供`23_R2_spvc_matched_pair/metrics.json`。若出现退化，再看同目录的`validation_worst_examples.json`和`train_worst_examples.json`；逐点数据保存在`train_pairs.npz`、`validation_pairs.npz`，无需先粘贴这些大文件。

结论判定：

- 有`new_unsafe_old_not`：确实找到新策略新增的单步unsafe反例，应先定位，再讨论训练修复；不抵消掉它来只报告净变化。
- 没有新增unsafe：只能说本批配对没有发现这类退化，仍检查动作漂移和其他outcome，再进入原协议CARLA在线开发回放。
- 加载检查失败：停止，不把它当成策略性能差。

第21步CP仍只对应原冻结策略；本步不宣布其已转移到第22步策略，也不触碰新正式校准数据。

## 最新结果：第22步已完成，原窄带零硬违反，宽区域存在违反

结果来源：用户回传终端输出。尚未读取服务器完整JSON、checkpoint及逐点NPZ；不据此宣称已独立重跑或审计模型。

| 检查范围 | 实际检查点数 | 硬违反 | 补偿后违反 | 最小硬裕量 |
|---|---:|---:|---:|---:|
| 原窄带 `legacy_band` | 69/10000 | 0/69 | 1 | +0.211631775 |
| 阈值以下安全区域 `below_unsafe_safe` | 1332/10000 | 103/1332（7.73%） | 162 | -2.230720520 |
| 全部安全网格 `all_safe` | 9253/10000 | 109/9253（1.18%） | 356 | -2.230720520 |

区域界：初始上界1.307321、不安全下界20.039461、全域下界0.405764，顺序正确。区域值比率为
`1-(init_upper-domain_lower)/(unsafe_lower-domain_lower)=0.954081128...`。
该数值只反映区域分离，不单独构成95.41%的安全保证。

### 本次可以得出的结论

1. 在本次最终模型、原噪声和legacy计算下，原窄带非空且硬违反为0；“DINO代理模型无法通过原窄带硬检查”的担忧未在这次结果中出现。
2. 原窄带只包含总网格的0.69%。同一checkpoint扩大范围后出现硬违反，说明范围筛选确实影响通过结果，不再只是猜测。
3. 这不是原cGAN与DINO的性能优劣对照；没有在这里重跑cGAN基线。也不是证明真实控制器发生了103次事故：这些是屏障不等式的违反点。
4. 窄带仍有1个补偿后违反。原硬检查用`expected-B>=0`，补偿检查还加上代码中的网格/局部梯度项；后者未全通过，不能称为完成连续域证明。该局部梯度项自身也沿用legacy实现，并未在本次升级为可信全域上界。
5. 宽区域违反率较低或降低不意味着更安全：1332点和9253点的分母不同，零违反的新增点会稀释比例；违反绝对数由103升到109。

当前实验目标“同一模型下比较原窄带与宽域检查”已完成。先保存第22步结果，不建议继续缩窄带宽或改容忍阈值。后续应读取`scope_comparison.json`、`metrics.json`核对模型哈希及训练停止原因；若要定位宽域违反，再分析`scope_samples.npz`。原SPVC训练可能更新PPO，第21步旧PPO的轨迹CP结果不能自动转移给第22步PPO，需先核对并对最终图像路径做开发回放。

## 当前正式结论：可以说明什么，与SPVC有什么异同

### A. 一句话结论

**当前R2 DINO双路径代理闭环已经达到本地沿用的SPVC实现的“非空窄带零硬违反”结果，并实现初始区与不安全区的屏障值分离；但尚未证明满足原论文定理的全部条件，也未通过本次实验建立实际图像闭环的端到端SBC保证。**

这里“与SPVC同口径”是指采用相同的训练机制、噪声设置和检查规则，不是所有模型、数据、输入维度和证明假设都完全相同。也不是已与原cGAN模型做完同预算的优劣对照。

### B. 与本地SPVC实现保持一致和发生变化的部分

| 比较项 | 本地原SPVC基线 | 当前第22步R2版本 | 结论边界 |
|---|---|---|---|
| 可验证前端 | 状态与环境变量经cGAN/感知近似链送控制器 | 物理距离经小代理q得到32维latent | 替换的是感知表示链，不是证明两条前端完全等价 |
| 实际图像路径 | 原感知链及其控制输入 | 图像→冻结DINO→R2投影→32维latent，与速度一起送PPO | 图像路径没有直接进入第22步状态网格验证 |
| 控制器 | 原感知输入对应的PPO | 适配32维latent＋速度的R2 PPO | 保留PPO方法与SPVC更新机制；不是原PPO的相同输入维度或相同权重 |
| SBC | 原神经SBC结构与训练方法 | 沿用原结构、平方输出、区域及下降损失 | 不使用第19步轨迹SBC权重 |
| 训练流程 | SBC与控制器交替更新 | 沿用原循环 | PPO可能被更新，不能直接继承旧PPO的统计保证 |
| 动力学和随机噪声 | 原AEBS与factor=0.05的均匀下一状态噪声 | 相同 | 此噪声不是第14步CP残差盒，也不自动覆盖实际DINO误差 |
| 状态网格 | 100×100 | 相同 | 10000为总网格数，不是窄带实际检查数 |
| 下降检查范围 | 归一化B位于不安全下界的95%～100%之间 | 相同，实际选中69点 | 是B值的窄带，不是物理距离固定的一条危险带 |
| 硬违反规则 | `expected-B>=0`计违反，使用原比例阈值 | 相同；本次0/69 | 硬检查通过不等于全部验证条件通过 |
| 网格补偿与区间计算 | 原局部梯度补偿及legacy区间实现 | 保留不变 | 原实现的已知审计限制也保留下来，不因复用而消失 |
| 距离单位 | 原数据std1归一化 | 原VT坐标保留，进入q前转换到R2尺度 | 必要的接口单位转换，不改变物理安全定义 |
| 额外诊断 | 本次不依赖其额外诊断 | 同一checkpoint上增加宽区域检查 | 只记录，不改变训练、停止条件或原硬判定 |

本次没有加入QP、安全过滤器、人工可恢复集或新的CP契约；也没有为了得到零违反进一步缩窄原95%～100%的带宽。

### C. “和原论文一样的保证”为什么还不能直接说

必须区分三个层次：

1. **原实现口径的实验结果：已有正面结果。** 窄带实际69点，硬违反0；区域界顺序正确。
2. **原论文定理所需的完整验证：尚未建立。** 原论文Theorem 2.2在规定的安全子水平区域要求期望下降，不是只检查不安全阈值下方最后5%的带。当前扩域诊断存在违反，且窄带补偿后也有1个违反。扩域诊断本身不是原定理的逐项完整实现，因此这里的结论是“证据尚不充分”，不是宣布原论文定理被推翻。
3. **从q代理到实际图像路径的转移：尚未由第22步证明。** `距离→q→PPO`和`图像→DINO/投影→PPO`共用控制器，但输入latent仍有差异。需要为相同最终控制器补上误差覆盖与安全传递论证。

期望下降意味着按指定噪声分布计算的下一步平均屏障值不增加；它不要求每次随机转移都下降。与此同时，不能因为采用期望而随意漏掉定理要求检查的状态。原方法假设、代码筛选、数值界是否有效，都应逐项核对。这一要求对原基线和改进方法相同，不是给DINO方案单方面增加标准。

### D. 对外可以使用与不能使用的表述

| 建议使用的表述 | 不应使用的表述 | 原因 |
|---|---|---|
| “在原SPVC实现的窄带硬检查下，当前R2代理模型实现0/69违反” | “全部10000个状态都通过” | 原窄带只选中了69个点 |
| “同checkpoint扩域后出现103/1332及109/9253硬违反，说明筛选影响检查结果” | “原SPVC能成功完全是因为缩小区域” | 本实验确定了这组模型的范围效应，没有证明唯一原因，也未重跑原cGAN对照 |
| “初始区与不安全区界分离，区域值比率为95.4081%” | “已经证明至少95.41%安全” | 概率解释还依赖动态条件及其他定理前提 |
| “窄带硬违反0，但补偿后违反1” | “SBC的连续域条件全部满足” | 两种检查不同，后者未零违反，且legacy界仍有审计限制 |
| “发现了屏障不等式违反点” | “发现了109次真实碰撞” | 证书条件失败不等于实际闭环发生事故 |
| “第21步对冻结旧策略的轨迹评分通过CP校准” | “第21步CP自动补齐第22步新PPO的证明” | 模型可能已改变，且两种证据的对象、时域与条件不同 |

### E. 第21步与第22步分别证明到哪里

- **第21步：** 459条轨迹均success，CP评分阈值为负。在完整性、事先冻结、独立同分布等条件经核验成立时，可支持登记分布及400步范围内的轨迹统计安全结论。它不是原神经B条件期望下降的验证。
- **第22步：** 使用原SPVC训练与legacy检查，对代理闭环得到窄带零硬违反及正确区域分离。它不使用第21步校准数据作为误差证明，也不自动得到真实图像闭环保证。
- **不能直接合并：** 不能把第21步的99%/99%与第22步的95.41%直接相乘、相加或任选其一作为“最终系统安全率”；需要先确认模型一致，再建立组合定理及各自假设。

### F. 可以直接放入汇报的总结段

> 我们在保持原SPVC训练机制、AEBS动力学、随机噪声和窄带硬检查规则不变的前提下，将原感知前端替换为冻结DINO与R2双路径latent表示。当前代理闭环在原窄带实际选中的69个网格点上实现零硬违反，初始区与不安全区屏障值成功分离。这说明当前方案能够达到原实现的窄带硬检查结果，并非在相同口径下无法训练出候选屏障。同一checkpoint扩大检查范围后出现违反，窄带补偿后也仍有1个违反，因此区域比率95.41%目前只作为数值指标，不能直接解释为已经证明的安全概率。后续需要核验原定理条件，并对最终PPO补充图像路径评估与代理误差转移，才能讨论原论文级别的端到端保证。

### G. 当前最小后续工作

- [x] 记录第22步同口径对照结果及上述结论边界。
- [ ] 读取完整`metrics.json`和`scope_comparison.json`，确认checkpoint哈希、输入尺度和训练停止原因。
- [ ] 固定第22步最终PPO/SBC，评估对应的实际图像路径，不继续调窄带或容忍阈值来追求通过。
- [ ] 按研究目标选择：继续补随机SBC的完整验证/误差转移，或单独报告具有明确前提的有限时域统计保证。

## 0. 实验设计与复现：原窄带与宽区域同checkpoint比较

本次不修改原VT训练损失、原噪声、legacy区间实现或原窄带判定。使用同一组最终PPO/SBC，额外执行一次完整网格诊断，比较三个范围：

| 输出名 | 范围 | 用途 |
|---|---|---|
| `legacy_band` | 原条件 `0.95*归一化不安全下界 < 归一化B < 归一化不安全下界` | 忠实复制原筛选，不额外排除状态 |
| `below_unsafe_safe` | 原不安全区之外，且B低于同一不安全下界 | 去掉95%下限的宽区域对照 |
| `all_safe` | 原网格中全部不在不安全区内的状态 | 更广的压力诊断，不额外删终止点 |

后两项不会影响原训练循环、停止条件或模型选择，也不能直接当作论文定理的完整验证。三个范围均采用原expected计算、原噪声和原硬违反条件 `expected-B>=0`，因此恰好等于0也计违反。额外单列原梯度补偿后违反数。不是拿CP残差盒和原均匀噪声互相比较。

重要修正：原环境以旧数据的`env.std1`归一化距离，R2的q可能使用另一尺度。本次只在适配器中作 `d_old_norm * env.std1 / R2_scale` 单位转换；不修改原物理状态域、动力学或q权重。两种尺度及比例记录在metrics中。若对旧run执行只诊断模式，则按旧metrics恢复其原输入解释，不偷偷改变已保存模型的行为。

### 当前实现进度

- [x] 增加同checkpoint三个范围的完整网格诊断，不提前遇错退出。
- [x] 输出真实检查点数及对应违反率，不再把筛选后零违反误读成检查了全部10000点。
- [x] 同时记录区域界、仅按区域值计算的比率、最终模型哈希与每点结果。
- [x] 空窄带不当作安全证据；非有限数值标为无效。
- [x] 增加`--diagnose-only-from`，可复查原SPVC格式的已有输出，不训练。
- [x] 本地5项轻量单元测试及Python语法检查通过（直接运行测试文件，绕过包初始化对torch的依赖）。
- [x] 用户已在服务器运行并回传第22步输出；本机未进行实际模型/auto_LiRPA运行测试。
- [x] 同checkpoint结果确认“窄带零硬违反、宽域有硬违反”；窄带补偿后仍有1个违反。
- [ ] 完整产物及模型哈希核验，最终PPO的图像路径开发评估。

### 0.1 本机执行：同步代码到原服务器

网络可连接服务器时运行；没有替你连接或运行服务器实验。`--backup`保留被替换文件的备份，无delete，不同步或覆盖实验结果。同步原VT三个文件是为了使服务器与本次阅读的本地实现一致；它们在本次没有算法改动。

```bash
cd /Users/johann/project/work_based_on_spvc/artical-F122
rsync -avR --backup --suffix=.before_scope_compare -e 'ssh -p 20020' \
  Aebs/dino_latent/run_spvc.py \
  Aebs/dino_latent/spvc_policy.py \
  Aebs/dino_latent/compare_spvc_scopes.py \
  Aebs/dino_latent/test_compare_spvc_scopes.py \
  Aebs/VT/train.py Aebs/VT/loop.py Aebs/VT/verify.py \
  root@10.11.154.191:/root/work_based_on_spvc/artical-F122/
```

### 0.2 服务器执行：干净的R2原SPVC基线＋自动范围对照

```bash
cd /root/work_based_on_spvc/artical-F122
/opt/miniconda3/bin/conda run --no-capture-output -n vt \
  python -m unittest Aebs.dino_latent.test_compare_spvc_scopes -v

/opt/miniconda3/bin/conda run --no-capture-output -n vt \
  python -m Aebs.dino_latent.run_spvc \
  --max-iteration-index 500 \
  --timeout-seconds 7200 \
  --stop-on-zero-violation \
  --compare-scopes \
  --output-dir results/dino_expanded_v1/22_R2_original_spvc_scope_compare
```

这里从R2 PPO09开始，由原SPVC交替训练PPO与新SBC；不载入轨迹SBC v3，二者结构和训练条件不同。500为最大迭代索引；7200秒为原训练循环时间预算，最后诊断另需时间。遇到原零违反及目标条件会提前停止。输出目录已有内容则拒绝覆盖，请换后缀。

原循环停止后可能已经更新过一次PPO，因此历史`loop_info`不一定对应最终模型。新增对照会重新包装并检查实际保存的模型，以`scope_comparison.json`为本次范围比较依据；文件内保存其哈希。

### 0.3 如果已有R2原SPVC结果：仅诊断，不重训

仅适用于`run_spvc`生成且包含`sbc.pt`、`latent_ppo_spvc.pt`和`metrics.json`的目录，不适用于第19步的`barrier.pt`。

```bash
cd /root/work_based_on_spvc/artical-F122
/opt/miniconda3/bin/conda run --no-capture-output -n vt \
  python -m Aebs.dino_latent.run_spvc \
  --diagnose-only-from results/dino_expanded_v1/17_R2_original_spvc \
  --output-dir results/dino_expanded_v1/22_R2_original_spvc_scope_diagnostic
```

没有已有结果就只运行0.2，不要先运行这条。若旧run使用了自定义表示/PPO路径，需同时传入原路径参数。

### 0.4 给我看哪些结果

回传最后完整的 `[R2 original-SPVC same-checkpoint scope comparison]` 块和`scope_comparison.json`。重点是三组的`checked`、`hard`、`compensated`和`min_margin`，以及初始/不安全/全域的界。

- 窄带非空且零违反、宽区域有违反：说明筛选对这组模型的通过情况影响很大。
- 窄带也违反：不能仅靠恢复原筛选解释失败。
- 窄带为空：这是空集检查，不能报告得到证书。
- 三组均零违反：是同一legacy计算下的更广样本证据，仍不是图像路径端到端证明。

本实验没有修改DINO/R2图像编码器，也不使用第21步新校准轨迹来训练或选择模型。若原SPVC更新了PPO，第21步针对旧PPO的统计结论不能直接用于这个新PPO。

## 1. 实验目标

本实验只回答一个问题：在不引入CP、残差契约、QP、安全过滤器、人工可恢复集或新验证器的情况下，把原SPVC的`cGAN＋state_net`前端替换为当前R2 DINO双路径表示后，原SPVC训练与验证流程能否跑通，以及能得到什么结果。

## 2. 完整数据流

部署图像路径：

```text
真实图像 → 冻结DINO → R2 projection → 32维latent ┐
                                                   ├→ PPO → 动作
速度 ─────────────────────────────────────────────┘
```

原SPVC训练和VT网格路径：

```text
物理距离 → 冻结q_psi → 32维latent ┐
                                   ├→ PPO → 原AEBS动力学
速度 ─────────────────────────────┘
                                      ↓
                         原人工均匀过程噪声
                                      ↓
                       原SBC训练与VT网格检查
```

两条路径使用同一R2 latent坐标和同一个PPO actor。VT状态网格无法直接生成每个连续状态对应的真实CARLA图像，因此沿用R2的物理代理分支`q_psi`。这一点必须写入结论边界。

## 3. 与原SPVC保持不变的内容

- 原`VTLearner`和交替训练循环；
- SBC结构`[2,16,8,1]`与平方输出；
- PPO actor结构及其原SPVC微调方法；
- `p_lip=2.0`、`l_lip=4.0`、`epsilon=0.1`；
- learner目标概率0.95、verifier目标概率0.9；
- `100×100`状态网格；
- `factor=0.05`定义的原均匀下一状态噪声；
- 原VT验证区域、概率计算和`0.1%`硬违反比例阈值；
- 原legacy IBP实现，不在本基线中替换或强化。

## 4. 唯一方法替换

```text
原版：状态＋z → cGAN → state_net → PPO
本版：距离 → q_psi latent → PPO
部署：图像 → frozen DINO/projection latent → PPO
```

表示模型固定为`08_R2_group_alignment`，初始控制器固定为`09_R2_ppo`。默认从随机SBC开始，不载入旧语义SBC，避免混入其他方法的证书形状。

## 5. 明确不使用的内容

- trajectory-level CP；
- 全局或动作条件残差盒；
- 新scenario鲁棒SBC；
- 可恢复集或viability计算；
- QP可微层；
- runtime safety filter；
- 新IBP或其他强化验证器；
- test集合。

## 6. 运行入口

```bash
cd /root/work_based_on_spvc/artical-F122

/opt/miniconda3/bin/conda run --no-capture-output -n vt \
python -m Aebs.dino_latent.run_spvc \
  --max-iteration-index 500 \
  --timeout-seconds 7200 \
  --stop-on-zero-violation \
  --output-dir results/dino_expanded_v1/17_R2_original_spvc
```

`500`只是增加允许的优化预算，不改变损失、噪声、网格或验证方法。`--stop-on-zero-violation`用于在第一次同时达到零硬违反、区域顺序正确和目标概率时停止，防止下一次PPO更新破坏该checkpoint。

## 7. 输出和判定

主要输出：

```text
results/dino_expanded_v1/17_R2_original_spvc/sbc.pt
results/dino_expanded_v1/17_R2_original_spvc/latent_ppo_spvc.pt
results/dino_expanded_v1/17_R2_original_spvc/metrics.json
```

最低成功条件：

1. `hard_violations=0/10000`；
2. `lb_unsafe > ub_init > domain_min`；
3. `actual_reach_prob >= 0.9`；
4. 保存的checkpoint与达到上述条件的迭代一致。

如果只有违反比例不超过`0.1%`，只能称为通过原代码宽松阈值，不能称为零违反。如果区域顺序错误，则不能计算有效概率下界。

## 8. 可以和不可以声称什么

可以声称：

> 在原SPVC噪声模型、原状态网格、原训练损失和原legacy VT实现下，R2 DINO前端对应的q代理控制路径完成了原SPVC基线训练与检查。

不可以声称：

> 真实图像DINO路径的全部感知误差已经被形式化覆盖。

原因是VT检查使用`distance→q_psi latent`，部署使用`image→DINO/projection latent`。本基线刻意不加入两条路径误差的CP或鲁棒包络。同时，原legacy VT区间实现已有已知审计限制，因此本结果应作为忠实基线，而不是比原论文更强的新形式化保证。
