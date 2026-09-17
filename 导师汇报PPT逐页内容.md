# SafePVC改进工作导师汇报PPT逐页内容

> 建议时长：25分钟主汇报 + 10分钟讨论  
> 主汇报：23页  
> 附录：6页  
> 汇报目标：让导师看清“原方法有什么问题、实际做了哪些实验、哪些路线失败、最终证明了什么”。

## 汇报时间分配

| 部分 | 页码 | 时间 |
|---|---:|---:|
| 问题与总体思路 | 1–6 | 5分钟 |
| 感知、契约与负结果 | 7–13 | 7分钟 |
| 安全教师与无filter PPO | 14–17 | 5分钟 |
| 形式验证与SBC | 18–21 | 6分钟 |
| 结论与讨论 | 22–23 | 2分钟 |

---

## 第1页 封面

### 标题

**面向视觉AEBS的感知契约、安全策略编译与连续域验证**

副标题：SafePVC论文复现与改进工作汇报

屏幕上只放：

- 汇报人姓名
- 导师姓名
- 日期
- 一张简洁的AEBS场景图

### 口头讲解

“这次工作从原SafePVC的复现和问题检查开始，最终建立了一套从图像感知、误差契约、无filter PPO到连续域形式验证的完整闭环。我会同时汇报成功结果和失败实验。”

---

## 第2页 最终结果摘要

### 标题

**实验链条已完成，最终策略与两层证书均通过**

### 屏幕内容

- 最终standalone PPO：4种感知模式均 `100% success / 0% unsafe / 0% timeout`
- 连续域IBP：3624个evaluated cells，2002个certified cells，`0 unresolved`
- 可恢复域占矩形operational domain的85.53%
- 物理不可恢复域占14.47%
- 有限状态SBC：`verified`，`B(R/T/U)=0/0/1`，条件 `beta=0`

### 建议图表

左侧放系统链条：图像、语义、PPO、动力学。右侧放三个最终数字：`800/800 success`、`0 unresolved`、`beta=0 conditional`。

### 口头讲解

“先给结论。最终部署只使用PPO，不在运行时调用safety filter。经验测试全800回合通过。在契约内感知误差下，可恢复域的连续验证没有留下未决cell。最后又把这个闭包关系写成了有限状态SBC。”

---

## 第3页 研究问题

### 标题

**视觉控制的安全问题同时涉及感知、策略和动力学**

### 屏幕内容

1. 图像估计的距离存在误差
2. PPO会将小感知误差放大成不同制动动作
3. 有限回合不出事，不能覆盖所有连续状态
4. 有些状态已经物理上无法刹停

核心问题：

> 如何训练一个不依赖运行时filter的控制器，并对所有契约内感知结果给出连续域安全证据？

### 口头讲解

强调“控制器经验表现好”和“连续状态已证明”是两个层次。

---

## 第4页 原SafePVC方法

### 标题

**原SafePVC使用视觉代理、PPO和可学习随机barrier**

### 屏幕内容

画原方法流程：

```text
state -> cGAN/视觉代理 -> state estimation -> PPO -> dynamics
                                      |
                                      +-> stochastic barrier certificate
```

原方法目标：

- 用学习的observation model代表高维视觉
- 用SBC给出概率安全下界
- 根据counterexample联合修改barrier和controller

### 口头讲解

这页只客观说原文在做什么，不立即否定。

---

## 第5页 代码与证明链审查

### 标题

**复现后发现四个会影响结论的口径问题**

### 屏幕内容

| 问题 | 后果 |
|---|---|
| 感知误差与process disturbance混用 | 同一误差可能被重复计算 |
| 训练观测与验证观测不一致 | controller在最坏误差下失效 |
| terminal、goal、unsafe和验证域界限不清 | barrier约束会互相冲突 |
| success和timeout统计不够严格 | “停住不撞”可能被误认为好策略 |

### 口头讲解

“所以我没有一开始就继续加epoch或调barrier权重，而是先统一数据含义、终止条件和验证域。”

---

## 第6页 全部工作路线

### 标题

**实验从原链路复现逐步转向物理引导的连续验证**

### 屏幕内容

可直接使用已生成的论文风格架构图：`figures/current-architecture-paper.png`。

用一条横向时间线，每个节点只写短语：

```text
原工程审查
  -> 语义encoder
  -> conformal契约
  -> robust PPO负结果
  -> neural SBC负结果
  -> anti-stall安全教师
  -> standalone PPO
  -> 形式反例修复
  -> split-IBP
  -> 有限状态SBC
```

颜色规则：灰色=完成的基础工作，红色=失败但有结论，绿色=最终采用。

### 口头讲解

“整个工作不是一次训练得到的。前半部分负责定义正确问题，中间部分排除不可行路线，后半部分才是最终控制器和证书。”

---

## 第7页 AEBS环境与评估口径

### 标题

**任务需要同时满足不碰撞和完成制动过程**

### 屏幕内容

- 状态：`distance d`、`speed v`
- 动作：制动强度，上限3 m/s²
- operational domain：`d=[5,16]m`，`v=[0.5,3]m/s`
- 五类互斥结局：`success / unsafe / stopped-safe / out-of-domain / timeout`
- 每个实验统一记录成功率、危险率、超时率、步数和return

### 建议图

二维 `distance-speed` 示意图，标出unsafe边界、goal/terminal带和初始区域。

---

## 第8页 语义感知模型

### 标题

**低维语义距离显著减少感知误差**

### 屏幕内容

| 模型 | MAE | RMSE |
|---|---:|---:|
| legacy real-image estimator | 2.8619 m | 3.4540 m |
| semantic encoder | **0.1004 m** | **0.1934 m** |

其他结果：

- 最大绝对误差：1.2678m
- 语义输入下的控制动作MAE：0.0443
- 数据划分：train 240、calibration 80、test 80

### 建议图表

用两根并列柱表示MAE和RMSE，避免放训练loss曲线。

### 口头讲解

“这里的改进不只是误差更小。它把高维图像问题转成了可校准、可验证的一维距离区间问题。”

---

## 第9页 语义误差契约

### 标题

**状态条件conformal契约覆盖率97.5%**

### 屏幕内容

| 真实距离 | 契约半径 |
|---|---:|
| 5.00–7.75m | 0.0834m |
| 7.75–10.50m | 2.7757m |
| 10.50–13.25m | 0.1649m |
| 13.25–16.00m | 0.4160m |

对照：

- global conformal：91.25% coverage
- Mondrian normalized conformal：92.50% coverage，第一分箱quantile=27.9975
- 最终state-conditional contract：97.50% coverage

### 口头讲解

主动指出7.75–10.5m分箱很宽。说明最终验证使用了这个完整宽区间，没有为了通过而缩小契约。

---

## 第10页 robust controller实验

### 标题

**简单混合最坏观测训练反而降低了控制性能**

### 屏幕内容

训练：20,000 timesteps，`50% exact + 25% uniform + 25% endpoint`

| 模式 | baseline unsafe | robust unsafe | baseline return | robust return |
|---|---:|---:|---:|---:|
| exact | 0% | 0% | 405.8 | 282.8 |
| uniform | 1% | **21.5%** | 405.5 | 280.7 |
| random-boundary | 17% | **28%** | 400.7 | 278.5 |
| worst-endpoint | 100% | 100% | 241.8 | 165.4 |

结论：该checkpoint不采用。

### 口头讲解

“这个负结果排除了一条很直觉的路线：将契约端点直接混进PPO训练，并不能保证学到安全策略。”

---

## 第11页 不确定性拆分

### 标题

**感知误差和过程残差必须分开建模**

### 屏幕内容

两条不同路径：

```text
感知误差: true state -> perceived distance -> PPO action变化
过程残差: predicted next state -> actual simulator next state
```

- 旧 `03_uncertainty`：speed support约 `[-0.1494, 0.1158]`，来自感知导致的动作差
- 新 `03_process_uncertainty`：16个cell均为零residual
- 最终MVP口径：契约内感知误差 + 确定性动力学

### 口头讲解

“零残差不代表真实车辆没有噪声，只代表现有simulator数据支持的是确定性MVP。”

---

## 第12页 neural SBC实验

### 标题

**多种neural SBC训练策略均未获得全域证书**

### 屏幕内容

不要把全部12个版本都放上去，主页只放代表性数据：

| 方法 | violations | min margin |
|---|---:|---:|
| first robust SBC | 4170/6400 | -0.529670 |
| grid training | 1096/6400 | -1.024010 |
| top-k warm start | 975/6320 | -1.065387 |
| terminal-value correction | 3655/5419 | -0.053706 |
| low-speed curriculum | 2576/4483 | -0.001793 |
| standalone-PPO neural SBC | 2769/4483 | -0.165517 |

已尝试：权重放大、staged training、hard-case max loss、top-k、goal/terminal域修正、viability domain、curriculum、softplus-square。

### 口头讲解

“这些不是只有设想，都已实现并运行。违反数不能直接横比，因为期间修正了验证域。共同结论是自由neural barrier没有证成。”

---

## 第13页 neural SBC失败原因

### 标题

**失败来自证书结构和域边界，单纯增加epoch无法解决**

### 屏幕内容

- 原矩形域包含最大制动也无法挽回的状态
- terminal、unsafe、initial和strict decrease条件在边界上存在冲突
- 自由神经barrier同时拟合区域值和一步下降，容易只得到折中解
- 固定网格LP共80,045条约束，因终止/unsafe插值冲突返回infeasible
- 同一LP报告一步robust unsafe successor为0，说明controller本身未必不安全

### 建议图

一张 `distance-speed` 图，用不同颜色标出initial、terminal、unsafe和不可恢复边界，比文字更容易讲清冲突。

---

## 第14页 safety filter路线

### 标题

**安全过滤器先解决防撞，anti-stall再解决长时停滞**

### 屏幕内容

| 模式 | baseline | global max | adaptive bin | anti-stall |
|---|---|---|---|---|
| exact | 100% success | 100% | 100% | **100%** |
| uniform | 96% success, 4% unsafe | 75.5% success | 90% success | **100%** |
| random-boundary | 86.5% success, 13.5% unsafe | 100% timeout | 100% timeout | **100%** |
| worst-endpoint | 100% unsafe | 100% success | 100% success | **100%** |

所有anti-stall结果均为0% unsafe和0% timeout。

### 口头讲解

“第一版filter确实防撞，但在random-boundary中所有回合都超时。所以我们将任务完成率作为同等重要指标，加入anti-stall后才得到可用教师。”

---

## 第15页 为什么有filter还要训练PPO

### 标题

**safety filter只作为离线教师，最终部署仍然是单一PPO**

### 屏幕内容

训练阶段：

```text
baseline PPO + semantic contract + anti-stall filter -> teacher actions
teacher actions + retention samples -> student PPO
```

部署阶段：

```text
semantic observation -> standalone PPO -> braking action
```

不加载filter JSON，不在线求解优化问题。

### 口头讲解

“filter提供可解释的安全行为，PPO负责把它编译成可直接部署的策略。这样后续才能直接对一个固定神经网络做输出界验证。”

---

## 第16页 standalone PPO蒸馏结果

### 标题

**第一版全超时，第二版通过低速标签修正恢复任务完成**

### 屏幕内容

| 版本 | teacher MAE | exact | uniform | random boundary | worst endpoint |
|---|---:|---|---|---|---|
| v1 | 0.3569 | 100% timeout | 100% timeout | 100% timeout | 100% timeout |
| v2 | 0.1057 | 100% success | 100% success | 100% success | 100% success |

v1失败原因：低速区轻微过制动经长时闭环累积，车辆无法到达目标。

v2修正：低速临界样本加密、恢复标签、增加过制动惩罚。

### 口头讲解

不要把“epoch更多”当成主要解释。真正的改动是低速标签和闭环行为。

---

## 第17页 证书迁移与形式反例

### 标题

**粗网格100%通过后，密集检查仍发现了窄边界反例**

### 屏幕内容

| 检查 | 规模 | 结果 |
|---|---:|---|
| 粗网格 | 4483 states、33 semantic samples/state | teacher/student坆0 violations，min margin=0.003987m |
| 高速局部加密 | 18,646 states、129 samples/state | student出现少量反例，min margin约-0.0226m |

最坏区域：`d≈7.14m`，`v≈2.60m/s`，感知契约端点。

### 建议图

一张局部 `distance-speed` 散点图，反例用红点标出。

### 口头讲解

“这一步说明经验通过和粗网格通过都还不够。形式工具的价值是告诉我们具体在哪个状态和哪个感知端点下制动不足。”

---

## 第18页 反例定向修复

### 标题

**联合回放同时保留了高速和低速临界行为**

### 屏幕内容

| 版本 | 改动 | 结果 |
|---|---|---|
| result 07 | 第一次高速局部修复 | violations 19降至16 |
| result 08 | 高速饱和目标 | 高速dense violations=0 |
| result 13 | 只修低速边界 | 高速反例回归，不采用 |
| result 14 | 478个低速 + 78个高速临界状态联合回放 | dense violations=0，四种rollout全通过 |

最终result 14：800回合中800 success、0 unsafe、0 timeout。

### 口头讲解

“result 13是一个很重要的负结果。它证明只修新反例会破坏之前已修好的区域。result 14因此使用高低速联合回放。”

---

## 第19页 物理可恢复域

### 标题

**离散停车距离区分了策略问题和物理不可恢复状态**

### 屏幕内容

$$
\mathcal R=\{(d,v):d-6-S_{\mathrm{disc}}(v;0.5,3,0.05)\ge0\}.
$$

关键数据：

- continuous recoverable/unrecoverable：5441/247
- discrete recoverable/unrecoverable：5419/269
- discrete recoverable but rollout-unsafe：0
- continuous formula过度乐观、被discrete formula排除的点：14
- initial-region minimum discrete margin：7.47m

### 建议图

画 `distance-speed` 可恢复边界，边界上方为R，下方为物理不可恢复区。

### 口头讲解

“这里的14.47%不是验证失败。这些状态在当前最大制动和离散时间步下本来就无法保证恢复。”

---

## 第20页 split-IBP验证方法

### 标题

**连续验证比较PPO动作下界与物理所需制动**

### 屏幕内容

对每个可恢复state cell：

1. 根据conformal contract生成连续语义输入区间
2. 用Tanh/Linear IBP计算PPO制动下界 $a_i^-$
3. 由离散动力学计算保持下一状态可恢复所需的 $a_{i,\mathrm{req}}^+$
4. 验证 $a_i^-\ge a_{i,\mathrm{req}}^+$
5. 界过松时自适应分割state cell和semantic interval

### 建议图

一张五步流程图。不要把IBP全部递推公式放在主页。

### 口头讲解

“采样是在区间内取很多点，IBP则给整个输入盒子一个保守输出下界。自适应分割用于减小这个保守性。”

---

## 第21页 连续验证收敛与最终结果

### 标题

**策略修复和数值界收紧将未决cell从703降到0**

### 屏幕内容

| 阶段 | 主要改动 | unresolved cells | unresolved area |
|---|---|---:|---:|
| result 11 | 首次全域IBP | 703 | 0.009752% |
| result 12 | 修正低速物理阈值 | 627 | 约0.01% |
| result 14 | 联合边界修复 | 248 | 0.003440% |
| result 15 | semantic split=64 | 135 | 0.001873% |
| result 16 | semantic split=256, depth=9 | **0** | **0%** |

result 16：

- evaluated=3624
- certified=2002
- certified area=85.5347%
- physically unrecoverable area=14.4653%
- runtime=20.77s

### 口头讲解

“前面几步同时包含问题定义修正、策略修复和数值界收紧。result 16的0 unresolved才是最终闭包结果。”

---

## 第22页 有限状态SBC

### 标题

**有限状态supermartingale barrier将连续闭包结果写成SBC形式**

### 屏幕内容

三个抽象状态：

- `R`：可恢复运行状态
- `T`：安全终止状态
- `U`：危险或物理不可恢复状态

result 16支持的robust relation：`R -> {R,T}`

LP条件：

$$
B(s')\le B(s),\quad B(T)=0,\quad B(U)\ge1,\quad B(R)\le\beta.
$$

结果：`verified`，`B(R/T/U)=0/0/1`，`beta=0`，LP residual=0，runtime=0.273s。

### 口头讲解

“这个SBC不再训练任意神经函数，也不强制终止集中每步严格下降。它是对result 16连续闭包证据的有限状态形式化，而不是一个独立重复的神经网络验证器。”

---

## 第23页 准确结论、贡献与后续工作

### 标题

**当前成果是契约内的闭环条件安全证据**

### 屏幕内容

已完成：

- 语义encoder和状态相关感知契约
- 将anti-stall safety teacher编译到无filter PPO
- 基于形式反例的高低速联合修复
- 物理可恢复集与sound split-IBP连续验证
- 有限状态SBC条件证书

必须保留的限制：

- 97.5%是单帧test coverage，不是整条轨迹安全率
- `beta=0`以契约成立和process residual=0为条件
- 85.53%是已证可恢复面积占比，不是安全概率
- 当前只有AEBS单benchmark和有限感知数据

下一步：独立轨迹数据校准、真实process residual、额外benchmark和消融实验。

### 口头收尾

“这项工作最终不是把一个neural SBC强行调到通过，而是找到了更符合AEBS物理结构的证明路线。现在控制器、反例修复、连续验证和SBC表达已经跑通。最大的剩余问题是将单帧感知coverage升级为轨迹级概率结论。”

---

# 答辩附录

## 附录A 全部neural SBC实验表

放完整版本列表：`robust_sbc`、`gridtrain_diag`、`unsigned_staged`、`hardcase_maxloss`、`topk_warmstart`、`filtered_zero_process`、`complete_domain`、`goal_guided`、`viability_domain`、`terminal_value`、`low_speed_curriculum`、`softplus_square`、`standalone_ppo`。

这页不在主汇报展开，导师追问“到底试了哪些SBC改动”时再打开。

## 附录B safety filter完整指标

放baseline、global-max、adaptive observed-bin、adaptive anti-stall在四种感知模式下的success、unsafe、timeout、steps、intervention和extra braking。

## 附录C result 14完整rollout结果

放4×200 episodes的success、unsafe、timeout、mean steps，并明确评估时没有调用filter。

## 附录D result 16验证参数

放operational domain、semantic split=256、state max depth=9、3624 evaluated、2002 certified、0 unresolved、20.77s runtime。

## 附录E 安全结论的数学表述

放下面两个公式：

$$
\pi_{14}(\xi)\ge a_{\min}(s),\qquad
s\in\mathcal R,\ \xi\in\mathcal C_\xi(s),
$$

$$
f(s,\pi_{14}(\xi))\in\mathcal R
\quad\text{or enters the safe terminal set}.
$$

然后列出五个假设：初始状态在R内、二维动力学正确、process residual=0、感知位于contract内、使用result 14 checkpoint与相同动作裁剪。

## 附录F 导师可能追问的问题

### 1. 为什么不继续调neural SBC？

回答：已尝试多种权重、训练课程、输出形式和验证域。失败集中在结构和边界冲突，而AEBS已有可直接计算的停车物理边界。物理证书更容易解释，也更容易连续验证。

### 2. 有了IBP，为什么还需要SBC？

回答：IBP证明神经策略在连续感知区间上的动作下界，物理闭包证明下一状态仍可恢复。有限状态SBC则将这个闭包关系统一写成supermartingale barrier表达。连续证据主要来自IBP，SBC是最终抽象层证书。

### 3. `beta=0` 是不是表示事故概率为0？

回答：不是无条件现实概率。它是在每步感知误差都位于契约内且process residual=0时，抽象系统离开可恢复集的上界。

### 4. 85.53%是不是安全率？

回答：不是。它是operational rectangle中物理可恢复且已证成的面积比例。剩下14.47%是当前最大制动下物理不可恢复的区域。

### 5. 97.5% coverage能否推出整条轨迹的安全率？

回答：不能直接推出。闭环中的感知误差可能随状态和历史变化。下一步需要独立轨迹数据上的trajectory-level calibration或有效的时间风险记账。

### 6. 这个工作的论文新意是什么？

回答：建议把组合贡献表述为：状态相关感知契约、safety teacher到无filter policy的编译、基于形式反例的联合边界修复、物理可恢复集与sound continuous verification的整体方法。有限状态SBC是证书表达的补充。正式投稿前还需要系统文献对比和额外benchmark。

---

# 制作PPT时的版式建议

- 主色使用深蓝和灰色，绿色只标通过，红色只标失败/反例。
- 每页保留一个中心问题，表格超过6行时移到附录。
- 主汇报只保留能支持当页结论的数字，完整metrics放附录。
- 流程图中明确区分“训练期filter”和“部署期无filter”。
- 第21页是整个实验的核心证据页，保证字号充足，不要堆放其他内容。
- 第23页必须保留限制条件，避免导师追问时出现“把条件安全说成无条件概率安全”的问题。
