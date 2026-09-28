# BarrierNet 与 Opt-ODENet 的 QP 部分详细对比

> 更新时间：2026-09-22  
> 对比对象：
> - Wei Xiao, Ramin Hasani, Xiao Li, Daniela Rus, **BarrierNet: A Safety-Guaranteed Layer for Neural Networks**，其后扩展发表于 IEEE Transactions on Robotics。
> - Keyan Miao, Liqun Zhao, Han Wang, Konstantinos Gatsis, Antonis Papachristodoulou, **Opt-ODENet: Neural ODE Controller Design with Differentiable Optimization Layers for Safety and Stability**，L4DC 2025；本文主要参考其 longer version。

论文链接：

- [BarrierNet, arXiv:2111.11277](https://arxiv.org/abs/2111.11277)
- [BarrierNet, IEEE T-RO](https://doi.org/10.1109/TRO.2023.3249564)
- [Opt-ODENet, L4DC 2025正式版](https://proceedings.mlr.press/v283/miao25a.html)
- [Opt-ODENet longer version, arXiv:2504.17139](https://arxiv.org/abs/2504.17139)
- [OptNet：两者使用的可微QP基础](https://proceedings.mlr.press/v70/amos17a.html)

## 1. 一句话结论

BarrierNet 和 Opt-ODENet 都把控制屏障函数约束放进一个可微二次规划（differentiable QP）中，并通过KKT条件把任务损失的梯度传回神经网络。但是二者对QP的定位不同：

> **BarrierNet把QP本身设计成一个高度参数化、环境相关、可以模仿参考控制器的神经网络层；Opt-ODENet把QP设计成一个结构更固定的CBF安全投影器，并把它嵌入Neural ODE轨迹优化和随伴梯度中。**

BarrierNet主要解决：怎样让传统HOCBF不过度保守，并与任意上游神经控制器一起端到端训练。

Opt-ODENet主要解决：怎样在没有预先设计的nominal controller和大规模动作标签时，直接通过连续时间轨迹损失学习控制器，同时让每个时刻的动作满足CBF安全约束。

## 2. 它们共同使用的QP基础

一般凸QP写成：

\[
z^*=\arg\min_z\frac12 z^TQz+q^Tz
\]

满足

\[
Az=b,\qquad Gz\le h,
\]

其中 `Q` 至少需要半正定；为了获得唯一解和稳定的局部微分，通常希望 `Q` 正定，并需要相应的约束正则性条件。

在控制问题中，QP决策变量 `z` 通常就是控制动作 `u`。前向传播时求解QP得到 `u*`；反向传播时不展开求解器的每一次迭代，而是对最优解满足的KKT条件做隐式微分。

QP的拉格朗日函数为：

\[
\mathcal L(u,\nu,\lambda)
=\frac12u^TQu+q^Tu
+\nu^T(Au-b)+\lambda^T(Gu-h),
\]

其中 `ν` 是等式约束乘子，`λ≥0` 是不等式约束乘子。KKT条件包括：

\[
Qu^*+q+A^T\nu^*+G^T\lambda^*=0,
\]

\[
Au^*-b=0,
\]

\[
D(\lambda^*)(Gu^*-h)=0.
\]

对这些方程求微分，可以得到一个KKT线性系统。求解这个线性系统，就能获得QP解对 `Q,q,A,b,G,h` 的局部导数，再由链式法则传给产生这些量的网络参数。

因此，两篇论文中的“可微QP”不是说QP目标函数本身处处光滑就足够了，而是说：**在活跃约束集合稳定、KKT矩阵可逆等正则条件下，最优解映射可以通过隐式函数定理进行局部微分。** 当活跃集切换、约束退化或QP接近不可行时，梯度可能不连续或数值不稳定。

## 3. BarrierNet 的QP

### 3.1 QP形式

BarrierNet的核心QP为：

\[
u^*(t)=\arg\min_{u(t)}
\frac12u(t)^TH(z\mid\theta_h)u(t)
+F(z\mid\theta_f)^Tu(t),
\]

满足每个安全约束对应的HOCBF不等式：

\[
L_f^m b_j(x)
+[L_gL_f^{m-1}b_j(x)]u
+O(b_j(x),z\mid\theta_p)
+p_m(z\mid\theta_{p_m})
\alpha_m(\psi_{m-1}(x,z\mid\theta_p))
\ge0,
\]

以及控制输入边界：

\[
u_{\min}\le u\le u_{\max}.
\]

这里：

- `u`：QP决策变量，也是最终控制动作；
- `z`：环境或感知特征，例如障碍物位置、速度；
- `x`：系统真实状态反馈；
- `b_j(x)≥0`：第 `j` 个安全集合；
- `m`：安全函数相对控制输入的相对阶；
- `H(z)`：QP二次项；
- `F(z)`：QP线性项，可解释为参考动作相关项；
- `p_i(z)>0`：环境相关、可训练的HOCBF penalty；
- `α_i`：class-K函数；
- `O(·)`：HOCBF递推中不直接包含当前控制动作的其余Lie导数项。

如果目标是跟踪参考动作 `u_ref`，常见投影目标

\[
\frac12\|u-u_{ref}\|_H^2
\]

展开后对应

\[
\frac12u^THu-(Hu_{ref})^Tu+\text{constant}.
\]

因此BarrierNet中的 `F(z)` 可以学习与参考控制相关的线性项，但论文给出的形式比固定的欧氏投影更一般：`H` 和 `F` 都可以依赖环境并由网络产生。

### 3.2 “softening”不是松弛安全约束

BarrierNet论文使用“softened HOCBF”这一说法，容易被理解成给约束添加slack并允许违反。实际上其主要做法是把传统HOCBF递推

\[
\psi_i=\dot\psi_{i-1}+\alpha_i(\psi_{i-1})
\]

改为

\[
\psi_i=\dot\psi_{i-1}+p_i(z)\alpha_i(\psi_{i-1}),
\qquad p_i(z)>0.
\]

它通过正的、环境相关的 `p_i(z)` 调节约束强度，减少固定class-K函数造成的保守性，但最终HOCBF不等式仍作为QP硬约束。也就是说：

- 被“软化”的是CBF条件的形状和保守程度；
- 没有在主公式中允许HOCBF约束被违反；
- `p_i(z)` 必须保持正值并满足论文安全定理所需的Lipschitz连续性。

### 3.3 哪些QP量可以训练

BarrierNet允许训练：

| QP量 | 是否可训练 | 含义 |
|---|---|---|
| `H(z|θ_h)` | 是 | 控制代价的局部度量或权重 |
| `F(z|θ_f)` | 是 | 参考动作/性能偏好的线性项 |
| `p_i(z|θ_p)` | 是 | HOCBF约束强度与保守度 |
| `G(x,z)` | 部分间接变化 | Lie导数结构由动力学和屏障决定，`p_i`影响右侧项 |
| `u_min,u_max` | 否 | 物理控制边界 |
| `b_j(x)` | 原论文中预先构造 | 安全集本身不是由QP学习出来的 |

BarrierNet因此不仅学习“QP前面的控制器”，还学习QP目标与约束中的若干参数。

### 3.4 前向和反向传播

前向：

1. 上游网络从 `z` 生成 `H、F、p_i`；
2. 根据 `x`、系统动力学和 `b_j` 构造HOCBF线性不等式；
3. 求解QP；
4. 将 `u*` 施加到系统。

反向：

BarrierNet把HOCBF约束统一写成 `Gu≤h`，其中对第 `j` 个约束：

\[
G_j=-L_gL_f^{m-1}b_j(x),
\]

\[
h_j=L_f^mb_j(x)+O(b_j,z)
+p_m(z)\alpha_m(\psi_{m-1}(x,z)).
\]

论文通过QP的KKT系统计算损失对 `H、F、G、h` 的梯度，再将 `∂loss/∂h` 通过链式法则传给 `p_i` 网络。

### 3.5 BarrierNet中QP的角色

BarrierNet的QP是：

- 神经网络的最后一层；
- 运行时安全控制器的一部分；
- 每个离散控制时刻都必须求解；
- 将参考控制/任务偏好与HOCBF安全约束融合；
- 通过环境相关的penalty自适应改变安全约束强度。

它不是训练完成后可以删除的临时模块。

## 4. Opt-ODENet 的QP

### 4.1 QP形式

Opt-ODENet先由控制网络输出：

\[
u_{nn}=\pi(x;\theta_1),
\]

然后求解：

\[
u_{safe}=\arg\min_u
\frac12u^TQ(u_{nn})u+q(u_{nn})^Tu,
\]

满足

\[
Au=b,\qquad Gu\le h.
\]

论文实际使用：

\[
Q=I,\qquad q=-u_{nn},
\]

没有等式约束，并令

\[
G=-\nabla B(x)^Tg(x),
\]

\[
h=\nabla B(x)^Tf(x)+\alpha(B(x);\theta_2).
\]

于是 `Gu≤h` 等价于：

\[
\nabla B(x)^T(f(x)+g(x)u)
+\alpha(B(x);\theta_2)\ge0,
\]

即标准CBF条件。

由于

\[
\frac12u^Tu-u_{nn}^Tu
=\frac12\|u-u_{nn}\|_2^2
-\frac12\|u_{nn}\|_2^2,
\]

第二项与优化变量 `u` 无关，所以Opt-ODENet的QP本质上是在求：

> **距离神经网络原始动作最近、同时满足CBF约束的安全动作。**

这是一个结构明确的欧氏投影，而不是像BarrierNet那样学习任意的 `H(z)` 与 `F(z)`。

### 4.2 哪些QP量可以训练

| QP量 | 是否可训练 | 含义 |
|---|---|---|
| `Q=I` | 否 | 固定欧氏距离 |
| `q=-u_nn` | 间接训练 | 由控制器网络输出决定 |
| `α(B;θ_2)` | 是 | CBF约束强度、安全与性能折中 |
| `B(x)` | 原论文中已知 | 安全集/CBF先验给定 |
| `f(x),g(x)` | 已知 | 控制仿射动力学 |
| `G,h` | 随状态变化 | 由 `B,f,g,α` 构造 |

论文既允许 `α` 是一个神经网络，也讨论线性形式：

\[
\alpha(B)=\kappa B,
\]

其中 `κ` 可以训练。

### 4.3 QP如何嵌入Neural ODE

Opt-ODENet的完整闭环为：

\[
u_{nn}(t)=\pi(x(t);\theta_1),
\]

\[
u_{safe}(t)=\operatorname{QP}
(x(t),u_{nn}(t);\theta_2),
\]

\[
\dot x(t)=f(x(t))+g(x(t))u_{safe}(t).
\]

ODE求解器沿时间积分闭环动力学。每次ODE函数评估都可能需要调用控制网络和QP，而不只是每条轨迹求解一次QP。

这使它与BarrierNet出现一个非常重要的工程区别：

- BarrierNet通常按照固定控制周期，一步调用一次QP；
- Opt-ODENet训练时，QP位于ODE solver内部，自适应ODE求解器可能在一个记录时间间隔内多次评估QP。

因此Opt-ODENet训练的计算图更长，QP求解次数和梯度计算成本通常更高，但它能直接优化连续时间整条轨迹。

### 4.4 训练目标

Opt-ODENet不要求先用专家或nominal controller生成动作标签。它从初始状态出发，在QP保护下积分闭环轨迹，并使用轨迹任务损失训练控制网络和 `α`。

稳定/收敛目标主要通过CLF型损失实现，例如：

\[
\mathcal V(x)=\max\left\{
0,\nabla V_x(x)^TF(x,u)+\gamma V_x(x)
\right\},
\]

再积分得到：

\[
\ell=\mathbb E\left[\int_{t_0}^{t_f}\mathcal V(x(t))dt\right].
\]

因此Opt-ODENet的职责分配是：

- CBF-QP：把安全当作运行时硬约束；
- CLF loss：在训练目标中推动稳定和收敛；
- Neural ODE：生成连续时间轨迹并连接长期任务损失；
- adjoint method：反向传播整条轨迹的梯度。

需要注意，CLF在主方法中是损失项而不是QP硬约束。其稳定性结论依赖于找到使相应Lyapunov violation loss达到要求的参数，不能简单理解成“只要QP可行就同时自动得到稳定性”。

### 4.5 反向传播的两层隐式结构

Opt-ODENet同时包含两类梯度机制：

1. 对QP的KKT条件做隐式微分，得到 `u_safe` 对 `u_nn`、`α`及QP参数的导数；
2. 对Neural ODE使用adjoint method，从终端/轨迹损失反向积分伴随状态。

简化地写：

\[
\frac{d\ell}{d\theta_1}
=\int
\frac{\partial\ell}{\partial x}
\frac{\partial x}{\partial u_{safe}}
\frac{\partial u_{safe}}{\partial u_{nn}}
\frac{\partial u_{nn}}{\partial\theta_1},dt,
\]

\[
\frac{d\ell}{d\theta_2}
=\int
\frac{\partial\ell}{\partial x}
\frac{\partial x}{\partial u_{safe}}
\frac{\partial u_{safe}}{\partial\alpha}
\frac{\partial\alpha}{\partial\theta_2},dt.
\]

实际实现通过伴随方程和KKT线性系统避免显式构造完整巨大Jacobian。

## 5. QP部分逐项对比

| 对比项 | BarrierNet | Opt-ODENet |
|---|---|---|
| QP决策变量 | 控制动作 `u` | 控制动作 `u` |
| QP输出 | 安全控制 `u*` | 安全控制 `u_safe` |
| 二次项 | `H(z|θ_h)`，可训练、可随环境变化 | `Q=I`，固定 |
| 线性项 | `F(z|θ_f)`，可训练 | `q=-u_nn`，由控制网络间接决定 |
| 目标解释 | 学习任务代价/参考控制与局部控制度量 | 把NN动作欧氏投影到CBF可行集 |
| 安全约束 | 参数化HOCBF，显式支持任意相对阶 | 主公式为一阶CBF，论文另做HOCBF扩展 |
| 可训练安全参数 | 环境相关的正penalty `p_i(z)` | class-K函数 `α(B;θ_2)`或系数 `κ` |
| CBF本身 | 预先定义 | 预先定义且论文明确假设已知 |
| 动力学 | 已知控制仿射动力学 | 已知、时不变控制仿射动力学 |
| 控制边界 | 主QP中明确包含 | 主方法式(9)没有突出输入边界，可作为一般QP约束加入 |
| 等式约束 | 主公式没有 | 一般推导包含；实际CBF-QP中没有 |
| slack变量 | 主公式没有 | 主公式没有 |
| nominal/reference controller | 允许跟踪任意nominal controller；原算法用nominal controller生成数据 | 不依赖预训练nominal controller；NN动作在训练中直接形成 |
| 训练数据 | 原流程依赖nominal controller生成的数据集 | 从初始状态滚动Neural ODE并最小化轨迹损失 |
| QP在网络中的位置 | 上游网络后的最终安全层 | 控制网络后、ODE动力学内部 |
| 反向传播 | QP KKT隐式微分 | QP KKT隐式微分 + ODE adjoint |
| 稳定性处理 | 不是原始QP核心；主要保证安全 | CLF放入轨迹损失，CBF放入QP |
| 推理阶段 | 每个控制时刻都要求解QP | 每个控制/ODE评估时刻都要求解QP |
| 主要可调折中 | `H、F、p_i` | 控制NN与 `α/κ` |
| 主要目标 | 环境自适应、减小HOCBF保守性、模仿/融合参考控制 | 无专家数据地联合学习任务控制与安全投影 |

## 6. 两个QP之间的数学关系

### 6.1 Opt-ODENet可以看作更受约束的投影型QP

如果在BarrierNet中固定：

\[
H=I,
\qquad F=-u_{nn},
\]

并把安全约束写成一阶CBF形式，那么目标部分就与Opt-ODENet一致。

因此从单步QP参数化能力看：

> BarrierNet的QP目标比Opt-ODENet更一般；Opt-ODENet采用了更明确、更容易解释和更容易保证强凸性的安全投影结构。

但不能因此说完整的Opt-ODENet只是BarrierNet的简单特例，因为Opt-ODENet的主要新增内容在QP外部：Neural ODE轨迹、CLF损失以及QP梯度与adjoint method的结合。

### 6.2 `p_i` 与 `α` 的关系

BarrierNet使用：

\[
p_i(z)\alpha_i(\psi_{i-1}),
\]

Opt-ODENet使用：

\[
\alpha(B;\theta_2),
\]

二者都通过学习class-K相关项调节CBF约束强度，但参数化重点不同：

- BarrierNet保留基础 `α_i`，额外学习环境相关正倍率 `p_i(z)`；
- Opt-ODENet直接学习 `α`，或学习线性class-K函数的斜率 `κ`；
- BarrierNet的输入 `z` 可以显式包含障碍物和环境特征；
- Opt-ODENet的 `α` 在主公式中主要以 `B(x)` 和参数 `θ_2` 为输入。

## 7. 安全保证的共同前提和差异

### 7.1 两者都不是“只要加入QP就无条件安全”

两者的安全结论都至少依赖：

1. 系统动力学 `f,g` 与真实系统一致或误差已经被鲁棒约束覆盖；
2. CBF/HOCBF `B` 或 `b_j` 正确定义安全集；
3. 初始状态位于相应安全/HOCBF集合；
4. QP在运行时可行并被足够准确地求解；
5. 控制输入确实按照QP输出施加；
6. 连续时间CBF条件与实际数字控制采样之间的误差得到处理；
7. 感知状态足够准确，或感知误差已进入鲁棒CBF设计。

### 7.2 BarrierNet的额外前提

- `p_i(z)` 始终为正；
- `p_i(z)` 满足论文要求的Lipschitz连续性；
- HOCBF递推集合条件成立；
- 参数化 `H` 必须保持QP凸性，实践中还应保持正定和良好条件数；
- 多个HOCBF和控制边界不能互相冲突。

BarrierNet论文还明确指出：网络只能在离散时刻提供控制，采样间轨迹可能在安全边界附近产生inter-sampling violation。这是连续时间安全定理落到数字控制器时的实际缺口。

### 7.3 Opt-ODENet的额外前提

- `B(x)` 已知且适用于给定安全任务；
- `f,g` 已知并满足控制仿射形式；
- 学到的class-K函数必须保持class-K所需性质，而不能只是任意神经网络输出；
- ODE solver和实际控制执行的离散化误差不能破坏CBF条件；
- CLF损失必须真正达到相应条件，才可把训练收敛解释为稳定性结论。

## 8. QP可行性问题

两篇论文的主QP都把CBF当作硬约束，而且都没有在核心公式中加入安全slack。因此当出现以下情况时，QP可能不可行：

- 控制上下限不足以满足CBF；
- 多个障碍物约束相互冲突；
- 系统已进入不可恢复状态；
- 感知误差导致错误的CBF约束；
- CBF设计与真实动力学不匹配；
- 学到的penalty或class-K参数把约束推得过紧。

工程实现中必须明确不可行处理策略，例如：

- 对性能/CLF约束加slack，但不轻易放松安全约束；
- 对安全约束使用分级优先级；
- 设计backup controller；
- 在进入QP前做可行域监测；
- 对控制边界和不确定性进行鲁棒CBF设计；
- 记录QP status、KKT residual、最小安全余量和求解时间。

如果直接在安全约束上加入slack，那么“QP总能返回动作”不等于“安全仍有保证”。必须单独量化slack对安全结论的影响。

## 9. 数值和实现层面的区别

### 9.1 强凸性

Opt-ODENet固定 `Q=I`，天然强凸；只要可行，控制解唯一，数值条件相对清楚。

BarrierNet允许学习 `H(z)`，表达能力更强，但实现时不能直接让网络任意输出矩阵。通常需要类似：

\[
H(z)=L(z)L(z)^T+\varepsilon I
\]

的参数化来保证正定，否则QP可能退化、失去凸性或产生不稳定梯度。

### 9.2 活跃约束切换

两种方法的安全动作都是分段光滑函数。当某个障碍物约束从非活跃变成活跃时，QP解的Jacobian可能跳变。训练中表现为：

- 梯度尖峰；
- 接近边界时优化震荡；
- 严格互补条件不满足时KKT矩阵病态；
- 小数值误差改变活跃集。

### 9.3 计算量

若控制维数小、约束数量少，两者的单次QP通常不大。真正差异来自调用次数：

- BarrierNet：通常每个控制周期调用一次；
- Opt-ODENet：QP位于ODE函数中，训练时ODE solver可能多次调用，且反向还需要adjoint与QP敏感度。

Opt-ODENet论文也把提高QP层计算效率列为后续工作。

## 10. 训练流程对比

### 10.1 BarrierNet

```text
构造 CBF/HOCBF
    ↓
准备 nominal controller 或专家优化器
    ↓
生成状态/环境特征 → 参考动作数据
    ↓
上游网络输出 H、F、p_i
    ↓
BarrierNet QP 输出安全动作
    ↓
与专家/任务目标计算损失
    ↓
通过 KKT 隐式微分更新网络
```

### 10.2 Opt-ODENet

```text
构造已知 CBF 与 CLF 型任务损失
    ↓
从初始状态开始
    ↓
控制网络输出 u_nn
    ↓
CBF-QP 将 u_nn 投影为 u_safe
    ↓
ODE solver 积分闭环轨迹
    ↓
计算轨迹 CLF/任务损失
    ↓
ODE adjoint + QP KKT 隐式微分
    ↓
更新控制网络和 α/κ
```

## 11. 哪一种QP更适合什么任务

### 更适合BarrierNet的情况

- 已有可靠的nominal controller或专家轨迹；
- 希望把安全层接到现有神经控制器后面；
- 安全约束具有较高相对阶，需要HOCBF；
- 障碍物和环境特征变化明显，希望约束强度随环境自适应；
- 希望学习非欧氏的控制修正代价 `H(z)`；
- 重点是降低CBF保守性和模仿高质量参考控制。

### 更适合Opt-ODENet的情况

- 没有专家安全动作或nominal controller；
- 动力学模型已知且可以使用ODE solver；
- 希望直接从轨迹任务目标学习控制器；
- 需要把长期稳定/收敛性能纳入训练；
- 希望QP保持简单、强凸、解释为最近安全动作投影；
- 能接受训练时反复求解ODE和QP带来的计算成本。

## 12. 对当前SafePVC/AEBS项目的意义

这两种方法都属于**运行时QP安全层**：部署时仍然要在每个控制时刻求解QP。因此它们和当前项目想要的“最终standalone PPO、不调用safety filter”目标存在根本差异。

如果直接采用BarrierNet或Opt-ODENet：

- 可以得到结构清楚的在线CBF保护；
- 但并没有真正去掉runtime safety filter；
- 仍需要预先给出适用于AEBS的CBF/HOCBF；
- 若CBF依赖人工停车边界，通用性问题依然存在；
- 视觉感知误差若未进入鲁棒CBF，QP安全结论仍可能不成立。

它们在当前项目中更合适的角色是：

1. 作为安全教师产生动作，再蒸馏到standalone policy；
2. 作为在线安全基线，与无filter策略比较性能和计算开销；
3. 用于生成安全轨迹，为通用scenario certificate预训练控制器；
4. 研究“运行时QP保证”和“离线PAC证书”之间的差异。

如果目标改成“允许部署时保留QP，优先获得在线安全约束”，推荐先实现Opt-ODENet式固定投影QP，因为：

- `Q=I` 实现简单；
- 强凸性天然成立；
- 与现有PPO动作的关系清晰；
- 可以直接观察QP干预量；
- 不需要一开始就学习 `H`；
- 更容易诊断不可行与数值问题。

在此基础上，再逐步加入BarrierNet的环境相关 `p_i(z)`，而不是一次性同时学习 `H、F、p_i`。

## 13. 最容易混淆的几个结论

### 误解1：Opt-ODENet的QP比BarrierNet“更高级”

不准确。Opt-ODENet的单步QP更简单、更受约束；其创新重点是把differentiable CBF-QP与Neural ODE、CLF轨迹损失和adjoint method结合。

### 误解2：BarrierNet的softened HOCBF允许安全约束被违反

不准确。它主要通过正的环境相关倍率调整HOCBF保守性，HOCBF不等式仍是硬约束。

### 误解3：QP训练完成后可以删除

两篇论文都不是这样。QP输出就是正式安全动作，推理时仍需要QP。

### 误解4：QP可行就同时证明稳定和安全

QP中的CBF主要负责安全。Opt-ODENet的稳定/收敛由CLF型训练损失承担；BarrierNet的任务性能取决于参考控制与训练目标。

### 误解5：可微QP处处光滑

不准确。QP解通常是分段光滑的；活跃约束切换、退化和不可行都会影响梯度。

## 14. 最终总结表

| 核心问题 | BarrierNet | Opt-ODENet |
|---|---|---|
| QP到底做什么 | 学习任务代价并满足参数化HOCBF | 将NN动作投影到CBF可行集 |
| 学习QP哪些部分 | `H、F、p_i` | `q=-u_nn`对应的控制NN与 `α/κ` |
| 安全约束 | 可训练penalty调节的HOCBF硬约束 | CBF硬约束 |
| 性能来源 | nominal controller/训练数据和可学习目标 | Neural ODE整轨迹损失 |
| 稳定性来源 | 不是原始QP主要内容 | CLF型loss |
| 反向传播 | KKT隐式微分 | KKT隐式微分加ODE adjoint |
| QP灵活性 | 高，但数值设计更复杂 | 低一些，但结构简单、强凸、易解释 |
| 是否需要运行时QP | 需要 | 需要 |
| 最大优势 | 环境自适应、HOCBF、降低保守性 | 不依赖专家动作、直接轨迹优化 |
| 最大风险 | 学习型QP参数的凸性、可行性和采样间安全 | ODE内反复QP的计算量、已知CBF/模型假设 |

最准确的关系可以概括为：

> **BarrierNet是在学习“怎样构造一个更灵活的安全QP层”；Opt-ODENet是在学习“怎样让一个结构化的安全QP层参与连续时间整轨迹最优控制训练”。**

## 参考文献

1. Xiao, W., Hasani, R., Li, X., and Rus, D. *BarrierNet: A Safety-Guaranteed Layer for Neural Networks*. arXiv:2111.11277, 2021.
2. Xiao, W., Wang, T.-H., Hasani, R., Chahine, M., Amini, A., Li, X., and Rus, D. *BarrierNet: Differentiable Control Barrier Functions for Learning of Safe Robot Control*. IEEE Transactions on Robotics, 2023.
3. Miao, K., Zhao, L., Wang, H., Gatsis, K., and Papachristodoulou, A. *Opt-ODENet: Neural ODE Controller Design with Differentiable Optimization Layers for Safety and Stability*. L4DC, 2025.
4. Amos, B., and Kolter, J. Z. *OptNet: Differentiable Optimization as a Layer in Neural Networks*. ICML, 2017.

