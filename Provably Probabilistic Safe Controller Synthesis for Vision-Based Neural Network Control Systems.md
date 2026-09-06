# Provably Probabilistic Safe Controller Synthesis for Vision-Based Neural Network Control Systems

Anonymous Author(s) 

## Abstract

The prevalence of nonlinear systems in safety-critical domains calls for controllers with safety guarantees, while vision-based control relies on high-dimensional images complicates both decision-making and formal safety analysis under uncertainty. This paper proposes a provably safe controller synthesis method for vision-based neural network control systems. We first employ a conditional generative adversarial network (cGAN) to approximate the mapping from system states to visual observations and combine it with RL-based pretraining to build a verifiable closed-loop structure. A data-driven model quantifies uncertainties from environmental perturbations, while martingale theory guides the learning of a stochastic barrier certificate (SBC) to provide rigorous probabilistic safety bounds. Furthermore, counterexamples from verification are used to alternately refine both the controller and certificate networks, ultimately yielding a controller with formally provable probabilistic safety guarantees. Experimental results on widely studied benchmarks demonstrate the efficiency and effectiveness of our approach. 

## Keywords

vision-based neural network control system, formal verification, martingales, stochastic barrier certificate, reinforcement learning 

## 1 Introduction

Neural network control systems (NNCSs) [12, 28] with strong nonlinear approximation are widely used in safety-critical fields like autonomous driving, robotics, and UAVs [6, 18]. However, NNs are inherently vulnerable to perturbations [3, 4], raising safety and reliability concerns. Vision-based NNCSs, which depend on high-dimensional images captured by sensors [15, 22], introduce significant complexity, rendering the synthesis and formal verification of safe controllers substantially more challenging. 

Despite significant advances in formal verification of NNCSs [7, 19, 31, 35], the verification of vision-based NNCSs remains at an early stage. The main challenge arises from the complexity of visual perception networks, which typically employ convolutional or transformer-based architectures, and the theoretical difficulty of reasoning about all possible high-dimensional visual inputs. A key strategy to address this challenge is the approximation of the visual observation model, enabling vision-based controllers to map states to image observations and determine control actions from these observations. Existing approaches often utilize conditional generative adversarial networks (cGANs [21]) to approximate the perception system and integrate the cGAN with the controller into a unified network. This results in a manageable mapping from the explicitly defined low-dimensional state space and latent variables to control actions, facilitating the direct application of existing NN verification techniques to assess the safety of vision-based NNCSs. 

However, conventional NN verification methods exhibit notable limitations when applied to this field. For instance, reachabilitybased frameworks that combine cGANs with system analysis [20] tend to produce large over-approximation errors, leading to overly conservative verification results, high computational costs, and frequent false alarms in multi-step temporal reasoning. Additionally, these approaches usually provide only qualitative safety assessments and fail to yield quantitative safety guarantees under environmental perturbations. In contrast, deductive verification methods based on barrier certificates (BCs [23]) can offer formal safety guarantees over infinite time horizons. Recent advances in neural BCs have significantly improved computational efficiency and scalability [9, 36], yet their extension to vision-based verification remains largely unexplored. Although some attempts [10, 32] combine safety synthesis for vision-based controllers with control barrier function (CBF [2]), these methods rely on online optimization during execution, introducing substantial computational overhead and still lacking formal global safety guarantees. In addition, the supervised learning paradigm often suffers from limited expressiveness and unstable loss convergence. 

To address these challenges, this paper proposes a provably probabilistic safety controller synthesis framework for vision-based NNCSs and establishes a formally computable lower bound on safety probability. The framework uses a cGAN-based visual observation model to approximate system states from raw observations. These estimated states are then used to pre-train a highperformance controller with reinforcement learning. Subsequently, state disturbances caused by environmental perturbations are modeled from data, and their evolution is tracked throughout policy updates. This leads to a verifiable closed-loop system that supports formal safety analysis under uncertainty. Building on this, we integrate martingale theory [5, 13] with stochastic barrier certificates (SBC) to verify the probabilistic safety of systems over infinite-horizon time and derive rigorous safety lower bounds. For computational efficiency, neural certificate networks are used to construct and train the SBC, while counterexamples generated during verification iteratively refine both the controller network and certificate candidates, ultimately producing a valid SBC and a controller provably safe with respect to it. 

To the best of our knowledge, our method is the first to jointly leverage RL and SBC to synthesize high-performance vision-based controllers while providing formal safety guarantees of probabilistic safety bounds over unbounded steps. The proposed approach achieves both strong control performance and high-probability safety, demonstrating its effectiveness and practicality for safe controller synthesis in vision-based NNCSs. The main contributions of this paper can be summarized as follows: 

• We propose a provably probabilistic safe controller synthesis framework for vision-based neural network control systems, providing formal probabilistic safety guarantees. 

• We develop a learnable stochastic barrier certificate which leverages martingale theory and data-driven disturbance to enable probabilistic verification with formally certified lower bounds on safety. 

• We synthesize provably safe controllers under the guidance of stochastic barrier certificates and iteratively refine both the certificate and controller networks using counterexamples, improving efficiency and practicality. 

## 2 Preliminaries

We consider a discrete-time dynamical system: 

$$
s _ {t + 1} = f (s _ {t}, u _ {t}), \quad o _ {t} = g (s _ {t}, z _ {t}), \quad u _ {t} = \pi (o _ {t})
$$

with initial state $s _ { 0 } \in S _ { 0 } .$ . Here $f : S \times U  S$ denotes the system dynamics, where $S \subseteq \mathbb { R } ^ { m }$ represents the continuous state space and $U \subseteq \mathbb { R } ^ { n }$ the control action space. The variables $s _ { t }$ and $u _ { t }$ denote the system state and control input at time step ??, respectively. The mapping ?? $: S \times Z \to O$ characterizes the observation process, where $z _ { t } \in Z$ captures unobserved environmental perturbations, $Z \subseteq \mathbb { R } ^ { p }$ denotes the perturbation space, $o _ { t } \in O$ denotes the corresponding visual observation, and $\bar { O } \subseteq \mathbb { R } ^ { H \times W }$ represents the observation space consisting of $H \times W$ grayscale images. The policy $\pi : O \to U$ represents a vision-based control mechanism that maps observations $o _ { t }$ to the actions $u _ { t }$ . In this formulation, $m , n , p$ ∈ N denote the dimensions of the respective spaces, and $H , W \in$ N denote the height and width of the image observations. 

In this setting, the system dynamics ?? are assumed to be known, while both the observation function ?? and the control policy ?? are unknown. To obtain a tractable surrogate for the observation process, we approximate ?? using a conditional generative adversarial network (cGAN) that learns a mapping from pairs $( s , z )$ of system state ?? and latent variable ?? to visual observations ?? [16], where the generator is trained on datasets of images annotated with their associated system states within a conditional adversarial learning framework [14, 21]. The stochasticity of the system arises from the unobserved environmental variable $z _ { t } ,$ which affects the visual observation $o _ { t }$ , propagates through the policy $\pi ( o _ { t } )$ , and ultimately affects the closed-loop state evolution. To facilitate uncertainty analysis, we introduce the following assumption. 

Assumption 1. Let ??0 denote a reference environmental condition. There exists a mapping $F : S \times Z \to S$ determined by the system dynamics $f _ { : }$ , the observation function ${ \mathit { g } } ,$ and the control policy ??, such that for a given state ?? ∈ ?? and environmental condition $z \in Z .$ , the next state is 

$$
s ^ {\prime} = F (s, z).
$$

Define the deviation in the system state due to environmental variations relative to the reference ??0 as 

$$
\Delta s = F (s, z) - F (s, z _ {0}) \in \Omega_ {\Delta} \subset \mathbb {R} ^ {m},
$$

where Δ?? is a random variable with Borel probability measure 

$$
\Delta s \sim \mu , \quad \mu \in \mathcal {P} (\Omega_ {\Delta}),
$$

independent of the current state ??. The closed-loop dynamics can then be written as 

$$
s ^ {\prime} = \tilde {F} (s, z _ {0}, \Delta s) = F (s, z _ {0}) + \Delta s,
$$

where Δ?? is sampled from the fixed distribution ??. 

Based on the above assumptions, we consider a stochastic discretetime system 

$$
s _ {t + 1} = \tilde {F} (s _ {t}, z _ {0}, \Delta s)
$$

with a Borel-measurable state space $S \subseteq \mathbb { R } ^ { m }$ , initial set $S _ { 0 } \subseteq S ,$ and disturbance space $\Omega _ { \Delta } \subseteq \mathbb { R } ^ { m }$ . We assume that ?? is compact and that $\tilde { F }$ is Lipschitz continuous, which ensures that the system evolution is well-defined under stochastic disturbances Δ??. A system trajectory is defined as the sequence $( s _ { t } , \Delta s ) _ { t \in  { \mathbb { N } } _ { 0 } }$ , which induces a discrete-time Markov process [25] with probability measure $\mathbb { P } _ { s _ { 0 } }$ and expectation operator $\mathbb { E } _ { s _ { 0 } }$ for any initial state $s _ { 0 } \in S _ { 0 }$ . Then, the probabilistic safety problem can be defined as follows: 

Definition 2.1 (Probabilistic Safety). Let $X _ { u } \subseteq S$ be a Borel measurable unsafe set in the state space $S \subseteq \mathbb { R } ^ { m }$ . Let $\textstyle p \in [ 0 , 1 )$ ) denote a probability threshold. The goal is to learn a control policy ?? such that, for any initial state ??0 $\in S _ { 0 } ,$ the system avoids entering the unsafe set $X _ { u }$ with probability at least $\mathcal { P } \cdot$ Formally, we want to find a control policy ?? satisfying 

$$
\mathbb {P} _ {s _ {0}} \left[ \operatorname{Safe} (X _ {u}) \right] \geq p,
$$

where Safe $( X _ { u } ) \ : = \ : \{ ( s _ { t } , \Delta s ) _ { t \in \mathbb { N } _ { 0 } } \mid \forall t \in \mathbb { N } _ { 0 } , s _ { t } \notin X _ { u } \}$ is the set of all trajectories that never enter the unsafe set $X _ { u } .$ . Here, $\mathbb { P } _ { s _ { 0 } }$ is the probability measure induced by the initial state ??0 and the stochastic disturbances $\Delta s .$ , ensuring that the system remains in the safe region $S \setminus X _ { u }$ with at least probability $\mathcal { P }$ over time ?? . 

To formally characterize such safety guarantees, we introduce a stochastic barrier certificate (SBC [24]) that captures the relationship among system states, disturbances, and safety constraints. Serving as a Lyapunov-like certificate, SBC ensures that the synthesized control policy satisfies the desired safety probability bound throughout the closed-loop evolution. 

Theorem 2.2. Let ?? be the state space, $S _ { 0 } \subseteq S$ the initial set, and $X _ { u } \subseteq S$ the unsafe set. Let $\boldsymbol { p } \in [ 0 , 1 )$ denote the probability threshold. A continuous function $B : S  \mathbb { R }$ is said to satisfy the SBC with respect to $S _ { 0 } , X _ { u }$ , and ?? if it satisfies: 

(i) $B ( s ) \geq 0$ for each ${ \mathfrak { s } } \in S ,$ 

(ii) $B ( s ) \leq 1$ for each $s \in S _ { 0 }$ , 

(iii) $B ( s ) \geq { \frac { 1 } { 1 - p } }$ for each $s \in X _ { u }$ 1 − ?? 

(iv) There exists $\epsilon > 0$ such that for each $\mathfrak { s } \in S \backslash X _ { u }$ at which $B ( s ) \leq { \frac { 1 } { 1 - p } } ;$ we have $B ( s ) \geq \mathbb { E } _ { \Delta s \sim \mu } \big [ B ( \tilde { F } ( s , z _ { 0 } , \Delta s ) ) \big ] + \epsilon .$ 

If an SBC is found, the probability of entering the unsafe region can be guaranteed to be less than $\begin{array} { r } { \boldsymbol { { 1 - } } \boldsymbol { { p } } , } \end{array}$ which means that the probability of system safety can be guaranteed to be at least $\mathcal { P } \cdot$ 

## 3 The Framework of Provably Probabilistic Safe Controller Synthesis

In this section, we present a closed-loop framework named SafePVC for synthesizing provably probabilistic safe controllers for visionbased neural network control systems. As illustrated in Fig. 1, the framework consists of two core components: an observation approximator and a controller generator. The observation approximator provides an explicit, analytically tractable visual input model and pretrains well-performance controller policies using RL, while the controller generator employs neural SBCs to assess the probabilistic safety of these policies under environmental perceptions and uses counterexamples identified during verification to refine the controller network. This iterative process continues until yields a vision-based DNN controller equipped with a valid SBC providing formal provable probabilistic safety guarantees. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-10/05eece32-1a1f-41a5-8953-345b517e590d/a46862711131967c79efbad151a501ebcfa3976bbccc4e33ff0c6390460f5d1a.jpg)



Figure 1: The pipeline of SafePVC. The left block presents the observation approximator, which uses a cGAN-based state–image mapping and RL-based pretraining to construct a verifiable closed-loop system (VCLS) with an initial controller $\pi _ { 0 } .$ After obtaining $\pi _ { 0 } ,$ the VCLS executes under environmental perturbations to obtain a data-driven state disturbance distribution. This distribution, together with the full $\mathbf { V C L S } ,$ is then passed to the controller generator in the right block. The pretrained controller is evaluated by synthesizing an SBC that certifies probabilistic safety under disturbance $\Delta s$ derived from discretized states ??˜. If the SBC violates posterior feasibility, counterexamples (cex) are extracted to alternately refine both the controller and SBC networks. The refined controller is reinserted into the VCLS to update the disturbance distribution, and this loop continues until both a valid SBC and a controller ?? with provable probabilistic safety are obtained.


## 3.1 Verifiable Closed-loop System Construction

In this subsection, we construct an end-to-end verifiable closed-loop control system that transforms the otherwise intractable visionbased NNCS into a form amenable to formal analysis. The system maps the state ?? and latent variable ?? to a vision-based control policy together with a quantitative estimation of environmental perturbations. As illustrated in the left block of Fig. 1, the procedure consists of three tightly coupled steps: i) a conditional generative model for perception, ii) a distillation process that produces a verificationfriendly MLP model, and iii) a reinforcement learning module for pre-training a robust control policy network. 

[cGAN-based Perception Model]. We employ a cGAN where generator $\mathcal { G } ( s , z )$ produces observations conditioned on state ?? and environmental perturbation ??, while discriminator D distinguishes real from generated observations. The model is pre-trained with orthogonal initialization using the objective: 

$$
\min _ {\mathcal {G}} \max _ {\mathcal {D}} \mathbb {E} _ {(o, s) \sim p _ {\text { data }}} [ \log D (o, s) ] + \mathbb {E} _ {s \sim p _ {s}, z \sim p _ {z}} [ \log (1 - D (\mathcal {G} (s, z), s)) ],
$$

where $ { p _ { \mathrm { d a t a } } }$ is the joint distribution of real state-observation pairs, $\mathit { p } _ { s }$ the state distribution, and $\scriptstyle { \mathcal { P } } z$ the environmental perturbation distribution. The discriminator maximizes the objective by classifying real and generated samples correctly, while the generator minimizes it by fooling the discriminator. 

[MLP Distillation with Lipschitz Regularization]. To improve verification efficiency, the trained cGAN generator is distilled into a compact MLP model ??(??, ??). Given paired samples produced by the teacher model ${ \mathcal { G } } ,$ the student MLP is optimized to minimize 

the following distillation loss: 

$$
\mathcal {L} _ {\mathrm{distill}} = \left\| \mathcal {G} (s, z) - g (s, z) \right\| _ {2} ^ {2} + \lambda_ {\mathrm{lip}} \mathcal {L} _ {\mathrm{lip}},
$$

where the squared ℓ2-norm term $\| \mathscr { G } ( s , z ) - g ( s , z ) \| _ { 2 } ^ { 2 }$ enforces output consistency, and $\mathcal { L } _ { \mathrm { l i p } }$ is the spectral Lipschitz loss controlling the global Lipschitz constant of ${ \mathit { g } } ,$ which ensures the distilled perception model retains the expressiveness of the cGAN while offering verifiability through bounded Lipschitz continuity. 

In details, the loss $\mathcal { L } _ { \mathrm { l i p } }$ is computed by estimating the spectral norm of each linear layer via power iteration and combining it with the Lipschitz factors of activation and BatchNorm layers, where the latter contribute through their scaling coefficients. This yields a principled approximation of the global Lipschitz constant of the neural network. 

[Vision-Based Controller Training with PPO]. We train a vision-based controller $\pi _ { \boldsymbol { \theta } } ( a | \boldsymbol { o } )$ using Proximal Policy Optimization (PPO) [29], where the policy makes decisions based on state estimates ??ˆ derived from perceptual observations $o = g ( s , z )$ . The PPO clipped surrogate objective is 

$$
L (\theta) = \hat {\mathbb {E}} _ {t} \left[ \min \left(r _ {t} (\theta) \hat {A} _ {t}, \operatorname{clip} \left(r _ {t} (\theta), 1 - \epsilon , 1 + \epsilon\right) \hat {A} _ {t}\right) \right],
$$

where $\hat { \mathbb { E } } _ { t }$ denotes the empirical expectation over timesteps, $r _ { t } ( \theta ) =$ $\frac { \pi _ { \theta } ( a _ { t } | o _ { t } ) } { \pi _ { \theta _ { \mathrm { o l d } } } ( a _ { t } | o _ { t } ) }$ is the probability ratio with $\theta _ { \mathrm { o l d } }$ denoting the policy parameters from the previous update, and $\hat { A } _ { t }$ is the advantage estimator. We pretrained a controller $\pi _ { 0 }$ using these state estimates via a third-party RL framework [26]. 

Remark 1 (Data-Driven Disturbance Estimation). After constructing the closed-loop system, a data-driven approach is employed to quantitatively characterize the disturbance term $\Delta s ,$ capturing how environmental perturbations affect system dynamics. 

For a given set of states $S = \{ s _ { 1 } , . . . , s _ { N _ { s } } \}$ , we generate ?? pertur-$s _ { i } ,$ $s _ { i , z _ { 0 } } ^ { \prime }$ under baseline control $z _ { 0 }$ and $s _ { i , z } ^ { \prime }$ under perturbed control $z ,$ and define the 

$\Delta s _ { i } = s _ { i , z } ^ { \prime } - s _ { i , z _ { 0 } } ^ { \prime }$ . By statistically analyzing the distribution of Δ?? across all states and perturbation samples, we obtain the corresponding state disturbance distribution, which serves as the basis for subsequent system verification. 

## 3.2 SBC-Guided Provably Probabilistic Safe Controller Synthesis

In this section, as shown in the right block of Fig. 1, we first employ the stochastic barrier certificate conditions from Theorem 2.2 to rigorously verify the probabilistic safety of the visual-based control ??0 previously obtained through reinforcement learning. And then, violations identified during verification trigger counterexample-guided refinement, iteratively improving the policy until SBC-guaranteed probabilistic safety is achieved. 

3.2.1 Neural SBC Synthesis. We adopt a learning-and-verification paradigm to construct a neural SBC, building on prior evidence that neural certificate construction yields effective [9, 36]. A neural network $B ( \pmb { s } )$ is used as a proxy to approximate a candidate stochastic barrier function by minimizing a composite loss function 

$$
\mathcal {L} _ {L} = \mathcal {L} _ {\mathrm {dec\_L}} + \lambda_ {R} \mathcal {L} _ {\mathrm{region}} + \lambda_ {L} \mathcal {L} _ {\mathrm {lip\_L}},
$$

designed to penalize violations of the four key structural conditions stated in Theorem 2.2. 

[Expected Decrease Condition $\mathcal {L} _ {\mathrm {dec\_ {L}}}$ . The dominant term $\mathcal { L } _ { \mathrm { d e c } \underline { { \tau } } }$ implements the martingale-based Expected Decrease Condition [17], which encourages the conditional expectation of the SBC value at the next state to lie strictly below a scaled current value:

$$
\mathcal {L} _ {\mathrm {dec\_ {L}}} = \mathbb {E} _ {\mathbf {s} _ {t}} \bigg [ \max \left(0, \mathbb {E} \big [ B (\mathbf {s} _ {t + 1}) \mid \mathbf {s} _ {t} \big ] - \gamma B (\mathbf {s} _ {t}) + \epsilon\right) \bigg ],
$$

where $\gamma ~ \in ~ \left( 0 , 1 \right]$ and $\epsilon \ > \ 0$ impose a strict decrease margin. The inner expectation is typically estimated via Monte-Carlo sampling [30] over next states $\mathbf { s } _ { t + 1 }$ drawn from the system dynamics conditioned on $\mathbf { s } _ { t }$ . 

[Region Constraints $\scriptstyle ( { \mathcal { L } } _ { \mathbf { r e g i o n } } ) ]$ . To satisfy the initial-state and safety requirements, $\mathcal { L } _ { \mathrm { r e g i o n } }$ penalizes violations of the boundary conditions on samples from the initial set ??0 and unsafe set $S _ { u }$ . 

By the way, a regularization term $\mathcal { L } _ { \mathrm { l i p \_ L } }$ is incorporated into the loss function of $B ( s )$ to constrain its Lipschitz $\ell _ { 2 }$ constant, thereby facilitating the next verification process outlined in Theorem 3.1. 

To formally check the validity of learned SBC candidates, each condition of the SBC in Theorem 2.2 must be rigorously verified. Condition (i) can be satisfied by appropriately designing the neural network architecture, while conditions (ii) and (iii) are verified via interval propagation techniques [11, 33]. Specifically, we leverage the state-of-the-art neural network verification tool ?? $- \beta -$ Crown [34] and determine satisfaction based on the computed interval bounds. For condition (iv), we establish the following theorem: 

Theorem 3.1. Let $\tilde { F } ( s , z _ { 0 } , \Delta s ) = f ( s , \pi ( g ( s , z _ { 0 } ) ) ) + \Delta s ,$ and ?? : $S \to \mathbb { R }$ be a Lipschitz continuous function with constant $L _ { B } .$ Suppose $f , \pi ,$ , and ?? have Lipschitz constants $L _ { f } , L _ { \pi } .$ , and $L _ { g } ,$ respectively, and let $\tau > 0$ bound $\| s - \tilde { s } \| _ { 2 } \le \tau f o r s , \tilde { s } \in S .$ Define $K \ : = \ : \tau$ · $L _ { B } \cdot \left( 1 + L _ { f } \sqrt { 1 + ( L _ { \pi } L _ { g } ) ^ { 2 } } \right)$ , if there exists $\tilde { s } \in \cal S$ such that $B ( \tilde { s } ) -$ $\mathbb { E } _ { \Delta s \sim \mu } \big [ B ( \tilde { F } ( \tilde { s } , z _ { 0 } , \Delta s ) ) \big ] - K \ge \epsilon ,$ , then for any ?? ∈ ?? with $\| s - \tilde { s } \| _ { 2 } \leq \tau$ , we have $\begin{array} { r } { B ( s ) - \mathbb { E } _ { \Delta s \sim \mu } \big [ B ( \tilde { F } ( s , z _ { 0 } , \Delta s ) ) \big ] \geq \epsilon } \end{array}$ . 

Proof. Let $s \in S$ satisfy $\| s - \tilde { s } \| _ { 2 } \leq \tau .$ . By the Lipschitz continuity of ?? with constant $L _ { B } ,$ we have 

$$
B (s) \geq B (\tilde {s}) - L _ {B} \| s - \tilde {s} \| _ {2} \geq B (\tilde {s}) - \tau L _ {B}. \tag {1}
$$

For the expectation term, using the Lipschitz properties of $B , f ,$ $\pi ,$ and ?? with constants $L _ { B } , L _ { f } , L _ { \pi } .$ , and $L _ { g } ,$ respectively, we have 

$$
\begin{array}{l} \mathbb {E} _ {\Delta s \sim \mu} [ B (\tilde {F} (s, z _ {0}, \Delta s)) ] = \mathbb {E} _ {\Delta s \sim \mu} [ B (f (s, \pi (g (s, z _ {0}))) + \Delta s) ] \\ \leq \mathbb {E} _ {\Delta s \sim \mu} [ B (f (\tilde {s}, \pi (g (\tilde {s}, z _ {0}))) + \Delta s) ] \\ + L _ {B} \left\| f (s, \pi (g (s, z _ {0}))) - f (\tilde {s}, \pi (g (\tilde {s}, z _ {0}))) \right\| _ {2} \\ \leq \mathbb {E} _ {\Delta s \sim \mu} [ B (\tilde {F} (\tilde {s}, z _ {0}, \Delta s)) ] \\ + L _ {B} L _ {f} \sqrt {\| s - \tilde {s} \| _ {2} ^ {2} + \| \pi (g (s , z _ {0})) - \pi (g (\tilde {s} , z _ {0})) \| _ {2} ^ {2}} \\ \leq \mathbb {E} _ {\Delta s \sim \mu} [ B (\tilde {F} (\tilde {s}, z _ {0}, \Delta s)) ] + \tau L _ {B} L _ {f} \sqrt {1 + (L _ {\pi} L _ {g}) ^ {2}}. \tag {2} \\ \end{array}
$$

Combining (1) and (2), we obtain 

$$
\begin{array}{l} B (s) - \mathbb {E} _ {\Delta s \sim \mu} [ B (\tilde {F} (s, z _ {0}, \Delta s)) ] \\ \geq B (\tilde {s}) - \mathbb {E} _ {\Delta s \sim \mu} \big [ B \big (\tilde {F} (\tilde {s}, z _ {0}, \Delta s) \big) \big ] - \tau L _ {B} \big (1 + L _ {f} \sqrt {1 + (L _ {\pi} L _ {g}) ^ {2}} \big) \\ = B (\tilde {s}) - \mathbb {E} _ {\Delta s \sim \mu} [ B (\tilde {F} (\tilde {s}, z _ {0}, \Delta s)) ] - K. \tag {3} \\ \end{array}
$$

By assumption, $\begin{array} { r } { B ( \tilde { s } ) - \mathbb { E } _ { \Delta s \sim \mu } [ B ( \tilde { F } ( \tilde { s } , z _ { 0 } , \Delta s ) ) ] - K \geq \epsilon , } \end{array}$ hence 

$$
B (s) - \mathbb {E} _ {\Delta s \sim \mu} \left[ B (\tilde {F} (s, z _ {0}, \Delta s)) \right] \geq \epsilon .
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-10/05eece32-1a1f-41a5-8953-345b517e590d/f33ac06e3059a0eb244e4a59ad5d716a38cfb3d1603e17f31537b4e990c595f0.jpg)


3.2.2 SBC-guided Controller Policy Synthesis. The policy ?? is optimized to aggressively satisfy the expected decrease condition relative to the current candidate SBC $B ( \pmb { s } )$ , while adhering to robustness and behavioral constraints. The controller’s loss function $\mathcal { L } _ { P }$ is defined as: 

$$
\mathcal {L} _ {P} = \mathcal {L} _ {\text { dec\_P }} + \lambda_ {P} \mathcal {L} _ {\text { lip\_P }} + \lambda_ {M} \mathcal {L} _ {\text { mse }}.
$$

[Decrease Maximization $\scriptstyle \left( \mathcal { L } _ { \mathbf { d e c } } \ \mathbf { p } \right) ]$ . The term $\mathcal { L } _ { \mathrm { d e c } , \mathrm { P } }$ shares the same martingale loss structure as $\mathcal { L } _ { \mathrm { d e c } \mathrm { ~ L ~ } }$ . When verification fails, the policy ?? collects counterexamples generated during the verification process and computes this loss accordingly. The policy is then updated to minimize the loss, thereby encouraging the selection of actions that lead to a greater expected decrease in the SBC value. 

[Lipschitz Regularization $( \mathcal { L } _ { \mathrm { I i p } } \mathbf { \Omega p } ) ]$ . The term $\mathcal { L } _ { \mathrm { l i p \_ F } }$ constrains the controller network’s local Lipschitz $\ell _ { 2 }$ constant with respect to its inputs. This regularization enhances smoothness and improves the policy’s robustness against input disturbances, in accordance with the requirements of Theorem 3.1. 

To balance safety and performance, we introduce a behavioral similarity constraint $\mathcal { L } _ { \mathrm { m s e } }$ that penalizes deviations from the pretrained baseline $\pi _ { 0 } { : }$ 

$$
\mathcal {L} _ {\mathrm{mse}} = \mathbb {E} _ {o} \left[ \| \pi (o) - \pi_ {0} (o) \| _ {2} ^ {2} \right].
$$

Remark 2 (Alternating Optimization). To improve the training efficiency and convergence of both the controller network and the SBC network, the synthesis process adopts an alternating optimization scheme. Specifically, the procedure iterates between two steps: 

• SBC Update (fixed ??): The SBC parameters are optimized to minimize $\mathcal { L } _ { L }$ yielding tighter probabilistic safety bounds for the current policy. 

• Policy Update (fixed ??): The policy parameters are optimized to minimize L?? , guiding the controller to produce actions that satisfy the probabilistic safety constraints established by the current SBC. 

This alternating scheme establishes a symbiotic relationship: the SBC certifies the safety of the policy, while the policy is refined to satisfy increasingly stringent safety conditions. Compared with methods that jointly train controllers and barrier functions using supervised learning (e.g., CBF-based approaches [1, 32]), this approach achieves improved training efficiency and faster convergence. 

## 3.3 Overall Algorithm

In this section, we present Algorithm 1 to summarize the procedure called SafePVC of synthesizing a provably probabilistic safe controller for a vision-based neural network control system. 


Algorithm 1 SafePVC


Require: dataset $D = \{(s_i, o_i)\}$ , system dynamics f, latent variable $z \sim U[-1, 1]$ , discrete system state space S

Ensure: stochastic barrier certificate SBC, controller network $\pi$ 1: $G, D \leftarrow \text{cGAN}(D, z)$ 2: $g \leftarrow \text{Distill}(G)$ 3: $\pi_0 \leftarrow \text{InitController}()$ 4: $\pi_0 \leftarrow \text{PPO}(\pi_0, f, S), \pi \leftarrow \pi_0$ 5: VCLS $\leftarrow \text{Concat}(g, \pi_0)$ 6: SBC $\leftarrow \text{InitSBC}()$ 7: $\Delta s \leftarrow \text{Distribution\_Estimation}(VCLS, f, z_0, z, S)$ 8: iters $\leftarrow 0$ 9: while (SBC, VCLS, $z_0, \Delta s, S$ ) not satisfied theorem2.2 do

10: Collect Cexs encountered during the verification process

11: SBC $\leftarrow \text{Update}_{\text{SBC}}(\text{SBC}, \text{Loss}_{\text{SBC}}(\text{Cexs}))$ 12: if iters mod 10 == 0 then

13: $\pi \leftarrow \text{Update}_\pi(\pi, \text{Loss}_{\text{VCLS}}(\text{Cexs}))$ 14: $\Delta s \leftarrow \text{Distribution\_Estimation}(VCLS, f, z_0, z, S)$ 15: end if

16: iters $\leftarrow \text{iters} + 1$ 17: end while

18: return SBC, $\pi$ 

In Algorithm 1, Lines 1–2 correspond to the training and distillation of the observation model. Lines 3–4 describe the pretraining of the controller using the PPO algorithm. Line 5 integrates the observation model and the controller network to construct the complete verifiable closed-loop system (VCLS). Line 7 computes the state disturbance distribution under the current VCLS structure and environmental perturbations. Line 9 determines whether the current VCLS and the SBC satisfy the predefined safety constraints. Finally, lines 10–15 describe the counterexample-guided alternating training process between the SBC and the controller network, which iteratively refines both safety verification and control performance. 

## 4 Experiments

In this section, we evaluate the proposed tool SafePVC in two simulated environments: the X-Plane 11 [27] flight simulator and the CARLA [8] driving simulator. For both benchmarks, verifiable closed-loop systems are first constructed via the proposed MLP distillation with Lipschitz regularization technique based on cGAN. Subsequently, two ablation studies are conducted to assess the effectiveness of SBC-based verification-guided controller training, as well as the algorithm robustness of the certified probabilistic safety lower bounds of the controllers under varying disturbance intensities. All experiments were conducted on a Windows 11 machine equipped with 32-GB RAM and an AMD Ryzen 7 5800H CPU running at 3.2 GHz. The source code and more experimental details are available at https://anonymous.4open.science/r/artical-F122. 

## 4.1 Experiment Setup

[X-Plane11 Trajectory Tracking]. It evaluates aircraft taxiing along a predefined path, requiring the controller to maintain stability and accurately follow trajectory constraints. The system dynamics are governed by the following discrete-time equations: 

$$
p _ {k + 1} = p _ {k} + v \Delta t \sin \theta_ {k}, \quad \theta_ {k + 1} = \theta_ {k} + \frac {v}{L} \Delta t \tan \phi_ {k}
$$

where the aircraft state is defined by its lateral deviation $p ~ \in$ [−11, 11] m and heading angle deviation $\theta \in [ - 3 0 , 3 0 ] ^ { \circ }$ . The parameters ?? = 5 m/s, $\Delta t = 0 . 0 5 s ,$ and $L = 5$ m represent the taxiing speed, controller update frequency, and wheelbase, respectively. 

[CARLA Emergency Braking]. It requires the vehicle to approach a target point as rapidly as possible while guaranteeing that its speed is reduced below a safe threshold before reaching a specified distance. The system dynamics are governed by the following discrete-time equations: 

$$
d _ {k + 1} = d _ {k} - v _ {k} \Delta t, \quad v _ {k + 1} = v _ {k} - a _ {k} \Delta t
$$

where the vehicle’s state is defined by its distance to the destination $d \in [ 5 , 1 6 ]$ m and its current velocity ?? ∈ [0, 3] m/s. The parameter $\Delta t = 0 . 0 5$ s denotes the controller’s update frequency. 

Furthermore, the control signals $\phi _ { k }$ (for the aircraft) and $a _ { k }$ (for the vehicle) are generated by an image-based controller. This controller first estimates the system state from images provided by a perception module, which approximates the actual system, and then computes the corresponding control actions via a DNN. During the pre-training phase of our experiment, the DNN controller was trained using the PPO reinforcement learning algorithm. The hyperparameters were set as: learning rate $= 3 \times 1 0 ^ { - 4 }$ , batch size = 64, and training steps = 200k. 

## 4.2 Training of the Perception Model

We adopt the method introduced in Section 3.1. In the X-Plane11 and CARLA simulation environments, the latent variable $z \sim U [ - 1 , 1 ]$ is set to have dimensions of 2 and 4, respectively. Accordingly, the model generates grayscale images with resolutions of 8 × 16 and 32 × 32. To enable the cGAN to learn environmental perception, we collected 10000 and 400 state-image data pairs from the X-Plane11 and Carla environments, respectively, during specified time intervals, followed by down-sampling. During our experiments, we observed that training a cGAN with a small-scale dataset led to mode collapse. Under such conditions, supervised fine-tuning of an MLP yielded superior results. As illustrated in Fig. 2, the proposed observation model effectively learns the mapping from system states to environmental perception. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-10/05eece32-1a1f-41a5-8953-345b517e590d/703d3b846a3378dcd3aaaac7897c71b7b5ea50e9d4cb9dab46bf5b05a7282509.jpg)



Figure 2: Comparison of Perception model approximations in the X-Plane11 (left) and CARLA (right) simulation environments. Each group displays the raw input, downsampled image, and generated grayscale image. The corresponding state parameters, distance ?? (m) and heading angle $\theta ( ^ { \circ } ) ,$ are shown above.


## 4.3 Performance Evaluation of SafePVC

The existing baseline methods typically measure controller safety over a limited time horizon, and are unable to ensure reliable safety for stochastic systems over infinite or long-term horizons. This part assesses the practical effectiveness of our approach in such extended scenarios. 


Table 1: Verifiable Probabilistic Safety Lower Bounds


<table><tr><td rowspan="2">Benchmark</td><td rowspan="2">Fixed</td><td colspan="2">Verification-Guided</td></tr><tr><td>Joint</td><td>Alternating</td></tr><tr><td>X-Plane11 Trajectory Tracking</td><td>84.9%(40 iters.)</td><td>OT</td><td>92.1%(35 iters.)</td></tr><tr><td>CARLA Emergency Braking</td><td>72.6%(100 iters.)</td><td>OT</td><td>94.2%(100 iters.)</td></tr></table>

Specifically, we first evaluate the safety assurance efficacy of the verification-guided training methodology through an ablation study. Within a framework where the perception model parameters are frozen and the DNN policy network is pre-trained, we compare the effect of SBC synthesis under two conditions: i) using verification counter-examples during post-training, and ii) fixing the policy network parameters. For condition i), we benchmark our approach against the method proposed in previous work [37]. The method in [37] employs a joint loss function to achieve the synchronous update of both the barrier function and the learning network. In contrast, our verification-guided training employs an alternating optimization strategy: specifically, we perform 1 round of policy network parameter updates after every 10 rounds of barrier function training. The learning rates for the barrier function and the policy network are set to $1 \bar { \times } 1 0 ^ { - 3 }$ and $5 \times 1 0 ^ { - 3 }$ , respectively. The results are recorded in Table 1, and the visualization of the synthesized SBC is illustrated in Fig. 3. 

Table 1 shows that the fixed controller provides moderate safety guarantees, while the joint update scheme failed to converge within 100 iterations (denoted as $^ { \mathrm { { e } } } \mathrm { { O T ^ { \prime \prime } } } )$ , indicating potential instability when updating both networks simultaneously. In contrast, our verification-guided alternating optimization consistently achieves higher verifiable safety bounds with fewer iterations. These results demonstrate that alternating updates effectively leverage counterexamples to improve the learned SBC, yielding more reliable safety guarantees than both fixed and joint-update strategies. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-10/05eece32-1a1f-41a5-8953-345b517e590d/cff4e7cfb4d087f840d76dfb91a1612520a203dfd10b76ec1dbc299a5b808160.jpg)



Figure 3: Visualization of the synthesized SBCs in verification-Guided training for X-Plane11 Trajectory Tracking (left) and CARLA Emergency Braking (right).


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-10/05eece32-1a1f-41a5-8953-345b517e590d/29960bb0510d177a4b271efd7e0aa8233e4f2beccea3c31c5d6b258916247a57.jpg)



Figure 4: Maximum verifiable safety probability vs. training steps under different disturbances for X-Plane11 Trajectory Tracking (left) and CARLA Emergency Braking (right).


Subsequently, we conducted a detailed analysis of the controller’s safety guarantees under varying state disturbances. It is important to note that these state disturbances are not derived from realworld environmental perturbations via system-level network analysis. Instead, they are directly modeled as uniformly distributed within manually specified bounds to approximate potential extreme and adverse conditions. We compute the controller’s safety probability under multiple disturbance magnitudes—corresponding to percentages of the span in each dimension of the state space—to approximately evaluate the algorithm’s robustness across different disturbance strengths. 

## 5 Conclusion

This paper presented a framework for synthesizing provably probabilistic safe controllers for vision-based neural network control systems. It employed a cGAN-based visual observation model and reinforcement learning to pretrain high-performance policies while capturing state disturbances induced by environmental perturbations. Stochastic barrier certificates guided by martingale theory provided formal probabilistic safety guarantees over unbounded time horizons, and counterexample-guided refinement improved both the controller and certificate networks. Experiments on benchmark tasks demonstrated the effectiveness, scalability, and robustness of the proposed approach. 

In future work, we plan to explore higher-resolution image generation techniques to capture more complex environmental latent variables, enabling the verification of more diverse vision-based NNCS properties and the synthesis of controllers. 

## References



[1] Hossein Abdi, Golnaz Raja, and Reza Ghabcheloo. 2023. Safe control using vision-based control barrier function (V-CBF). In IEEE International Conference on Robotics and Automation. IEEE, 782–788. 





[2] Aaron D Ames, Samuel Coogan, Magnus Egerstedt, Gennaro Notomista, Koushil Sreenath, and Paulo Tabuada. 2019. Control barrier functions: Theory and applications. In 2019 18th European control conference (ECC). Ieee, 3420–3431. 





[3] Adith Boloor, Karthik Garimella, Xin He, Christopher Gill, Yevgeniy Vorobeychik, and Xuan Zhang. 2020. Attacking vision-based perception in end-to-end autonomous driving models. Journal of Systems Architecture 110 (2020), 101766. 





[4] Feiyang Cai, Jiani Li, and Xenofon Koutsoukos. 2020. Detecting adversarial examples in learning-enabled cyber-physical systems using variational autoencoder for regression. In 2020 IEEE Security and Privacy Workshops (SPW). IEEE, 208–214. 





[5] Aleksandar Chakarov and Sriram Sankaranarayanan. 2013. Probabilistic program analysis with martingales. In International Conference on Computer Aided Verification. Springer, 511–526. 





[6] Chenyi Chen, Ari Seff, Alain Kornhauser, and Jianxiong Xiao. 2015. Deepdriving: Learning affordance for direct perception in autonomous driving. In Proceedings of the IEEE international conference on computer vision. 2722–2730. 





[7] Liqian Chen, Yuan Zhou, and Ji Wang. 2025. Verifying Neural Network Controlled Systems by Combining Taylor Models. In Engineering of Complex Computer Systems: 29th International Conference, ICECCS 2025, Hangzhou, China, July 2–4, 2025, Proceedings. Springer Nature, 358. 





[8] Alexey Dosovitskiy, German Ros, Felipe Codevilla, Antonio Lopez, and Vladlen Koltun. 2017. CARLA: An open urban driving simulator. In Conference on robot learning. PMLR, 1–16. 





[9] Alec Edwards, Andrea Peruffo, and Alessandro Abate. 2024. Fossil 2.0: Formal certificate synthesis for the verification and control of dynamical models. In Proceedings of the 27th ACM International Conference on Hybrid Systems: Computation and Control. 1–10. 





[10] Yousef Emam, Gennaro Notomista, Paul Glotfelter, Zsolt Kira, and Magnus Egerstedt. 2022. Safe reinforcement learning using robust control barrier functions. IEEE Robotics and Automation Letters (2022). 





[11] Sven Gowal, Krishnamurthy Dvijotham, Robert Stanforth, Rudy Bunel, Chongli Qin, Jonathan Uesato, Relja Arandjelovic, Timothy Mann, and Pushmeet Kohli. 2018. On the effectiveness of interval bound propagation for training verifiably robust models. arXiv preprint arXiv:1810.12715 (2018). 





[12] Martin T Hagan, Howard B Demuth, and Orlando De Jesús. 2002. An introduction to the use of neural networks in control systems. International Journal of Robust and Nonlinear Control: IFAC-Affiliated Journal 12, 11 (2002), 959–985. 





[13] Wassily Hoeffding. 1963. Probability inequalities for sums of bounded random variables. Journal of the American statistical association 58, 301 (1963), 13–30. 





[14] Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A Efros. 2017. Image-toimage translation with conditional adversarial networks. In Proceedings of the IEEE conference on computer vision and pattern recognition. 1125–1134. 





[15] Ismet Burak Kadron, Divya Gopinath, Corina S Păsăreanu, and Huafeng Yu. 2021. Case study: Analysis of autonomous center line tracking neural networks. In International Workshop on Numerical Software Verification. Springer, 104–121. 





[16] Sydney M Katz, Anthony L Corso, Christopher A Strong, and Mykel J Kochenderfer. 2022. Verification of image-based neural network controllers using generative models. Journal of Aerospace Information Systems 19, 9 (2022), 574–584. 





[17] Mathias Lechner, Krishnendu Chatterjee, and Thomas A Henzinger. 2022. Stability verification in stochastic control systems via neural network supermartingales. In Proceedings of the aaai conference on artificial intelligence, Vol. 36. 7326–7336. 





[18] Sergey Levine, Peter Pastor, Alex Krizhevsky, Julian Ibarz, and Deirdre Quillen. 2018. Learning hand-eye coordination for robotic grasping with deep learning and large-scale data collection. The International journal of robotics research 37, 4-5 (2018), 421–436. 





[19] Diego Manzanas Lopez, Sung Woo Choi, Hoang-Dung Tran, and Taylor T Johnson. 2023. NNV 2.0: The neural network verification tool. In International Conference on Computer Aided Verification. Springer, 397–412. 





[20] Xinhang Ma, Junlin Wu, Hussein Sibai, Yiannis Kantaros, and Yevgeniy Vorobeychik. 2025. Learning Vision-Based Neural Network Controllers with Semi-Probabilistic Safety Guarantees. arXiv preprint arXiv:2503.00191 (2025). 





[21] Mehdi Mirza and Simon Osindero. 2014. Conditional generative adversarial nets. arXiv preprint arXiv:1411.1784 (2014). 





[22] Jose Maurıcio ST Motta, Guilherme C De Carvalho, and RS McMaster. 2001. Robot calibration using a 3D vision-based measurement system with a single camera. Robotics and Computer-Integrated Manufacturing 17, 6 (2001), 487–497. 





[23] Stephen Prajna and Ali Jadbabaie. 2004. Safety verification of hybrid systems using barrier certificates. In International Workshop on Hybrid Systems: Computation and Control. Springer, 477–492. 





[24] Stephen Prajna, Ali Jadbabaie, and George J Pappas. 2004. Stochastic safety verification using barrier certificates. In 2004 43rd IEEE conference on decision and control (CDC)(IEEE Cat. No. 04CH37601), Vol. 1. IEEE, 929–934. 





[25] Martin L Puterman. 2014. Markov decision processes: discrete stochastic dynamic programming. John Wiley & Sons. 





[26] Antonin Raffin, Ashley Hill, Adam Gleave, Anssi Kanervisto, Maximilian Ernestus, and Noah Dormann. 2021. Stable-baselines3: Reliable reinforcement learning implementations. Journal of machine learning research 22, 268 (2021), 1–8. 





[27] Laminar Research. 2017. X-Plane Flight Simulator. https://www.x-plane.com 





[28] Jagannathan Sarangapani. 2018. Neural network control of nonlinear discrete-time systems. CRC press. 





[29] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal Policy Optimization Algorithms. arXiv:1707.06347 [cs.LG] https: //arxiv.org/abs/1707.06347 





[30] Alexander Shapiro. 2003. Monte Carlo sampling methods. Handbooks in operations research and management science 10 (2003), 353–425. 





[31] Xinyu Wang, Liqian Chen, Zengyu Liu, Minghao Li, and Banghu Yin. 2025. Verifying Neural Network Controlled Systems by Combining Forward and Backward Reachability Analysis. In 2025 25th International Conference on Software Quality, Reliability and Security (QRS). IEEE, 315–326. 





[32] Wei Xiao, Tsun-Hsuan Wang, Makram Chahine, Alexander Amini, Ramin Hasani, and Daniela Rus. 2022. Differentiable control barrier functions for vision-based end-to-end autonomous driving. arXiv preprint arXiv:2203.02401 (2022). 





[33] Kaidi Xu, Zhouxing Shi, Huan Zhang, Yihan Wang, Kai-Wei Chang, Minlie Huang, Bhavya Kailkhura, Xue Lin, and Cho-Jui Hsieh. 2020. Automatic perturbation analysis for scalable certified robustness and beyond. Advances in Neural Information Processing Systems 33 (2020), 1129–1141. 





[34] Huan Zhang, Shiqi Wang, Kaidi Xu, Linyi Li, Bo Li, Suman Jana, Cho-Jui Hsieh, and J Zico Kolter. 2022. General cutting planes for bound-propagation-based neural network verification. Advances in neural information processing systems 35 (2022), 1656–1670. 





[35] Huan Zhang, Tsui-Wei Weng, Pin-Yu Chen, Cho-Jui Hsieh, and Luca Daniel. 2018. Efficient neural network robustness certification with general activation functions. Advances in neural information processing systems 31 (2018). 





[36] Hanrui Zhao, Niuniu Qi, Lydia Dehbi, Xia Zeng, and Zhengfeng Yang. 2023. Formal synthesis of neural barrier certificates for continuous systems via counterexample guided learning. ACM Transactions on Embedded Computing Systems 22, 5s (2023), 1–21. 





[37] Ðorđe Žikelić, Mathias Lechner, Thomas A Henzinger, and Krishnendu Chatterjee. 2023. Learning control policies for stochastic systems with reach-avoid guarantees. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 37. 11926–11935. 

