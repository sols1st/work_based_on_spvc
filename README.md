# SPVC 改进项目

项目沿用 SPVC 的视觉反馈控制与非负神经屏障表示，将 cGAN 前端换为 DINO 低维观测。AEBS、车道保持和五维弯道遇障制动采用**轨迹级 CP 感知误差校准与确定性鲁棒屏障证书**。CP 估计新轨迹的视觉误差全程落入包络的概率下界。验证器检查包络内的每个允许后继，由此推导登记模型在采样时刻避险的条件。原 SPVC 的随机屏障证书（SBC）使用后继屏障值的条件期望。

## 从哪里开始

| 位置 | 内容 |
|---|---|
| [artical-F122/](artical-F122/) | 主实验代码在 [Aebs/](artical-F122/Aebs/)，运行记录在 [results/](artical-F122/results/)，回归测试在 [tests/](artical-F122/tests/)。 |
| [docs/](docs/README.md) | 框架、实验报告、论文与导师汇报。[references/](docs/references/README.md) 存放原论文，[assets/framework_figures/](docs/assets/framework_figures/) 存放早期框架图。 |
| [external/](external/) | 第三方实现和视觉骨干，包括 SPVT、srlnbc 与本机 DINO 资源。 |
| `VISION-SLS/` | 本地上游检出。停车场实验默认从这里加载模型与资源。 |
| [site/](site/index.html) | 可直接打开的实验展示网站。 |
| `work/` | 本机工作记录、临时下载（`work/tmp/`）和误复制的旧路径镜像（`work/path_mirror/`）。 |

原先放在根目录的三份本机 AEBS 结果现位于 `artical-F122/results/`，目录名仍为 `real_aebs_artifact_v1`、`real_aebs_artifact_v2` 和 `real_image_aebs_probe_v1`。这些本机副本沿用原有 Git 忽略规则。

## 实验与文档入口

- **[项目文档总入口](docs/README.md)** 按问题和实验找完整步骤、原论文、导师汇报与历史版本。
- [当前实验与早期基线](docs/experiments/README.md) 可核对最新报告和对应的实验版本。
- [当前框架](docs/framework/SPVC当前框架_逐步流程与安全保证_20261109.md)：图像→DINO→CP→鲁棒屏障→验证。
- [当前方案与验证结果](docs/experiments/当前方案与验证结果.md)：四个实验的统一状态。
- [收益证明总览](docs/experiments/跨实验对照/收益证明_总览.md)：当前方案相对原 SPVC 的收益与边界。
- [可视化网站](site/index.html)：总览、实验配置、图片、方法对比。

11 月 17 日的实验包括 AEBS 与车道鲁棒综合、五维弯道制动、六维无人机悬停及八维起重机消摆。[近期报告与结果](docs/experiments/README.md)

## 早期基线结果

| 实验 | 结论 | 主文档 |
|---|---|---|
| AEBS 真实 RGB | 轨迹 CP 0.418359 m；留出 99/100；鲁棒屏障 292,024 区间 0/0；行为 4,800/4,800 | [AEBS](docs/experiments/AEBS/AEBS真实RGB图像_完整实验_20261108.md) |
| 车道保持 | 轨迹 CP 0.090093；留出 95/100；鲁棒屏障 0/0；图像闭环 12/12 | [车道](docs/experiments/车道保持/车道保持_轨迹级CP校准与SBC衔接_20261107.md) |
| 停车场 | 受保护闭环 30/30；启动审计 233/240；依赖在线后备 | [专题](docs/experiments/停车场/停车场神经SBC_构造证明与实验_20261103.md) |
| 移动机器人 | 行为 24/24；CP 盒内 28/32 反例；完整鲁棒屏障未通过 | [专题](docs/experiments/三维移动机器人/移动机器人_三维视觉实验_20261108.md) |

## 最新对照结论

在 400、10k、24k 张目标域原图的同像素、同调参对照中，重训的原结构灰度头在三个预算、三个种子的九对比较中均有更小的整轨迹测距误差半径。车道保持中，固定已补偿的视觉前端、控制器和神经屏障，联合约束视觉误差与转向偏差后，五种曲率均通过额外 ±0.045 rad 的转向扰动验证。原生随机 SBC、盲测与保证的适用条件见[三项完整对照实验](docs/experiments/跨实验对照/SPVC三项对照_AEBS公平实验与车道重新综合_20261117.md)。

## 文档分区

- [框架与安全保证](docs/framework/README.md) 收录方法流程、误差定义、CP 与屏障理论。
- [按任务查实验](docs/experiments/README.md) 收录七个场景及跨实验对照。[实验报告导读](docs/experiments/实验报告导读.md) 按研究问题列出前端、后端和数据预算对照。
- [论文原件与阅读笔记](docs/references/README.md) 和[导师汇报材料](docs/presentation/README.md) 分别用于查出处和准备组会。
- [历史归档](docs/archive/README.md) 可追溯各阶段报告和整理前的快照。

## 范围

性能、模型安全条件和 CP 概率前提分别报告；没有宣称真实车辆或完整浮点程序的无条件安全保证。详细条件见[当前框架](docs/framework/SPVC当前框架_逐步流程与安全保证_20261109.md)。
