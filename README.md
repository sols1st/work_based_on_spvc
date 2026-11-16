# SPVC 改进项目

把原 SPVC 的 cGAN 视觉前端替换为 **DINO + 低维视觉观测**，保留 PPO 控制、CP 误差包络和非负 SBC 验证。实验覆盖 AEBS、车道保持、停车场、移动机器人；安全结论均限定在各自数据分布、动力学和数值验证范围内。

## 入口

- **[当前文档](docs/current/README.md)**：只有 6 篇，先读这里。
- [当前框架](docs/current/SPVC当前框架_逐步流程与安全保证_20261109.md)：图像→DINO→CP→SBC→验证。
- [当前方案与验证结果](docs/current/当前方案与验证结果.md)：四个实验的统一状态。
- [收益证明总览](docs/current/收益证明_总览.md)：当前方案相对原 SPVC 的收益与边界。
- [可视化网站](site/index.html)：总览、实验配置、图片、方法对比。

## 当前主结果

| 实验 | 结论 | 主文档 |
|---|---|---|
| AEBS 真实 RGB | 轨迹 CP 0.418359 m；留出 99/100；SBC 292,024 区间 0/0；行为 4,800/4,800 | [AEBS](docs/current/AEBS真实RGB图像_完整实验_20261108.md) |
| 车道保持 | 轨迹 CP 0.090093；留出 95/100；SBC 0/0；图像闭环 12/12 | [车道](docs/current/车道保持_轨迹级CP校准与SBC衔接_20261107.md) |
| 停车场 | 受保护闭环 30/30；启动审计 233/240；依赖在线后备 | [专题](docs/topics/停车场神经SBC_构造证明与实验_20261103.md) |
| 移动机器人 | 行为 24/24；CP 盒内 28/32 反例；完整 SBC 未通过 | [专题](docs/topics/移动机器人_三维视觉实验_20261108.md) |

## 收益证明的一句话

- 原 cGAN 原生轨迹 q=1.9156 m/92 对 DINO q=0.4184 m/99，**部署系统更好，但不是同预算架构因果**；
- 400/10k/24k 同数据预算：**400 DINO 好；10k/24k 真实图 cGAN 头 q 更小；24k cGAN 覆盖更低**，无单向赢家；
- 联合 C 5 对种子无稳定收益；
- 车道共同半径 r_cert=0.12，0.16 失败。

## 文档分区

- [docs/current/](docs/current/README.md)：6 篇当前结论；
- [docs/details/](docs/details/README.md)：收益证明、数据预算、cGAN 对照、C 消融、半径扫描；
- [docs/topics/](docs/topics/README.md)：停车场、移动机器人；
- [docs/presentation/](docs/presentation/README.md)：汇报稿、PPT 大纲；
- [docs/archive/](docs/archive/)：历史报告与整理前快照；
- [docs/references/](docs/references/)：论文与代码学习资料。

## 范围

性能、模型安全条件和 CP 概率前提分别报告；没有宣称真实车辆或完整浮点程序的无条件安全保证。详细条件见[当前框架](docs/current/SPVC当前框架_逐步流程与安全保证_20261109.md)。
