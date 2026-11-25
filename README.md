# SPVC 改进项目

项目沿用 SPVC 的视觉反馈控制和非负神经屏障表示，把 cGAN 图像代理接口换成 DINO 低维观测与状态代理。轨迹级 CP 校准相机与代理之间的误差，鲁棒屏障检查允许误差下的闭环后继。

## 从这里开始

- **[文档首页](docs/README.md)** · 当前汇报、框架说明和两项主对照。
- **[实验索引](docs/experiments/README.md)** · 按任务或研究问题找报告。
- **[实验工程](artical-F122/README.md)** · [代码](artical-F122/Aebs/)、[运行结果](artical-F122/results/)和[测试](artical-F122/tests/)。

论文原件见[参考资料](docs/references/README.md)，旧报告、旧讲稿和阶段记录见[历史归档](docs/archive/README.md)。[文件清单](docs/文档清单.md)可按文件名查找。

第三方实现位于 [external/](external/)，停车场实验默认从本地 `VISION-SLS/` 读取资源。本机工作记录在 `work/`。
