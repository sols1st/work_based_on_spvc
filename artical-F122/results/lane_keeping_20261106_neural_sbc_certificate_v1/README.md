# 原 PPO＋ReLU 神经 SBC：车道保持结果包

2026-11-06。本包验证 11 月 5 日的图像/误差训练 PPO，权重不变，修复最后一个 Tanh 的重载遗漏。无在线过滤器，无 LQR 控制替换。

- [主审计](certificate.json)：仿射相关界，初始网格 60×60，误差每维 2 段。
- [交叉复核](independent_ibp.json)：普通 IBP，初始网格 75×75，误差每维 8 段。
- [运行回归](rollouts.json)：720 条误差轨迹、12 条实际渲染图像轨迹均无越界。
- [代码问题审计](bug_audit.json)、[测试记录](tests.txt)。
- 策略 actor.pt；证书 sbc.pt；图像头 patch_projection.pt；原校准文件 cp_patch_95.json。
- [详细解释](../../../docs/current/车道保持_原PPO神经SBC与代码审计_20261106.md)。

## 条件与结论

速度 4 米/秒；初态横向 ±0.25 米、航向 ±0.06 弧度；安全阈值 ±0.85 米/±0.30 弧度；已知曲率取 −0.01、0、0.01；零过程扰动。误差两个归一化分量均在 ±0.08。

B(x)=200 relu(max_i a_i^T x−1)，32 个仿射面，使用归一化真实状态。最大值用 ReLU 网络实现。初始上界 0，危险下界约 28.0829，所有相关连续状态和误差区间的动态不增检查通过。它是构造式神经证书，不是旧 Tanh 网络通过训练得到的结果。

这是数值连续域区间证据，不是完整浮点程序形式证明。12 条图像轨迹中 2 条各有一帧超出包络，仍安全但不满足全程证书前提。不得声称无条件 95% 轨迹/实车保证。

## 复现

从 artical-F122 目录运行，输出目录须为新的空路径：

~~~bash
../.venv-vision-sls/bin/python -m Aebs.lane_keeping.certify_neural \
  --actor results/lane_keeping_20261106_neural_sbc_certificate_v1/actor.pt \
  --output-dir results/lane_sbc_recheck_affine \
  --grid 60 --depth 16 --error-splits 2

../.venv-vision-sls/bin/python -m Aebs.lane_keeping.certify_neural \
  --actor results/lane_keeping_20261106_neural_sbc_certificate_v1/actor.pt \
  --output-dir results/lane_sbc_recheck_ibp \
  --grid 75 --depth 16 --error-splits 8 --method ibp

../.venv-vision-sls/bin/python -m pytest tests/test_lane_certificate.py -q
~~~

图像回归用 audit_neural_rollouts 模块。--package 应为一个含 actor.pt/sbc.pt 且尚无 rollouts.json 的新目录；--source 指向原 lane_keeping_20261105 包；--dino-repo 与 --dino-weights 指向本机 DINOv2 仓库和 vits14 权重。源码快照和 environment.json 记录本次环境；manifest.json 绑定包内文件。

旧 lane_keeping_20261105 结果不覆盖：它的 LQR 区间上下界和初态数字有问题，旧 passed 标记不再作为本轮证据。旧文件保持原样，供追踪。
