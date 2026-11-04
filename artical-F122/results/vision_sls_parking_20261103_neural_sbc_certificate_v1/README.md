# 停车场结构化神经 SBC 结果包

最终对象：几何 ReLU 网络＋原训练网络的有界 Tanh/Sigmoid 项＋线性时钟。权重是 `neural_sbc.pt`，定义为 `Aebs.vision_sls_parking.geometric_neural_sbc.GeometricNeuralSBC`。

**通过范围：新增在线后备检查的增广闭环可达不变域。不是原无过滤器 SPVC 的任意八维安全状态域。**初始盒子为原 15 个起点各分量 ±0.02，时域 30，原 RK4 零过程扰动模型。初始 SBC ≤0.305，危险 SBC ≥20，有效非终端单步变化 ≤−0.005。详见工作区 `docs/current/停车场神经SBC_构造证明与实验_20261103.md`；本包 `PROOF.md` 为对应快照（其中相对链接按原工作区位置解释）。

原点和扰动两组实际图像实验各 15/15 成功，无碰撞、无越界。原始实验归档在 `original/`、`fresh/`；`certificate.json` 给出逐场景独立区间审计；`remote_certificate_audit.json` 是服务器独立重算。`validation.json` 记录本地和服务器 CPU 的 59 项通过测试及一次 GPU 资源不足的中间失败。

启动 CP 使用单独的拟合、校准、审计样本。`startup_model.json` 保存误差集合；`startup_samples.npz` 保存独立渲染对应的状态和观测向量，不是原始图片文件。匹配该组启动分布和固定视觉/渲染协议时，CP 与不变性支持单次新实验至少 95% 的模型安全概率；不是固定校准集的逐状态保证或实车保证。启动之后的视觉误差可任意变化，但任务到达并非在所有这些变化下都有保证。

`ppo.zip`、`projection.pt`、`references.npz` 是运行所需冻结候选；DINO-G 大权重仍在服务器外部权重目录。`source_learned_sbc.pt` 是保留学习项的原权重，不应当作独立通过的原 SBC。`source_snapshot/` 保存实现；`initial_braking_cover.npz` 和 `reference_cover.npz` 保存连续盒子外包；`adversarial/` 保存反例回归；`parking_neural_sbc.png` 与 PDF 是结果图；`manifest.json` 是文件 SHA-256 清单。

载入网络：

```python
import torch
from Aebs.vision_sls_parking.planning_feedback import ReferenceBank
from Aebs.vision_sls_parking.planned_sbc import PlannedBarrier
from Aebs.vision_sls_parking.geometric_neural_sbc import GeometricNeuralSBC
bank = ReferenceBank(package_path / 'references.npz')
sbc = GeometricNeuralSBC(PlannedBarrier(bank))
sbc.load_state_dict(torch.load(package_path / 'neural_sbc.pt', map_location='cpu', weights_only=False))
sbc.eval()
# sbc(state8, reference_id, time): x 与 observer 拼接；离散编号和时间是 batch 长度的整数张量。
```

在线运行须使用 `BackupGovernor(..., initial_jitter=.02, startup_model=...)`；仅加载 PPO 或 SBC 权重不提供本包的闭环保证。仿真可用 `python -m Aebs.vision_sls_parking.evaluate_backup --candidate <本包> --startup-model <本包>/startup_model.json --sensor images --bound .02 --device cpu --threads 8 --output-dir <新目录>`，服务器上配合 `xvfb-run -a`。
