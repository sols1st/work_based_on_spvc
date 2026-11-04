# 规划、视觉反馈与 PPO 修正候选

冻结参考库＋延迟观测器＋时变 LQR＋有界 PPO 修正。具体对照结果见 summary.json。控制器使用首帧图像与目标选参考，不使用运行时真实状态；参考库基于原 15 个开发起点。

**尚无 SBC 安全证书，也没有轨迹级 CP 覆盖保证。** 原 20 维历史策略证书不适用于此模型。source_snapshot 保存本次实现；projection.pt 的原适配域记录在 config.json。

在 GPU 服务器 artical-F122 目录运行，输出目录需为新目录：

```bash
CUDA_VISIBLE_DEVICES=0 xvfb-run -a /opt/miniconda3/envs/vt/bin/python -m Aebs.vision_sls_parking.evaluate_planned_images --references results/vision_sls_parking_20261103_planned_candidate/references.npz --projection results/vision_sls_parking_20261103_planned_candidate/projection.pt --residual-policy results/vision_sls_parking_20261103_planned_candidate/ppo.zip --modes feedback residual --output-dir results/parking_planned_candidate_recheck
```

## 后续 SBC 求解

新闭环 SBC 已训练并保存，但没有通过；CP 盒内存在从原起点出发的超速反例。模型权重和参考轨迹未修改，详见 sbc_status.json 及 [SBC 求解报告](/Users/johann/project/work_based_on_spvc/docs/current/停车场新控制器_SBC求解与反例_20261103.md)。
