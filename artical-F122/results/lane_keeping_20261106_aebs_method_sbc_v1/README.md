# 车道保持：AEBS 同类训练式神经 SBC

2026-11-06。使用 AEBS 的 NonnegativeSBC 类，[2,16,8,1]、Tanh 隐藏层＋ReLU 输出，从随机初始化训练得到。无几何教师、LQR 构造分支或在线动作过滤器。

原 PPO 权重不变；两个 SBC 初始化种子都首次主验证通过，未触发策略反馈。反馈与任务回退模块已实现、测试，但本包不宣称观察到了交替更新的性能收益。

## 证据

- [主审计](certificate.json)：初态上界 0.08582801，危险下界 23.65349579，三个曲率零未解决区间。
- [普通 IBP 复核](independent_ibp.json)：初态上界 0.60715254，危险下界 20.14060243，零未解决区间。
- [粗 IBP 未通过记录](ibp8_unresolved.json)：直道 47,136 个未解决盒，没有点反例；误差每维从 8 段加细到 16 段后通过，半径保持 0.08。
- [训练配置](training/config.json)、[历史](training/history.json)：种子 41，2000 次更新达到开发检查通过。
- [第二个 SBC 种子](second_seed/certificate.json)：种子 42，1750 次更新；同一个 PPO，非第二个控制器种子。
- [回归](rollouts.json)：720 条误差轨迹和 12 条图像轨迹无越界；[13 项软件测试](tests.txt)。
- [逐步说明](../../../docs/current/车道保持_AEBS同类训练流程与神经SBC_20261106.md)。

## 使用边界

4 米/秒、0.1 秒离散步长、2.6 米轴距；初态 ±0.25 米/±0.06 弧度；安全边界 ±0.85 米/±0.30 弧度；已知曲率 −0.01/0/0.01；零过程扰动。每步两个归一化观测误差都在 ±0.08 内。

危险区域验证覆盖归一化 [-1.3,1.3]² 中的全部越界部分；任意安全状态的一步后继都在该大盒内，因此覆盖所有首次越界。未声称网络在无限平面上的危险下界。

这是连续域数值区间证据，非完整浮点形式证明。12 条图像回归中只有 10 条全程满足包络；另两条各一帧超过但仍安全。旧场景重跑属于开发回归，不是新独立图像测试或无条件 CP 安全概率。

## 复现

在 artical-F122 目录，选择不存在的输出目录：

~~~bash
# 完整流程；若首轮通过，不额外更新策略
../.venv-vision-sls/bin/python -m Aebs.lane_keeping.synthesize \
  --actor results/lane_keeping_20261105/ppo_image/actor.pt \
  --output-dir results/lane_aebs_reproduce --seed 41 --epochs 6000 --rounds 4

# 冻结模型主审计
../.venv-vision-sls/bin/python -m Aebs.lane_keeping.verify_learned \
  --actor results/lane_keeping_20261106_aebs_method_sbc_v1/actor.pt \
  --sbc results/lane_keeping_20261106_aebs_method_sbc_v1/sbc.pt \
  --output-dir results/lane_aebs_audit --grid 40 --depth 16 --error-splits 4

# 普通 IBP 复核
../.venv-vision-sls/bin/python -m Aebs.lane_keeping.verify_learned \
  --actor results/lane_keeping_20261106_aebs_method_sbc_v1/actor.pt \
  --sbc results/lane_keeping_20261106_aebs_method_sbc_v1/sbc.pt \
  --output-dir results/lane_aebs_ibp --method ibp --grid 50 --depth 16 --error-splits 16

../.venv-vision-sls/bin/python -m pytest tests/test_lane_joint.py tests/test_lane_certificate.py -q
~~~

加载 SBC 时使用 Aebs.vision_sls.sbc_cp.load_sbc_checkpoint；输入归一化真实状态 [e/0.85,theta/0.30]。它离线用于证明，不作为部署时额外动作修正器。

actor.pt 为策略，sbc.pt 为学习到的屏障，patch_projection.pt 为图像头，cp_patch_95.json 保留原参考校准结果。0.08 是本次声明验证包络，原 CP 文件半径约 0.07203；不能把人工扩大包络写成新的独立校准。

source_snapshot 保存本次相关源码，environment.json 记录环境和外部 DINO 权重散列，manifest.json 校验包内文件。manifest 本身不包含自身散列。原 11 月 5 日包与上一轮几何证书包保留，不覆盖。
