# R2 DINO＋原版SPVC最小替换实验

更新日期：2026-10-08。

## 1. 实验目标

本实验只回答一个问题：在不引入CP、残差契约、QP、安全过滤器、人工可恢复集或新验证器的情况下，把原SPVC的`cGAN＋state_net`前端替换为当前R2 DINO双路径表示后，原SPVC训练与验证流程能否跑通，以及能得到什么结果。

## 2. 完整数据流

部署图像路径：

```text
真实图像 → 冻结DINO → R2 projection → 32维latent ┐
                                                   ├→ PPO → 动作
速度 ─────────────────────────────────────────────┘
```

原SPVC训练和VT网格路径：

```text
物理距离 → 冻结q_psi → 32维latent ┐
                                   ├→ PPO → 原AEBS动力学
速度 ─────────────────────────────┘
                                      ↓
                         原人工均匀过程噪声
                                      ↓
                       原SBC训练与VT网格检查
```

两条路径使用同一R2 latent坐标和同一个PPO actor。VT状态网格无法直接生成每个连续状态对应的真实CARLA图像，因此沿用R2的物理代理分支`q_psi`。这一点必须写入结论边界。

## 3. 与原SPVC保持不变的内容

- 原`VTLearner`和交替训练循环；
- SBC结构`[2,16,8,1]`与平方输出；
- PPO actor结构及其原SPVC微调方法；
- `p_lip=2.0`、`l_lip=4.0`、`epsilon=0.1`；
- learner目标概率0.95、verifier目标概率0.9；
- `100×100`状态网格；
- `factor=0.05`定义的原均匀下一状态噪声；
- 原VT验证区域、概率计算和`0.1%`硬违反比例阈值；
- 原legacy IBP实现，不在本基线中替换或强化。

## 4. 唯一方法替换

```text
原版：状态＋z → cGAN → state_net → PPO
本版：距离 → q_psi latent → PPO
部署：图像 → frozen DINO/projection latent → PPO
```

表示模型固定为`08_R2_group_alignment`，初始控制器固定为`09_R2_ppo`。默认从随机SBC开始，不载入旧语义SBC，避免混入其他方法的证书形状。

## 5. 明确不使用的内容

- trajectory-level CP；
- 全局或动作条件残差盒；
- 新scenario鲁棒SBC；
- 可恢复集或viability计算；
- QP可微层；
- runtime safety filter；
- 新IBP或其他强化验证器；
- test集合。

## 6. 运行入口

```bash
cd /root/work_based_on_spvc/artical-F122

/opt/miniconda3/bin/conda run --no-capture-output -n vt \
python -m Aebs.dino_latent.run_spvc \
  --max-iteration-index 500 \
  --timeout-seconds 7200 \
  --stop-on-zero-violation \
  --output-dir results/dino_expanded_v1/17_R2_original_spvc
```

`500`只是增加允许的优化预算，不改变损失、噪声、网格或验证方法。`--stop-on-zero-violation`用于在第一次同时达到零硬违反、区域顺序正确和目标概率时停止，防止下一次PPO更新破坏该checkpoint。

## 7. 输出和判定

主要输出：

```text
results/dino_expanded_v1/17_R2_original_spvc/sbc.pt
results/dino_expanded_v1/17_R2_original_spvc/latent_ppo_spvc.pt
results/dino_expanded_v1/17_R2_original_spvc/metrics.json
```

最低成功条件：

1. `hard_violations=0/10000`；
2. `lb_unsafe > ub_init > domain_min`；
3. `actual_reach_prob >= 0.9`；
4. 保存的checkpoint与达到上述条件的迭代一致。

如果只有违反比例不超过`0.1%`，只能称为通过原代码宽松阈值，不能称为零违反。如果区域顺序错误，则不能计算有效概率下界。

## 8. 可以和不可以声称什么

可以声称：

> 在原SPVC噪声模型、原状态网格、原训练损失和原legacy VT实现下，R2 DINO前端对应的q代理控制路径完成了原SPVC基线训练与检查。

不可以声称：

> 真实图像DINO路径的全部感知误差已经被形式化覆盖。

原因是VT检查使用`distance→q_psi latent`，部署使用`image→DINO/projection latent`。本基线刻意不加入两条路径误差的CP或鲁棒包络。同时，原legacy VT区间实现已有已知审计限制，因此本结果应作为忠实基线，而不是比原论文更强的新形式化保证。
