# R2跨服务器CARLA校准轨迹采集与回传手册

更新日期：2026-10-07。

本手册用于在CARLA服务器采集R2独立闭环校准轨迹，将结果回传到模型服务器，并生成trajectory-level物理状态误差契约。采集阶段冻结DINO、R2 projection和R2-PPO，不读取现有train、validation或test图像，不更新任何模型。

## 两台服务器的职责

| 服务器 | 用途 | 本手册中的名称 |
| --- | --- | --- |
| 运行CARLA的图形服务器 | 生成同步RGB图像并执行图像路径闭环 | CARLA服务器 |
| `root@10.11.154.191 -p 20020` | 保存当前模型、校准误差契约及后续SBC结果 | 模型服务器 |

当前模型服务器项目目录：

```text
/root/work_based_on_spvc/artical-F122
```

采集器文件：

```text
Aebs/connect/collect_r2_calibration_trajectories.py
```

校准器文件：

```text
Aebs/dino_latent/calibrate_transition_contract.py
```

## CARLA服务器需要的文件

建议将完整项目目录和`external`目录复制到CARLA服务器。至少需要以下文件：

```text
artical-F122/Aebs/
artical-F122/Combined_network/
artical-F122/results/dino_expanded_v1/02_dino_h5/cache_manifest.json
artical-F122/results/dino_expanded_v1/08_R2_group_alignment/dino_safety_latent.pt
artical-F122/results/dino_expanded_v1/09_R2_ppo/latent_ppo.zip
artical-F122/results/dino_expanded_v1/09_R2_ppo/metrics.json
external/dinov2/
external/dinov2_weights/dinov2_vits14_pretrain.pth
```

修复后的采集器还必须重新复制到CARLA服务器，不能继续使用发生过位姿错配的旧版本。可以先从模型服务器拉取：

```bash
scp -P 20020 \
  root@10.11.154.191:/root/work_based_on_spvc/artical-F122/Aebs/connect/collect_r2_calibration_trajectories.py \
  /path/to/artical-F122/Aebs/connect/
```

## 环境检查

在CARLA服务器进入项目并激活包含CARLA、PyTorch、stable-baselines3、h5py和NumPy的环境：

```bash
cd /path/to/artical-F122
conda activate carla_env
```

检查关键依赖：

```bash
python - <<'PY'
import carla, h5py, numpy, torch
import stable_baselines3
print("CARLA client:", carla.__file__)
print("CUDA available:", torch.cuda.is_available())
PY
```

如果`CUDA available`为`False`，可以把后续命令中的`--device cuda`改成`--device cpu`，但DINO逐帧推理会明显变慢。

## 启动和检查CARLA

使用专用、渲染开启的CARLA实例。世界中不能已有车辆、行人或传感器。采集器不会自动加载或切换地图。

CARLA启动完成后，在项目目录检查当前地图和spawn点：

```bash
python -m Aebs.connect.collect_r2_calibration_trajectories \
  --inspect \
  --host 127.0.0.1 \
  --port 2000 \
  --output-dir results/dino_expanded_v1/calibration_inspect_unused
```

记录输出中的地图短名，例如：

```text
Town10HD_Opt
```

默认使用`spawn-index=4`。应确认该位置与原AEBS采集场景一致，并且道路近似水平。

## 必须先运行烟雾采集

不要直接运行459条正式采集。先用一个全新的目录运行2条轨迹：

```bash
python -m Aebs.connect.collect_r2_calibration_trajectories \
  --host 127.0.0.1 \
  --port 2000 \
  --expected-map Town10HD_Opt \
  --spawn-index 4 \
  --episodes 2 \
  --horizon 400 \
  --seed 1701 \
  --settle-ticks 10 \
  --device cuda \
  --output-dir results/dino_expanded_v1/10_calibration_smoke_v2
```

将`Town10HD_Opt`替换成inspect实际输出的地图短名。成功条件是末尾摘要满足：

```text
status = complete
completed_trajectories = 2
h5不为null
h5_sha256不为null
cleanup_errors为空
```

烟雾目录只用于检查流程，不能作为正式校准数据。

## 正式采集459条轨迹

烟雾采集成功后，重新启动或清空CARLA世界，并使用新的输出目录：

```bash
python -m Aebs.connect.collect_r2_calibration_trajectories \
  --host 127.0.0.1 \
  --port 2000 \
  --expected-map Town10HD_Opt \
  --spawn-index 4 \
  --episodes 459 \
  --horizon 400 \
  --seed 1701 \
  --settle-ticks 10 \
  --device cuda \
  --output-dir results/dino_expanded_v1/10_calibration_trajectories_v2
```

固定设置的含义：

- 每条轨迹初始距离均匀采样于`[15,16] m`；
- 初始速度均匀采样于`[2.5,3] m/s`；
- 每条轨迹独立均匀选择4种天气和3种车色；
- 最长400步，时间步长`0.05 s`；
- 使用图像latent路径的R2-PPO推进状态；
- q路径不参与状态推进，仅在后续离线校准时用于构造对照残差；
- seed固定为1701，不根据结果更换。

正式完成后应生成：

```text
results/dino_expanded_v1/10_calibration_trajectories_v2/config.json
results/dino_expanded_v1/10_calibration_trajectories_v2/summary.json
results/dino_expanded_v1/10_calibration_trajectories_v2/calibration_trajectories.h5
```

只有`summary.json`中`status=complete`且H5存在时，才能继续校准。

## 本次camera distance mismatch错误的原因和处理

此前运行在第21条轨迹报错：

```text
camera distance mismatch requested=15.082832 actual=6.005231
```

原因是新轨迹把相机从上一条轨迹终点约6米移回约15米时，CARLA第一帧偶发保留旧相机位姿。修复版不再固定等待1个tick，而是在最多10个同步tick内逐帧核对真实几何，只接受距离误差不超过0.01米的帧；旧帧不会写入H5。

旧失败目录状态为`incomplete`，不得续跑、重命名为正式结果或交给校准器。保留它用于追溯，新的烟雾和正式采集必须使用新目录。

以下信息只是警告，不是此次失败原因：

```text
torch.load weights_only=False FutureWarning
xFormers is not available
PPO on GPU with MlpPolicy
```

日志中的`success`、`unsafe`和`stopped_safe_outside_goal`是图像闭环真实开发结果，也不是采集器错误。必须保留并汇报。此前已完成的20条轨迹中有5条unsafe，这说明同步CARLA传感器图像与旧截图训练分布之间可能存在明显差异；正式数据仍可用于量化误差契约，但当前R2-PPO不能因此被描述为安全。

## 将采集结果回传到模型服务器

先在模型服务器创建目标目录：

```bash
ssh -p 20020 root@10.11.154.191 \
  "mkdir -p /root/work_based_on_spvc/artical-F122/results/dino_expanded_v1/10_calibration_trajectories_v2"
```

然后从CARLA服务器发送三个文件：

```bash
scp -P 20020 \
  results/dino_expanded_v1/10_calibration_trajectories_v2/config.json \
  results/dino_expanded_v1/10_calibration_trajectories_v2/summary.json \
  results/dino_expanded_v1/10_calibration_trajectories_v2/calibration_trajectories.h5 \
  root@10.11.154.191:/root/work_based_on_spvc/artical-F122/results/dino_expanded_v1/10_calibration_trajectories_v2/
```

## 在模型服务器生成误差契约

回传完成后登录模型服务器：

```bash
ssh root@10.11.154.191 -p 20020
cd /root/work_based_on_spvc/artical-F122
```

运行正式trajectory-level校准：

```bash
/opt/miniconda3/bin/conda run --no-capture-output -n vt \
python -m Aebs.dino_latent.calibrate_transition_contract \
  --calibration-h5 results/dino_expanded_v1/10_calibration_trajectories_v2/calibration_trajectories.h5 \
  --representation results/dino_expanded_v1/08_R2_group_alignment/dino_safety_latent.pt \
  --ppo-dir results/dino_expanded_v1/09_R2_ppo \
  --training-cache-manifest results/dino_expanded_v1/02_dino_h5/cache_manifest.json \
  --horizon 400 \
  --epsilon 0.01 \
  --beta 0.01 \
  --output-dir results/dino_expanded_v1/11_transition_contract_v1
```

输出包括：

```text
config.json
contract.json
metrics.json
calibration_scores.npz
```

该契约只给出登记分布下的轨迹级视觉闭环残差覆盖。在SBC验证器证明契约内所有物理残差都满足barrier条件之前，仍不能称为控制器安全证书。

## 失败恢复规则

- 输出目录已存在时不要覆盖，改用带新后缀的目录。
- `status=incomplete`的H5不得用于校准。
- CARLA中出现额外车辆、行人或传感器时，清空或重启专用世界后重跑。
- 地图名不一致时不要修改检查逻辑，使用正确CARLA地图重新启动。
- 位姿在10个tick后仍不匹配时，保存完整报错和`summary.json`，不要放宽0.01米阈值。
- 采集中断后不从partial H5继续；使用相同固定设置和新目录完整重跑。
