# Git 仓库体积审计与清理建议

2026-11-09。**已按确认执行取消跟踪；没有改写历史。** 工作区文件保留，只是移出 Git 索引。

## 1. 当前体积

| 项目 | 大小 |
|---|---:|
| 工作区 | 5.5 GB |
| `.git` | 清理前 2.0 GB → 清理后 **1.5 GB** |
| 已跟踪文件（清理前） | **1.70 GB / 3,085 个** |
| 已跟踪文件（清理后） | **0.35 GB / 2,688 个** |
| `.git` garbage（临时 pack） | 527 MB（已用 `git gc --prune=now` 清掉） |
| Git pack | 1.44 GiB，1 个 pack，0 loose |

清理命令（已执行）：

```bash
git gc --prune=now
```

## 2. 大在哪里

| 类别 | 规模 |
|---|---:|
| `artical-F122/` 已跟踪 | 2,516 个文件 |
| 其中 `artical-F122/results/` | **1,934 个文件** |
| 结果二进制约 | **795 个，约 0.96 GB** |
| `work/` 已跟踪 | 284 个文件，约 0.6 GB |
| `tmp/` 已跟踪 | 91 个文件，约 54 MB |

最大的几个已跟踪文件：

| 文件 | 大小 |
|---|---:|
| `artical-F122/results/mobile_robot_20261108_perception_r1/features.pt` | **629 MB** |
| `work/mobile_robot_20261108/features_portable.pt` | **629 MB** |
| `artical-F122/results/native_controllers_speed_residual_20261019_v1/reference_panel.npz` | 54 MB |
| `artical-F122/results/vision_sls_parking_20261025_same_state_v1/rendered_panel.npz` | 53 MB |
| `artical-F122/results/real_aebs_repair_sbc_20261108_v2/verify_trajectory/counterexamples.npz` | 21 MB |
| `artical-F122/results/real_aebs_repair_sbc_20261108_v1/verify_rgb/counterexamples.npz` | 20 MB |
| `artical-F122/auto_LiRPA/tests/data/maxpool_test_data` | 13 MB |

其他明显不该进 Git 的：

- `tmp/`：CARLA 预览抓图，91 个文件；
- `work/`：服务器快照、日志、warmstart 权重；
- 根目录 `real_aebs_artifact_v1/`、`real_aebs_artifact_v2/`、`real_image_aebs_probe_v1/`：重复结果目录；
- `Users/johann/project/work_based_on_spvc/docs/README.md`：误把绝对路径带进仓库；
- `VISION-SLS`：在索引中是 **gitlink（mode 160000）**，但没有 `.gitmodules`；它是独立 checkout，不是普通文件。

## 3. 已执行：取消跟踪生成物

已提交一次清理 commit（`Stop tracking generated artifacts and add gitignore`），共 401 个文件变化、54,929 行删除。实际执行了：

```bash
git rm -r --cached work tmp Users
git rm -r --cached real_aebs_artifact_v1 real_aebs_artifact_v2 real_image_aebs_probe_v1
git rm --cached VISION-SLS
git rm --cached \
  artical-F122/results/mobile_robot_20261108_perception_r1/features.pt \
  work/mobile_robot_20261108/features_portable.pt \
  artical-F122/results/native_controllers_speed_residual_20261019_v1/reference_panel.npz \
  artical-F122/results/vision_sls_parking_20261025_same_state_v1/rendered_panel.npz \
  artical-F122/results/real_aebs_repair_sbc_20261108_v2/verify_trajectory/counterexamples.npz \
  artical-F122/results/real_aebs_repair_sbc_20261108_v1/verify_rgb/counterexamples.npz
```

工作区文件仍在原位置；`work/`、`tmp/`、大二进制已被 `.gitignore` 覆盖。取消 `VISION-SLS` gitlink 后，它作为本地独立 checkout 保留，未提交。

## 4. 已加入 `.gitignore`

新增规则覆盖：

- OS/编辑器：`.DS_Store`、`.obsidian/`、`.claude/`、`.idea/`、`.vscode/`；
- Python：`__pycache__/`、`.pytest_cache/`、`*.egg-info/`、`build/`、`dist/`、venv；
- 外部 DINO 与独立 `VISION-SLS/` checkout；
- 原始数据：`carla_data_spvc_expanded/`、`data/lane_keeping*`、`*.h5`；
- 大二进制：`*.pt`、`*.pth`、`*.ckpt`、`*.safetensors`、`*.onnx`、`*.npz`、`*.npy`、`*.zip`、`*.tar*`；
- 临时/服务器快照：`tmp/`、`work/`、`logs/`、`*.log`；
- 误拷目录：`Users/`、根目录重复 artifact 目录。

**注意：`.gitignore` 不会让已跟踪文件消失。** 当前 `git status` 只有 `.gitignore` 修改和 `VISION-SLS`，因为其余大文件都已被跟踪。

## 5. 取消跟踪命令（已执行，供复核）

取消跟踪只影响索引，不删除工作区文件：

```bash
# 1) 立即取消明显的大目录/重复目录
git rm -r --cached work tmp Users
git rm -r --cached real_aebs_artifact_v1 real_aebs_artifact_v2 real_image_aebs_probe_v1

# 2) 取消两个 629 MB 特征文件
git rm --cached \
  artical-F122/results/mobile_robot_20261108_perception_r1/features.pt \
  work/mobile_robot_20261108/features_portable.pt

# 3) 取消结果中的大二进制（按 20 MB 以上清单逐项确认）
git rm --cached \
  artical-F122/results/native_controllers_speed_residual_20261019_v1/reference_panel.npz \
  artical-F122/results/vision_sls_parking_20261025_same_state_v1/rendered_panel.npz \
  artical-F122/results/real_aebs_repair_sbc_20261108_v2/verify_trajectory/counterexamples.npz \
  artical-F122/results/real_aebs_repair_sbc_20261108_v1/verify_rgb/counterexamples.npz

git add .gitignore
git commit -m "Stop tracking generated artifacts and add gitignore"
```

撤销误操作：

```bash
git reset HEAD -- <path>   # 只撤销索引变更
```

## 6. 为什么 `.git` 不会马上变小

即使执行上面的 `git rm --cached`，旧提交里仍然保存这些 blob；`.git` 仍接近 1.5 GB，`git clone` 也仍会下载历史。

要让 **历史** 变小，需要重写历史；这会改写 commit hash，影响远程协作者：

```bash
# 先做镜像备份
git clone --mirror . /tmp/work_based_on_spvc_backup.git

# 删除 20 MB 以上 blob（示例；先 dry-run / 备份）
git filter-repo --strip-blobs-bigger-than 20M

# 或用路径删除
git filter-repo --path work --path tmp --path Users --invert-paths
```

重写后需要 force-push；`origin/main` 是共享远程，执行前必须确认。

## 7. 推荐长期结构

- **代码/文档/小 JSON/README/精选小图**：Git。
- **大权重、特征、npz、zip、原始图像**：服务器/对象存储/发布包；Git 只留 SHA-256 和下载地址。
- 必须版本化的大文件：Git LFS（`git lfs track "*.pt"` 等），但 LFS 也有配额。
- `work/`、`tmp/` 不再作为源码目录提交；需要留证据时，导出精简报告到 `results/` 或 `docs/`。
- `VISION-SLS` 若是外部参考代码，建议正式改成 submodule；否则从索引移除并保留本地 checkout。
