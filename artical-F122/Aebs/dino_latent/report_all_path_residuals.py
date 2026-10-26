"""Render frozen residual results without retraining or rerunning calibration."""
import argparse
import json
from pathlib import Path


def threshold_count(summary, value):
    return summary["thresholds"]["%.12g" % value]["count_above"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--audit", type=Path, default=Path("results/cgan_latent_cp_tail_audit_20261018/metrics.json"))
    args = parser.parse_args()
    data = json.loads((args.directory / "metrics.json").read_text())
    audit = json.loads(args.audit.read_text())
    if not data.get("models_unchanged") or data.get("status", "").startswith("running"):
        raise ValueError("Need completed frozen-model calibration")
    names = list(data["paths"])
    lines = ["# 全路径物理后继残差与CP半径", "",
        "完成日期：2026年10月19日。所有速度数值单位为 m/s。", "",
        "使用第20步保留的459条参考PPO轨迹；不是每个候选自己闭环生成的轨迹。模型在读取本批数组前登记并冻结。",
        "残差是同一当前物理状态下，图像路径与代理路径动作造成的下一速度之差；距离坐标残差为0。",
        "每条轨迹取所有合格时刻的绝对残差最大值，再按99%轨迹覆盖、99%置信度校准。N=459时所需秩为最大值，故表中CP半径恰好等于本批最大绝对残差。",
        "CP陈述依赖独立同分布、模型未使用本批调参及参考轨迹分布等前提；正半径不是安全判定失败，半径也不是安全证书。",
        "时序路径h表示先预测未来第h帧表示，再在该目标帧真实物理状态上比较同一固定控制器的后继速度，不能理解为完整h步物理轨迹误差。",
        "时序模型有3帧上下文，仅评价有完整上下文和目标帧的位置；h1/h3/h10的时刻集合不同。未来动作是参考轨迹记录的查询动作，不是候选自主闭环。", "",
        "## 全部已登记路径", "",
        "| 路径和预测步长 | 状态或状态抽样组合数 | MAE | RMSE | P95绝对值 | P99绝对值 | CP半径及最大绝对值 | 超过0.15的组合数 | 超过0.15的轨迹数 | 超过旧精确半径的轨迹数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, result in data["paths"].items():
        s, t = result["step"], result["trajectory_score"]
        lines.append("| %s | %d | %.9f | %.9f | %.9f | %.9f | %.9f | %d | %d/459 | %d/459 |" % (
            name, s["count"], s["mae"], s["rmse"], s["p95_abs"], s["p99_abs"], result["cp_residual_radius_mps"],
            threshold_count(s, .15), threshold_count(t, .15), threshold_count(t, .1506854683160781)))
    joint = data["joint_reference_distribution_radius"]
    lines += ["", "共%d个路径与步长组合；旧pilot与速度消融中的concat基线有重复权重，不是%d种独立算法。" % (len(names), len(names)),
        "逐路径置信度为%.9f；多个分别99%%置信的结果不能直接合称同时99%%置信。" % next(iter(data["paths"].values()))["per_fixed_path_confidence"],
        "另把每条参考轨迹上的全部登记路径、步长及固定生成抽样先取最大值，再整体校准，得到共同半径%.9f，置信度%.9f。它覆盖的是这组已登记残差，不是联合安全概率。" % (joint["q_hat_mps"], joint["achieved_confidence"]),
        "", "## 原cGAN与DINO的静态配对绝对误差", "",
        "下表使用相同2460张开发图片及73800个非终止图像速度组合。MC32多出32次生成抽样，不能当作独立轨迹。",
        "native为原state_net与随仓库提供的基础PPO，不是已核实的最终原SPVC actor；shared为生成图也通过同一个R2编码器和当前actor，以固定控制接口。", "",
        "| 路径 | 组合数 | MAE | RMSE | P95 | P99 | 最大绝对值 | 超过0.15的组合数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    physical = audit["static_same_images"]["physical"]
    for name in ("original_R2_own_image", "cgan_native_01_z_zero", "cgan_native_01_uniform_mc32", "cgan_native_signed_uniform_mc32", "cgan_shared_z_zero", "cgan_shared_uniform_mc32"):
        s = physical[name]
        lines.append("| %s | %d | %.9f | %.9f | %.9f | %.9f | %.9f | %d |" % (name, s["count"], s["mae"], s["rmse"], s["p95_abs"], s["p99_abs"], s["max_abs"], threshold_count(s, .15)))
    lines += ["", "## 不能混淆的口径", "",
        "- 原生cGAN的逐轨迹RGB链路没有CP结果：参考H5只保存DINO特征，不能恢复逐帧原RGB。已做静态原生链路对照及逐轨迹共享R2接口对照，未用最近图片代替。",
        "- cGAN固定MC8只覆盖预先固定的8个噪声向量；z0是另一个固定向量，不是所有连续生成噪声的保证。",
        "- 原R2/PPO/SBC及所有候选权重哈希保持不变；没有训练、没有新的真实闭环评估、没有使用RGB test。",
        "- 本批第20步数据此前用于另一个冻结轨迹SBC的校准，不是刚采集的完全盲测。它未用于本轮模型拟合，但本次检查后已消耗；若据此改模型或挑选赢家，必须另留独立校准数据。",
        "- CP半径不等于运行中每一步都会出现该残差。允许半径内任意扰动持续发生的全局盒模型，比真实残差轨迹强得多；不能用覆盖率替代安全证明。", "",
        "数据源：[完整结果](metrics.json)、[冻结登记和哈希](frozen_methods.json)、[逐轨迹分数](all_trajectory_scores.npz)。"]
    (args.directory / "全路径CP残差结果.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    one = [(n, r) for n, r in data["paths"].items() if not n.endswith(("/h3", "/h10"))]
    fig, ax = plt.subplots(figsize=(11, max(7, .29 * len(one))))
    ys = list(range(len(one)))
    ax.scatter([r["step"]["mae"] for _, r in one], ys, label="Mean absolute residual", marker="o", s=24)
    ax.scatter([r["cp_residual_radius_mps"] for _, r in one], ys, label="Trajectory CP radius", marker="x", s=32)
    ax.axvline(.15, color="red", linestyle="--", linewidth=1, label="0.15 m/s reference (not a safety threshold)")
    ax.set_yticks(ys); ax.set_yticklabels([n for n, _ in one], fontsize=8)
    ax.invert_yaxis(); ax.set_xlabel("Absolute next-speed residual (m/s)")
    ax.set_title("459 reference-policy trajectories: low mean does not imply a small CP radius")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0, -.065)); ax.grid(axis="x", alpha=.2)
    fig.tight_layout(); fig.savefig(args.directory / "all_path_cp_radii.png", dpi=180); plt.close(fig)
    print("report:", args.directory / "全路径CP残差结果.md")


if __name__ == "__main__":
    main()
