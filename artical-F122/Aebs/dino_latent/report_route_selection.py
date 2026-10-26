"""Role-separated tables/figures from completed development results only."""
import argparse
import json
from pathlib import Path

import numpy as np
from Aebs.dino_latent.latent_method_models import paired_cluster_bootstrap
from Aebs.dino_latent.prepare_expanded_data import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory",type=Path,required=True)
    a=p.parse_args()
    data=json.loads((a.directory/"metrics.json").read_text())
    if data.get("status")!="role_separated_development_route_audit_complete" or not data["source_files_unchanged"]:
        raise ValueError("Completed frozen-source audit required")
    extra={}
    pairs={"matched_MLP_task_vs_plain":("new/mlp32_parameter_matched_task_aware","latest/mlp32_parameter_matched"),
           "Transformer_task_vs_plain":("latest/transformer32_task_aware","latest/transformer32"),
           "task_MLP_vs_task_Transformer":("new/mlp32_parameter_matched_task_aware","latest/transformer32_task_aware")}
    for label,(candidate,baseline) in pairs.items():
        extra[label]={}
        for h in (1,3,10):
            arrays=[np.load(a.directory/(name.replace('/','__')+"__h%d.npz"%h)) for name in (candidate,baseline)]
            if not np.array_equal(arrays[0]["windows"],arrays[1]["windows"]):raise ValueError("Unpaired analysis rows")
            extra[label][str(h)]=paired_cluster_bootstrap(np.abs(arrays[0]["residual"]),np.abs(arrays[1]["residual"]),arrays[0]["windows"][:,0])
    (a.directory/"objective_comparisons.json").write_text(json.dumps(dict(source_metrics_sha256=sha256(a.directory/"metrics.json"),
        comparisons=extra,scope="old development trajectories; paired whole-trajectory mean differences, NOT training-seed uncertainty"),indent=2))
    lines=["# 潜在路径整体适用性对照", "", "日期：2026年10月19日。新训练仅一组参数匹配任务MLP；原R2、PPO、SBC未变。", "",
        "## 如何读这份对照", "",
        "静态表示回答当前图像编码成什么；时序预测器回答历史表示和动作如何预测未来。多数时序模型继续使用原R2，因此不是替代R2的图像编码器。已知目标真实距离的q仅作验证代理参照，不纳入同信息量预测排名。", "",
        "所有数字来自旧开发数据，未读取第20步H5、RGB test或重新校准CP，也没有运行新候选自己的闭环。共同R2读出及保留actor属于任务训练的评价接口，不能据此宣布新表示的原生控制性能。", "",
        "## 后续同划分十组预测器", "",
        "以下十组使用相同321条训练、69条选择、69条内部审计轨迹；新的任务MLP复用相同步数、批量、初始化规则和任务目标。内部审计是开发数据，不是新最终测试。", ""]
    for horizon in (1,3,10):
        lines += ["### 第%d帧预测后的物理后继一致性"%horizon, "",
            "速度单位m/s。CPU耗时为整个h步预测调用的batch1中位毫秒，不含DINO、控制器和截图。", "",
            "| 路径 | 组合数 | MAE | P95 | 最大绝对误差 | 6至7米且0.5至1m/s的MAE | 参数量 | CPU毫秒 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for name,r in data["forecasters"].items():
            if r["legacy"]:continue
            h=str(horizon);m=r["horizons"][h]["latest_same_split69"]
            s=m["all"];near=m.get("braking_switch_d6_7_v0p5_1")
            lines.append("| %s | %d | %.9f | %.9f | %.9f | %s | %d | %.6f |"%(name,s["count"],s["mae"],s["p95_abs"],s["max_abs"],
                "%.9f"%near["mae"] if near else "无样本",r["parameters"],r["CPU_predictor_timing"][h]["1"]["median_ms"]))
        ref=data["reference_proxy"][str(horizon)]["latest_same_split69"]["all"]
        lines += ["", "相同目标状态、但额外已知目标真实距离的当前q参照：MAE %.9f，最大%.9f；不可与上表按同信息量排榜。"%(ref["mae"],ref["max_abs"]), ""]
    lines += ["## 训练目标与结构的二乘二对照", "",
        "前两组按整条轨迹等权计算任务目标相对普通目标的平均绝对误差差值，负数有利于任务目标；第三组为相同任务目标的MLP减Transformer，负数有利于MLP。区间仅针对这组冻结模型和旧数据，不代表多次训练的不确定性。", "",
        "| 比较对象 | 步长 | 平均差 m/s | 95%配对区间 | 改善轨迹比例 |", "|---|---:|---:|---|---:|"]
    for name,hs in extra.items():
        for h,s in hs.items():
            lines.append("| %s | %s | %.9f | [%.9f, %.9f] | %.2f%% |"%(name,h,s["mean_paired_delta"],*s["ci95"],100*s["lower_error_trajectory_fraction"]))
    n=data["config"]["cohort_counts"]["common_nontraining"]
    lines += ["", "## 早期与后续模型的共同未拟合子集", "",
        "共有%d条轨迹，各模型都未用它们拟合预测器权重，但早期模型曾将其所在验证池用于checkpoint选择。因此仍不是独立测试。早期训练1200步、batch256，后续800步、batch128，也不是跨批次纯结构消融。"%n, "",
        "仅此面板比较全部21条登记模型记录，其中部分concat基线权重重复。", ""]
    for horizon in (1,3):
        h=str(horizon)
        lines += ["### 共同子集第%d帧"%horizon,"",
            "| 路径 | 组合数 | MAE m/s | P95 m/s | 最大误差 m/s | CPU毫秒 |", "|---|---:|---:|---:|---:|---:|"]
        for name,r in data["forecasters"].items():
            s=r["horizons"][h]["common_nontraining"]["all"]
            lines.append("| %s | %d | %.9f | %.9f | %.9f | %.6f |"%(name,s["count"],s["mae"],s["p95_abs"],s["max_abs"],r["CPU_predictor_timing"][h]["1"]["median_ms"]))
        lines.append("")
    lines += ["## 静态表示现有证据与接入状态", "",
        "此表复用相同2460张静态开发图的既有结果；图像与代理均通过冻结R2读出，因此偏向保留R2，PCA/AE/VAE尚未各自训练原生PPO/SBC。不能拿此表作所有表示的最终优劣证明。", "",
        "| 静态路径 | 原生相对latent RMSE | 图像距离MAE m | 代理距离MAE m | 相对原R2图像的全部速度后继MAE m/s | 原生配套闭环证据 |",
        "|---|---:|---:|---:|---:|---|"]
    for name,r in data["static_representation_existing_evidence"]["results"].items():
        lines.append("| %s | %.6f | %.6f | %.6f | %.6f | %s |"%(name,r["relative_native_rmse"],r["image_distance_error_m"]["mae"],r["proxy_distance_error_m"]["mae"],
            r["all_pair_proxy_vs_original_speed_gap_mps"]["mae"],"已有R2主线与旧口径SBC，非图像端到端证书" if name=="original_R2" else "未重训原生PPO/SBC"))
    lines += ["", "## 不应遗漏的验收门槛", "",
        "1. 当前主线先选静态表示，再决定是否需要世界模型；二者不是互相替代的同一选择。",
        "2. 图像和q的匹配不能以牺牲真实控制任务换取；需要同时报告success、unsafe、timeout和安全停在目标外。",
        "3. 本轮预测误差和单步危险分歧不能冒称新策略自身回合成功率。",
        "4. 只有选定一个完整候选后才进入新的匹配PPO、SBC和独立图像闭环校准；不把全部路线都投入昂贵后续实验。",
        "5. 在旧参考数据上挑选后，不复用它作为新独立CP校准集；不为得到小半径而降低原目标。", "",
        "数据与配置：[完整JSON](metrics.json)、[登记和划分](config.json)、[同结构目标差异](objective_comparisons.json)。"]
    (a.directory/"潜在路径整体适用性对照.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    names=["latest/mlp32_parameter_matched","new/mlp32_parameter_matched_task_aware","latest/transformer32","latest/transformer32_task_aware","latest/gru32"]
    labels=["Matched MLP","Matched MLP + task","Transformer","Transformer + task","GRU"]
    fig,axes=plt.subplots(1,3,figsize=(13,4.2))
    for ax,key,title in zip(axes,("mae","max_abs","latency"),("Mean next-speed error (m/s)","Maximum next-speed error (m/s)","10-step predictor CPU latency (ms)")):
        values=[]
        for name in names:
            r=data["forecasters"][name]
            values.append(r["CPU_predictor_timing"]["10"]["1"]["median_ms"] if key=="latency" else r["horizons"]["10"]["latest_same_split69"]["all"][key])
        ax.barh(range(len(names)),values,color=["#a8bccb","#426b8a","#c0c6ce","#bc7147","#7e9d73"])
        ax.set_yticks(range(len(names)));ax.set_yticklabels(labels,fontsize=9);ax.invert_yaxis();ax.set_title(title,fontsize=10)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.grid(axis="x",alpha=.2)
    fig.suptitle("Same 69 old development trajectories: accuracy, tails, and predictor cost",fontsize=12)
    fig.tight_layout();fig.savefig(a.directory/"accuracy_tail_cost.png",dpi=180);plt.close(fig)
    print("report:",a.directory/"潜在路径整体适用性对照.md")


if __name__=="__main__":main()
