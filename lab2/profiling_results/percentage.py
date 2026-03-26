import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Patch

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Noto Sans CJK SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

df = pd.read_csv("sum_all_compare_float_error_dense_long.csv")
df = df[df["dataset"].isin(["large_small", "cancellation"])].copy()

df["rel_error_safe"] = np.maximum(df["rel_error"], 1e-30)
df["log_rel_error"] = np.log10(df["rel_error_safe"])

focus_algorithms = ["naive", "way8", "recursive", "stdacc", "kahan"]

algo_name_map = {
    "naive": "朴素累加",
    "way8": "8路链式累加",
    "recursive": "递归二叉归约",
    "stdacc": "标准库累加",
    "kahan": "Kahan补偿求和",
}

dataset_title_map = {
    "large_small": "大数-小数混合",
    "cancellation": "抵消型数据",
}

n_min, n_max = df["n"].min(), df["n"].max()
bins = np.logspace(np.log10(n_min), np.log10(n_max), 8)
df["n_bin"] = pd.cut(df["n"], bins=bins, include_lowest=True)

def format_interval(interval):
    return f"{int(interval.left)}–{int(interval.right)}"

bin_categories = df["n_bin"].cat.categories
bin_labels = [format_interval(iv) for iv in bin_categories]

out_dir = Path("paper_figures")
out_dir.mkdir(exist_ok=True)

colors = {
    "naive": "#d9eaf7",
    "way8": "#f7d9d9",
    "recursive": "#d9f7df",
    "stdacc": "#f7f1d9",
    "kahan": "#ead9f7",
}

for dataset in ["large_small", "cancellation"]:
    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)

    sub = df[(df["dataset"] == dataset) & (df["algorithm"].isin(focus_algorithms))].copy()

    n_groups = len(bin_categories)
    n_algos = len(focus_algorithms)

    group_centers = np.arange(n_groups) * 2.0
    box_width = 0.28
    offsets = np.linspace(-(n_algos - 1) / 2, (n_algos - 1) / 2, n_algos) * box_width

    legend_handles = []

    for algo_idx, algo in enumerate(focus_algorithms):
        data = []
        positions = []

        for i, iv in enumerate(bin_categories):
            vals = sub.loc[
                (sub["algorithm"] == algo) & (sub["n_bin"] == iv),
                "log_rel_error"
            ].dropna().to_numpy()

            if len(vals) > 0:
                pos = group_centers[i] + offsets[algo_idx]
                data.append(vals)
                positions.append(pos)

        bp = ax.boxplot(
            data,
            positions=positions,
            widths=box_width * 0.9,
            patch_artist=True,
            showfliers=False
        )

        for box in bp["boxes"]:
            box.set(facecolor=colors[algo], edgecolor="black", linewidth=1.0, alpha=0.85)
        for median in bp["medians"]:
            median.set(color="black", linewidth=1.3)
        for whisker in bp["whiskers"]:
            whisker.set(color="black", linewidth=1.0)
        for cap in bp["caps"]:
            cap.set(color="black", linewidth=1.0)

        # 叠加散点：解决“箱体塌成黑线”看不清的问题
        for vals, pos in zip(data, positions):
            jitter = np.random.uniform(-box_width * 0.12, box_width * 0.12, size=len(vals))
            ax.scatter(
                np.full(len(vals), pos) + jitter,
                vals,
                s=14,
                color=colors[algo],
                edgecolors="black",
                linewidths=0.3,
                alpha=0.75,
                zorder=3
            )

        legend_handles.append(
            Patch(facecolor=colors[algo], edgecolor="black", label=algo_name_map[algo])
        )

    ax.set_title(f"{dataset_title_map[dataset]}：不同算法误差分布箱线图")
    ax.set_xlabel("问题规模区间 n")
    ax.set_ylabel("log10(相对误差)")
    ax.set_xticks(group_centers)
    ax.set_xticklabels(bin_labels, rotation=30, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(handles=legend_handles, loc="upper center", ncol=len(focus_algorithms), frameon=False)

    # 对 cancellation 单独放大纵轴，避免都挤在 0 附近
    if dataset == "cancellation":
        y = sub["log_rel_error"].to_numpy()
        ax.set_ylim(y.min() - 0.02, y.max() + 0.01)

    plt.savefig(out_dir / f"{dataset}_分组误差箱线图_带散点.pdf", bbox_inches="tight")
    plt.savefig(out_dir / f"{dataset}_分组误差箱线图_带散点.png", bbox_inches="tight")
    plt.show()