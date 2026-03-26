import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext
from pathlib import Path

# =========================
# 1. 论文风格全局设置
# =========================
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.8,
    "lines.markersize": 4,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.3,
})

# 输出目录
out_dir = Path("paper_figures")
out_dir.mkdir(exist_ok=True)

# =========================
# 2. 读入数据
# =========================
df = pd.read_csv("sum_all_compare_float_error_dense_long.csv")
summary_df = pd.read_csv("sum_all_compare_float_error_summary.csv")

# 只分析误差最有意义的两类数据
df_focus = df[df["dataset"].isin(["large_small", "cancellation"])].copy()

# 代表性算法：适合论文主图
focus_algorithms = ["naive", "way8", "recursive", "stdacc", "kahan"]

algo_labels = {
    "naive": "Naive",
    "way8": "8-way",
    "recursive": "Recursive Pairwise",
    "stdacc": "std::accumulate",
    "kahan": "Kahan",
}

# 黑白/论文友好线型
algo_styles = {
    "naive":     {"linestyle": "-",  "marker": "o"},
    "way8":      {"linestyle": "--", "marker": "s"},
    "recursive": {"linestyle": "-.", "marker": "^"},
    "stdacc":    {"linestyle": ":",  "marker": "D"},
    "kahan":     {"linestyle": "-",  "marker": "x"},
}

dataset_titles = {
    "large_small": "Large-small mixture",
    "cancellation": "Cancellation",
}

# =========================
# 5. 图3：每种算法的最大相对误差
# =========================
summary_plot = summary_df.copy()

order = ["naive", "way2", "way4", "way8", "way16", "recursive", "iterative", "divide", "stdacc", "kahan"]
summary_plot["algorithm"] = pd.Categorical(summary_plot["algorithm"], categories=order, ordered=True)
summary_plot = summary_plot.sort_values("algorithm")

fig, ax = plt.subplots(figsize=(8.0, 4.2), constrained_layout=True)

bars = ax.bar(
    summary_plot["algorithm"],
    summary_plot["max_rel_error"],
    edgecolor="black",
    linewidth=0.8
)

ax.set_yscale("log")
ax.set_ylabel("Maximum relative error")
ax.set_xlabel("Algorithm")
ax.set_title("Maximum relative error across all datasets and sizes")
ax.grid(True, axis="y", which="both")

plt.xticks(rotation=30, ha="right")

plt.savefig(out_dir / "fig_max_relative_error_bar.pdf", bbox_inches="tight")
plt.savefig(out_dir / "fig_max_relative_error_bar.png", bbox_inches="tight")
plt.show()