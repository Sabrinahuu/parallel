import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 可选：设置中文字体，避免中文乱码
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Noto Sans CJK SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

df = pd.read_csv("sum_all_compare_float_error_dense_long.csv")
df = df[df["dataset"].isin(["large_small", "cancellation"])].copy()

algo_order = ["naive", "way2", "way4", "way8", "way16", "recursive", "iterative", "divide", "stdacc", "kahan"]

# 算法名改成中文
algo_name_map = {
    "naive": "朴素累加",
    "way2": "2路链式累加",
    "way4": "4路链式累加",
    "way8": "8路链式累加",
    "way16": "16路链式累加",
    "recursive": "递归二叉归约",
    "iterative": "循环二叉归约",
    "divide": "分治求和",
    "stdacc": "标准库累加",
    "kahan": "Kahan补偿求和",
}

# 数据集标题改成中文
dataset_title_map = {
    "large_small": "大数-小数混合：log10(相对误差)",
    "cancellation": "抵消型数据：log10(相对误差)",
}

out_dir = Path("paper_figures")
out_dir.mkdir(exist_ok=True)

fig, axes = plt.subplots(2, 1, figsize=(11, 6), constrained_layout=True)

for ax, dataset in zip(axes, ["large_small", "cancellation"]):
    sub = df[df["dataset"] == dataset].copy()
    pivot = sub.pivot_table(index="algorithm", columns="n", values="rel_error", aggfunc="mean")
    pivot = pivot.reindex(algo_order)

    data = np.log10(np.maximum(pivot.to_numpy(), 1e-30))

    im = ax.imshow(data, aspect="auto", interpolation="nearest")
    ax.set_title(dataset_title_map[dataset])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([algo_name_map[x] for x in pivot.index])

    col_positions = np.linspace(0, len(pivot.columns) - 1, 8, dtype=int)
    ax.set_xticks(col_positions)
    ax.set_xticklabels([f"{pivot.columns[i]}" for i in col_positions], rotation=30, ha="right")
    ax.set_xlabel("问题规模 n")
    ax.set_ylabel("算法")

cbar = fig.colorbar(im, ax=axes, shrink=0.92)
cbar.set_label("log10(相对误差)")

plt.savefig(out_dir / "误差热力图.pdf", bbox_inches="tight")
plt.savefig(out_dir / "误差热力图.png", bbox_inches="tight")
plt.show()