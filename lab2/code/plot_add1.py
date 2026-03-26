import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams

# ======================================
# 1. 中文字体设置：自动寻找 Matplotlib 可见字体
# ======================================
candidate_fonts = [
    "SimHei",
    "Microsoft YaHei",
    "Noto Sans CJK SC",
    "Noto Serif CJK SC",
    "Noto Sans CJK JP",
    "Noto Serif CJK JP",
    "Source Han Sans SC",
    "Source Han Serif SC",
    "WenQuanYi Micro Hei",
    "PingFang SC",
    "Heiti SC",
    "STHeiti",
    "Arial Unicode MS",
]

available_fonts = {f.name for f in fm.fontManager.ttflist}
chosen_font = None
for font in candidate_fonts:
    if font in available_fonts:
        chosen_font = font
        break

if chosen_font is None:
    raise RuntimeError(
        "未找到可用的中文字体。\n"
        "请先执行：fc-cache -fv\n"
        "并删除 Matplotlib 缓存：rm -rf ~/.cache/matplotlib\n"
        "然后重试。"
    )

print(f"当前使用中文字体: {chosen_font}")

rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = [chosen_font]
rcParams["axes.unicode_minus"] = False

# ======================================
# 2. 读取数据
# ======================================
df = pd.read_csv("sum_all_compare_results.csv")
df = df.sort_values("n").reset_index(drop=True)

algorithms = [
    "naive",
    "way2",
    "way4",
    "way8",
    "way16",
    "recursive",
    "iterative",
    "divide",
    "stdacc",
]

label_map = {
    "naive": "普通求和",
    "way2": "2路展开",
    "way4": "4路展开",
    "way8": "8路展开",
    "way16": "16路展开",
    "recursive": "递归求和",
    "iterative": "迭代求和",
    "divide": "分治求和",
    "stdacc": "标准库求和",
}

color_map = {
    "naive": "#8D6E63",
    "way2": "#90A4AE",
    "way4": "#7986CB",
    "way8": "#FF8A65",
    "way16": "#4DB6AC",
    "recursive": "#BA68C8",
    "iterative": "#9CCC65",
    "divide": "#E57373",
    "stdacc": "#B0BEC5",
}

# ======================================
# 3. 缓存边界
# 假设元素类型为 int(4B)
# 32KB L1, 256KB L2, 8MB L3
# ======================================
cache_points_n = {
    "L1缓存": 32 * 1024 // 4,          # 8192
    "L2缓存": 256 * 1024 // 4,         # 65536
    "L3缓存": 8 * 1024 * 1024 // 4,    # 2097152
}

cache_points_kb = {
    "L1缓存": 32,
    "L2缓存": 256,
    "L3缓存": 8192,
}

# ======================================
# 4. 图1：每元素时间图（核心性能图）
# ======================================
plt.figure(figsize=(13, 8))

for alg in algorithms:
    plt.plot(
        df["data_kb"],
        df[f"{alg}_time_per_elem_ns"],
        marker="o",
        linewidth=1.2,
        markersize=3.0,
        label=label_map[alg],
        color=color_map[alg]
    )

for label, xpos in cache_points_kb.items():
    plt.axvline(x=xpos, color="#D32F2F", linestyle=":", linewidth=1.5, alpha=0.75)
    plt.text(
        xpos,
        plt.ylim()[1] * 0.95,
        label,
        color="#D32F2F",
        fontsize=11,
        rotation=90,
        ha="center",
        va="top"
    )

plt.xscale("log")
plt.xlabel("数据规模（KB）", fontsize=14)
plt.ylabel("每元素时间（ns/elem）", fontsize=14)
plt.title("各算法每元素时间对比", fontsize=16)
plt.grid(True, which="both", alpha=0.3)
plt.legend(title="算法名称", ncol=3, fontsize=10, title_fontsize=11)
plt.tight_layout()
plt.savefig("图1_每元素时间对比.png", dpi=300, bbox_inches="tight")
plt.show()

# ======================================
# 5. 图2：相对 naive 的加速比
# ======================================
plt.figure(figsize=(13, 8))

speedup_algs = ["way2", "way4", "way8", "way16", "recursive", "iterative", "divide", "stdacc"]

for alg in speedup_algs:
    plt.plot(
        df["data_kb"],
        df[f"{alg}_speedup_vs_naive"],
        marker="o",
        linewidth=1.2,
        markersize=3.0,
        label=label_map[alg],
        color=color_map[alg]
    )

plt.axhline(y=1.0, color="black", linestyle="--", linewidth=1.2, alpha=0.8)

for label, xpos in cache_points_kb.items():
    plt.axvline(x=xpos, color="#D32F2F", linestyle=":", linewidth=1.5, alpha=0.75)
    plt.text(
        xpos,
        plt.ylim()[1] * 0.95,
        label,
        color="#D32F2F",
        fontsize=11,
        rotation=90,
        ha="center",
        va="top"
    )

plt.xscale("log")
plt.xlabel("数据规模（KB）", fontsize=14)
plt.ylabel("相对普通求和的加速比", fontsize=14)
plt.title("各算法相对普通求和的加速收益", fontsize=16)
plt.grid(True, which="both", alpha=0.3)
plt.legend(title="算法名称", ncol=3, fontsize=10, title_fontsize=11)
plt.tight_layout()
plt.savefig("图2_相对加速比.png", dpi=300, bbox_inches="tight")
plt.show()

# ======================================
# 6. 图3：稳定性图（avg/min）
# 比值越接近 1，说明抖动越小、稳定性越好
# ======================================
plt.figure(figsize=(13, 8))

for alg in algorithms:
    stability = df[f"{alg}_avg_ms"] / df[f"{alg}_min_ms"]
    plt.plot(
        df["data_kb"],
        stability,
        marker="o",
        linewidth=1.2,
        markersize=3.0,
        label=label_map[alg],
        color=color_map[alg]
    )

for label, xpos in cache_points_kb.items():
    plt.axvline(x=xpos, color="#D32F2F", linestyle=":", linewidth=1.5, alpha=0.75)
    plt.text(
        xpos,
        plt.ylim()[1] * 0.98,
        label,
        color="#D32F2F",
        fontsize=11,
        rotation=90,
        ha="center",
        va="top"
    )

plt.xscale("log")
plt.xlabel("数据规模（KB）", fontsize=14)
plt.ylabel("平均时间 / 最小时间", fontsize=14)
plt.title("各算法运行稳定性对比", fontsize=16)
plt.grid(True, which="both", alpha=0.3)
plt.legend(title="算法名称", ncol=3, fontsize=10, title_fontsize=11)
plt.tight_layout()
plt.savefig("图3_运行稳定性对比.png", dpi=300, bbox_inches="tight")
plt.show()

# ======================================
# 7. 图4：缓存区间热力图
# 行：算法
# 列：L1 / L2 / L3 / 内存
# 值：每元素平均时间
# ======================================
def classify_regime(data_kb):
    if data_kb <= 32:
        return "L1内"
    elif data_kb <= 256:
        return "L2内"
    elif data_kb <= 8192:
        return "L3内"
    else:
        return "主存区间"

df["cache_regime"] = df["data_kb"].apply(classify_regime)

regimes = ["L1内", "L2内", "L3内", "主存区间"]

heatmap_data = []
for alg in algorithms:
    row = []
    for regime in regimes:
        sub = df[df["cache_regime"] == regime]
        value = sub[f"{alg}_time_per_elem_ns"].mean()
        row.append(value)
    heatmap_data.append(row)

heatmap_data = np.array(heatmap_data)

plt.figure(figsize=(10, 6))
im = plt.imshow(heatmap_data, aspect="auto")

plt.xticks(range(len(regimes)), regimes, fontsize=12)
plt.yticks(range(len(algorithms)), [label_map[a] for a in algorithms], fontsize=12)
plt.title("各算法在不同缓存区间的平均每元素时间", fontsize=15)

# 在格子内标数值
for i in range(heatmap_data.shape[0]):
    for j in range(heatmap_data.shape[1]):
        plt.text(
            j, i,
            f"{heatmap_data[i, j]:.3f}",
            ha="center",
            va="center",
            fontsize=10,
            color="black"
        )

cbar = plt.colorbar(im)
cbar.set_label("平均每元素时间（ns/elem）", fontsize=12)

plt.tight_layout()
plt.savefig("图4_缓存区间热力图.png", dpi=300, bbox_inches="tight")
plt.show()