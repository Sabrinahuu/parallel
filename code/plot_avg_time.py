import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams

# =========================
# 1. 自动寻找可用中文字体
# =========================
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
        "请先安装中文字体，例如：\n"
        "Ubuntu / WSL: sudo apt install fonts-noto-cjk\n"
        "安装后重新运行脚本。"
    )

print(f"当前使用中文字体: {chosen_font}")

rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = [chosen_font]
rcParams["axes.unicode_minus"] = False

# =========================
# 2. 读取数据
# =========================
df = pd.read_csv("sum_all_compare_results.csv")
x_col = "n"

avg_cols = [
    "naive_avg_ms",
    "way2_avg_ms",
    "way4_avg_ms",
    "way8_avg_ms",
    "way16_avg_ms",
    "divide_avg_ms",
    "stdacc_avg_ms",
    "recursive_avg_ms",
    "iterative_avg_ms",
]

# =========================
# 3. 中文图例名称
# =========================
label_map = {
    "naive_avg_ms": "普通求和",
    "way2_avg_ms": "2路展开",
    "way4_avg_ms": "4路展开",
    "way8_avg_ms": "8路展开",
    "way16_avg_ms": "16路展开",
    "recursive_avg_ms": "递归求和",
    "iterative_avg_ms": "迭代求和",
    "divide_avg_ms": "分治求和",
    "stdacc_avg_ms": "标准库求和",
}

color_map = {
    # -------- 快算法组：冷色 / 中性色 --------
    "naive_avg_ms": "#5E6C84",      # 灰蓝
    "way2_avg_ms": "#4C78A8",       # 蓝
    "way4_avg_ms": "#72B7B2",       # 青
    "way8_avg_ms": "#54A24B",       # 绿
    "way16_avg_ms": "#9D755D",      # 棕
    "divide_avg_ms": "#B279A2",     # 柔紫
    "stdacc_avg_ms": "#BAB0AC",     # 浅灰

    # -------- 慢算法组：高对比强调色 --------
    "recursive_avg_ms": "#E45756",  # 红
    "iterative_avg_ms": "#F58518",  # 橙
}

# 按 n 排序
df = df.sort_values(x_col)

# 计算单次执行时间
for col in avg_cols:
    df[col + "_single"] = df[col] / df["inner_loop"]

# =========================
# 4. 画图
# =========================
plt.figure(figsize=(13, 8))

slow_cols = ["recursive_avg_ms", "iterative_avg_ms"]

for col in avg_cols:
    is_slow = col in slow_cols

    plt.plot(
        df[x_col],
        df[col + "_single"],
        marker="o",
        linewidth=2.2 if is_slow else 1.4,
        markersize=3.6 if is_slow else 2.6,
        label=label_map[col],
        color=color_map[col],
        alpha=0.95 if is_slow else 0.9,
        zorder=5 if is_slow else 3
    )

plt.xscale("log")
plt.yscale("log")

plt.xlim(df[x_col].min(), df[x_col].max())

y_min = df[[col + "_single" for col in avg_cols]].min().min()
y_max = df[[col + "_single" for col in avg_cols]].max().max()
plt.ylim(y_min * 0.8, y_max * 1.2)

# 中文标题和坐标轴
plt.xlabel("数据规模 n", fontsize=14)
plt.ylabel("单次运行时间（毫秒）", fontsize=14)
plt.title("各算法单次运行时间对比（双对数坐标）", fontsize=16)

# =========================
# 5. 添加 L1 / L2 / L3 缓存标记
# =========================
cache_points = {
    "L1缓存": 8 * 1024,         # 32KB / 4B = 8192
    "L2缓存": 64 * 1024,        # 256KB / 4B = 65536
    "L3缓存": 2 * 1024 * 1024,  # 8MB / 4B = 2097152
}

for label, xpos in cache_points.items():
    plt.axvline(
        x=xpos,
        color="#D32F2F",
        linestyle=":",
        linewidth=1.8,
        alpha=0.8
    )
    plt.text(
        xpos,
        y_max * 1.05,
        label,
        color="#D32F2F",
        fontsize=12,
        rotation=90,
        ha="center",
        va="bottom"
    )

plt.grid(True, which="both", alpha=0.3)
plt.legend(title="算法名称", ncol=3, fontsize=10, title_fontsize=11)

plt.tight_layout()
plt.savefig("avg_runtime_all_log.png", dpi=300, bbox_inches="tight")




# =========================
# 7. 高性能算法：相对普通求和的加速比图
# speedup = naive_time / alg_time
# 大于 1 表示比普通求和更快
# =========================
speedup_cols = [
    "way2_avg_ms",
    "way4_avg_ms",
    "way8_avg_ms",
    "way16_avg_ms",
    "divide_avg_ms",
    "stdacc_avg_ms",
]

plt.figure(figsize=(13, 8))

for col in speedup_cols:
    speedup = df["naive_avg_ms"] / df[col]

    plt.plot(
        df[x_col],
        speedup,
        marker="o",
        linewidth=1.8,
        markersize=3.0,
        label=label_map[col],
        color=color_map[col],
        alpha=0.95
    )

plt.xscale("log")
plt.xlim(df[x_col].min(), df[x_col].max())

speedup_y_min = min((df["naive_avg_ms"] / df[col]).min() for col in speedup_cols)
speedup_y_max = max((df["naive_avg_ms"] / df[col]).max() for col in speedup_cols)
plt.ylim(0.75,1.8)

plt.axhline(
    y=1.0,
    color="black",
    linestyle="--",
    linewidth=1.4,
    alpha=0.8
)

for label, xpos in cache_points.items():
    plt.axvline(
        x=xpos,
        color="#D32F2F",
        linestyle=":",
        linewidth=1.6,
        alpha=0.75
    )
    plt.text(
        xpos,
        speedup_y_max * 1.01,
        label,
        color="#D32F2F",
        fontsize=12,
        rotation=90,
        ha="center",
        va="bottom"
    )

plt.xlabel("数据规模 n", fontsize=14)
plt.ylabel("相对普通求和的加速比", fontsize=14)
plt.title("高性能算法相对普通求和的加速比对比", fontsize=16)

plt.grid(True, which="both", alpha=0.3)
plt.legend(title="高性能算法", ncol=3, fontsize=10, title_fontsize=11)
plt.tight_layout()
plt.savefig("fast_speedup_vs_naive.png", dpi=300, bbox_inches="tight")
plt.show()



# =========================
# 8. 高性能算法：缓存区间平均加速比热力图
# 行：算法
# 列：L1 / L2 / L3 / 主存
# 值：相对普通求和的平均加速比
# =========================
import numpy as np

fast_cols = [
    "naive_avg_ms",
    "way2_avg_ms",
    "way4_avg_ms",
    "way8_avg_ms",
    "way16_avg_ms",
    "divide_avg_ms",
    "stdacc_avg_ms",
]

# 按缓存区间分类
def classify_regime(n):
    if n <= 8 * 1024:                  # L1: 32KB / 4B
        return "L1内"
    elif n <= 64 * 1024:               # L2: 256KB / 4B
        return "L2内"
    elif n <= 2 * 1024 * 1024:         # L3: 8MB / 4B
        return "L3内"
    else:
        return "主存区间"

df["cache_regime"] = df[x_col].apply(classify_regime)

regimes = ["L1内", "L2内", "L3内", "主存区间"]

# 构造热力图数据：值为相对普通求和的平均加速比
heatmap_data = []
heatmap_labels = []

for col in fast_cols:
    row = []
    for regime in regimes:
        sub = df[df["cache_regime"] == regime]

        if col == "naive_avg_ms":
            value = 1.0
        else:
            speedup = (sub["naive_avg_ms"] / sub[col]).mean()
            value = speedup

        row.append(value)

    heatmap_data.append(row)
    heatmap_labels.append(label_map[col])

heatmap_data = np.array(heatmap_data)

plt.figure(figsize=(10, 6))
im = plt.imshow(heatmap_data, aspect="auto", cmap="YlOrRd")

plt.xticks(range(len(regimes)), regimes, fontsize=12)
plt.yticks(range(len(heatmap_labels)), heatmap_labels, fontsize=12)
plt.title("高性能算法在不同缓存区间的平均加速比热力图", fontsize=15)

# 在格子中写数值
for i in range(heatmap_data.shape[0]):
    for j in range(heatmap_data.shape[1]):
        plt.text(
            j, i,
            f"{heatmap_data[i, j]:.2f}",
            ha="center",
            va="center",
            fontsize=10,
            color="black"
        )

cbar = plt.colorbar(im)
cbar.set_label("相对普通求和的平均加速比", fontsize=12)

plt.tight_layout()
plt.savefig("fast_algorithms_speedup_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()