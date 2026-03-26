import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==============================
# 中文显示设置
# ==============================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==============================
# 莫兰迪柔和色系
# ==============================
morrandi_colors = {
    '朴素累加': '#A1887F',    # 褐灰
    '两路累加': '#90A4AE',    # 蓝灰
    '四路累加': '#7986CB',    # 蓝紫
    '八路累加': '#FF8A65',    # 橙
    '十六路累加': '#4DB6AC',  # 青绿
    '递归分治': '#BA68C8',    # 紫
    '循环分治': '#9CCC65',    # 黄绿
    '分治算法': '#E57373',    # 柔红
    '标准库': '#B0BEC5'       # 灰蓝
}

# ==============================
# 读取 CSV
# ==============================
csv_file = "sum_all_compare_results.csv"
df = pd.read_csv(csv_file)

# 过滤无效值
time_columns = [
    'naive_avg_ms', 'way2_avg_ms', 'way4_avg_ms', 'way8_avg_ms',
    'way16_avg_ms', 'recursive_avg_ms', 'iterative_avg_ms',
    'divide_avg_ms', 'stdacc_avg_ms'
]
for col in time_columns:
    df = df[df[col] > 0]

# 按 n 排序
df = df.sort_values(by='n').reset_index(drop=True)

# ==============================
# 你的算法列映射
# ==============================
time_plot_columns = {
    '朴素累加': 'naive_avg_ms',
    '两路累加': 'way2_avg_ms',
    '四路累加': 'way4_avg_ms',
    '八路累加': 'way8_avg_ms',
    '十六路累加': 'way16_avg_ms',
    '递归分治': 'recursive_avg_ms',
    '循环分治': 'iterative_avg_ms',
    '分治算法': 'divide_avg_ms',
    '标准库': 'stdacc_avg_ms'
}

speedup_plot_columns = {
    '两路累加': 'way2_speedup_vs_naive',
    '四路累加': 'way4_speedup_vs_naive',
    '八路累加': 'way8_speedup_vs_naive',
    '十六路累加': 'way16_speedup_vs_naive',
    '递归分治': 'recursive_speedup_vs_naive',
    '循环分治': 'iterative_speedup_vs_naive',
    '分治算法': 'divide_speedup_vs_naive',
    '标准库': 'stdacc_speedup_vs_naive'
}

# ==============================
# cache 边界（按 int 4B 粗略对应）
# L1 ~ 32KB  -> 8192
# L2 ~ 256KB -> 65536
# L3 ~ 8MB   -> 2097152
# ==============================
cache_sizes = [
    (8192, 'L1 缓存边界'),
    (65536, 'L2 缓存边界'),
    (2097152, 'L3 缓存边界')
]

# ==============================
# 通用样式函数
# ==============================
def setup_common_style(fig, ax):
    fig.patch.set_facecolor('#F5F5F5')
    ax.set_facecolor('#F5F5F5')
    ax.grid(True, which='both', alpha=0.3, linestyle='--', color='#CCCCCC')
    return ax

def add_cache_lines(ax, ymin=None, ymax=None):
    for size, label in cache_sizes:
        ax.axvline(x=size, color='#D32F2F', linestyle=':', alpha=0.7, linewidth=1.5)
        if ymin is not None and ymax is not None:
            text_y = ymax / 1.6 if ymax > 0 else 1
            ax.text(size * 1.03, text_y, label, rotation=90,
                    color='#D32F2F', alpha=0.9, fontsize=9)

def add_watermark(fig, text):
    fig.text(0.99, 0.01, text,
             color='#888888', fontsize=10,
             ha='right', va='bottom', alpha=0.7)

# ==============================
# 对数分箱平滑
# ==============================
def log_bin_median(df_input, x_col, y_col, bins=100):
    x = df_input[x_col].values
    y = df_input[y_col].values

    x = np.array(x)
    y = np.array(y)

    valid = (x > 0) & (y > 0)
    x = x[valid]
    y = y[valid]

    if len(x) == 0:
        return pd.DataFrame(columns=[x_col, y_col])

    bin_edges = np.logspace(np.log10(x.min()), np.log10(x.max()), bins)
    bin_ids = np.digitize(x, bin_edges)

    result_x = []
    result_y = []

    for b in range(1, len(bin_edges)):
        mask = bin_ids == b
        if np.any(mask):
            result_x.append(np.median(x[mask]))
            result_y.append(np.median(y[mask]))

    return pd.DataFrame({x_col: result_x, y_col: result_y})

# ==============================
# 图 1：对数坐标执行时间比较
# ==============================
fig1, ax1 = plt.subplots(figsize=(12, 8), dpi=120)
ax1 = setup_common_style(fig1, ax1)

ax1.set_title('不同数组大小下的算法执行时间', fontsize=16, fontweight='bold', color='#333333')
ax1.set_xlabel('数组大小 n', fontsize=13, color='#333333')
ax1.set_ylabel('执行时间 (ms)', fontsize=13, color='#333333')

for label, col in time_plot_columns.items():
    smooth_df = log_bin_median(df, 'n', col, bins=100)
    ax1.plot(
        smooth_df['n'],
        smooth_df[col],
        color=morrandi_colors[label],
        linewidth=2.2,
        marker='o',
        markersize=3,
        label=label
    )

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_ylim(1e-5, 1e2)

add_cache_lines(ax1, ymin=1e-5, ymax=1e2)
ax1.legend(loc='upper left', fontsize=10, frameon=True)
add_watermark(fig1, '向量求和算法执行时间分析（对数坐标）')

plt.tight_layout()
plt.savefig('sum_execution_time_log.png', dpi=300, bbox_inches='tight')
plt.show()

# ==============================
# 图 2：加速比对比
# ==============================
fig2, ax2 = plt.subplots(figsize=(12, 8), dpi=120)
ax2 = setup_common_style(fig2, ax2)

ax2.set_title('各优化算法相对于朴素累加的加速比', fontsize=16, fontweight='bold', color='#333333')
ax2.set_xlabel('数组大小 n', fontsize=13, color='#333333')
ax2.set_ylabel('加速比', fontsize=13, color='#333333')

for label, col in speedup_plot_columns.items():
    smooth_df = log_bin_median(df, 'n', col, bins=100)
    ax2.plot(
        smooth_df['n'],
        smooth_df[col],
        color=morrandi_colors[label],
        linewidth=2.2,
        marker='o',
        markersize=3,
        label=label
    )

ax2.axhline(y=1.0, color='#999999', linestyle='--', linewidth=1.5)
ax2.set_xscale('log')
ax2.set_ylim(0, 8)

add_cache_lines(ax2, ymin=0, ymax=8)
ax2.legend(loc='best', fontsize=10, frameon=True)
add_watermark(fig2, '向量求和算法加速比分析')

plt.tight_layout()
plt.savefig('sum_speedup_ratios.png', dpi=300, bbox_inches='tight')
plt.show()

# ==============================
# 图 3：小规模数据性能比较
# 这里取 n <= 10000
# ==============================
fig3, ax3 = plt.subplots(figsize=(12, 8), dpi=120)
ax3 = setup_common_style(fig3, ax3)

ax3.set_title('小规模数据下的算法性能对比（n ≤ 10000）', fontsize=16, fontweight='bold', color='#333333')
ax3.set_xlabel('数组大小 n', fontsize=13, color='#333333')
ax3.set_ylabel('执行时间 (ms)', fontsize=13, color='#333333')

small_df = df[df['n'] <= 10000]

small_time_columns = {
    '朴素累加': 'naive_avg_ms',
    '两路累加': 'way2_avg_ms',
    '四路累加': 'way4_avg_ms',
    '八路累加': 'way8_avg_ms',
    '十六路累加': 'way16_avg_ms',
    '递归分治': 'recursive_avg_ms',
    '循环分治': 'iterative_avg_ms',
    '分治算法': 'divide_avg_ms'
}

for label, col in small_time_columns.items():
    ax3.plot(
        small_df['n'],
        small_df[col],
        color=morrandi_colors[label],
        linewidth=2.0,
        marker='o',
        markersize=4,
        label=label
    )

# 小规模图只显示 L1
ax3.axvline(x=8192, color='#D32F2F', linestyle=':', alpha=0.7, linewidth=1.5)
ax3.text(8192 * 1.02, ax3.get_ylim()[1] * 0.85, 'L1 缓存边界',
         rotation=90, color='#D32F2F', alpha=0.9, fontsize=9)

ax3.legend(loc='upper left', fontsize=10, frameon=True)
add_watermark(fig3, '小规模数据算法性能对比')

plt.tight_layout()
plt.savefig('sum_small_dataset_performance.png', dpi=300, bbox_inches='tight')
plt.show()