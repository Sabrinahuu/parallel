import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

processes = [1, 2, 4, 8]

# 原始时间数据
time_node1 = [1.233, 1.238, 0.831, 0.803]
time_node2 = [1.108, 0.978, 1.330, 1.566]

# 计算加速比（以各自1进程为基准）
speedup_node1 = [time_node1[0] / t for t in time_node1]
speedup_node2 = [time_node2[0] / t for t in time_node2]
speedup_ideal = [1, 2, 4, 8]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('MPI 并行化结果分析', fontsize=14, fontweight='bold', y=1.02)

# ── 左图：加速比 ──────────────────────────────────────────
ax1 = axes[0]
ax1.plot(processes, speedup_ideal, 'k--', linewidth=1.2, label='理想加速比', zorder=1)
ax1.plot(processes, speedup_node1, 'o-', color='#2878B5', linewidth=2,
         markersize=8, label='nodes=1', zorder=3)
ax1.plot(processes, speedup_node2, 's--', color='#C82423', linewidth=2,
         markersize=8, label='nodes=2', zorder=3)

# 标注数值
for i, (p, s1, s2) in enumerate(zip(processes, speedup_node1, speedup_node2)):
    ax1.annotate(f'{s1:.2f}×', (p, s1), textcoords='offset points',
                 xytext=(8, 4), fontsize=9, color='#2878B5')
    ax1.annotate(f'{s2:.2f}×', (p, s2), textcoords='offset points',
                 xytext=(8, -14), fontsize=9, color='#C82423')

ax1.set_xlabel('线程数', fontsize=11)
ax1.set_ylabel('加速比(×)', fontsize=11)
ax1.set_title('加速比 vs. 线程数', fontsize=12)
ax1.set_xticks(processes)
ax1.set_xticklabels(['1\n(串行)', '2', '4', '8'])
ax1.set_ylim(0, 9.5)
ax1.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
ax1.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.0f×'))
ax1.legend(fontsize=10, loc='upper left')
ax1.grid(True, alpha=0.3, linestyle=':')
ax1.axhline(y=1, color='gray', linewidth=0.8, alpha=0.5)

# 负加速区域标注
ax1.fill_between(processes[1:], speedup_node2[1:], 1,
                 where=[s < 1 for s in speedup_node2[1:]],
                 alpha=0.12, color='#C82423', label='_nolegend_')
ax1.annotate('negative\nspeedup', xy=(6, 0.77), fontsize=8,
             color='#C82423', ha='center')

# ── 右图：原始时间对比 ────────────────────────────────────
ax2 = axes[1]
x = np.arange(len(processes))
width = 0.35

bars1 = ax2.bar(x - width/2, time_node1, width, label='nodes=1',
                color='#2878B5', alpha=0.85, edgecolor='white', linewidth=0.5)
bars2 = ax2.bar(x + width/2, time_node2, width, label='nodes=2',
                color='#C82423', alpha=0.85, edgecolor='white', linewidth=0.5)

# 标注数值
for bar in bars1:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, h + 0.02,
             f'{h:.3f}s', ha='center', va='bottom', fontsize=8.5, color='#2878B5')
for bar in bars2:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, h + 0.02,
             f'{h:.3f}s', ha='center', va='bottom', fontsize=8.5, color='#C82423')

ax2.set_xlabel('线程数', fontsize=11)
ax2.set_ylabel('Guess Time (s)', fontsize=11)
ax2.set_title('Guess Time vs. 线程数', fontsize=12)
ax2.set_xticks(x)
ax2.set_xticklabels(['1\n(串行)', '2', '4', '8'])
ax2.set_ylim(0, 2.0)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, linestyle=':', axis='y')

plt.tight_layout()
plt.savefig(r'E:\并行\lab5\code\mpi_speedup_Chinese.png', dpi=150,
            bbox_inches='tight', facecolor='white')
print("saved: mpi_speedup_Chinese.png")