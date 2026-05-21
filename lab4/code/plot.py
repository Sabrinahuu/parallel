import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'Microsoft YaHei'

labels = [
    '串行',
    'pthread\n(无锁)',
    'OpenMP\n(guided)',
    'Priority\nQueue(概率降序)',
    'MultiQueue',
    '最大堆'
]

times = [8.02, 7.42, 7.253, 5.914, 0.429, 0.500]

colors = [
    '#B0BEC5',
    '#90CAF9',
    '#64B5F6',
    '#42A5F5',
    '#1565C0',
    '#1976D2',
]

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(labels, times, color=colors, width=0.55, edgecolor='white', linewidth=1.2)

for bar, t in zip(bars, times):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.08,
        f'{t:.3f}s',
        ha='center', va='bottom',
        fontsize=11, fontweight='bold', color='#333333'
    )

ax.set_ylabel('Guess Time (seconds)', fontsize=12)
ax.set_title('Guess Time 对比', fontsize=13, fontweight='bold', pad=15)
ax.set_ylim(0, 9.5)
ax.yaxis.grid(True, linestyle='--', alpha=0.5)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 加速比标注
baseline = times[0]
for bar, t in zip(bars[1:], times[1:]):
    speedup = baseline / t
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        0.15,
        f'×{speedup:.1f}',
        ha='center', va='bottom',
        fontsize=9, color='white', fontweight='bold'
    )

plt.tight_layout()
plt.savefig(r'E:\并行\lab4\guess_time_chart_Chinese.png', dpi=150, bbox_inches='tight')
print("Saved.")