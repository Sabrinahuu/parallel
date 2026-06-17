import matplotlib.pyplot as plt
import numpy as np

versions = ['串行\nbaseline', '初版GPU\n(单次调用)', '批量GPU\n(多PT)', '双缓冲\n流水线', '自适应\n调度']
guess_time = [7.48595, 10.3675, 12.3188, 10.4061, 9.91942]
hash_time  = [6.38797, 6.35678, 6.28694, 6.41672, 6.31594]
speedup    = [7.48595 / t for t in guess_time]  # 相对baseline的比值

colors = ['#2ecc71', '#e74c3c', '#e74c3c', '#e74c3c', '#e74c3c']
# baseline用绿色，GPU版本用红色系深浅区分
gpu_colors = ['#e74c3c', '#c0392b', '#e67e22', '#f39c12']
bar_colors = ['#2ecc71'] + gpu_colors

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('PCFG GPU并行化性能对比', fontsize=14, fontweight='bold', y=1.02)

# ── 图1：Guess time 对比 ──
ax1 = axes[0]
bars = ax1.bar(versions, guess_time, color=bar_colors, edgecolor='white', linewidth=0.8)
ax1.axhline(y=7.48595, color='#2ecc71', linestyle='--', linewidth=1.5, label='baseline基准线')
ax1.set_title('Guess Time 对比', fontsize=12, fontweight='bold')
ax1.set_ylabel('时间 (秒)', fontsize=10)
ax1.set_ylim(0, 14)
for bar, val in zip(bars, guess_time):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             f'{val:.2f}s', ha='center', va='bottom', fontsize=9)
ax1.legend(fontsize=9)
ax1.grid(axis='y', alpha=0.3)

# ── 图2：相对baseline的性能比（越接近1越好，>1表示加速）──
ax2 = axes[1]
bars2 = ax2.bar(versions, speedup, color=bar_colors, edgecolor='white', linewidth=0.8)
ax2.axhline(y=1.0, color='#2ecc71', linestyle='--', linewidth=1.5, label='baseline基准线(1.0x)')
ax2.set_title('相对Baseline性能比', fontsize=12, fontweight='bold')
ax2.set_ylabel('性能比 (baseline / 当前版本)', fontsize=10)
ax2.set_ylim(0, 1.3)
for bar, val in zip(bars2, speedup):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.2f}x', ha='center', va='bottom', fontsize=9)
ax2.legend(fontsize=9)
ax2.grid(axis='y', alpha=0.3)

# ── 图3：Guess time 与 Hash time 对比（说明GPU只影响了生成阶段）──
ax3 = axes[2]
x = np.arange(len(versions))
width = 0.35
bars3a = ax3.bar(x - width/2, guess_time, width, label='Guess time',
                 color=['#3498db' if i == 0 else '#85c1e9' for i in range(5)],
                 edgecolor='white')
bars3b = ax3.bar(x + width/2, hash_time, width, label='Hash time',
                 color=['#e67e22' if i == 0 else '#f0b27a' for i in range(5)],
                 edgecolor='white')
ax3.set_title('Guess Time vs Hash Time', fontsize=12, fontweight='bold')
ax3.set_ylabel('时间 (秒)', fontsize=10)
ax3.set_xticks(x)
ax3.set_xticklabels(versions, fontsize=8)
ax3.set_ylim(0, 15)
ax3.legend(fontsize=9)
ax3.grid(axis='y', alpha=0.3)
# 标注数值
for bar, val in zip(bars3a, guess_time):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{val:.1f}', ha='center', va='bottom', fontsize=7.5)
for bar, val in zip(bars3b, hash_time):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{val:.1f}', ha='center', va='bottom', fontsize=7.5)

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.tight_layout()
plt.savefig('gpu_performance.png', dpi=150, bbox_inches='tight')
plt.show()
print("图表已保存为 gpu_performance.png")