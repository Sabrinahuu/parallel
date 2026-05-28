import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

cases = ['纯流水线\n(MPI 1:5)', 'MPI+OpenMP\n(Hash)', 'MPI+OpenMP\n+生成优化']

guess_time = [0.78951, 0.91267, 0.95029]
hash_time  = [0.617735, 0.318329, 0.415439]

x = np.arange(len(cases))

fig, ax1 = plt.subplots(figsize=(7, 5))

fig.suptitle('三种并行方法的性能对比',
             fontsize=14, fontweight='bold')

# Guess Time vs Hash Time 折线+填充
ax1.plot(x, guess_time, 'o-', color='#2878B5', linewidth=2.5,
         markersize=9, label='Guess Time', zorder=3)
ax1.plot(x, hash_time,  's-', color='#C82423', linewidth=2.5,
         markersize=9, label='Max Hash Time', zorder=3)

# 填充两条线之间的区域，面积越小说明流水线越均衡
ax1.fill_between(x, guess_time, hash_time,
                 alpha=0.12, color='gray', label='差距')

# 标注数值
for i, (g, h) in enumerate(zip(guess_time, hash_time)):
    ax1.annotate(f'{g:.3f}', (x[i], g), textcoords='offset points',
                 xytext=(0, 10), ha='center', fontsize=9, color='#2878B5')
    ax1.annotate(f'{h:.3f}', (x[i], h), textcoords='offset points',
                 xytext=(0, -18), ha='center', fontsize=9, color='#C82423')

# 标注差值
for i, (g, h) in enumerate(zip(guess_time, hash_time)):
    mid = (g + h) / 2
    diff = abs(g - h)
    ax1.annotate(f'gap={diff:.3f}s', (x[i], mid),
                 textcoords='offset points', xytext=(18, 0),
                 fontsize=8, color='gray',
                 arrowprops=dict(arrowstyle='-', color='gray',
                                 lw=0.8, linestyle='dashed'))

ax1.set_xticks(x)
ax1.set_xticklabels(cases, fontsize=9)
ax1.set_ylabel('Time (s)', fontsize=11)
ax1.set_title('Guess Time vs Hash Time\n(更小的差距对应更好的流水线平衡)',
              fontsize=11)
ax1.set_ylim(0, 1.25)
ax1.legend(fontsize=10, loc='upper left')
ax1.grid(True, alpha=0.3, linestyle=':')

plt.tight_layout()
plt.savefig('pipeline_compare_left_only.png', dpi=150,
            bbox_inches='tight', facecolor='white')
print("saved: pipeline_compare_left_only.png")