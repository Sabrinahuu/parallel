import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据
ratios     = ['1:1', '1:2', '1:3', '1:4', '1:5', '1:6', '1:7']
guess_time = [2.95552, 1.51864, 1.19963, 1.13939, 0.78951, 0.810959, 0.831011]
hash_time  = [2.90563, 1.48493, 1.05306, 0.882177, 0.617735, 0.600768, 0.605991]

x = np.arange(len(ratios))
width = 0.35

fig, ax = plt.subplots(figsize=(11, 6))

bars1 = ax.bar(x - width/2, guess_time, width,
               label='Guess Time', color='#2878B5', alpha=0.88,
               edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x + width/2, hash_time, width,
               label='Max Hash Time', color='#C82423', alpha=0.88,
               edgecolor='white', linewidth=0.5)

# 标注数值
for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.02,
            f'{h:.3f}', ha='center', va='bottom', fontsize=8, color='#2878B5')

for bar in bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.02,
            f'{h:.3f}', ha='center', va='bottom', fontsize=8, color='#C82423')

# 标注最优点（1:5时两者最接近且guess time最小）
ax.annotate('最好的比例！\n(1:5)', xy=(4, 0.78951),
            xytext=(4.5, 1.4),
            fontsize=9, color='green',
            arrowprops=dict(arrowstyle='->', color='green', lw=1.5))

ax.set_xlabel('生成进程：哈希进程', fontsize=12)
ax.set_ylabel('Time (s)', fontsize=12)
ax.set_title('流水线: Guess Time vs Hash Time\n不同比例下',
             fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(ratios, fontsize=11)
ax.set_ylim(0, 3.6)
ax.legend(fontsize=11)
ax.grid(True, axis='y', alpha=0.3, linestyle=':')

# 添加说明文字
ax.text(0.01, 0.97,
        '当 Guess Time ≈ Hash Time → 最好的比例',
        transform=ax.transAxes, fontsize=9, color='gray',
        verticalalignment='top')

plt.tight_layout()
plt.savefig('pipeline_performance_Chinese.png', dpi=150,
            bbox_inches='tight', facecolor='white')
print("saved: pipeline_performance_Chinese.png")