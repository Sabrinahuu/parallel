# -*- coding: utf-8 -*-
"""
优化前后不同进程数下 Guess Time / Hash Time 扩展性对比图
"""
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# ------------------ 中文字体配置 ------------------
font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
zh_font = fm.FontProperties(fname=font_path)
plt.rcParams["axes.unicode_minus"] = False

# ------------------ 数据 ------------------
nprocs = [1, 2, 4, 8]

guess_old = [1.32483, 0.904204, 0.82374, 0.97684]
guess_new = [0.480152, 0.375222, 0.305828, 0.221733]

hash_old = [2.85679, 2.57162, 2.59221, 2.79857]
hash_new = [1.12489, 0.566839, 0.28587, 0.142149]

# ------------------ 配色 ------------------
color_old = "#4C72B0"   # 蓝色：优化前
color_new = "#C44E52"   # 红色：优化后
color_ideal = "gray"

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# ================= 子图1：Guess Time =================
ax = axes[0]
ax.plot(nprocs, guess_old, "o-", color=color_old, linewidth=2, markersize=7,
        label="优化前（集中式 Gatherv）")
ax.plot(nprocs, guess_new, "s-", color=color_new, linewidth=2, markersize=7,
        label="优化后（本地生成+本地哈希）")

# 标注数值
for x, y in zip(nprocs, guess_old):
    ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=9, color=color_old)
for x, y in zip(nprocs, guess_new):
    ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                xytext=(0, -14), ha="center", fontsize=9, color=color_new)

ax.set_xlabel("进程数", fontproperties=zh_font, fontsize=12)
ax.set_ylabel("Guess Time (s)", fontsize=12)
ax.set_title("Guess Time 随进程数变化", fontproperties=zh_font, fontsize=13)
ax.set_xticks(nprocs)
ax.set_xscale("log", base=2)
ax.set_xticklabels(nprocs)
ax.grid(alpha=0.3, linestyle="--")
ax.legend(prop=zh_font, fontsize=10, loc="upper right")

# ================= 子图2：Hash Time =================
ax = axes[1]
ax.plot(nprocs, hash_old, "o-", color=color_old, linewidth=2, markersize=7,
        label="优化前（rank0集中哈希）")
ax.plot(nprocs, hash_new, "s-", color=color_new, linewidth=2, markersize=7,
        label="优化后（各进程本地NEON哈希）")

for x, y in zip(nprocs, hash_old):
    ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=9, color=color_old)
for x, y in zip(nprocs, hash_new):
    ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                xytext=(0, -14), ha="center", fontsize=9, color=color_new)

ax.set_xlabel("进程数", fontproperties=zh_font, fontsize=12)
ax.set_ylabel("Hash Time (s)", fontsize=12)
ax.set_title("Hash Time 随进程数变化", fontproperties=zh_font, fontsize=13)
ax.set_xticks(nprocs)
ax.set_xscale("log", base=2)
ax.set_xticklabels(nprocs)
ax.grid(alpha=0.3, linestyle="--")
ax.legend(prop=zh_font, fontsize=10, loc="upper right")

fig.suptitle("新旧架构在不同进程数下的扩展性对比", fontproperties=zh_font, fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/scalability_comparison.png", dpi=200, bbox_inches="tight")
print("图1已保存")

# ================= 图2：加速比柱状图 =================
fig2, ax = plt.subplots(figsize=(8, 5.5))

speedup_guess = [o / n for o, n in zip(guess_old, guess_new)]
speedup_hash = [o / n for o, n in zip(hash_old, hash_new)]

x = np.arange(len(nprocs))
width = 0.35

bars1 = ax.bar(x - width/2, speedup_guess, width, label="Guess Time 加速比", color="#55A868")
bars2 = ax.bar(x + width/2, speedup_hash, width, label="Hash Time 加速比", color="#8172B2")

for b in bars1:
    h = b.get_height()
    ax.annotate(f"{h:.2f}x", (b.get_x() + b.get_width()/2, h),
                textcoords="offset points", xytext=(0, 3), ha="center", fontsize=10)
for b in bars2:
    h = b.get_height()
    ax.annotate(f"{h:.2f}x", (b.get_x() + b.get_width()/2, h),
                textcoords="offset points", xytext=(0, 3), ha="center", fontsize=10)

ax.set_xlabel("进程数", fontproperties=zh_font, fontsize=12)
ax.set_ylabel("加速比（优化前/优化后）", fontproperties=zh_font, fontsize=12)
ax.set_title("不同进程数下优化带来的加速比", fontproperties=zh_font, fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(nprocs)
ax.legend(prop=zh_font, fontsize=10)
ax.grid(alpha=0.3, linestyle="--", axis="y")

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/speedup_barchart.png", dpi=200, bbox_inches="tight")
print("图2已保存")