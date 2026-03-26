import matplotlib.pyplot as plt

# 中文方法名（去掉 bad）
functions_cn = [
    "分块优化",
    "分块展开4次+寄存器复用",
    "缓存优化",
    "外层展开2次",
    "循环展开4次",
    "循环展开4次+寄存器复用",
    "向量化优化",
]

cpu_time = [
    20.246,
    51.948,
    24.266,
    16.265,
    26.188,
    24.342,
    21.498,
]

instructions_retired = [
    36_095_740_000,
    102_147_470_000,
    50_621_490_000,
    70_065_030_000,
    92_425_700_000,
    92_689_260_000,
    52_208_840_000,
]

# 若本机支持中文，可正常显示；如果仍乱码，可改成你系统里已有的中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# ---------- CPU Time 柱状图 ----------
plt.figure(figsize=(12, 6))
plt.bar(functions_cn, cpu_time, color="#7FA9C9")
plt.xticks(rotation=30, ha="right")
plt.ylabel("CPU时间 (s)")
plt.title("不同优化方法的 CPU 时间对比")
plt.tight_layout()
plt.show()

# ---------- Instructions Retired 柱状图 ----------
plt.figure(figsize=(12, 6))
plt.bar(functions_cn, instructions_retired, color="#9BBE9B")
plt.xticks(rotation=30, ha="right")
plt.ylabel("IR")
plt.title("不同优化方法的IR对比")
plt.ticklabel_format(style="plain", axis="y")
plt.tight_layout()
plt.show()