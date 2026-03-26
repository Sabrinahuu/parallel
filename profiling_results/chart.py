import pandas as pd
import numpy as np

# 读取长表
df = pd.read_csv("sum_all_compare_float_error_dense_long.csv")

# 只保留关心的数据集和算法
datasets = ["large_small", "cancellation"]
algorithms = ["naive", "way8", "recursive", "stdacc", "kahan"]

df = df[df["dataset"].isin(datasets) & df["algorithm"].isin(algorithms)].copy()

# 目标规模点
target_ns = [1000, 6000000]

# 若某个 n 不一定精确存在，就取最接近的
selected_rows = []
for dataset in datasets:
    sub = df[df["dataset"] == dataset]
    available_ns = np.array(sorted(sub["n"].unique()))
    for target_n in target_ns:
        nearest_n = available_ns[np.argmin(np.abs(available_ns - target_n))]
        chosen = sub[sub["n"] == nearest_n].copy()
        chosen["target_n"] = target_n
        chosen["selected_n"] = nearest_n
        selected_rows.append(chosen)

sel = pd.concat(selected_rows, ignore_index=True)

# 转成宽表
table_df = sel.pivot_table(
    index=["dataset", "selected_n"],
    columns="algorithm",
    values="rel_error",
    aggfunc="first"
).reset_index()

# 排序
dataset_order = {"large_small": 0, "cancellation": 1}
table_df["dataset_order"] = table_df["dataset"].map(dataset_order)
table_df = table_df.sort_values(["dataset_order", "selected_n"]).drop(columns="dataset_order")

# 中文名称
dataset_name = {
    "large_small": "大数-小数混合",
    "cancellation": "抵消型数据"
}
algo_name = {
    "naive": "朴素累加",
    "way8": "8路链式累加",
    "recursive": "递归二叉归约",
    "stdacc": "标准库累加",
    "kahan": "Kahan补偿求和"
}

# 科学计数法格式化
def fmt(x):
    if pd.isna(x):
        return "--"
    return f"{x:.2e}"

# 生成 LaTeX
lines = []
lines.append(r"\begin{table}[htbp]")
lines.append(r"\centering")
lines.append(r"\caption{代表规模下不同算法的相对误差对比}")
lines.append(r"\label{tab:rel_error_representative}")
lines.append(r"\scriptsize")
lines.append(r"\setlength{\tabcolsep}{4pt}")
lines.append(r"\begin{tabular}{llccccc}")
lines.append(r"\hline")
lines.append(r"数据集 & $n$ & 朴素累加 & 8路链式累加 & 递归二叉归约 & 标准库累加 & Kahan补偿求和 \\")
lines.append(r"\hline")

for _, row in table_df.iterrows():
    ds = dataset_name[row["dataset"]]
    n = int(row["selected_n"])
    naive = fmt(row.get("naive"))
    way8 = fmt(row.get("way8"))
    recursive = fmt(row.get("recursive"))
    stdacc = fmt(row.get("stdacc"))
    kahan = fmt(row.get("kahan"))

    lines.append(
        f"{ds} & {n} & {naive} & {way8} & {recursive} & {stdacc} & {kahan} \\\\"
    )

lines.append(r"\hline")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

latex_code = "\n".join(lines)
print(latex_code)