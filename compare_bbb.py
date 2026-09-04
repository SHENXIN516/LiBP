import pandas as pd

# =========================
# 1. 文件路径
# =========================
TRUE_FILE = "true.csv"
PRED_FILE = "pred.csv"

OUTPUT_ERROR = "bbb_mispredicted.csv"

# =========================
# 2. 读取数据
# =========================
true_df = pd.read_csv(TRUE_FILE)
pred_df = pd.read_csv(PRED_FILE)

# =========================
# 3. 标签映射
# =========================
label_map = {
    1: "BBB+",
    0: "BBB-"
}
true_df["True_BBB"] = true_df["label"].map(label_map)

# =========================
# 4. 合并（按 SMILES）
# =========================
merged_df = pd.merge(
    true_df,
    pred_df,
    on="SMILES",
    how="inner"
)

# =========================
# 5. 仅保留预测错误
# =========================
error_df = merged_df[merged_df["True_BBB"] != merged_df["Prediction"]].copy()

# =========================
# 6. 按预测结果拆列
# =========================
error_df["Pred_BBB_Plus"] = error_df["Prediction"].apply(
    lambda x: "BBB+" if x == "BBB+" else ""
)
error_df["Pred_BBB_Minus"] = error_df["Prediction"].apply(
    lambda x: "BBB-" if x == "BBB-" else ""
)

error_df["Pred_BBB_Plus_Prob"] = error_df.apply(
    lambda row: row["Prob_BBB_Plus"] if row["Prediction"] == "BBB+" else "",
    axis=1
)
error_df["Pred_BBB_Minus_Prob"] = error_df.apply(
    lambda row: row["Prob_BBB_Minus"] if row["Prediction"] == "BBB-" else "",
    axis=1
)

# =========================
# 7. 最终输出列（仅错误）
# =========================
final_cols = [
    "SMILES",
    "Name",
    "True_BBB",
    "Pred_BBB_Plus",
    "Pred_BBB_Plus_Prob",
    "Pred_BBB_Minus",
    "Pred_BBB_Minus_Prob",
    "Confidence"
]

final_error_df = error_df[final_cols]

# =========================
# 8. 导出
# =========================
final_error_df.to_csv(OUTPUT_ERROR, index=False)

# =========================
# 9. 控制台统计
# =========================
fp = ((error_df["True_BBB"] == "BBB-") & (error_df["Prediction"] == "BBB+")).sum()
fn = ((error_df["True_BBB"] == "BBB+") & (error_df["Prediction"] == "BBB-")).sum()
# =========================
# 8. 排序：预测为 BBB+ 的放前面
# =========================
final_error_df = final_error_df.sort_values(
    by="Pred_BBB_Plus",
    ascending=False
)
final_error_df.to_csv(OUTPUT_ERROR, index=False)

print("===== BBB Misprediction Summary =====")
print(f"Total mispredicted molecules : {len(final_error_df)}")
print(f"False Positive (BBB- → BBB+) : {fp}")
print(f"False Negative (BBB+ → BBB-) : {fn}")
print("Saved:")
print(f" - Mispredictions -> {OUTPUT_ERROR}")
print("====================================")
