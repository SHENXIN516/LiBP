# -*- coding: utf-8 -*-
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MolStandardize

# -----------------------------
# 配置输入输出文件
# -----------------------------
input_file = "peptide.xlsx"     # 替换为你的 Excel 文件路径
output_file = "peptide.csv"    # 输出 CSV 文件路径
smiles_col = "SMILES"         # Excel 中 SMILES 列名
label_col = "label"           # Excel 中标签列名

# -----------------------------
# 读取 Excel 文件
# -----------------------------
df = pd.read_excel(input_file)

# -----------------------------
# SMILES 标准化函数
# -----------------------------
def standardize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        # 去盐、去溶剂
        normalizer = MolStandardize.normalize.Normalizer()
        mol = normalizer.normalize(mol)
        # Canonical SMILES
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return None

# -----------------------------
# 标准化 SMILES
# -----------------------------
df["canonical_SMILES"] = df[smiles_col].apply(standardize_smiles)

# 删除无法解析的 SMILES
df = df[df["canonical_SMILES"].notnull()]

# -----------------------------
# 删除重复分子
# -----------------------------
df = df.drop_duplicates(subset="canonical_SMILES")

# -----------------------------
# 输出 CSV
# -----------------------------
df_final = df[[ "canonical_SMILES", label_col ]]
df_final.to_csv(output_file, index=False)
print(f"Standardized and deduplicated dataset saved to {output_file}")