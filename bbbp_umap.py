from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 屏蔽冗余警告
RDLogger.DisableLog("rdApp.warning")
warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# 期刊投稿级绘图配置
# ==========================================
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False

# ==========================================
# 参考新图的色彩方案：淡蓝 vs 淡粉
# ==========================================
LABEL_COLORS = {
    "BBB+": "#A0C8E1",  # 柔和淡蓝 (参考图中 EGFT/HER2)
    "BBB-": "#F4A5A5"   # 柔和淡粉 (参考图中 EGFR/HER2/VEGFR)
}
CONTINUOUS_CMAP = "turbo"  # 保留右侧图的高区分度彩虹色

def _morgan_bits(mol: Chem.Mol, n_bits: int = 2048, radius: int = 2) -> np.ndarray:
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((1,), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def _load_data(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    df = df.dropna(subset=[c for c in df.columns if 'smiles' in c.lower()])
    
    col_map = {c.upper(): c for c in df.columns}
    smiles_col = col_map.get("SMILES")
    label_col = col_map.get("LABEL")
    
    if not smiles_col or label_col is None:
        raise ValueError(f"CSV must contain 'SMILES' and 'label' columns.")

    records = []
    for _, row in df.iterrows():
        smiles = str(row[smiles_col]).strip()
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: continue

        records.append({
            "SMILES": smiles,
            "LABEL": int(row[label_col]),
            "label_name": "BBB+" if int(row[label_col]) == 1 else "BBB-",
            "cLogP": Crippen.MolLogP(mol),
            "fps": _morgan_bits(mol),
        })
    return pd.DataFrame.from_records(records)

def _style_axes(ax, title, xlabel="Component 1", ylabel="Component 2"):
    """参考上传图的坐标轴风格：保留刻度和标签"""
    ax.set_title(title, loc="center", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    # 稍微淡化边框
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    ax.grid(True, linestyle='--', alpha=0.3)

def save_individual_plots(df: pd.DataFrame, output_prefix: Path) -> None:
    # 提取指纹并降维
    x = np.stack(df["fps"].values)
    
    # UMAP 计算
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.15, metric="jaccard", random_state=42)
    umap_xy = reducer.fit_transform(x)

    # t-SNE 计算
    x_pca = PCA(n_components=min(50, x.shape[1]), random_state=42).fit_transform(x)
    tsne_xy = TSNE(n_components=2, perplexity=30, init="pca", random_state=42).fit_transform(x_pca)

    labels = df["label_name"].to_numpy()
    clogp = df["cLogP"].to_numpy()

    plot_configs = [
        {
            "name": "UMAP_Class",
            "xy": umap_xy,
            "type": "categorical",
            "title": "UMAP: Chemical Space Distribution",
            "xlabel": "UMAP 1", "ylabel": "UMAP 2"
        },
        {
            "name": "tSNE_Class",
            "xy": tsne_xy,
            "type": "categorical",
            "title": "t-SNE: Scaffold Clusters",
            "xlabel": "t-SNE 1", "ylabel": "t-SNE 2"
        },
        {
            "name": "UMAP_cLogP",
            "xy": umap_xy,
            "type": "continuous",
            "title": "UMAP: Lipophilicity (cLogP)",
            "xlabel": "UMAP 1", "ylabel": "UMAP 2"
        }
    ]

    for config in plot_configs:
        fig, ax = plt.subplots(figsize=(7, 6.5), dpi=300)
        
        if config["type"] == "categorical":
            for label in ["BBB+", "BBB-"]:
                mask = labels == label
                ax.scatter(
                    config["xy"][mask, 0], config["xy"][mask, 1],
                    s=25, c=LABEL_COLORS[label],
                    alpha=0.6, edgecolors="white", linewidths=0.5, label=label
                )
            ax.legend(frameon=True, loc="upper right", fontsize=10)
        else:
            sc = ax.scatter(
                config["xy"][:, 0], config["xy"][:, 1],
                c=clogp, s=20, cmap=CONTINUOUS_CMAP,
                alpha=0.7, edgecolors="none"
            )
            cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("cLogP", fontsize=11)

        _style_axes(ax, config["title"], config["xlabel"], config["ylabel"])
        
        # 导出独立的 SVG
        out_file = output_prefix.parent / f"{output_prefix.stem}_{config['name']}.svg"
        fig.savefig(out_file, format="svg", bbox_inches="tight")
        plt.close(fig)
        print(f"[SUCCESS] Saved: {out_file}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="bbbp_cleaned.csv")
    parser.add_argument("--output", default="Figure_Output")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input file {input_path} not found.")
        return

    df = _load_data(input_path)
    save_individual_plots(df, Path(args.output))

if __name__ == "__main__":
    main()