from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# 设置全局 SVG 字体可编辑
mpl.rcParams["svg.fonttype"] = "none"

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

try:
    from adjust_text import adjust_text
    _has_adjust = True
except ImportError:
    _has_adjust = False

RDLogger.DisableLog("rdApp.warning")
warnings.filterwarnings("ignore")

COLOR_PALETTE = ["#F4A5A5", "#A0C8E1"] 
METRIC_PALETTE = ["#7EA1C4", "#A2C4A2"]

def load_fps(path: Path, nbits: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    smi_col = "SMILES" if "SMILES" in df.columns else "smiles"
    smiles = df[smi_col].astype(str).tolist()
    labels = df["label"].astype(int).to_numpy()

    fps = np.zeros((len(smiles), nbits), dtype=np.uint8)
    valid_idx = []
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m:
            arr = AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=nbits)
            fps[i, list(arr.GetOnBits())] = 1
            valid_idx.append(i)
    return fps[valid_idx], labels[valid_idx]

def panel_C_bit_activation(fps, labels, out, topn=6):
    mean_pos = fps[labels == 1].mean(axis=0)
    mean_neg = fps[labels == 0].mean(axis=0)
    diff = mean_pos - mean_neg

    mu = np.mean(diff)
    sigma = np.std(diff)

    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(mean_neg, mean_pos, c=diff, cmap="RdBu_r", s=25, alpha=0.5, edgecolors="none")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#AAAAAA", linewidth=1)
    
    idx_list = np.argsort(np.abs(diff))
    texts = []
    count = 0
    # 逆序遍历，跳过位点387
    for i in reversed(idx_list):
        if i == 387: continue
        if count >= 5: break
        t = ax.text(mean_neg[i], mean_pos[i], f"Bit {i}", fontsize=7.5, fontweight='bold')
        texts.append(t)
        count += 1
    
    if _has_adjust:
        adjust_text(texts, arrowprops=dict(arrowstyle='->', color='#333333', lw=0.6), ax=ax)
    
    # 统计信息标注 (使用原始字符串 fr 避免警告)
    stats_text = fr"$\mu_{{diff}} = {mu:.4f}$" + "\n" + fr"$\sigma_{{diff}} = {sigma:.4f}$"
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.6, edgecolor='none'))

    ax.set_xlabel("Mean activation (BBB-)")
    ax.set_ylabel("Mean activation (BBB+)")
    ax.set_title("Bit-activation Differential Analysis", fontweight="bold")

    # 保留色条，但不添加英文标签
    cb = plt.colorbar(sc, ax=ax)
    cb.outline.set_visible(False)
    
    plt.tight_layout()
    plt.savefig(out, format="svg", bbox_inches="tight")
    plt.close()

def panel_D_violin_sparsity(fps, labels, out):
    counts = fps.sum(axis=1)
    df = pd.DataFrame({"sparsity": counts, "label": ["BBB+" if l==1 else "BBB-" for l in labels]})
    
    plt.figure(figsize=(6, 5)) 
    sns.violinplot(x="label", y="sparsity", data=df, palette=COLOR_PALETTE, width=0.6, inner="box")

    # 分组标注平均值和标准差
    stats = df.groupby("label")["sparsity"].agg(["mean", "std"])
    for i, group in enumerate(["BBB-", "BBB+"]):
        m, s = stats.loc[group, "mean"], stats.loc[group, "std"]
        stat_str = fr"$\mu = {m:.1f}$" + "\n" + fr"$\sigma = {s:.1f}$"
        plt.text(i, plt.gca().get_ylim()[1] * 0.85, stat_str, ha='center', fontsize=9, 
                 fontweight='bold', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

    plt.title("Fingerprint Sparsity", fontweight="bold")
    plt.ylabel("Active Bits")
    plt.tight_layout()
    plt.savefig(out, format="svg", bbox_inches="tight")
    plt.close()

def panel_E_correlation_heatmap(fps, out, topk=20):
    var = fps.var(axis=0)
    idx = np.argsort(var)[-topk:][::-1]
    sel = fps[:, idx]
    corr = np.corrcoef(sel.T)
    
    plt.figure(figsize=(7, 6))
    # 限制色域使对角线不刺眼，增加蓝色辨识度
    ax = sns.heatmap(corr, cmap="RdBu_r", center=0, square=True,
                vmin=-0.6, vmax=0.6, 
                xticklabels=idx, yticklabels=idx,
                linewidths=0.8, linecolor='white',
                cbar=True)

    plt.title("Fingerprint Bit Correlation", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out, format="svg", bbox_inches="tight")
    plt.close()

# 其他函数 (LDA, PCA, Clustering) 保持逻辑不变，仅确保不出现(A)(B)
def panel_A_lda(fps, labels, out):
    lda = LinearDiscriminantAnalysis(n_components=1).fit_transform(fps, labels)
    pca = PCA(n_components=1, random_state=0).fit_transform(fps)
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=lda[:, 0], y=pca[:, 0], hue=labels, palette=COLOR_PALETTE, s=25, alpha=0.7)
    plt.xlabel("LD1")
    plt.ylabel("PCA1")
    plt.title("LDA-PCA Projection", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out, format="svg")
    plt.close()

def panel_B_pca(fps, labels, out):
    X = PCA(n_components=2, random_state=0).fit_transform(fps)
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, palette=COLOR_PALETTE, s=20, alpha=0.7)
    plt.title("PCA Dimensionality Reduction", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out, format="svg")
    plt.close()

def panel_F_clustering_quality(fps, labels, out):
    methods = {"PCA": PCA(n_components=2, random_state=0).fit_transform(fps),
               "t-SNE": TSNE(n_components=2, init="pca", random_state=0).fit_transform(fps)}
    m_list = list(methods.keys())
    db = [davies_bouldin_score(methods[m], labels) for m in m_list]
    ch = [calinski_harabasz_score(methods[m], labels) for m in m_list]

    x = np.arange(len(m_list)); width = 0.35
    fig, ax1 = plt.subplots(figsize=(6, 5))
    ax1.bar(x - width/2, db, width, color=METRIC_PALETTE[0], label="DB Index (↓)")
    ax1.set_ylabel("DB Index", color=METRIC_PALETTE[0], fontweight="bold")
    ax2 = ax1.twinx()
    ax2.bar(x + width/2, ch, width, color=METRIC_PALETTE[1], label="CH Index (↑)")
    ax2.set_ylabel("CH Index", color=METRIC_PALETTE[1], fontweight="bold")
    ax1.set_xticks(x); ax1.set_xticklabels(m_list, fontweight="bold")
    plt.title("Clustering Quality Metrics", fontweight="bold")
    plt.tight_layout(); plt.savefig(out, format="svg", bbox_inches="tight"); plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="dataset/bbbp_cleaned4.csv")
    parser.add_argument("--out-dir", default="refined_plots")
    args = parser.parse_args()
    outdir = Path(args.out_dir); outdir.mkdir(parents=True, exist_ok=True)

    fps, labels = load_fps(Path(args.input))
    panel_A_lda(fps, labels, outdir / "panel_A.svg")
    panel_B_pca(fps, labels, outdir / "panel_B.svg")
    panel_C_bit_activation(fps, labels, outdir / "panel_C.svg")
    panel_D_violin_sparsity(fps, labels, outdir / "panel_D.svg")
    panel_E_correlation_heatmap(fps, outdir / "panel_E.svg")
    panel_F_clustering_quality(fps, labels, outdir / "panel_F.svg")
    print(f"Done. Files saved to: {outdir}")

if __name__ == "__main__":
    main()