from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["font.family"] = "DejaVu Sans"

import matplotlib.pyplot as plt
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog("rdApp.warning")


LABEL_MAP = {
    1: "BBB+",
    0: "BBB-",
}

PALETTE = {
    "BBB+": "#2F6BFF",
    "BBB-": "#D95F02",
}


def _load_dataframe(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    if "SMILES" not in df.columns or "label" not in df.columns:
        raise ValueError("Input CSV must contain SMILES and label columns.")

    df = df.copy()
    df["label_name"] = df["label"].map(LABEL_MAP)
    df = df[df["label_name"].notna()].copy()

    records = []
    invalid = 0
    for _, row in df.iterrows():
        smiles = str(row["SMILES"]).strip()
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            invalid += 1
            continue

        records.append(
            {
                "SMILES": smiles,
                "label": int(row["label"]),
                "label_name": row["label_name"],
                "mw": Descriptors.MolWt(mol),
                "tpsa": Descriptors.TPSA(mol),
            }
        )

    clean_df = pd.DataFrame.from_records(records)
    if clean_df.empty:
        raise ValueError("No valid molecules were found in the input CSV.")

    if invalid:
        print(f"[WARN] Skipped {invalid} invalid SMILES entries.")

    return clean_df


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)


def _annotate_bars(ax, bars, values):
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.015,
            f"{value:,}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#222222",
        )


def build_figure(df: pd.DataFrame, output_path: Path) -> None:
    counts = df["label_name"].value_counts().reindex(["BBB+", "BBB-"])
    total = len(df)

    fig = plt.figure(figsize=(13.5, 5.0), facecolor="white")
    gs = fig.add_gridspec(1, 3, wspace=0.32)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    # Panel A: class balance
    bars = ax1.bar(
        counts.index,
        counts.values,
        color=[PALETTE[label] for label in counts.index],
        width=0.62,
        edgecolor="#222222",
        linewidth=0.8,
    )
    _style_axes(ax1)
    ax1.set_title("A  Class balance", loc="left", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Count")
    ax1.set_ylim(0, max(counts.values) * 1.18)
    _annotate_bars(ax1, bars, counts.values)
    ax1.text(
        0.5,
        -0.18,
        f"n = {total:,} molecules\nBBB+ {counts['BBB+']:,} ({counts['BBB+'] / total:.1%})\nBBB- {counts['BBB-']:,} ({counts['BBB-'] / total:.1%})",
        transform=ax1.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color="#444444",
    )

    # Panel B: MW distribution
    data_mw = [df.loc[df["label_name"] == label, "mw"] for label in ["BBB+", "BBB-"]]
    box1 = ax2.boxplot(
        data_mw,
        tick_labels=["BBB+", "BBB-"],
        patch_artist=True,
        widths=0.55,
        showfliers=False,
        medianprops={"color": "#111111", "linewidth": 1.4},
        boxprops={"linewidth": 1.0, "edgecolor": "#444444"},
        whiskerprops={"linewidth": 1.0, "color": "#444444"},
        capprops={"linewidth": 1.0, "color": "#444444"},
    )
    for patch, label in zip(box1["boxes"], ["BBB+", "BBB-"]):
        patch.set_facecolor(PALETTE[label])
        patch.set_alpha(0.22)
    _style_axes(ax2)
    ax2.set_title("B  Molecular weight", loc="left", fontsize=13, fontweight="bold")
    ax2.set_ylabel("MW")
    ax2.set_ylim(0, max(df["mw"]) * 1.05)
    for i, label in enumerate(["BBB+", "BBB-"]):
        median = df.loc[df["label_name"] == label, "mw"].median()
        ax2.text(i + 1, median, f"{median:.1f}", ha="center", va="bottom", fontsize=9, color="#333333")

    # Panel C: TPSA distribution
    data_tpsa = [df.loc[df["label_name"] == label, "tpsa"] for label in ["BBB+", "BBB-"]]
    box2 = ax3.boxplot(
        data_tpsa,
        tick_labels=["BBB+", "BBB-"],
        patch_artist=True,
        widths=0.55,
        showfliers=False,
        medianprops={"color": "#111111", "linewidth": 1.4},
        boxprops={"linewidth": 1.0, "edgecolor": "#444444"},
        whiskerprops={"linewidth": 1.0, "color": "#444444"},
        capprops={"linewidth": 1.0, "color": "#444444"},
    )
    for patch, label in zip(box2["boxes"], ["BBB+", "BBB-"]):
        patch.set_facecolor(PALETTE[label])
        patch.set_alpha(0.22)
    _style_axes(ax3)
    ax3.set_title("C  Topological polar surface area", loc="left", fontsize=13, fontweight="bold")
    ax3.set_ylabel("TPSA")
    ax3.set_ylim(0, max(df["tpsa"]) * 1.05)
    for i, label in enumerate(["BBB+", "BBB-"]):
        median = df.loc[df["label_name"] == label, "tpsa"].median()
        ax3.text(i + 1, median, f"{median:.1f}", ha="center", va="bottom", fontsize=9, color="#333333")

    fig.suptitle(
        "BBBP cleaned4 dataset summary",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.03,
        "Descriptors calculated with RDKit from valid SMILES entries only.",
        ha="center",
        fontsize=9,
        color="#555555",
    )

    fig.savefig(output_path, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] Saved figure to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a 3-panel SVG summary for BBBP cleaned4.")
    parser.add_argument("--input", default="dataset/bbbp_cleaned4.csv", help="Input CSV file.")
    parser.add_argument("--output", default="dataset/bbbp_cleaned4_si.svg", help="Output SVG path.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = _load_dataframe(input_path)
    build_figure(df, output_path)


if __name__ == "__main__":
    main()