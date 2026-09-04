#!/usr/bin/env python3
"""Summarize counterfactual pairs into a publication-style transformation/descriptor figure.

Outputs:
- a CSV with per-case transformation labels and descriptor deltas
- an SVG figure with two clean tables:
  1) Transformation -> Effect
  2) Descriptor Increase / Decrease counts
"""

from __future__ import annotations

import argparse
import os
from collections import Counter, OrderedDict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Lipinski, rdMolDescriptors


def count_smarts(mol: Chem.Mol, smarts: str) -> int:
    patt = Chem.MolFromSmarts(smarts)
    if patt is None:
        return 0
    return len(mol.GetSubstructMatches(patt))


def mol_stats(mol: Chem.Mol) -> Dict[str, float]:
    return {
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "logp": float(Crippen.MolLogP(mol)),
        "hbd": float(Lipinski.NumHDonors(mol)),
        "hba": float(Lipinski.NumHAcceptors(mol)),
        "amide": float(count_smarts(mol, "[NX3][CX3](=[OX1])")),
        "hydroxyl": float(count_smarts(mol, "[OX2H]")),
        "aromatic_rings": float(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "heteroatoms": float(sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in (1, 6))),
    }


def classify_transformation(seed: Chem.Mol, cf: Chem.Mol) -> List[str]:
    seed_s = mol_stats(seed)
    cf_s = mol_stats(cf)
    labels: List[str] = []

    if seed_s["hydroxyl"] > cf_s["hydroxyl"] and cf_s["hbd"] < seed_s["hbd"]:
        labels.append("remove hydroxyl")
    if seed_s["amide"] > cf_s["amide"]:
        labels.append("reduce amide")
    if cf_s["aromatic_rings"] > seed_s["aromatic_rings"]:
        labels.append("add aromatic ring")
    if cf_s["heteroatoms"] < seed_s["heteroatoms"]:
        labels.append("remove heteroatoms")

    if not labels:
        labels.append("other")
    return labels


def orient_pair(row: pd.Series) -> Tuple[str, str, float, float]:
    """Match the panel orientation: left is BBB+ side, right is BBB- side."""
    seed_pred = int(row["seed_pred"])
    cf_pred = int(row["counterfactual_pred"])
    seed_smiles = str(row["seed_smiles"])
    cf_smiles = str(row["counterfactual_smiles"])
    seed_prob = float(row["seed_prob_bbb_plus"])
    cf_prob = float(row["counterfactual_prob_bbb_plus"])

    if seed_pred == 1 and cf_pred == 0:
        return seed_smiles, cf_smiles, seed_prob, cf_prob
    if seed_pred == 0 and cf_pred == 1:
        return cf_smiles, seed_smiles, cf_prob, seed_prob

    if seed_prob >= cf_prob:
        return seed_smiles, cf_smiles, seed_prob, cf_prob
    return cf_smiles, seed_smiles, cf_prob, seed_prob


def descriptor_direction(delta: float) -> str:
    if delta > 0:
        return "Increase"
    if delta < 0:
        return "Decrease"
    return "No change"


def add_table(ax, title: str, headers: List[str], rows: List[List[str]], col_widths=None):
    ax.set_axis_off()
    ax.set_title(title, fontsize=18, fontweight="bold", pad=18)
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.0, 1.55)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#444444")
        cell.set_linewidth(0.8)
        if r == 0:
            cell.set_facecolor("#e9eef5")
            cell.set_text_props(weight="bold", color="#222222")
        else:
            cell.set_facecolor("white")

    return table


def main():
    parser = argparse.ArgumentParser(description="Summarize CF pairs into publishable tables")
    parser.add_argument(
        "--csv",
        default="/home/shenxin/LiBP/results/interpretability_suite_v2/B_counterfactual_case_studies/counterfactual_cases.csv",
    )
    parser.add_argument(
        "--out_dir",
        default="/home/shenxin/LiBP/results/interpretability_suite_v2/B_counterfactual_case_studies",
    )
    parser.add_argument("--out_prefix", default="cf_summary")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if df.empty:
        raise RuntimeError(f"CSV is empty: {args.csv}")

    records = []
    for _, row in df.iterrows():
        left_smi, right_smi, left_prob, right_prob = orient_pair(row)
        left = Chem.MolFromSmiles(left_smi)
        right = Chem.MolFromSmiles(right_smi)
        if left is None or right is None:
            continue

        left_s = mol_stats(left)
        right_s = mol_stats(right)
        labels = classify_transformation(right, left)
        delta_tpsa = left_s["tpsa"] - right_s["tpsa"]
        delta_hbd = left_s["hbd"] - right_s["hbd"]
        delta_logp = left_s["logp"] - right_s["logp"]

        records.append(
            {
                "case_id": int(row["case_id"]),
                "transformation": labels[0],
                "transformation_all": "; ".join(labels),
                "effect": "BBB+",
                "similarity": float(row["tanimoto_similarity"]),
                "left_smiles": left_smi,
                "right_smiles": right_smi,
                "left_prob": left_prob,
                "right_prob": right_prob,
                "delta_tpsa": delta_tpsa,
                "delta_hbd": delta_hbd,
                "delta_logp": delta_logp,
                "tpsa_dir": descriptor_direction(delta_tpsa),
                "hbd_dir": descriptor_direction(delta_hbd),
                "logp_dir": descriptor_direction(delta_logp),
            }
        )

    out_df = pd.DataFrame(records)
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f"{args.out_prefix}.csv")
    out_df.to_csv(csv_path, index=False)

    # Transformation summary table.
    transform_order = ["remove hydroxyl", "reduce amide", "add aromatic ring", "remove heteroatoms", "other"]
    transform_counts: Counter = Counter()
    if not out_df.empty and "transformation_all" in out_df.columns:
        for s in out_df["transformation_all"].astype(str):
            for label in [x.strip() for x in s.split(";") if x.strip()]:
                transform_counts[label] += 1
    transform_rows: List[List[str]] = []
    for t in transform_order:
        if t in transform_counts:
            transform_rows.append([t, "BBB+", str(transform_counts[t])])
    if not transform_rows:
        transform_rows = [["n/a", "n/a", "0"]]

    # Descriptor direction counts.
    desc_rows: List[List[str]] = []
    for desc, col in [("TPSA", "tpsa_dir"), ("HBD", "hbd_dir"), ("LogP", "logp_dir")]:
        s = out_df[col].value_counts().to_dict() if col in out_df.columns else {}
        inc = int(s.get("Increase", 0))
        dec = int(s.get("Decrease", 0))
        no = int(s.get("No change", 0))
        desc_rows.append([desc, str(inc), str(dec), str(no)])

    fig = plt.figure(figsize=(11.5, 7.2), facecolor="white")
    gs = fig.add_gridspec(2, 1, height_ratios=[1.1, 0.9], hspace=0.22)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])

    add_table(
        ax1,
        "Transformation  Effect",
        ["Transformation", "Effect", "Count"],
        transform_rows,
        col_widths=[0.55, 0.18, 0.12],
    )
    add_table(
        ax2,
        "Descriptor  Increase  Decrease",
        ["Descriptor", "Increase", "Decrease", "No change"],
        desc_rows,
        col_widths=[0.35, 0.18, 0.18, 0.18],
    )

    fig.text(
        0.5,
        0.02,
        f"Based on {len(out_df)} CF pairs; similarity median={out_df['similarity'].median():.3f}",
        ha="center",
        va="bottom",
        fontsize=11,
        color="#444444",
    )

    svg_path = os.path.join(args.out_dir, f"{args.out_prefix}.svg")
    png_path = os.path.join(args.out_dir, f"{args.out_prefix}.png")
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {svg_path}")
    print(f"Wrote: {png_path}")


if __name__ == "__main__":
    main()
