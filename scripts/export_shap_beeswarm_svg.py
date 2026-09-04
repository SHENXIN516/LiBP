#!/usr/bin/env python3
"""Export existing SHAP beeswarm as SVG using saved SHAP matrix.

This script reproduces the sample selection used in the suite (rng seed 42),
loads the previously saved SHAP values matrix (`global_shap_values_matrix.csv`),
recomputes feature values for the same molecule list, and writes an SVG.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/shenxin/LiBP"
if ROOT not in sys.path:
    sys.path.append(ROOT)

# import the suite module by path (script folder is not a package)
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.warning")


ATOM_TYPES = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B"]
MOTIF_SMARTS = {
    "amide": "[NX3][CX3](=[OX1])",
    "amine": "[NX3;H2,H1;!$(NC=O)]",
    "hydroxyl": "[OX2H]",
    "carboxyl": "C(=O)[OX2H1,OX1-]",
    "aromatic_ring": "a1aaaaa1",
    "halogen": "[F,Cl,Br,I]",
    "sulfonamide": "S(=O)(=O)N",
    "hetero5_ring": "[r5;!#6]",
    "hetero6_ring": "[r6;!#6]",
}


def motif_feature_names():
    names = [f"atom_count_{a}" for a in ATOM_TYPES]
    names += [f"motif_count_{k}" for k in MOTIF_SMARTS.keys()]
    names += ["num_atoms", "num_bonds", "tpsa", "logp", "hbd", "hba", "rot_bonds", "ring_count"]
    return names


def detect_smiles_col(df: pd.DataFrame) -> str:
    for c in ["SMILES", "smiles", "sequence", "mol"]:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot find SMILES column. Available: {list(df.columns)}")


def canonicalize_smiles(smiles: str):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def parse_one_csv(csv_path: str, benchmark_id: int, smiles_col: str | None, label_col: str):
    df = pd.read_csv(csv_path)
    s_col = smiles_col if smiles_col and smiles_col in df.columns else detect_smiles_col(df)
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in {csv_path}")

    rows = []
    for i, r in df.iterrows():
        can = canonicalize_smiles(r[s_col])
        if can is None:
            continue
        mol = Chem.MolFromSmiles(can)
        if mol is None:
            continue
        # minimal row object
        rows.append(type("R", (), {"benchmark_id": benchmark_id, "idx": int(i), "smiles": can, "label": int(r[label_col]), "mol": mol}))
    return rows


def featurize_molecule(mol: Chem.Mol):
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

    feats = []
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    for a in ATOM_TYPES:
        feats.append(float(symbols.count(a)))

    for smarts in MOTIF_SMARTS.values():
        patt = Chem.MolFromSmarts(smarts)
        cnt = len(mol.GetSubstructMatches(patt)) if patt is not None else 0
        feats.append(float(cnt))

    feats += [
        float(mol.GetNumAtoms()),
        float(mol.GetNumBonds()),
        float(rdMolDescriptors.CalcTPSA(mol)),
        float(Crippen.MolLogP(mol)),
        float(Lipinski.NumHDonors(mol)),
        float(Lipinski.NumHAcceptors(mol)),
        float(Descriptors.NumRotatableBonds(mol)),
        float(rdMolDescriptors.CalcNumRings(mol)),
    ]
    return np.asarray(feats, dtype=np.float32)

OUT_DIR = os.path.join(ROOT, "results/interpretability_suite_v2/A_global_boxplot")
SHAP_VALUES_CSV = os.path.join(OUT_DIR, "global_shap_values_matrix.csv")
DATA_DIR = os.path.join(ROOT, "dataset")


def load_rows(dataset_dir: str, start: int = 0, end: int = 9):
    rows = []
    for b in range(start, end + 1):
        path = os.path.join(dataset_dir, f"benchmark_{b}.csv")
        if not os.path.exists(path):
            continue
        rs = parse_one_csv(path, benchmark_id=b, smiles_col=None, label_col="label")
        rows.extend(rs)
    return rows


def main():
    if not os.path.exists(SHAP_VALUES_CSV):
        raise FileNotFoundError(f"Missing SHAP values CSV: {SHAP_VALUES_CSV}")

    rows = load_rows(DATA_DIR, 0, 9)
    if not rows:
        raise RuntimeError("No rows found in dataset directory; ensure benchmark CSVs are present.")

    # Recompute feature matrix exactly as in the suite
    f_names = motif_feature_names()
    X_all = np.stack([featurize_molecule(r.mol) for r in rows], axis=0)

    rng = np.random.default_rng(42)
    n_samples = min(800, X_all.shape[0])
    sample_idx = rng.choice(X_all.shape[0], size=n_samples, replace=False)
    X_beeswarm = X_all[sample_idx]

    shap_df = pd.read_csv(SHAP_VALUES_CSV)
    shap_values = shap_df.values

    # Align shapes: shap_values should be (n_samples, n_features)
    if shap_values.shape[0] != X_beeswarm.shape[0]:
        # try transpose fallback
        if shap_values.shape[1] == X_beeswarm.shape[0]:
            shap_values = shap_values.T
        else:
            raise RuntimeError(
                f"SHAP matrix shape {shap_values.shape} does not match expected samples {X_beeswarm.shape}"
            )

    # Save features used (for provenance)
    feat_df = pd.DataFrame(X_beeswarm, columns=f_names)
    feat_df.to_csv(os.path.join(OUT_DIR, "global_shap_feature_matrix.csv"), index=False)

    # Create beeswarm and save as SVG
    import shap  # local import (requires shap installed)

    plt.figure(figsize=(10, max(6, int(0.32 * min(20, len(f_names)) + 3))))
    shap.summary_plot(
        shap_values,
        features=X_beeswarm,
        feature_names=f_names,
        max_display=20,
        show=False,
        plot_size=None,
    )
    plt.tight_layout()
    out_svg = os.path.join(OUT_DIR, "global_shap_beeswarm_top_features.svg")
    plt.savefig(out_svg, format="svg", dpi=240)
    plt.close()

    print("Wrote SVG:", out_svg)


if __name__ == "__main__":
    main()
