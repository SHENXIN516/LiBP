#!/usr/bin/env python3
"""Interpretability Suite v2

A. Global interpretability across benchmark_0..9 with boxplots
B. Counterfactual case studies (prediction flip after minimal structural edit)
C. Cross-modality (BBB+ small molecule vs BBB+ peptide)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, rdFingerprintGenerator, rdFMCS
from rdkit import RDLogger
from sklearn.ensemble import RandomForestRegressor

import torch

ROOT = "/home/shenxin/LiBP"
if ROOT not in sys.path:
    sys.path.append(ROOT)

from export_attention_single import aggregate_node_attention, load_checkpoint, mol_to_graph

RDLogger.DisableLog("rdApp.warning")


CANDIDATE_SMILES_COLUMNS = ["SMILES", "smiles", "sequence", "mol"]
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


@dataclass
class MolRow:
    benchmark_id: int
    idx: int
    smiles: str
    label: int
    mol: Chem.Mol


def detect_smiles_col(df: pd.DataFrame) -> str:
    for c in CANDIDATE_SMILES_COLUMNS:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot find SMILES column. Available columns: {list(df.columns)}")


def canonicalize_smiles(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def parse_one_csv(csv_path: str, benchmark_id: int, smiles_col: Optional[str], label_col: str) -> List[MolRow]:
    df = pd.read_csv(csv_path)
    s_col = smiles_col if smiles_col and smiles_col in df.columns else detect_smiles_col(df)
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in {csv_path}")

    rows: List[MolRow] = []
    for i, r in df.iterrows():
        can = canonicalize_smiles(r[s_col])
        if can is None:
            continue
        mol = Chem.MolFromSmiles(can)
        if mol is None:
            continue
        rows.append(MolRow(benchmark_id=benchmark_id, idx=int(i), smiles=can, label=int(r[label_col]), mol=mol))
    return rows


def motif_feature_names() -> List[str]:
    names = [f"atom_count_{a}" for a in ATOM_TYPES]
    names += [f"motif_count_{k}" for k in MOTIF_SMARTS.keys()]
    names += ["num_atoms", "num_bonds", "tpsa", "logp", "hbd", "hba", "rot_bonds", "ring_count"]
    return names


def featurize_molecule(mol: Chem.Mol) -> np.ndarray:
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

    feats: List[float] = []
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


class ModelRunner:
    def __init__(self, ckpt_path: str, device: str):
        self.device = torch.device(device)
        self.model, _, self.preprocessing = load_checkpoint(ckpt_path, self.device)
        self.explicit_h = bool(self.preprocessing.get("explicit_H", False))
        self.use_chirality = bool(self.preprocessing.get("use_chirality", False))

    def predict(self, smiles: str, return_attention: bool = False):
        data = mol_to_graph(smiles, explicit_h=self.explicit_h, use_chirality=self.use_chirality).to(self.device)
        with torch.no_grad():
            if return_attention:
                logits, attentions = self.model(data, return_attention=True)
            else:
                logits = self.model(data)
                attentions = None
            probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()
            pred = int(np.argmax(probs))

        if return_attention:
            node_scores = aggregate_node_attention(attentions, num_nodes=data.x.size(0))
            return float(probs[1]), pred, node_scores
        return float(probs[1]), pred


def predict_probs(rows: Sequence[MolRow], runner: ModelRunner) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for r in rows:
        if r.smiles in out:
            continue
        p1, _ = runner.predict(r.smiles, return_attention=False)
        out[r.smiles] = p1
    return out


def run_global_boxplot(rows: Sequence[MolRow], probs: Dict[str, float], out_dir: str, top_k: int):
    import shap

    os.makedirs(out_dir, exist_ok=True)
    f_names = motif_feature_names()
    b_ids = sorted({r.benchmark_id for r in rows})

    per_bench_rows = []
    for b in b_ids:
        b_rows = [r for r in rows if r.benchmark_id == b]
        X = np.stack([featurize_molecule(r.mol) for r in b_rows], axis=0)
        y = np.asarray([probs[r.smiles] for r in b_rows], dtype=np.float32)

        surrogate = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
        surrogate.fit(X, y)

        explainer = shap.TreeExplainer(surrogate)
        shap_values = np.asarray(explainer.shap_values(X))
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 0]

        global_imp = np.mean(np.abs(shap_values), axis=0)
        for f, v in zip(f_names, global_imp):
            per_bench_rows.append({"benchmark": f"benchmark_{b}", "feature": f, "mean_abs_shap": float(v)})

        pd.DataFrame({"feature": f_names, "mean_abs_shap": global_imp}).sort_values(
            "mean_abs_shap", ascending=False
        ).to_csv(os.path.join(out_dir, f"benchmark_{b}_global_shap.csv"), index=False)

    all_df = pd.DataFrame(per_bench_rows)
    all_df.to_csv(os.path.join(out_dir, "global_shap_all_benchmarks_long.csv"), index=False)

    # Median + IQR summary for robustness-oriented ranking.
    summary = (
        all_df.groupby("feature", as_index=False)["mean_abs_shap"]
        .agg(
            median="median",
            q1=lambda x: np.quantile(x, 0.25),
            q3=lambda x: np.quantile(x, 0.75),
        )
    )
    summary["iqr"] = summary["q3"] - summary["q1"]
    summary = summary.sort_values(["median", "iqr"], ascending=[False, True])
    summary.to_csv(os.path.join(out_dir, "global_shap_median_iqr_ranking.csv"), index=False)

    med = all_df.groupby("feature", as_index=False)["mean_abs_shap"].median().sort_values("mean_abs_shap", ascending=False)
    top_features = med["feature"].head(top_k).tolist()
    top_df = all_df[all_df["feature"].isin(top_features)].copy()

    order = top_features[::-1]
    data_for_box = [top_df[top_df["feature"] == f]["mean_abs_shap"].values for f in order]

    plt.figure(figsize=(11, max(6, int(0.35 * len(order) + 3))))
    bp = plt.boxplot(data_for_box, vert=False, patch_artist=True, tick_labels=order, showfliers=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#9ecae1")
        patch.set_alpha(0.7)

    # Overlay benchmark points to show variance explicitly.
    for i, f in enumerate(order, start=1):
        vals = top_df[top_df["feature"] == f]["mean_abs_shap"].values
        y = np.full_like(vals, fill_value=i, dtype=float)
        jitter = np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else np.array([0.0])
        plt.scatter(vals, y + jitter, s=28, alpha=0.85, color="#08519c")

    plt.xlabel("mean(|SHAP|) across molecules in each benchmark")
    plt.title("Global Interpretability Generalization (benchmark_0-9)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "global_shap_boxplot_top_features.png"), dpi=240)
    plt.close()

    # Median + IQR ranking plot (top features by median SHAP).
    top_summary = summary.head(top_k).copy()
    top_summary = top_summary.iloc[::-1]
    y = np.arange(len(top_summary))
    x = top_summary["median"].values
    xerr_low = x - top_summary["q1"].values
    xerr_high = top_summary["q3"].values - x

    plt.figure(figsize=(11, max(6, int(0.34 * len(top_summary) + 3))))
    plt.errorbar(
        x,
        y,
        xerr=[xerr_low, xerr_high],
        fmt="o",
        color="#08519c",
        ecolor="#6baed6",
        elinewidth=2,
        capsize=4,
    )
    plt.yticks(y, top_summary["feature"].tolist())
    plt.xlabel("median mean(|SHAP|) across benchmark_0-9")
    plt.title("Feature Ranking by Median + IQR")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "global_shap_median_iqr_ranking.png"), dpi=240)
    plt.close()

    # SHAP beeswarm summary plot across all benchmarks (similar to classic SHAP figure style).
    X_all = np.stack([featurize_molecule(r.mol) for r in rows], axis=0)
    y_all = np.asarray([probs[r.smiles] for r in rows], dtype=np.float32)

    surrogate_all = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    surrogate_all.fit(X_all, y_all)

    explainer_all = shap.TreeExplainer(surrogate_all)
    # Compute SHAP on a representative subset to keep runtime manageable.
    rng = np.random.default_rng(42)
    n_samples = min(800, X_all.shape[0])
    sample_idx = rng.choice(X_all.shape[0], size=n_samples, replace=False)
    X_beeswarm = X_all[sample_idx]

    shap_values_all = np.asarray(explainer_all.shap_values(X_beeswarm))
    if shap_values_all.ndim == 3:
        shap_values_all = shap_values_all[:, :, 0]

    # Export matrix for reproducibility / downstream plotting.
    shap_matrix_df = pd.DataFrame(shap_values_all, columns=f_names)
    shap_matrix_df.to_csv(os.path.join(out_dir, "global_shap_values_matrix.csv"), index=False)

    # Use SHAP's beeswarm-style summary plot with feature value coloring.
    plt.figure(figsize=(10, max(6, int(0.32 * top_k + 3))))
    shap.summary_plot(
        shap_values_all,
        features=X_beeswarm,
        feature_names=f_names,
        max_display=top_k,
        show=False,
        plot_size=None,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "global_shap_beeswarm_top_features.png"), dpi=240)
    plt.close()


def _score_to_color(v: float):
    from matplotlib import colormaps

    cmap = colormaps.get_cmap("YlOrRd")
    r, g, b, _ = cmap(float(v))
    return (float(r), float(g), float(b))


def render_attention_mol(smiles: str, atom_scores: np.ndarray, title: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    scores = np.asarray(atom_scores, dtype=np.float32)
    if scores.size != mol.GetNumAtoms():
        scores = np.resize(scores, (mol.GetNumAtoms(),))

    if float(scores.max()) - float(scores.min()) < 1e-9:
        norm = np.zeros_like(scores)
    else:
        norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)

    highlight_atoms = list(range(mol.GetNumAtoms()))
    atom_colors = {i: _score_to_color(float(norm[i])) for i in highlight_atoms}
    atom_radii = {i: 0.26 + 0.24 * float(norm[i]) for i in highlight_atoms}

    img = Draw.MolToImage(
        mol,
        size=(640, 500),
        legend=title,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=atom_colors,
        highlightAtomRadii=atom_radii,
    )
    return np.asarray(img)


def _top_attention_atoms(scores: np.ndarray, top_n: int) -> set:
    arr = np.asarray(scores, dtype=np.float32)
    if arr.size == 0:
        return set()
    top_n = max(1, min(int(top_n), int(arr.size)))
    ids = np.argsort(arr)[-top_n:]
    return {int(i) for i in ids.tolist()}


def get_common_atom_sets(mol_a: Chem.Mol, mol_b: Chem.Mol) -> Tuple[set, set]:
    try:
        mcs = rdFMCS.FindMCS([mol_a, mol_b], timeout=10)
    except Exception:
        return set(), set()
    if mcs.numAtoms <= 0:
        return set(), set()
    patt = Chem.MolFromSmarts(mcs.smartsString)
    if patt is None:
        return set(), set()

    match_a = mol_a.GetSubstructMatch(patt)
    match_b = mol_b.GetSubstructMatch(patt)
    return set(match_a), set(match_b)


def render_counterfactual_mol(
    smiles: str,
    atom_scores: np.ndarray,
    title: str,
    common_atoms: set,
    top_n_attention: int = 6,
) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    n_atoms = mol.GetNumAtoms()
    diff_atoms = set(range(n_atoms)) - set(common_atoms)
    top_attn = _top_attention_atoms(atom_scores, top_n=top_n_attention)

    highlight_atoms = sorted(set(common_atoms) | set(diff_atoms) | set(top_attn))

    # Colors: common scaffold (blue), changed part (orange), top attention (red).
    color_common = (0.45, 0.67, 0.86)
    color_diff = (0.98, 0.70, 0.38)
    color_top = (0.86, 0.20, 0.21)

    atom_colors: Dict[int, Tuple[float, float, float]] = {}
    atom_radii: Dict[int, float] = {}
    for i in highlight_atoms:
        if i in common_atoms:
            atom_colors[i] = color_common
            atom_radii[i] = 0.18
        if i in diff_atoms:
            atom_colors[i] = color_diff
            atom_radii[i] = 0.30
        if i in top_attn:
            atom_colors[i] = color_top
            atom_radii[i] = 0.37

    img = Draw.MolToImage(
        mol,
        size=(640, 500),
        legend=title,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=atom_colors,
        highlightAtomRadii=atom_radii,
    )
    return np.asarray(img)


def fragment_change_summary(seed_mol: Chem.Mol, cf_mol: Chem.Mol, common_seed_atoms: set, common_cf_atoms: set) -> str:
    seed_diff = sorted(set(range(seed_mol.GetNumAtoms())) - set(common_seed_atoms))
    cf_diff = sorted(set(range(cf_mol.GetNumAtoms())) - set(common_cf_atoms))

    seed_symbols = [seed_mol.GetAtomWithIdx(i).GetSymbol() for i in seed_diff]
    cf_symbols = [cf_mol.GetAtomWithIdx(i).GetSymbol() for i in cf_diff]
    return f"seed_diff_atoms={seed_symbols}; cf_diff_atoms={cf_symbols}"


def mol_descriptors(mol: Chem.Mol) -> Dict[str, float]:
    from rdkit.Chem import Crippen, Lipinski, rdMolDescriptors

    return {
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "logp": float(Crippen.MolLogP(mol)),
        "hbd": float(Lipinski.NumHDonors(mol)),
        "hba": float(Lipinski.NumHAcceptors(mol)),
    }


def build_prediction_maps(rows: Sequence[MolRow], probs: Dict[str, float]) -> Tuple[Dict[str, int], Dict[str, Chem.Mol]]:
    pred_map = {s: int(p >= 0.5) for s, p in probs.items()}
    mol_map: Dict[str, Chem.Mol] = {}
    for r in rows:
        if r.smiles not in mol_map:
            mol_map[r.smiles] = r.mol
    return pred_map, mol_map


def best_real_counterfactual_for_seed(
    seed_smiles: str,
    seed_prob: float,
    seed_pred: int,
    fp_map: Dict[str, DataStructs.cDataStructs.ExplicitBitVect],
    pred_map: Dict[str, int],
    prob_map: Dict[str, float],
    mol_map: Dict[str, Chem.Mol],
    min_similarity: float = 0.75,
    max_atom_diff: int = 8,
    prefer_lower_similarity: bool = False,
):
    best = None
    seed_fp = fp_map[seed_smiles]

    for cand_smiles, cand_pred in pred_map.items():
        if cand_smiles == seed_smiles:
            continue
        if cand_pred == seed_pred:
            continue

        atom_diff = abs(mol_map[cand_smiles].GetNumHeavyAtoms() - mol_map[seed_smiles].GetNumHeavyAtoms())
        if atom_diff > max_atom_diff:
            continue

        sim = float(DataStructs.TanimotoSimilarity(seed_fp, fp_map[cand_smiles]))
        if sim < min_similarity:
            continue

        cand_prob = prob_map[cand_smiles]
        delta = abs(cand_prob - seed_prob)
        score = (1.0 - sim if prefer_lower_similarity else sim) + 0.1 * delta
        if (best is None) or (score > best["score"]):
            best = {
                "seed_smiles": seed_smiles,
                "cf_smiles": cand_smiles,
                "seed_prob": seed_prob,
                "cf_prob": cand_prob,
                "seed_pred": seed_pred,
                "cf_pred": cand_pred,
                "similarity": sim,
                "score": score,
            }
    return best


def run_counterfactual_case_studies(
    rows: Sequence[MolRow],
    probs: Dict[str, float],
    runner: ModelRunner,
    out_dir: str,
    n_cases: int,
    min_similarity: float = 0.75,
    max_atom_diff: int = 8,
    prefer_lower_similarity: bool = False,
):
    os.makedirs(out_dir, exist_ok=True)

    pred_map, mol_map = build_prediction_maps(rows, probs)
    unique_smiles = sorted(mol_map.keys())
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp_map = {s: fpgen.GetFingerprint(mol_map[s]) for s in unique_smiles}

    # Prioritize decision-boundary molecules; they are easier to find high-similarity flips for.
    candidates = sorted(unique_smiles, key=lambda s: abs(probs[s] - 0.5))

    found = []
    used_smiles = set()
    for s in candidates:
        if s in used_smiles:
            continue
        cf = best_real_counterfactual_for_seed(
            seed_smiles=s,
            seed_prob=probs[s],
            seed_pred=pred_map[s],
            fp_map=fp_map,
            pred_map=pred_map,
            prob_map=probs,
            mol_map=mol_map,
            min_similarity=min_similarity,
            max_atom_diff=max_atom_diff,
            prefer_lower_similarity=prefer_lower_similarity,
        )
        if cf is None:
            continue
        found.append(cf)
        used_smiles.add(s)
        used_smiles.add(cf["cf_smiles"])
        if len(found) >= n_cases:
            break

    records = []
    for i, cf in enumerate(found, start=1):
        seed_mol = mol_map[cf["seed_smiles"]]
        c_mol = mol_map[cf["cf_smiles"]]
        seed_desc = mol_descriptors(seed_mol)
        cf_desc = mol_descriptors(c_mol)

        common_seed, common_cf = get_common_atom_sets(seed_mol, c_mol)
        change_summary = fragment_change_summary(seed_mol, c_mol, common_seed, common_cf)

        seed_p1, seed_pred, seed_attn = runner.predict(cf["seed_smiles"], return_attention=True)
        c_p1, c_pred, c_attn = runner.predict(cf["cf_smiles"], return_attention=True)
        cf["seed_prob"] = seed_p1
        cf["cf_prob"] = c_p1
        cf["seed_pred"] = seed_pred
        cf["cf_pred"] = c_pred

        seed_title = f"Factual | p(BBB+)={cf['seed_prob']:.3f} | pred={cf['seed_pred']}"
        cf_title = f"Counterfactual | p(BBB+)={cf['cf_prob']:.3f} | pred={cf['cf_pred']}"

        seed_img = render_counterfactual_mol(
            cf["seed_smiles"],
            seed_attn,
            seed_title,
            common_atoms=common_seed,
            top_n_attention=6,
        )
        cf_img = render_counterfactual_mol(
            cf["cf_smiles"],
            c_attn,
            cf_title,
            common_atoms=common_cf,
            top_n_attention=6,
        )

        fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.5))
        axes[0].imshow(seed_img)
        axes[0].axis("off")
        axes[1].imshow(cf_img)
        axes[1].axis("off")
        fig.suptitle(
            f"CF Case {i} | real-library matched counterfactual | sim={cf['similarity']:.3f}",
            fontsize=12,
        )
        fig.tight_layout()
        png_path = os.path.join(out_dir, f"counterfactual_case_{i}.png")
        fig.savefig(png_path, dpi=240)
        plt.close(fig)

        records.append(
            {
                "case_id": i,
                "seed_smiles": cf["seed_smiles"],
                "counterfactual_smiles": cf["cf_smiles"],
                "seed_prob_bbb_plus": cf["seed_prob"],
                "counterfactual_prob_bbb_plus": cf["cf_prob"],
                "seed_pred": cf["seed_pred"],
                "counterfactual_pred": cf["cf_pred"],
                "flipped": int(cf["seed_pred"] != cf["cf_pred"]),
                "counterfactual_in_benchmark_0_9": 1,
                "tanimoto_similarity": cf["similarity"],
                "edit_atom_idx": -1,
                "old_symbol": "library_match",
                "new_symbol": "library_match",
                "local_fragment": change_summary,
                "delta_tpsa": cf_desc["tpsa"] - seed_desc["tpsa"],
                "delta_logp": cf_desc["logp"] - seed_desc["logp"],
                "delta_hbd": cf_desc["hbd"] - seed_desc["hbd"],
                "delta_hba": cf_desc["hba"] - seed_desc["hba"],
                "image": png_path,
            }
        )

    pd.DataFrame(records).to_csv(os.path.join(out_dir, "counterfactual_cases.csv"), index=False)


def peptide_backbone_atoms(mol: Chem.Mol) -> set:
    backbone = set()
    patt = Chem.MolFromSmarts("[NX3][CX3](=[OX1])")
    if patt is None:
        return backbone
    for match in mol.GetSubstructMatches(patt):
        if len(match) == 3:
            n_id, c_id, o_id = match
            backbone.update([n_id, c_id, o_id])
    return backbone


def choose_cross_modality_pair(rows: Sequence[MolRow], probs: Dict[str, float]) -> Tuple[MolRow, MolRow]:
    amide_patt = Chem.MolFromSmarts("[NX3][CX3](=[OX1])")
    small, pep = [], []

    for r in rows:
        if r.label != 1:
            continue
        amide_count = len(r.mol.GetSubstructMatches(amide_patt)) if amide_patt is not None else 0
        heavy = r.mol.GetNumHeavyAtoms()
        p = probs[r.smiles]
        if amide_count <= 1 and heavy <= 35:
            small.append((p, r))
        if amide_count >= 2 and heavy >= 20:
            pep.append((p + 0.05 * amide_count, r))

    if not small or not pep:
        raise RuntimeError("Cannot find BBB+ small molecule and peptide candidates.")

    small.sort(reverse=True, key=lambda x: x[0])
    pep.sort(reverse=True, key=lambda x: x[0])
    return small[0][1], pep[0][1]


def run_cross_modality(rows: Sequence[MolRow], probs: Dict[str, float], runner: ModelRunner, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    sm, pep = choose_cross_modality_pair(rows, probs)
    sm_p1, sm_pred, sm_attn = runner.predict(sm.smiles, return_attention=True)
    pp_p1, pp_pred, pp_attn = runner.predict(pep.smiles, return_attention=True)

    sm_img = render_attention_mol(sm.smiles, sm_attn, f"Small Molecule | p={sm_p1:.3f} | pred={sm_pred}")
    pep_img = render_attention_mol(pep.smiles, pp_attn, f"Peptide | p={pp_p1:.3f} | pred={pp_pred}")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    axes[0].imshow(sm_img)
    axes[0].axis("off")
    axes[1].imshow(pep_img)
    axes[1].axis("off")
    fig.suptitle("Cross-modality: BBB+ small molecule vs BBB+ peptide", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cross_modality_attention.png"), dpi=240)
    plt.close(fig)

    pep_backbone = peptide_backbone_atoms(pep.mol)
    arr = np.asarray(pp_attn, dtype=np.float32)
    if arr.max() - arr.min() < 1e-9:
        arr_n = np.zeros_like(arr)
    else:
        arr_n = (arr - arr.min()) / (arr.max() - arr.min() + 1e-12)

    rows_out = []
    for a in pep.mol.GetAtoms():
        i = a.GetIdx()
        rows_out.append(
            {
                "atom_id": i,
                "symbol": a.GetSymbol(),
                "attention_score": float(arr[i]),
                "attention_norm": float(arr_n[i]),
                "group": "backbone" if i in pep_backbone else "sidechain",
            }
        )

    pd.DataFrame(rows_out).to_csv(os.path.join(out_dir, "peptide_backbone_sidechain_attention.csv"), index=False)
    with open(os.path.join(out_dir, "cross_modality_selection.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "small_molecule_smiles": sm.smiles,
                "small_molecule_prob_bbb_plus": sm_p1,
                "peptide_smiles": pep.smiles,
                "peptide_prob_bbb_plus": pp_p1,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def main():
    parser = argparse.ArgumentParser(description="Interpretability Suite v2")
    parser.add_argument("--dataset_dir", default="/home/shenxin/LiBP/dataset")
    parser.add_argument("--benchmark_start", type=int, default=0)
    parser.add_argument("--benchmark_end", type=int, default=9)
    parser.add_argument("--smiles_col", default="")
    parser.add_argument("--label_col", default="label")
    parser.add_argument("--ckpt", default="/home/shenxin/LiBP/ckpt/02best_model_epoch154_acc0.8693.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", default="/home/shenxin/LiBP/results/interpretability_suite_v2")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--num_cf_cases", type=int, default=3)
    parser.add_argument("--cf_min_similarity", type=float, default=0.75)
    parser.add_argument("--cf_max_atom_diff", type=int, default=8)
    parser.add_argument("--cf_prefer_lower_similarity", action="store_true")
    args = parser.parse_args()

    all_rows: List[MolRow] = []
    loaded_files = []
    for b in range(args.benchmark_start, args.benchmark_end + 1):
        path = os.path.join(args.dataset_dir, f"benchmark_{b}.csv")
        if not os.path.exists(path):
            continue
        rs = parse_one_csv(path, benchmark_id=b, smiles_col=args.smiles_col or None, label_col=args.label_col)
        all_rows.extend(rs)
        loaded_files.append(path)

    if not all_rows:
        raise RuntimeError("No benchmark rows loaded.")

    os.makedirs(args.out_dir, exist_ok=True)

    runner = ModelRunner(args.ckpt, args.device)
    probs = predict_probs(all_rows, runner)

    a_dir = os.path.join(args.out_dir, "A_global_boxplot")
    b_dir = os.path.join(args.out_dir, "B_counterfactual_case_studies")
    c_dir = os.path.join(args.out_dir, "C_cross_modality")

    run_global_boxplot(all_rows, probs, a_dir, top_k=args.top_k)
    run_counterfactual_case_studies(
        all_rows,
        probs,
        runner,
        b_dir,
        n_cases=args.num_cf_cases,
        min_similarity=args.cf_min_similarity,
        max_atom_diff=args.cf_max_atom_diff,
        prefer_lower_similarity=args.cf_prefer_lower_similarity,
    )
    run_cross_modality(all_rows, probs, runner, c_dir)

    manifest = {
        "loaded_files": loaded_files,
        "checkpoint": args.ckpt,
        "num_valid_molecules": len(all_rows),
        "outputs": {
            "A_global": a_dir,
            "B_counterfactual": b_dir,
            "C_cross_modality": c_dir,
        },
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("Done interpretability suite v2.")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
