#!/usr/bin/env python3
"""Render a publication-style counterfactual molecule gallery as SVG.

This figure is visual-only: no summary tables, no count charts.
It lays out 8 CF pairs as clean cards with left/right molecules and a slim arrow.
Large peptide-like molecules can be rendered with PyMOL when available.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import os
import tempfile
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdFMCS
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import RDLogger
from rdkit.Geometry import Point2D

RDLogger.DisableLog("rdApp.warning")


def get_common_sets(mol_a: Chem.Mol, mol_b: Chem.Mol) -> Tuple[set, set]:
    try:
        mcs = rdFMCS.FindMCS([mol_a, mol_b], timeout=10)
    except Exception:
        return set(), set()
    if mcs.numAtoms <= 0:
        return set(), set()
    patt = Chem.MolFromSmarts(mcs.smartsString)
    if patt is None:
        return set(), set()
    ma = mol_a.GetSubstructMatch(patt)
    mb = mol_b.GetSubstructMatch(patt)
    return set(ma), set(mb)


def _white_rgba():
    return (1.0, 1.0, 1.0, 1.0)


def draw_mol_with_box(smiles: str, common_atoms: set, diff_atoms: set, width: int = 620, height: int = 320) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.ones((height, width, 3), dtype=np.uint8) * 255

    mol = Chem.Mol(mol)
    Draw.rdDepictor.Compute2DCoords(mol)

    common_color = (0.80, 0.88, 0.98)
    diff_color = (0.99, 0.77, 0.42)

    highlight_atoms = sorted(set(common_atoms) | set(diff_atoms))
    atom_colors = {}
    atom_radii = {}
    for i in common_atoms:
        atom_colors[int(i)] = common_color
        atom_radii[int(i)] = 0.24
    for i in diff_atoms:
        atom_colors[int(i)] = diff_color
        atom_radii[int(i)] = 0.30

    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    opts = drawer.drawOptions()
    opts.clearBackground = True
    try:
        opts.backgroundColour = _white_rgba()
    except Exception:
        pass
    opts.padding = 0.05
    opts.bondLineWidth = 2

    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=atom_colors,
        highlightAtomRadii=atom_radii,
    )
    if diff_atoms:
        pts = [drawer.GetDrawCoords(int(i)) for i in diff_atoms]
        min_x = min(float(p.x) for p in pts)
        max_x = max(float(p.x) for p in pts)
        min_y = min(float(p.y) for p in pts)
        max_y = max(float(p.y) for p in pts)
        pad = 18.0
        p1 = Point2D(min_x - pad, min_y - pad)
        p2 = Point2D(max_x + pad, max_y + pad)
        drawer.SetColour((0.34, 0.34, 0.34))
        drawer.SetLineWidth(1.2)
        drawer.DrawRect(p1, p2)

    drawer.FinishDrawing()
    img = Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGB")
    return np.asarray(img)


def draw_mol_with_pymol(smiles: str, width: int = 620, height: int = 320) -> Optional[np.ndarray]:
    try:
        import pymol2  # type: ignore
    except Exception:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    try:
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        AllChem.UFFOptimizeMolecule(mol, maxIters=150)
    except Exception:
        return None

    with tempfile.TemporaryDirectory() as td:
        sdf_path = os.path.join(td, "tmp.sdf")
        png_path = os.path.join(td, "tmp.png")
        w = Chem.SDWriter(sdf_path)
        w.write(mol)
        w.close()

        try:
            with pymol2.PyMOL() as pm:
                cmd = pm.cmd
                cmd.reinitialize()
                cmd.load(sdf_path, "mol")
                cmd.hide("everything", "all")
                cmd.show("sticks", "mol")
                cmd.set("stick_radius", 0.18)
                cmd.set("ray_opaque_background", 0)
                cmd.set("ray_trace_mode", 1)
                cmd.set("ray_shadow", 0)
                cmd.set("depth_cue", 0)
                cmd.bg_color("black")
                # Element palette tuned for a dark journal-style render.
                cmd.color("orange", "mol and elem C")
                cmd.color("red", "mol and elem O")
                cmd.color("deepsalmon", "mol and elem N")
                cmd.color("yellow", "mol and elem S")
                cmd.color("cyan", "mol and elem F")
                cmd.color("magenta", "mol and elem Cl")
                cmd.set("stick_color", "orange")
                cmd.orient("mol")
                cmd.zoom("mol", 2.15)
                cmd.png(png_path, width=width, height=height, dpi=300, ray=1)
        except Exception:
            return None

        if not os.path.exists(png_path):
            return None
        return np.asarray(Image.open(png_path).convert("RGB"))


def maybe_draw(smiles: str, common_atoms: set, diff_atoms: set, use_pymol_for_peptides: bool, peptide_heavy_atom_threshold: int, width: int, height: int) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.ones((height, width, 3), dtype=np.uint8) * 255
    if use_pymol_for_peptides and mol.GetNumHeavyAtoms() >= peptide_heavy_atom_threshold:
        py_img = draw_mol_with_pymol(smiles, width=width, height=height)
        if py_img is not None:
            return py_img
    return draw_mol_with_box(smiles, common_atoms, diff_atoms, width=width, height=height)


def build_pairs(df: pd.DataFrame) -> List[Dict[str, float]]:
    pairs: List[Dict[str, float]] = []
    for _, r in df.sort_values("case_id").iterrows():
        seed_smi = str(r["seed_smiles"])
        cf_smi = str(r["counterfactual_smiles"])
        seed_pred = int(r["seed_pred"])
        cf_pred = int(r["counterfactual_pred"])
        seed_p = float(r["seed_prob_bbb_plus"])
        cf_p = float(r["counterfactual_prob_bbb_plus"])

        # Orient so left is BBB+ whenever possible.
        if seed_pred == 1 and cf_pred == 0:
            left, right = seed_smi, cf_smi
            left_p, right_p = seed_p, cf_p
        elif seed_pred == 0 and cf_pred == 1:
            left, right = cf_smi, seed_smi
            left_p, right_p = cf_p, seed_p
        else:
            if seed_p >= cf_p:
                left, right = seed_smi, cf_smi
                left_p, right_p = seed_p, cf_p
            else:
                left, right = cf_smi, seed_smi
                left_p, right_p = cf_p, seed_p

        pairs.append(
            {
                "case_id": int(r["case_id"]),
                "left_smiles": left,
                "right_smiles": right,
                "left_prob": left_p,
                "right_prob": right_p,
                "delta_p": left_p - right_p,
                "similarity": float(r.get("tanimoto_similarity", float("nan"))),
            }
        )
    return pairs


def add_card(fig, x0, y0, w, h):
    patch = FancyBboxPatch(
        (x0, y0),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.0,
        edgecolor="#d8d8d8",
        facecolor="white",
        transform=fig.transFigure,
        zorder=0,
    )
    fig.add_artist(patch)


def plot_gallery(pairs: Sequence[Dict[str, float]], out_svg: str, use_pymol_for_peptides: bool, peptide_heavy_atom_threshold: int):
    pairs = list(pairs)[:8]
    if len(pairs) < 4:
        raise RuntimeError(f"Need at least 4 pairs, got {len(pairs)}")

    fig = plt.figure(figsize=(20, 10.8), facecolor="white")
    gs = fig.add_gridspec(2, 4, left=0.03, right=0.97, top=0.92, bottom=0.05, wspace=0.06, hspace=0.12)

    fig.text(0.03, 0.965, "Counterfactual pair gallery", fontsize=24, fontweight="bold", ha="left", va="top")
    fig.text(0.03, 0.935, "Low-similarity real-library matched pairs; left = BBB+ side, right = BBB- side.", fontsize=12.5, color="#444444", ha="left", va="top")

    for idx, pair in enumerate(pairs):
        r = idx // 4
        c = idx % 4
        ax = fig.add_subplot(gs[r, c])
        ax.set_axis_off()
        ax.set_facecolor("white")

        # background card in figure coords for the subplot cell
        bbox = ax.get_position()
        add_card(fig, bbox.x0 + 0.005, bbox.y0 + 0.005, bbox.width - 0.01, bbox.height - 0.01)

        left_smi = pair["left_smiles"]
        right_smi = pair["right_smiles"]
        left_mol = Chem.MolFromSmiles(left_smi)
        right_mol = Chem.MolFromSmiles(right_smi)
        if left_mol is None or right_mol is None:
            continue

        l_common, r_common = get_common_sets(left_mol, right_mol)
        l_diff = set(range(left_mol.GetNumAtoms())) - l_common
        r_diff = set(range(right_mol.GetNumAtoms())) - r_common

        # Render two panels and place them inside the card.
        left_img = maybe_draw(left_smi, l_common, l_diff, use_pymol_for_peptides, peptide_heavy_atom_threshold, 560, 290)
        right_img = maybe_draw(right_smi, r_common, r_diff, use_pymol_for_peptides, peptide_heavy_atom_threshold, 560, 290)

        inner = ax.inset_axes([0.02, 0.18, 0.96, 0.68])
        inner.set_axis_off()
        inner_left = inner.inset_axes([0.00, 0.03, 0.44, 0.92])
        inner_mid = inner.inset_axes([0.46, 0.34, 0.08, 0.28])
        inner_right = inner.inset_axes([0.56, 0.03, 0.44, 0.92])
        for a in (inner_left, inner_mid, inner_right):
            a.set_axis_off()
            a.set_facecolor("white")

        inner_left.imshow(left_img)
        inner_right.imshow(right_img)
        inner_mid.add_patch(
            FancyArrowPatch(
                (0.1, 0.5),
                (0.9, 0.5),
                arrowstyle="-|>",
                mutation_scale=18,
                linewidth=1.8,
                color="#6b6b6b",
                transform=inner_mid.transAxes,
            )
        )
        inner_mid.text(0.5, 0.70, f"Δp={pair['delta_p']:+.3f}", fontsize=10.5, ha="center", va="bottom", color="#333333", transform=inner_mid.transAxes)
        inner_mid.text(0.5, 0.24, f"sim={pair['similarity']:.2f}", fontsize=9.5, ha="center", va="top", color="#7a7a7a", transform=inner_mid.transAxes)

        ax.text(0.03, 0.97, f"({idx+1})", transform=ax.transAxes, fontsize=15, fontweight="bold", ha="left", va="top", color="#222222")
        ax.text(0.97, 0.97, "BBB+ → BBB-", transform=ax.transAxes, fontsize=10.5, ha="right", va="top", color="#5a5a5a")

    os.makedirs(os.path.dirname(out_svg), exist_ok=True)
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Render a clean visual CF gallery as SVG")
    parser.add_argument(
        "--csv",
        default="/home/shenxin/LiBP/results/interpretability_suite_v2/B_counterfactual_case_studies_low_sim/counterfactual_cases.csv",
    )
    parser.add_argument(
        "--out_svg",
        default="/home/shenxin/LiBP/results/interpretability_suite_v2/B_counterfactual_case_studies_low_sim/cf_gallery_visual_only.svg",
    )
    parser.add_argument("--use_pymol_for_peptides", action="store_true")
    parser.add_argument("--require_pymol", action="store_true")
    parser.add_argument("--peptide_heavy_atom_threshold", type=int, default=120)
    args = parser.parse_args()

    if args.require_pymol and importlib.util.find_spec("pymol2") is None:
        raise RuntimeError("PyMOL is required but pymol2 is not importable in this environment.")

    df = pd.read_csv(args.csv)
    if df.empty:
        raise RuntimeError("CSV is empty")

    pairs = build_pairs(df)
    plot_gallery(pairs, args.out_svg, args.use_pymol_for_peptides, args.peptide_heavy_atom_threshold)
    print(f"Wrote: {args.out_svg}")


if __name__ == "__main__":
    main()
