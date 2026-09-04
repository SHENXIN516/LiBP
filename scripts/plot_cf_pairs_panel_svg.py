#!/usr/bin/env python3
"""Plot CF pairs as a publication-style two-panel SVG figure.

Left panel: BBB permeable (pred=1)
Right panel: BBB impermeable (pred=0)
"""

from __future__ import annotations

import argparse
import io
import os
from typing import Dict, List, Optional, Sequence, Tuple
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw, rdFMCS, AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import RDLogger
from rdkit.Geometry import Point2D
import importlib.util

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


def draw_mol_with_box(
    smiles: str,
    common_atoms: set,
    diff_atoms: set,
    width: int = 560,
    height: int = 260,
) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.ones((height, width, 3), dtype=np.uint8) * 255

    # Work on a copy with 2D coords to ensure deterministic rendering.
    mol = Chem.Mol(mol)
    Draw.rdDepictor.Compute2DCoords(mol)

    common_color = (0.95, 0.70, 0.70)
    diff_color = (0.15, 0.85, 0.90)

    highlight_atoms = sorted(set(common_atoms) | set(diff_atoms))
    atom_colors = {}
    atom_radii = {}

    for i in common_atoms:
        atom_colors[int(i)] = common_color
        atom_radii[int(i)] = 0.26
    for i in diff_atoms:
        atom_colors[int(i)] = diff_color
        atom_radii[int(i)] = 0.30

    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    opts = drawer.drawOptions()
    opts.clearBackground = True
    try:
        opts.backgroundColour = (1.0, 1.0, 1.0)
    except Exception:
        pass
    opts.padding = 0.04
    opts.bondLineWidth = 2

    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=atom_colors,
        highlightAtomRadii=atom_radii,
    )

    # Draw a rectangle around changed fragment (diff atoms).
    if diff_atoms:
        pts = [drawer.GetDrawCoords(int(i)) for i in diff_atoms]
        min_x = min(float(p.x) for p in pts)
        max_x = max(float(p.x) for p in pts)
        min_y = min(float(p.y) for p in pts)
        max_y = max(float(p.y) for p in pts)

        pad = 18.0
        p1 = Point2D(min_x - pad, min_y - pad)
        p2 = Point2D(max_x + pad, max_y + pad)
        drawer.SetColour((0.0, 0.0, 0.0))
        drawer.SetLineWidth(1.2)
        drawer.DrawRect(p1, p2)

    drawer.FinishDrawing()
    png = drawer.GetDrawingText()
    img = Image.open(io.BytesIO(png)).convert("RGB")
    return np.asarray(img)


def draw_mol_with_pymol(smiles: str, width: int = 560, height: int = 260) -> Optional[np.ndarray]:
    """Render molecule with PyMOL as fallback for very large/peptidic molecules.

    Returns None if PyMOL is unavailable or rendering fails.
    """
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
        AllChem.UFFOptimizeMolecule(mol, maxIters=100)
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
                cmd.set("stick_radius", 0.16)
                cmd.bg_color("white")
                cmd.set("ray_opaque_background", 0)
                cmd.util.cbag("mol")
                cmd.orient("mol")
                cmd.zoom("mol", 2.0)
                cmd.png(png_path, width=width, height=height, dpi=300, ray=1)
        except Exception:
            return None

        if not os.path.exists(png_path):
            return None
        return np.asarray(Image.open(png_path).convert("RGB"))


    def draw_mol_with_coords(smiles: str, width: int = 1200, height: int = 520) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        """Draw molecule and return image plus atom-draw coordinates (pixel space).

        Returns (img_array, coords) where coords[i] is (x,y) in pixels for atom i.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.ones((height, width, 3), dtype=np.uint8) * 255, []

        mol = Chem.Mol(mol)
        Draw.rdDepictor.Compute2DCoords(mol)

        drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
        opts = drawer.drawOptions()
        opts.clearBackground = True
        try:
            opts.backgroundColour = (1.0, 1.0, 1.0)
        except Exception:
            pass
        drawer.DrawMolecule(mol)

        # collect atom draw coords
        coords: List[Tuple[float, float]] = []
        for i in range(mol.GetNumAtoms()):
            try:
                p = drawer.GetDrawCoords(i)
                coords.append((float(p.x), float(p.y)))
            except Exception:
                coords.append((0.0, 0.0))

        drawer.FinishDrawing()
        png = drawer.GetDrawingText()
        img = Image.open(io.BytesIO(png)).convert("RGB")
        return np.asarray(img), coords


def roman(n: int) -> str:
    vals = [
        (10, "x"),
        (9, "ix"),
        (5, "v"),
        (4, "iv"),
        (1, "i"),
    ]
    out = []
    x = n
    for v, s in vals:
        while x >= v:
            out.append(s)
            x -= v
    return "".join(out)


def build_pairs(df: pd.DataFrame) -> List[Dict[str, float]]:
    pairs: List[Dict[str, float]] = []
    for _, r in df.sort_values("case_id").iterrows():
        seed_smi = str(r["seed_smiles"])
        cf_smi = str(r["counterfactual_smiles"])
        seed_pred = int(r["seed_pred"])
        cf_pred = int(r["counterfactual_pred"])
        seed_p = float(r["seed_prob_bbb_plus"])
        cf_p = float(r["counterfactual_prob_bbb_plus"])

        if seed_pred == 1 and cf_pred == 0:
            left, right = seed_smi, cf_smi
            left_p, right_p = seed_p, cf_p
        elif seed_pred == 0 and cf_pred == 1:
            left, right = cf_smi, seed_smi
            left_p, right_p = cf_p, seed_p
        else:
            # Fallback: use probability ordering if both are same side by predicted label.
            if seed_p >= cf_p:
                left, right = seed_smi, cf_smi
                left_p, right_p = seed_p, cf_p
            else:
                left, right = cf_smi, seed_smi
                left_p, right_p = cf_p, seed_p

        pair = {
            "left_smiles": left,
            "right_smiles": right,
            "left_prob": left_p,
            "right_prob": right_p,
            "delta_p": left_p - right_p,
        }
        # include chemical-property deltas if present in the dataframe
        if "delta_logp" in r.index and not pd.isna(r["delta_logp"]):
            csv_dl = float(r["delta_logp"])  # counterfactual - seed
            # compute left_minus_right for logP depending on which side is left
            if left == str(r.get("seed_smiles", "")):
                pair["delta_logp"] = -csv_dl
            else:
                pair["delta_logp"] = csv_dl
        if "delta_tpsa" in r.index and not pd.isna(r["delta_tpsa"]):
            csv_dt = float(r["delta_tpsa"])  # counterfactual - seed
            pair["delta_tpsa"] = -csv_dt if left == str(r.get("seed_smiles", "")) else csv_dt
        if "delta_hbd" in r.index and not pd.isna(r["delta_hbd"]):
            csv_dh = float(r["delta_hbd"])  # counterfactual - seed
            pair["delta_hbd"] = -csv_dh if left == str(r.get("seed_smiles", "")) else csv_dh
        if "delta_hba" in r.index and not pd.isna(r["delta_hba"]):
            csv_ha = float(r["delta_hba"])  # counterfactual - seed
            pair["delta_hba"] = -csv_ha if left == str(r.get("seed_smiles", "")) else csv_ha

        pairs.append(pair)
    return pairs


def maybe_draw(
    smiles: str,
    common_atoms: set,
    diff_atoms: set,
    use_pymol_for_peptides: bool,
    peptide_heavy_atom_threshold: int,
    width: int,
    height: int,
) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.ones((height, width, 3), dtype=np.uint8) * 255

    if use_pymol_for_peptides and mol.GetNumHeavyAtoms() >= peptide_heavy_atom_threshold:
        py_img = draw_mol_with_pymol(smiles, width=width, height=height)
        if py_img is not None:
            return py_img

    return draw_mol_with_box(smiles, common_atoms, diff_atoms, width=width, height=height)


def plot_panel(
    pairs: Sequence[Dict[str, float]],
    out_svg: str,
    use_pymol_for_peptides: bool,
    peptide_heavy_atom_threshold: int,
):
    # Publication layout: 2 rows x 4 cols, left 4 cases as (i-iv), right 4 cases as (v-viii).
    if len(pairs) < 4:
        raise RuntimeError(
            f"Need at least 4 CF pairs to build 8-case panel; got {len(pairs)}. "
            "Please regenerate counterfactual_cases.csv with --num_cf_cases >= 4."
        )

    pairs = list(pairs)[:4]
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(18.5, 8.3))
    fig.patch.set_facecolor('white')

    for i, pair in enumerate(pairs, start=1):
        left_smi = str(pair["left_smiles"])
        right_smi = str(pair["right_smiles"])
        delta_p = float(pair["delta_p"])

        lmol = Chem.MolFromSmiles(left_smi)
        rmol = Chem.MolFromSmiles(right_smi)
        if lmol is None or rmol is None:
            continue

        l_common, r_common = get_common_sets(lmol, rmol)
        l_diff = set(range(lmol.GetNumAtoms())) - l_common
        r_diff = set(range(rmol.GetNumAtoms())) - r_common

        left_img = maybe_draw(
            left_smi,
            l_common,
            l_diff,
            use_pymol_for_peptides=use_pymol_for_peptides,
            peptide_heavy_atom_threshold=peptide_heavy_atom_threshold,
            width=620,
            height=300,
        )
        right_img = maybe_draw(
            right_smi,
            r_common,
            r_diff,
            use_pymol_for_peptides=use_pymol_for_peptides,
            peptide_heavy_atom_threshold=peptide_heavy_atom_threshold,
            width=620,
            height=300,
        )

        row = (i - 1) // 2
        col = (i - 1) % 2
        ax_l = axes[row, col]
        ax_r = axes[row, col + 2]

        ax_l.imshow(left_img)
        ax_r.imshow(right_img)
        ax_l.set_facecolor('white')
        ax_r.set_facecolor('white')
        ax_l.axis("off")
        ax_r.axis("off")

        left_id = roman(i)
        right_id = roman(i + 4)

        ax_l.text(0.03, 0.02, f"({left_id})", transform=ax_l.transAxes, fontsize=18, va="bottom", ha="left")
        ax_r.text(0.03, 0.02, f"({right_id})", transform=ax_r.transAxes, fontsize=18, va="bottom", ha="left")

        # Add delta-p beside each molecule (strictly paired value, opposite sign on right).
        ax_l.text(
            0.98,
            0.04,
            f"Δp={delta_p:+.3f}",
            transform=ax_l.transAxes,
            fontsize=12,
            va="bottom",
            ha="right",
            color="#0a3d62",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 1.2},
        )
        ax_r.text(
            0.98,
            0.04,
            f"Δp={-delta_p:+.3f}",
            transform=ax_r.transAxes,
            fontsize=12,
            va="bottom",
            ha="right",
            color="#7f0000",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 1.2},
        )

    fig.text(0.25, 0.985, "BBB permeable", ha="center", va="top", fontsize=26)
    fig.text(0.75, 0.985, "BBB impermeable", ha="center", va="top", fontsize=26)
    # lighter separator line (not black)
    fig.add_artist(plt.Line2D([0.5, 0.5], [0.06, 0.95], transform=fig.transFigure, color="#bdbdbd", linewidth=1.6))

    plt.tight_layout(rect=[0.02, 0.04, 0.98, 0.95])
    os.makedirs(os.path.dirname(out_svg), exist_ok=True)
    fig.savefig(out_svg, format="svg", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot CF pairs panel as SVG")
    parser.add_argument(
        "--csv",
        default="/home/shenxin/LiBP/results/interpretability_suite_v2/B_counterfactual_case_studies/counterfactual_cases.csv",
    )
    parser.add_argument(
        "--out_svg",
        default="/home/shenxin/LiBP/results/interpretability_suite_v2/B_counterfactual_case_studies/cf_pairs_panel_style_8cases.svg",
    )
    parser.add_argument("--max_cases", type=int, default=4, help="Number of CF pairs to use; 4 pairs -> 8 shown cases")
    parser.add_argument(
        "--use_pymol_for_peptides",
        action="store_true",
        help="Use PyMOL rendering for very large molecules to avoid 2D overlap",
    )
    parser.add_argument("--peptide_heavy_atom_threshold", type=int, default=120)
    parser.add_argument("--preserve_three", action="store_true", help="Preserve first three CF cases and render custom 3-case layout")
    parser.add_argument("--require_pymol", action="store_true", help="If set, fail if pymol2 is not importable (force PyMOL usage)")
    args = parser.parse_args()

    # If user requires PyMOL, ensure pymol2 is importable before proceeding.
    if args.require_pymol:
        found = importlib.util.find_spec("pymol2") is not None
        if not found:
            raise RuntimeError("PyMOL (pymol2) is required but not available in the current environment. Install pymol-open-source from conda-forge and retry.")

    df = pd.read_csv(args.csv)
    if df.empty:
        raise RuntimeError("counterfactual_cases.csv is empty")

    df = df.copy()
    if args.preserve_three and len(df) >= 3:
        # Use the first three cases exactly as provided by the user
        df3 = df.head(3).copy()
        pairs3 = build_pairs(df3)
        # draw custom 3-case layout: 2 small molecules on top row, peptide centered below
        def plot_three_case_layout(pairs, out_svg, use_pymol_for_peptides, peptide_heavy_atom_threshold):
            # Expect pairs to be list of 3 dicts
            if len(pairs) < 3:
                raise RuntimeError("Need at least 3 CF cases for preserve_three layout")
            left_smi = pairs[0]["left_smiles"]
            right_smi = pairs[1]["left_smiles"] if pairs[1]["left_smiles"] else pairs[1]["right_smiles"]
            center_smi = pairs[2]["left_smiles"]
            delta0 = float(pairs[0]["delta_p"])
            delta1 = float(pairs[1]["delta_p"])
            delta2 = float(pairs[2]["delta_p"])

            fig = plt.figure(figsize=(14, 10))
            fig.patch.set_facecolor('white')
            # Top row: two small molecules
            ax1 = plt.subplot2grid((3, 4), (0, 0), colspan=2, rowspan=1)
            ax2 = plt.subplot2grid((3, 4), (0, 2), colspan=2, rowspan=1)
            # Bottom row: centered big peptide across middle columns
            ax3 = plt.subplot2grid((3, 4), (1, 1), colspan=2, rowspan=2)

            # draw images
            lmol = Chem.MolFromSmiles(left_smi)
            rmol = Chem.MolFromSmiles(right_smi)
            cmol = Chem.MolFromSmiles(center_smi)
            l_common, _ = get_common_sets(lmol, Chem.MolFromSmiles(pairs[0]["right_smiles"]) if pairs[0].get("right_smiles") else lmol)
            r_common, _ = get_common_sets(rmol, Chem.MolFromSmiles(pairs[1]["right_smiles"]) if pairs[1].get("right_smiles") else rmol)
            c_common, _ = get_common_sets(cmol, Chem.MolFromSmiles(pairs[2].get("right_smiles", center_smi)))

            img1 = maybe_draw(left_smi, l_common, set(range(lmol.GetNumAtoms())) - l_common, use_pymol_for_peptides, peptide_heavy_atom_threshold, 560, 300)
            img2 = maybe_draw(right_smi, r_common, set(range(rmol.GetNumAtoms())) - r_common, use_pymol_for_peptides, peptide_heavy_atom_threshold, 560, 300)
            # For the center peptide, if it's large, create a composite: small full image + zoomed crop
            if cmol.GetNumHeavyAtoms() >= peptide_heavy_atom_threshold:
                # full small thumbnail
                thumb_h = 360
                thumb_w = 560
                thumb = maybe_draw(center_smi, c_common, set(range(cmol.GetNumAtoms())) - c_common, use_pymol_for_peptides, peptide_heavy_atom_threshold, thumb_w, thumb_h)

                # large render + atom coords for cropping
                big_w, big_h = 1400, 700
                big_img, coords = draw_mol_with_coords(center_smi, width=big_w, height=big_h)

                # find bbox around diff atoms using coords
                diff_idxs = sorted(list(set(range(cmol.GetNumAtoms())) - c_common))
                if diff_idxs and coords:
                    xs = [coords[i][0] for i in diff_idxs if i < len(coords)]
                    ys = [coords[i][1] for i in diff_idxs if i < len(coords)]
                    if xs and ys:
                        min_x, max_x = max(0, int(min(xs) - 40)), min(big_w, int(max(xs) + 40))
                        min_y, max_y = max(0, int(min(ys) - 40)), min(big_h, int(max(ys) + 40))
                        crop = big_img[min_y:max_y, min_x:max_x]
                    else:
                        crop = big_img
                else:
                    crop = big_img

                # assemble composite canvas
                canvas_h = 520
                canvas_w = 1200
                canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
                # paste thumbnail on left
                timg = Image.fromarray(thumb)
                timg = timg.resize((int(canvas_w * 0.36), int(canvas_h * 0.92)), Image.LANCZOS)
                canvas.paste(timg, (20, int((canvas_h - timg.size[1]) / 2)))

                # paste crop on right (fit to remaining area)
                cimg = Image.fromarray(crop)
                max_right_w = canvas_w - (20 + timg.size[0] + 20)
                max_right_h = canvas_h - 40
                cimg.thumbnail((max_right_w, max_right_h), Image.LANCZOS)
                cx = 20 + timg.size[0] + 20
                cy = int((canvas_h - cimg.size[1]) / 2)
                canvas.paste(cimg, (cx, cy))

                img3 = np.asarray(canvas)
            else:
                img3 = maybe_draw(center_smi, c_common, set(range(cmol.GetNumAtoms())) - c_common, use_pymol_for_peptides, peptide_heavy_atom_threshold, 1200, 520)

            ax1.imshow(img1); ax1.axis('off'); ax1.set_facecolor('white')
            ax2.imshow(img2); ax2.axis('off'); ax2.set_facecolor('white')
            ax3.imshow(img3); ax3.axis('off'); ax3.set_facecolor('white')

            ax1.text(0.03, 0.03, f"(i)", transform=ax1.transAxes, fontsize=18, va="bottom", ha="left")
            ax2.text(0.03, 0.03, f"(ii)", transform=ax2.transAxes, fontsize=18, va="bottom", ha="left")
            ax3.text(0.03, 0.03, f"(iii)", transform=ax3.transAxes, fontsize=18, va="bottom", ha="left")

            # For the second case, prefer labeling with ΔlogP if available
            d0 = delta0
            d2 = delta2
            d1_label = None
            if "delta_logp" in pairs[1]:
                d1_label = pairs[1]["delta_logp"]
            else:
                d1_label = delta1

            ax1.text(0.98, 0.04, f"Δp={d0:+.3f}", transform=ax1.transAxes, fontsize=12, ha='right', va='bottom', color='#0a3d62', bbox={'facecolor':'white','alpha':0.85,'edgecolor':'none','pad':1.2})
            ax2.text(0.98, 0.04, f"ΔlogP={d1_label:+.3f}", transform=ax2.transAxes, fontsize=12, ha='right', va='bottom', color='#0a3d62', bbox={'facecolor':'white','alpha':0.85,'edgecolor':'none','pad':1.2})
            ax3.text(0.98, 0.04, f"Δp={d2:+.3f}", transform=ax3.transAxes, fontsize=12, ha='right', va='bottom', color='#7f0000', bbox={'facecolor':'white','alpha':0.85,'edgecolor':'none','pad':1.2})

            fig.text(0.25, 0.985, "BBB permeable", ha="center", va="top", fontsize=26)
            fig.text(0.75, 0.985, "BBB impermeable", ha="center", va="top", fontsize=26)
            fig.savefig(out_svg, format='svg', dpi=300, bbox_inches='tight')
            plt.close(fig)

        out_svg3 = args.out_svg
        plot_three_case_layout(pairs3, out_svg3, args.use_pymol_for_peptides, args.peptide_heavy_atom_threshold)
        print(f"Wrote: {out_svg3}")
        return

    # default full-panel behavior
    df = df.head(max(1, args.max_cases)).copy()
    pairs = build_pairs(df)
    plot_panel(
        pairs,
        args.out_svg,
        use_pymol_for_peptides=args.use_pymol_for_peptides,
        peptide_heavy_atom_threshold=args.peptide_heavy_atom_threshold,
    )
    print(f"Wrote: {args.out_svg}")


if __name__ == "__main__":
    main()
