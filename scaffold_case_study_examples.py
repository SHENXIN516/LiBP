from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Scaffolds import MurckoScaffold


def _get_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    scaff = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return scaff or ""


def _legend(row: pd.Series) -> str:
    label = int(row["label"])
    pred = int(row["pred"])
    prob = float(row["prob_positive"])
    correct = int(row["correct"])
    return f"T:{label} P:{pred} Prob:{prob:.3f}"


def _draw_grid(frame: pd.DataFrame, out_path: Path, title: str) -> None:
    mols = []
    legends = []
    for _, row in frame.iterrows():
        mol = Chem.MolFromSmiles(str(row["smiles"]))
        if mol is None:
            continue
        mols.append(mol)
        legends.append(_legend(row))

    if not mols:
        print(f"[WARN] No valid molecules for {title}")
        return

    svg = Draw.MolsToGridImage(
        mols,
        molsPerRow=4,
        subImgSize=(330, 260),
        legends=legends,
        useSVG=True,
    )

    if isinstance(svg, bytes):
        out_path.write_text(svg.decode("utf-8"), encoding="utf-8")
    else:
        out_path.write_text(str(svg), encoding="utf-8")

    print(f"[OK] {title} -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export case-study grids for mixed-correctness scaffolds."
    )
    parser.add_argument(
        "--preds",
        default="preds_with_correctness.csv",
        help="CSV with columns: smiles,label,prob_positive,pred,correct",
    )
    parser.add_argument(
        "--mixed",
        default="case_study/scaffold_mixed_scaffolds.csv",
        help="CSV with mixed scaffolds summary.",
    )
    parser.add_argument(
        "--outdir",
        default="case_study",
        help="Directory for output SVG/CSV files.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=3,
        help="Number of scaffolds to export.",
    )
    parser.add_argument(
        "--max-per-scaffold",
        type=int,
        default=16,
        help="Max samples to show per scaffold.",
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Include empty scaffold in selection.",
    )
    args = parser.parse_args()

    preds_path = Path(args.preds)
    mixed_path = Path(args.mixed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(preds_path)
    df["smiles"] = df["smiles"].astype(str).str.strip()
    df["scaffold"] = df["smiles"].map(_get_scaffold)

    mixed = pd.read_csv(mixed_path)
    if not args.include_empty:
        mixed = mixed[mixed["scaffold"].astype(str).str.len() > 0]

    mixed = mixed.sort_values(
        by=["error_rate", "total", "errors"], ascending=[False, False, False]
    )
    mixed = mixed.head(args.topk)

    summary_rows = []
    for idx, row in mixed.reset_index(drop=True).iterrows():
        scaff = row["scaffold"]
        sub = df[df["scaffold"] == scaff].copy()
        sub = sub.sort_values(by=["correct", "prob_positive"], ascending=[True, False])
        sub = sub.head(args.max_per_scaffold)

        csv_path = outdir / f"case_scaffold_{idx+1}.csv"
        sub.to_csv(csv_path, index=False)

        svg_path = outdir / f"case_scaffold_{idx+1}.svg"
        _draw_grid(sub, svg_path, f"scaffold_{idx+1}")

        summary_rows.append(
            {
                "scaffold": scaff,
                "total": int(row["total"]),
                "errors": int(row["errors"]),
                "corrects": int(row["corrects"]),
                "error_rate": float(row["error_rate"]),
                "csv": csv_path.name,
                "svg": svg_path.name,
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary_path = outdir / "case_study_scaffolds_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("=== Case Study Exports ===")
    print(summary)
    print(f"Summary -> {summary_path}")


if __name__ == "__main__":
    main()
