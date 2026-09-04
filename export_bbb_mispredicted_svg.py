from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw


def _load_molecules(frame: pd.DataFrame) -> tuple[list[Chem.Mol], list[str]]:
    mols: list[Chem.Mol] = []
    legends: list[str] = []

    for _, row in frame.iterrows():
        smiles = str(row["SMILES"]).strip()
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"[WARN] Skip invalid SMILES: {smiles}")
            continue

        inchikey = Chem.MolToInchiKey(mol)
        mols.append(mol)
        legends.append(f"InChIKey: {inchikey}")

    return mols, legends


def _save_svg(frame: pd.DataFrame, output_path: Path, title: str) -> None:
    mols, legends = _load_molecules(frame)
    if not mols:
        print(f"[WARN] No valid molecules found for {title}; skipping {output_path.name}")
        return

    svg = Draw.MolsToGridImage(
        mols,
        molsPerRow=4,
        subImgSize=(350, 285),
        legends=legends,
        useSVG=True,
    )

    if isinstance(svg, bytes):
        svg_text = svg.decode("utf-8")
    else:
        svg_text = str(svg)

    output_path.write_text(svg_text, encoding="utf-8")
    print(f"[OK] {title}: {len(mols)} molecules -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export BBB mispredicted molecules as separate FP/FN SVG grids."
    )
    parser.add_argument(
        "--input",
        default="bbb_mispredicted.csv",
        help="Input CSV file with BBB mispredictions.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory where SVG files will be written.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    fp_df = df[(df["True_BBB"] == "BBB-") & (df["Pred_BBB_Plus"] == "BBB+")].copy()
    fn_df = df[(df["True_BBB"] == "BBB+") & (df["Pred_BBB_Minus"] == "BBB-")].copy()

    _save_svg(fp_df, output_dir / "bbb_mispredicted_fp.svg", "False Positive")
    _save_svg(fn_df, output_dir / "bbb_mispredicted_fn.svg", "False Negative")


if __name__ == "__main__":
    main()