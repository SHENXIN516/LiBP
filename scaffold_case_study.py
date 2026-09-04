from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def _get_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    scaff = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return scaff or ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze scaffold-level error patterns and mixed-correctness cases."
    )
    parser.add_argument(
        "--input",
        default="preds_with_correctness.csv",
        help="CSV with columns: smiles,label,prob_positive,pred,correct",
    )
    parser.add_argument(
        "--outdir",
        default=".",
        help="Directory to write analysis CSV files.",
    )
    parser.add_argument(
        "--min-mixed-size",
        type=int,
        default=3,
        help="Min samples per scaffold to be considered for mixed-case study.",
    )
    parser.add_argument(
        "--max-examples-per-scaffold",
        type=int,
        default=20,
        help="Max rows to keep per scaffold in case examples export.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    if "smiles" not in df.columns:
        raise ValueError("Input must contain a 'smiles' column.")

    df["smiles"] = df["smiles"].astype(str).str.strip()
    df["scaffold"] = df["smiles"].map(_get_scaffold)

    df["correct"] = df["correct"].astype(int)
    df["label"] = df["label"].astype(int)
    df["pred"] = df["pred"].astype(int)

    df["is_error"] = 1 - df["correct"]
    df["error_type"] = "OK"
    df.loc[(df["label"] == 1) & (df["pred"] == 0), "error_type"] = "FN"
    df.loc[(df["label"] == 0) & (df["pred"] == 1), "error_type"] = "FP"

    # Scaffold-level summary
    grouped = df.groupby("scaffold", dropna=False)
    summary = grouped.agg(
        total=("scaffold", "size"),
        errors=("is_error", "sum"),
        corrects=("correct", "sum"),
        fp=("error_type", lambda x: (x == "FP").sum()),
        fn=("error_type", lambda x: (x == "FN").sum()),
    ).reset_index()

    summary["error_rate"] = summary["errors"] / summary["total"]
    summary["has_mixed"] = (summary["errors"] > 0) & (summary["corrects"] > 0)

    summary = summary.sort_values(
        by=["error_rate", "total", "errors"], ascending=[False, False, False]
    )
    summary.to_csv(outdir / "scaffold_error_summary.csv", index=False)

    # Mixed-case scaffolds for possible activity cliffs
    mixed = summary[(summary["has_mixed"]) & (summary["total"] >= args.min_mixed_size)]
    mixed.to_csv(outdir / "scaffold_mixed_scaffolds.csv", index=False)

    # Case-study examples: keep limited rows per scaffold
    if not mixed.empty:
        mixed_scaffolds = set(mixed["scaffold"].tolist())
        case_df = df[df["scaffold"].isin(mixed_scaffolds)].copy()
        case_df["rank"] = case_df.groupby("scaffold").cumcount()
        case_df = case_df[case_df["rank"] < args.max_examples_per_scaffold]
        case_df = case_df.drop(columns=["rank"])
        case_df.to_csv(outdir / "scaffold_case_examples.csv", index=False)

    # Quick console summary
    print("=== Scaffold Case Study Summary ===")
    print(f"Total samples        : {len(df)}")
    print(f"Total scaffolds      : {summary['scaffold'].nunique()}")
    print(f"Mixed scaffolds      : {len(mixed)}")
    print(f"Output directory     : {outdir.resolve()}")
    print("Outputs:")
    print(" - scaffold_error_summary.csv")
    print(" - scaffold_mixed_scaffolds.csv")
    if not mixed.empty:
        print(" - scaffold_case_examples.csv")


if __name__ == "__main__":
    main()
