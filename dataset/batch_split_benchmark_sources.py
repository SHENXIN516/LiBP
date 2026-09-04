#!/usr/bin/env python3
"""Batch classify benchmark files against a reference CSV and split into two outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

CANDIDATE_SEQ_COLUMNS = ["sequence", "SMILES", "smiles", "canonical_smiles", "mol"]


def detect_sequence_column(df: pd.DataFrame, file_name: str) -> str:
    for col in CANDIDATE_SEQ_COLUMNS:
        if col in df.columns:
            return col
    raise ValueError(
        f"Could not find sequence/SMILES column in {file_name}. "
        f"Available columns: {list(df.columns)}"
    )


def normalize_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def process_one(benchmark_path: Path, reference_set: set[str], source_col: str) -> tuple[int, int, int]:
    df = pd.read_csv(benchmark_path)
    seq_col = detect_sequence_column(df, str(benchmark_path))

    matched = normalize_series(df[seq_col]).isin(reference_set)
    df[source_col] = matched.map({True: "from_reference_csv", False: "not_from_reference_csv"})

    with_source_path = benchmark_path.with_name(benchmark_path.stem + "_with_source.csv")
    from_path = benchmark_path.with_name(benchmark_path.stem + "_from_reference.csv")
    not_from_path = benchmark_path.with_name(benchmark_path.stem + "_not_from_reference.csv")

    df.to_csv(with_source_path, index=False)
    df[df[source_col] == "from_reference_csv"].to_csv(from_path, index=False)
    df[df[source_col] == "not_from_reference_csv"].to_csv(not_from_path, index=False)

    total = len(df)
    n_from = int(matched.sum())
    n_not = total - n_from
    return total, n_from, n_not


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch process benchmark_0..N against a reference CSV")
    parser.add_argument("--dataset-dir", required=True, help="Directory containing benchmark_*.csv")
    parser.add_argument("--reference", required=True, help="Reference CSV path")
    parser.add_argument("--start", type=int, default=0, help="Start benchmark index (inclusive)")
    parser.add_argument("--end", type=int, default=9, help="End benchmark index (inclusive)")
    parser.add_argument("--source-col", default="source_flag", help="Output source column name")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    reference_path = Path(args.reference)

    reference_df = pd.read_csv(reference_path)
    ref_seq_col = detect_sequence_column(reference_df, str(reference_path))
    reference_set = set(normalize_series(reference_df[ref_seq_col]))

    grand_total = 0
    grand_from = 0
    grand_not = 0

    for i in range(args.start, args.end + 1):
        benchmark_path = dataset_dir / f"benchmark_{i}.csv"
        if not benchmark_path.exists():
            print(f"[skip] missing: {benchmark_path}")
            continue

        total, n_from, n_not = process_one(benchmark_path, reference_set, args.source_col)
        grand_total += total
        grand_from += n_from
        grand_not += n_not
        print(f"benchmark_{i}: total={total}, from_reference={n_from}, not_from_reference={n_not}")

    print("----")
    print(f"summary: total={grand_total}, from_reference={grand_from}, not_from_reference={grand_not}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
