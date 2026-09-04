#!/usr/bin/env python3
"""Classify whether each benchmark record comes from a reference CSV.

Usage:
    python classify_benchmark_source.py \
        --benchmark benchmark_0.csv \
        --reference bbbp_cleaned4.csv \
        --output benchmark_0_with_source.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


CANDIDATE_SEQ_COLUMNS = [
    "sequence",
    "SMILES",
    "smiles",
    "canonical_smiles",
    "mol",
]


def detect_sequence_column(df: pd.DataFrame, file_name: str) -> str:
    for col in CANDIDATE_SEQ_COLUMNS:
        if col in df.columns:
            return col
    raise ValueError(
        f"Could not find sequence/SMILES column in {file_name}. "
        f"Available columns: {list(df.columns)}"
    )


def normalize_series(s: pd.Series) -> pd.Series:
    # Normalize textual IDs to reduce mismatch from spaces or missing values.
    return s.astype(str).str.strip()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mark benchmark rows as from-reference or not-from-reference."
    )
    parser.add_argument("--benchmark", required=True, help="Path to benchmark CSV")
    parser.add_argument("--reference", required=True, help="Path to reference CSV")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument(
        "--source-col",
        default="source_flag",
        help="Name of the output source column (default: source_flag)",
    )

    args = parser.parse_args()

    benchmark_path = Path(args.benchmark)
    reference_path = Path(args.reference)
    output_path = Path(args.output)

    if not benchmark_path.exists():
        print(f"Benchmark file not found: {benchmark_path}", file=sys.stderr)
        return 1
    if not reference_path.exists():
        print(f"Reference file not found: {reference_path}", file=sys.stderr)
        return 1

    benchmark_df = pd.read_csv(benchmark_path)
    reference_df = pd.read_csv(reference_path)

    benchmark_seq_col = detect_sequence_column(benchmark_df, str(benchmark_path))
    reference_seq_col = detect_sequence_column(reference_df, str(reference_path))

    benchmark_norm = normalize_series(benchmark_df[benchmark_seq_col])
    reference_set = set(normalize_series(reference_df[reference_seq_col]))

    matched = benchmark_norm.isin(reference_set)

    benchmark_df[args.source_col] = matched.map(
        {True: "from_reference_csv", False: "not_from_reference_csv"}
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_df.to_csv(output_path, index=False)

    total = len(benchmark_df)
    n_from = int(matched.sum())
    n_not = total - n_from

    print(f"Done. Output: {output_path}")
    print(f"Benchmark sequence column: {benchmark_seq_col}")
    print(f"Reference sequence column: {reference_seq_col}")
    print(f"from_reference_csv: {n_from}")
    print(f"not_from_reference_csv: {n_not}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
