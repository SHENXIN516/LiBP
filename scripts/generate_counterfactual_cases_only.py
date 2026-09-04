#!/usr/bin/env python3
"""Generate only counterfactual cases CSV (no SHAP / no cross-modality).

This is a lightweight helper to populate enough CF pairs for panel plotting.
"""

from __future__ import annotations

import argparse
import os
import sys
import importlib.util


def load_suite_module(path: str):
    spec = importlib.util.spec_from_file_location("interpretability_suite_v2", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    parser = argparse.ArgumentParser(description="Generate CF cases only")
    parser.add_argument("--dataset_dir", default="/home/shenxin/LiBP/dataset")
    parser.add_argument("--benchmark_start", type=int, default=0)
    parser.add_argument("--benchmark_end", type=int, default=9)
    parser.add_argument("--smiles_col", default="")
    parser.add_argument("--label_col", default="label")
    parser.add_argument("--ckpt", default="/home/shenxin/LiBP/ckpt/02best_model_epoch154_acc0.8693.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--out_dir",
        default="/home/shenxin/LiBP/results/interpretability_suite_v2/B_counterfactual_case_studies",
    )
    parser.add_argument("--num_cf_cases", type=int, default=4)
    parser.add_argument("--cf_min_similarity", type=float, default=0.75)
    parser.add_argument("--cf_max_atom_diff", type=int, default=8)
    parser.add_argument("--cf_prefer_lower_similarity", action="store_true")
    args = parser.parse_args()

    suite_path = "/home/shenxin/LiBP/scripts/interpretability_suite_v2.py"
    suite = load_suite_module(suite_path)

    all_rows = []
    for b in range(args.benchmark_start, args.benchmark_end + 1):
        path = os.path.join(args.dataset_dir, f"benchmark_{b}.csv")
        if not os.path.exists(path):
            continue
        rs = suite.parse_one_csv(path, benchmark_id=b, smiles_col=args.smiles_col or None, label_col=args.label_col)
        all_rows.extend(rs)

    if not all_rows:
        raise RuntimeError("No benchmark rows loaded.")

    runner = suite.ModelRunner(args.ckpt, args.device)
    probs = suite.predict_probs(all_rows, runner)
    os.makedirs(args.out_dir, exist_ok=True)
    suite.run_counterfactual_case_studies(
        all_rows,
        probs,
        runner,
        args.out_dir,
        n_cases=args.num_cf_cases,
        min_similarity=args.cf_min_similarity,
        max_atom_diff=args.cf_max_atom_diff,
        prefer_lower_similarity=args.cf_prefer_lower_similarity,
    )
    print(f"Wrote CF CSV to: {os.path.join(args.out_dir, 'counterfactual_cases.csv')}")


if __name__ == "__main__":
    main()
