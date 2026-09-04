#!/usr/bin/env bash
set -euo pipefail

python /home/shenxin/LiBP/scripts/interpretability_suite.py \
  --input_csv /home/shenxin/LiBP/dataset/benchmark_0.csv \
  --label_col label \
  --ckpt /home/shenxin/LiBP/ckpt/02best_model_epoch154_acc0.8693.pt \
  --out_dir /home/shenxin/LiBP/results/interpretability_suite \
  --num_pairs 3 \
  --top_k 20

echo "All outputs saved in /home/shenxin/LiBP/results/interpretability_suite"
