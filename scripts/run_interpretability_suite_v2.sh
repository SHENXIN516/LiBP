#!/usr/bin/env bash
set -euo pipefail

source /home/shenxin/miniconda3/etc/profile.d/conda.sh
conda activate bbbp-split

python /home/shenxin/LiBP/scripts/interpretability_suite_v2.py \
  --dataset_dir /home/shenxin/LiBP/dataset \
  --benchmark_start 0 \
  --benchmark_end 9 \
  --label_col label \
  --ckpt /home/shenxin/LiBP/ckpt/02best_model_epoch154_acc0.8693.pt \
  --out_dir /home/shenxin/LiBP/results/interpretability_suite_v2 \
  --top_k 20 \
  --num_cf_cases 3

echo "All outputs saved in /home/shenxin/LiBP/results/interpretability_suite_v2"
