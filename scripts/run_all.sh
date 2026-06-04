#!/bin/bash
# Run all training jobs sequentially (each waits for the previous to finish).
set -e
export CUDA_VISIBLE_DEVICES=1
mkdir -p /home/arooba/compute-aware-vit-variant-c/scripts/logs
cd /home/arooba/compute-aware-vit-variant-c

echo "=== [1/4] Baseline ==="
conda run -n ai_assisted_env python -u train_baseline.py \
  --data_dir /home/arooba/compute-aware-vit-thesis/data/ \
  --out_dir outputs/baseline \
  2>&1 | tee scripts/logs/baseline.out

echo "=== [2/4] Static 49 tokens (25%) ==="
conda run -n ai_assisted_env python -u train_static.py \
  --keep_tokens 49 \
  --data_dir /home/arooba/compute-aware-vit-thesis/data/ \
  --out_dir outputs/static_49 \
  2>&1 | tee scripts/logs/static_49.out

echo "=== [3/4] Static 98 tokens (50%) ==="
conda run -n ai_assisted_env python -u train_static.py \
  --keep_tokens 98 \
  --data_dir /home/arooba/compute-aware-vit-thesis/data/ \
  --out_dir outputs/static_98 \
  2>&1 | tee scripts/logs/static_98.out

echo "=== [4/5] Adaptive v1 (budget levels: 49/98/196) ==="
conda run -n ai_assisted_env python -u train_adaptive.py \
  --data_dir /home/arooba/compute-aware-vit-thesis/data/ \
  --out_dir outputs/adaptive \
  --budget_weight 0.1 \
  2>&1 | tee scripts/logs/adaptive.out

echo "=== [5/5] Adaptive v2 (soft blending fix, lambda warmup, tau annealing) ==="
conda run -n ai_assisted_env python -u train_adaptive_v2.py \
  --data_dir /home/arooba/compute-aware-vit-thesis/data/ \
  --out_dir outputs/adaptive_v2 \
  --budget_weight 0.1 \
  --lambda_warmup_epochs 5 \
  --tau_init 3.0 \
  --tau_final 0.5 \
  2>&1 | tee scripts/logs/adaptive_v2.out

echo "=== All jobs complete ==="
