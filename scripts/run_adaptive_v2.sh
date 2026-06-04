#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
mkdir -p /home/arooba/compute-aware-vit-variant-c/scripts/logs
cd /home/arooba/compute-aware-vit-variant-c
nohup conda run -n ai_assisted_env python -u train_adaptive_v2.py \
  --data_dir /home/arooba/compute-aware-vit-thesis/data/ \
  --out_dir outputs/adaptive_v2 \
  --budget_weight 0.1 \
  --lambda_warmup_epochs 5 \
  --tau_init 3.0 \
  --tau_final 0.5 \
  > scripts/logs/adaptive_v2.out 2>&1 &
echo "started PID $!"
