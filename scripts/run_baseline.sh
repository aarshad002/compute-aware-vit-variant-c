#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
mkdir -p /home/arooba/compute-aware-vit-variant-c/scripts/logs
cd /home/arooba/compute-aware-vit-variant-c
nohup conda run -n ai_assisted_env python -u train_baseline.py \
  --data_dir /home/arooba/compute-aware-vit-thesis/data/ \
  --out_dir outputs/baseline \
  > scripts/logs/baseline.out 2>&1 &
echo "started PID $!"
