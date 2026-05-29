#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
mkdir -p /home/arooba/compute-aware-vit-variant-c/scripts/logs
cd /home/arooba/compute-aware-vit-variant-c
nohup conda run -n ai_assisted_env python -u train_baseline.py \
  > scripts/logs/baseline.out 2>&1 &
echo "started PID $!"
