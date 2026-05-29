#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
mkdir -p /home/arooba/compute-aware-vit-variant-c/scripts/logs
cd /home/arooba/compute-aware-vit-variant-c
nohup conda run -n ai_assisted_env python -u train_static_pruned.py --depth 4 \
  > scripts/logs/static_pruned_4.out 2>&1 &
echo "started PID $!"
