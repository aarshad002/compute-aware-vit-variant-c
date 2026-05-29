#!/bin/bash
# Run all training jobs sequentially, then evaluate.
# Each job is launched with nohup and the script waits for it to finish
# before starting the next.
set -e

export CUDA_VISIBLE_DEVICES=1
mkdir -p /home/arooba/compute-aware-vit-variant-c/scripts/logs
cd /home/arooba/compute-aware-vit-variant-c

echo "============================================================"
echo " Compute-Aware ViT — Sequential Training Pipeline"
echo "============================================================"

# ---- 1. Full DeiT-tiny baseline ----
echo ""
echo "[1/5] Training baseline (12 blocks) ..."
nohup conda run -n ai_assisted_env python -u train_baseline.py \
  > scripts/logs/baseline.out 2>&1 &
PID=$!
echo "  PID $PID — tailing logs/baseline.out"
wait $PID
echo "  Baseline complete."

# ---- 2. Static pruned depth=4 ----
echo ""
echo "[2/5] Training static pruned depth=4 ..."
nohup conda run -n ai_assisted_env python -u train_static_pruned.py --depth 4 \
  > scripts/logs/static_pruned_4.out 2>&1 &
PID=$!
echo "  PID $PID — tailing logs/static_pruned_4.out"
wait $PID
echo "  Static pruned 4 complete."

# ---- 3. Static pruned depth=8 ----
echo ""
echo "[3/5] Training static pruned depth=8 ..."
nohup conda run -n ai_assisted_env python -u train_static_pruned.py --depth 8 \
  > scripts/logs/static_pruned_8.out 2>&1 &
PID=$!
echo "  PID $PID — tailing logs/static_pruned_8.out"
wait $PID
echo "  Static pruned 8 complete."

# ---- 4. Adaptive multi-exit model ----
echo ""
echo "[4/5] Training adaptive multi-exit model ..."
nohup conda run -n ai_assisted_env python -u train_adaptive.py \
  > scripts/logs/adaptive.out 2>&1 &
PID=$!
echo "  PID $PID — tailing logs/adaptive.out"
wait $PID
echo "  Adaptive training complete."

# ---- 5. Threshold sweep evaluation ----
echo ""
echo "[5/5] Evaluating adaptive model (threshold sweep) ..."
nohup conda run -n ai_assisted_env python -u eval_adaptive.py \
  > scripts/logs/eval_adaptive.out 2>&1 &
PID=$!
echo "  PID $PID — tailing logs/eval_adaptive.out"
wait $PID
echo "  Evaluation complete."

echo ""
echo "============================================================"
echo " All jobs finished."
echo " Metrics: results/{baseline,static_4,static_8,adaptive}/"
echo " Eval:    results/adaptive_eval/threshold_sweep.json"
echo "============================================================"
