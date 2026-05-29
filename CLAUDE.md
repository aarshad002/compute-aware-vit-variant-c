# Project Context

This is a Master's thesis on compute-aware adaptive inference in Vision 
Transformers. The core problem: Vision Transformers process all image 
tokens equally regardless of image difficulty. This wastes computation 
on easy images that could be classified with far fewer tokens, while 
hard images may need the full model capacity.

Conda environment name: ai_assisted_env

# Environment Notes

Follow these exactly — do not deviate:

- Server is a Linux GPU server. Do NOT use sbatch or SLURM.
- GPU: always set export CUDA_VISIBLE_DEVICES=1
- Dataset: CIFAR-100 already downloaded at:
  /home/arooba/compute-aware-vit-thesis/data/
- Model: deit_tiny_patch16_224 from timm, pretrained=True
- Use python -u for unbuffered output
- Hyperparameters — use exactly:
    batch_size: 32
    epochs: 20
    learning_rate: 0.0001
    weight_decay: 0.0001
    seed: 42
    optimizer: Adam (not AdamW, not SGD)
- No learning rate scheduler — constant lr throughout
- nohup script pattern for each training job:
    #!/bin/bash
    export CUDA_VISIBLE_DEVICES=1
    mkdir -p /home/arooba/compute-aware-vit-variant-c/scripts/logs
    cd /home/arooba/compute-aware-vit-variant-c
    nohup conda run -n ai_assisted_env python -u <script> \
      > scripts/logs/<name>.out 2>&1 &
    echo "started PID $!"

# Your Task

Think carefully about this problem and design the best solution you can.
Do not follow a prescribed structure. I want your genuine architectural 
judgement, not a standard implementation.

## Analysis required

Before writing any code, write a DESIGN.md that addresses:

1. What are the key challenges in building a compute-aware ViT 
   inference pipeline on CIFAR-100?

2. What adaptive mechanism do you propose and why? Consider the 
   full design space — what alternatives exist and why did you 
   reject them?

3. Where do you expect your approach to succeed and where might 
   it fail?

4. What would you do differently with more time or data?

## Minimum deliverables

Your implementation must include:

- A dense ViT baseline (deit_tiny_patch16_224, CIFAR-100)
  for reference
- At least one static pruning baseline for comparison
- An adaptive inference mechanism that allocates different 
  compute to different images based on their difficulty
- FLOPs measurement per model configuration using fvcore
- metrics.json per run containing: model name, parameters, 
  flops_giga, best_val_acc, epoch_history
- nohup run scripts for every training job in scripts/
- A run_all.sh that runs all jobs sequentially

## What I will evaluate

- Does the adaptive mechanism genuinely route easy images to 
  cheaper compute and hard images to more expensive compute?
- How well does accuracy hold up as compute is reduced?
- Is the design justified and well-reasoned in DESIGN.md?
- Is the code clean, modular, and well-documented?

## What I will do

I will run your code on the GPU server using the scripts you create.
Do not run training yourself.
Implement everything, verify code structure is correct, then stop 
and wait for my confirmation.