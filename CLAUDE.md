# Project Context

This is a Master's thesis on compute-aware adaptive inference in Vision 
Transformers. The core problem: Vision Transformers process all image 
tokens equally regardless of image difficulty. This wastes computation 
on easy images that could be classified correctly with far fewer tokens, 
while hard images may genuinely need more tokens to classify correctly.

The key idea: instead of processing all 196 patch tokens for every image, 
we want to process only the most informative tokens — and crucially, use 
fewer tokens for easy images and more tokens for hard images. This reduces 
average compute while maintaining accuracy.

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

Think carefully about this problem and design the best solution you can 
within the token pruning paradigm. Do not follow a prescribed structure. 
I want your genuine architectural judgement.

## The mechanism you must implement

The adaptive mechanism MUST work by controlling how many patch tokens 
are processed by the Vision Transformer. Specifically:

- DeiT-tiny processes 196 patch tokens + 1 CLS token = 197 total tokens
- Your system must prune (remove) patch tokens so that easy images 
  are processed with fewer tokens and hard images with more tokens
- The CLS token must never be pruned
- Different images in the same batch can use different token budgets
- The token budget decision must be made per image based on some 
  measure of image difficulty or content

This is token-level adaptive computation — the model processes 
different numbers of spatial tokens for different images. This is 
NOT depth reduction, NOT early exit, NOT knowledge distillation.

## What you have full freedom to design

Within the token pruning constraint above, you have complete freedom:

- How to score which tokens are most informative 
  (L2 norm? attention scores? learned scorer? something else?)
- When to prune (which transformer layer?)
- How to decide the token budget per image 
  (fixed ratios? learned controller? confidence-based? something else?)
- How many budget levels to support (2? 3? 4? continuous?)
- How to train the system end-to-end

## Analysis required

Before writing any code, write a DESIGN.md that addresses:

1. What token scoring method do you propose and why?
   What alternatives did you consider and reject?

2. How do you decide the token budget per image?
   What signal tells you an image is easy vs hard?

3. At which transformer layer do you prune?
   What are the trade-offs of pruning earlier vs later?

4. How do you train the system?
   What loss function? Joint training or staged?

5. Where do you expect your approach to succeed and fail?

## Minimum deliverables

Your implementation must include:

- A dense ViT baseline (deit_tiny_patch16_224, CIFAR-100)
- At least one static token pruning baseline 
  (fixed token budget applied to every image equally)
- An adaptive token budget mechanism that allocates different 
  numbers of patch tokens to different images based on difficulty
- FLOPs measurement per configuration using fvcore
- metrics.json per run: model name, parameters, flops_giga, 
  best_val_acc, epoch_history
- nohup run scripts for every training job in scripts/
- A run_all.sh that runs all training jobs sequentially

## What I will evaluate

- Does the system genuinely route easy images to fewer tokens 
  and hard images to more tokens?
- How well does accuracy hold up as average token count decreases?
- Is the design well-reasoned in DESIGN.md?
- Is the code clean, modular, and well-documented?

## What I will do

I will run your code on the GPU server using the scripts you create.
Do not run training yourself.
Implement everything, verify code structure is correct, then stop 
and wait for my confirmation.