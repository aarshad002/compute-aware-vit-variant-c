# Project Context

This is a Master's thesis on compute-aware adaptive inference in Vision 
Transformers. The core problem: Vision Transformers process all image 
tokens equally regardless of image difficulty. This wastes computation 
on easy images that could be classified correctly with far fewer tokens.

The environment is already set up (see SETUP.md in the root directory).
Conda environment name: ai_assisted_env

# Your Task

I want you to think carefully about this problem and propose the best 
solution you can. Do not follow a prescribed structure.

## What I need from you

1. Analyse the problem. What are the key challenges in building a 
   compute-aware ViT inference pipeline?

2. Propose an architecture. What is the best way to implement adaptive 
   token budget selection? Consider whether simple confidence thresholding 
   is optimal or whether something better exists.

3. Implement your proposed solution. It must include at minimum:
   - A dense ViT baseline (deit_tiny_patch16_224, CIFAR-100)
   - At least one static pruning baseline for comparison
   - An adaptive mechanism that allocates different compute to 
     different images based on difficulty
   - Measurement of accuracy and FLOPs per configuration

4. Write a DESIGN.md file explaining:
   - Why you chose this architecture
   - What alternatives you considered and rejected
   - Where you expect this approach to succeed or fail

## Technical constraints

- Framework: PyTorch
- Model: deit_tiny_patch16_224 from timm
- Dataset: CIFAR-100, images resized to 224x224
- FLOPs measurement: fvcore
- Seed: 42

## What I will do

I will run your code on a GPU cluster. Do not run training yourself.
Focus on producing the best possible solution, not the most familiar one.
I am more interested in good design decisions than in following a 
specific structure I already have in mind.