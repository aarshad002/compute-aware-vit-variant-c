# Design: Compute-Aware Adaptive Inference in Vision Transformers
## Variant C — Confidence-Based Multi-Exit ViT on CIFAR-100

---

## 1. Key Challenges

### 1.1 Resolution Mismatch and What It Implies

CIFAR-100 images are 32×32 pixels. DeiT-tiny uses 16×16 patches, which at 224×224 resolution
gives 196 patches per image. To use the pretrained model, we upsample to 224×224, meaning
each "original" CIFAR pixel is expanded to a 7×7 block of interpolated values. This has two
implications:

- **Adjacent patches are highly correlated**: The spatial token redundancy that motivates
  token-pruning approaches (DynamicViT, EViT) is artificially amplified. Every token in an
  upsampled CIFAR image carries locally similar information to its neighbors.
- **Pretrained positional embeddings still help**: Because the upsampling is deterministic,
  patch positions are consistent across images, and ImageNet-pretrained positional encodings
  remain meaningful.

**Consequence for design**: Token-pruning approaches are a poor fit for upsampled CIFAR images —
there's little structural token diversity to prune. Depth-based exits (which are about *how much
computation to spend*, not *which spatial regions to focus on*) are the right axis of adaptation.

### 1.2 Class Taxonomy and Heterogeneous Difficulty

CIFAR-100 organizes 100 classes into 20 superclasses. Within a superclass (e.g., large carnivores:
bear, leopard, lion, tiger, wolf), inter-class discriminability is much lower than between
superclasses (e.g., distinguishing a bicycle from an aquatic mammal). This creates a bimodal
difficulty distribution:

- **Easy images**: belong to visually distinct superclasses, or have clear canonical appearances.
  These images should classify correctly well before the final transformer block.
- **Hard images**: fine-grained within-superclass cases, unusual poses, or ambiguous images
  that even humans struggle with. These need the full model capacity, and sometimes it still won't help.

A well-designed adaptive mechanism should implicitly discover this hierarchy without explicit
superclass labels — the confidence signal at intermediate layers is a proxy for discriminability.

### 1.3 The Pretrain-Finetune Distribution Shift

DeiT-tiny was pretrained on ImageNet-1k with full 224×224 natural images. CIFAR-100's upsampled
images are a significant distribution shift: the texture is blocky, the effective spatial frequency
is lower, and the class vocabulary partially overlaps with ImageNet but not perfectly. This argues
for fine-tuning all layers (not just the head) and using a small learning rate (1e-4) to adapt
without destroying pretrained features.

### 1.4 Difficulty as a Moving Target During Training

The confidence of the model at intermediate layers changes as training progresses. Early in
training, even the final exit may be uncertain on most images. Defining "easy" via a fixed
confidence threshold must accommodate a model that starts poorly calibrated and improves over
epochs. This is handled implicitly: we don't use per-image routing loss — we use a supervised
classification loss at each exit and allow the model to naturally become better calibrated.

### 1.5 The Gap Between FLOPs and Wall-Clock Speedup

On GPU, the practical speedup from early exits depends on batch composition. If images in a
batch exit at different layers, the batch cannot be terminated early in standard PyTorch — you'd
need to dynamically regroup images. For this thesis, we measure *expected FLOPs* (the weighted
average of FLOPs across the exit distribution), which is the canonical metric in the adaptive
inference literature and is directly comparable across methods.

---

## 2. Proposed Adaptive Mechanism and Design Space

### 2.1 The Full Design Space

Before stating my choice, I want to be explicit about what I considered and rejected.

**Option A: Token Pruning (DynamicViT, EViT)**

These methods learn to score each spatial token at pruning stages (every 4 blocks typically)
and drop low-scoring tokens. Subsequent attention computations are cheaper because sequence
length N decreases, reducing the O(N²d) attention and improving MLP throughput.

*Why rejected*: As established in §1.1, the upsampled CIFAR tokens are spatially correlated,
making reliable token scoring harder. More critically: at DeiT-tiny's scale (197 tokens, d=192),
reducing N by 50% saves only ~25% of total FLOPs because the MLP blocks (which scale as O(Nd²),
not O(N²d)) dominate. The implementation complexity is high, and the saving is modest.

**Option B: Depth-wise Mixture of Experts**

Route different images through different "expert" subnetworks at each layer, inspired by
Switch Transformer. This would create a natural compute-difficulty alignment if experts
specialize by difficulty.

*Why rejected*: Requires training from scratch (cannot use pretrained DeiT weights), adds
significant parameter overhead (each expert is a full copy of a layer), and is prone to
expert collapse where all inputs choose the same path. Out of scope for a 20-epoch fine-tuning
regime.

**Option C: Attention Head Masking**

Dynamically zero out entire attention heads per image based on a lightweight routing network
(as in Sparse is Enough, HAT).

*Why rejected*: Coarser than I want — FLOPs saved per masked head are small relative to the
rest of the block. More importantly, the mismatch between training (all heads active) and
inference (some heads zeroed) creates a distribution shift that is difficult to correct without
special training objectives.

**Option D: Early-Exit Cascades (BERT-PABEE, Shallow-Deep Networks)**

Attach lightweight classifiers at intermediate layers. During inference, exit as soon as
confidence exceeds a threshold. This is the method I chose.

**Option E: Adaptive Patch Resolution**

Process fewer patches for easy images (e.g., 49-patch center crop vs. 196-patch full image).

*Why rejected*: DeiT's positional embeddings are fixed-length. Interpolating them for different
sequence lengths is possible but fragile, especially with a pretrained model and a 20-epoch
fine-tuning budget that won't allow the positional encodings to fully adapt.

### 2.2 My Design: Confidence-Based Multi-Exit ViT

I attach exit classifiers after transformer blocks 4, 8, and 12 (the three evenly-spaced
thirds of the 12-block DeiT-tiny). Each classifier is a LayerNorm applied to the CLS token
followed by a linear projection to 100 classes.

```
Input → PatchEmbed → [Blocks 1-4] → Exit-0 (LayerNorm + Linear → 100 classes)
                   → [Blocks 5-8] → Exit-1 (LayerNorm + Linear → 100 classes)
                   → [Blocks 9-12] → Exit-2 (LayerNorm + Linear → 100 classes)
```

**Why the CLS token?** DeiT's CLS token aggregates information from all patches via attention
at each layer. It is the natural carrier of global semantic information needed for classification,
and it exists at every intermediate layer.

**Why LayerNorm at each exit?** The internal representations before the final block's LayerNorm
are not normalized to the same scale expected by the linear head. Adding per-exit LayerNorms
allows each classifier to learn its own normalization, decoupling exit calibration from the
backbone's internal scale.

**Why blocks 4, 8, 12?** Even thirds of the network give a clean compute profile:
- Exit 0 (4 blocks): ~33% of full model FLOPs
- Exit 1 (8 blocks): ~67% of full model FLOPs
- Exit 2 (12 blocks): 100% of full model FLOPs

The coarser the exits, the more stable the confidence signal at each exit (fewer exits =
more compute per exit = more informative representation). Three exits is sufficient to
demonstrate the accuracy-FLOPs trade-off curve.

### 2.3 Training Objective

All exits are trained jointly with equal-weight cross-entropy losses:

```
L = (1/3) * [CE(exit_0, y) + CE(exit_1, y) + CE(exit_2, y)]
```

Equal weights reflect that we want all exits to be genuinely useful, not just the final one.
An alternative is to weight later exits more heavily (e.g., [0.2, 0.3, 0.5]) to ensure the
full model remains strong, but this risks under-training early exits. The equal-weight scheme
means every exit is optimized as a full classifier from its available representation.

### 2.4 Inference Routing

At inference, we scan exits in order. An image exits at the first exit where:

```
max_i softmax(logits)[i] >= threshold τ
```

Images that do not reach threshold τ at exits 0 or 1 always exit at exit 2. By sweeping τ
from 0.1 to 0.99, we trace the full accuracy-FLOPs Pareto frontier.

**Why max probability over entropy?** Both are valid confidence proxies. Max softmax probability
is simpler, more interpretable (directly: "how confident are we in the top prediction?"), and
slightly less sensitive to the long tail of near-zero probabilities for irrelevant classes.

### 2.5 Static Pruning Baselines

To understand what the adaptive mechanism adds beyond simply using a shallower model, I train
two static baselines:

- **Static-4**: DeiT-tiny with only the first 4 transformer blocks, trained from scratch with
  a new head. This has the same FLOPs as Exit-0 of the adaptive model.
- **Static-8**: DeiT-tiny with only the first 8 transformer blocks. Same FLOPs as Exit-1.

These answer the question: *is the adaptive routing actually better than just using a smaller
model?* If the adaptive model at low-threshold achieves higher accuracy than Static-4 at the
same FLOPs, the joint multi-exit training provides a genuine benefit (the early exits benefit
from the gradient signal of deeper layers during joint training).

---

## 3. Expected Successes and Anticipated Failures

### 3.1 Where I Expect Success

**Superclass-level separation**: Images from visually distinct superclasses (vehicles vs.
natural objects vs. people) should be confidently classified at Exit-0 or Exit-1. The first
4 DeiT blocks are sufficient to extract coarse semantic features, and the ImageNet pretraining
has already "seen" many of these categories.

**Compute savings at moderate thresholds**: At τ=0.7, I expect 35-55% of images to exit before
block 12, delivering an average FLOPs reduction of 25-40% with less than 3% accuracy drop from
the full model. The Pareto curve should clearly dominate the static pruned baselines at
intermediate compute budgets — specifically, at any given FLOPs level, the adaptive model's
accuracy should exceed Static-4's accuracy.

**Calibration improvement over training**: The model's exit distribution should shift toward
earlier exits as training progresses (the CLS representations become more separable earlier),
which is a signal that the joint training is working correctly.

### 3.2 Where I Anticipate Failure

**Fine-grained within-superclass cases**: Classes like oak_tree vs. maple_tree, or bottle vs.
can vs. cup will likely require all 12 blocks. The exit distribution will not be uniform — it
will be bimodal, with a large fraction of easy images exiting at Block 4 and the difficult
fine-grained cases always reaching Block 12.

**Early exit overconfidence**: Softmax max-probability is a known poor calibration metric for
neural networks. A model that is wrong but confident will incorrectly exit early, producing
mis-classifications that wouldn't happen with the full model. This effect is larger for
Exit-0 (weakest representation) and can manifest as a sharp accuracy cliff at very high τ.

**The 4-block static baseline may be competitive with early exit**: Because Static-4 trains
its head specifically for the 4-block representation without being "distracted" by the need
to also be useful at deeper depths, it might match or exceed Exit-0's accuracy at the same
FLOPs. This would show that joint training's benefit for early exits is limited — a known
limitation discussed in the SDN (Shallow-Deep Networks) paper.

**CIFAR-100 task difficulty ceiling**: DeiT-tiny is a small model, and fine-tuning for 20
epochs on CIFAR-100 with lr=1e-4 may not fully close the domain gap from ImageNet. The full
12-block model may plateau around 65-72% accuracy, leaving limited headroom for the adaptive
mechanism to demonstrate savings while preserving accuracy.

---

## 4. What I Would Do Differently with More Time

### 4.1 Learned Confidence Thresholds

Rather than a single global τ swept post-hoc, I would learn per-exit thresholds jointly with
the model using a differentiable expected-exit-depth loss:

```
L_compute = λ * E[depth_used] / max_depth
```

This would train the model to minimize expected depth while preserving accuracy, naturally
finding the optimal trade-off without post-hoc threshold search.

### 4.2 Exit-to-Exit Distillation

The final exit (full model) could act as a teacher for earlier exits, using knowledge
distillation:

```
L_kd = KL(softmax(exit_12/T) || softmax(exit_4/T))
```

This would improve the calibration and accuracy of early exits beyond what the ground-truth
supervision alone provides, because the teacher's soft labels carry information about
inter-class similarity.

### 4.3 Per-Superclass Calibration

Because CIFAR-100 has a known 20-superclass structure, a per-superclass confidence threshold
(learned or searched) would give much better routing than a single global τ. Easy superclasses
(vehicles, household objects) need τ=0.5 to exit early; hard superclasses (large carnivores,
flowers) need τ=0.95 or higher. This hierarchical routing would dramatically improve the
accuracy-FLOPs trade-off.

### 4.4 Two-Stage Training

First train the full 12-block model to convergence (giving strong final-exit representations),
then freeze blocks 1-4 and train only Exit-0's head. This prevents the early exit's
optimization pressure from degrading the quality of the full model's features.

### 4.5 Backbone Architecture Search

DeiT-tiny's uniform block structure (all 12 blocks identical) is not optimal for multi-exit.
A model designed for multi-exit might use wider/deeper early blocks (to make early
representations more discriminative) and narrower later blocks (since the full model has
already done most of the hard work). With more time, I would run a lightweight NAS to find
the optimal layer width schedule for the adaptive inference use case.

### 4.6 Larger Dataset or Backbone

CIFAR-100 with 32×32 resolution is a constrained testbed. The adaptive mechanism's value
is most apparent when:
- The dataset has a genuinely wide range of sample difficulty
- The full model has high capacity (so easy images are "wasted" on it)
- The domain supports natural image complexity gradients

ImageNet-1k with DeiT-small or DeiT-base would be a better demonstration substrate for the
thesis's core claims. The findings on CIFAR-100 are directionally valid but underpowered.

---

## 5. Implementation Summary

| Component | File | Purpose |
|---|---|---|
| Data loading | `src/data.py` | CIFAR-100 with ImageNet-style transforms |
| Full DeiT-tiny | `src/models/baseline.py` | 12-block fine-tuned model |
| Truncated DeiT-tiny | `src/models/static_pruned.py` | First N blocks + new head |
| Multi-exit DeiT | `src/models/adaptive_vit.py` | 3 exits at blocks 4, 8, 12 |
| Training loop | `src/train.py` | Generic and adaptive training |
| FLOPs + metrics | `src/metrics.py` | fvcore FLOPs, metrics.json saving |
| Utilities | `src/utils.py` | Seed, device, logging |
| Baseline entry | `train_baseline.py` | Trains and saves full model |
| Static entry | `train_static_pruned.py` | Trains depth-4 or depth-8 model |
| Adaptive entry | `train_adaptive.py` | Trains multi-exit model |
| Evaluation | `eval_adaptive.py` | Threshold sweep, Pareto curve |

**Checkpoints**: `checkpoints/{model_name}/best_model.pth`

**Results**: `results/{model_name}/metrics.json`

**Adaptive eval**: `results/adaptive_eval/threshold_sweep.json`
