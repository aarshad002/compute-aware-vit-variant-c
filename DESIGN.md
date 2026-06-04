# Design: Compute-Aware Adaptive Token Pruning for DeiT-tiny on CIFAR-100

## 1. Token Scoring Method

**Proposed: CLS-to-patch attention weights from block 3 (averaged across heads)**

After 4 transformer blocks, the CLS token has aggregated enough context to
assign meaningful importance to patch tokens. The attention weight from CLS
to each patch token reflects how much the model "cares about" that patch when
forming its global representation — this is a direct proxy for informativeness.

Implementation: extract the softmax attention matrix from the last pre-pruning
block, take row 0 (CLS query), columns 1: (patch keys), average across heads.
Result: [B, 196] importance scores per image.

**Alternatives considered and rejected:**

- *L2 norm of patch features*: Content-rich patches (edges, textures) have large
  norms, but this ignores global context — a bright background patch has high norm
  but zero discriminative value. Attention is more semantically grounded.

- *Gradient-based saliency*: Requires a backward pass per image at inference. Too
  expensive to use for routing; contradicts the efficiency goal.

- *Learned per-token scorer (MLP)*: Adds parameters and can overfit. The CLS
  attention is free — it's already computed in the forward pass.

- *Random selection*: Useful as an ablation baseline to confirm attention-based
  scoring is actually doing something useful.

## 2. Token Budget Decision Per Image

**Proposed: CLS-features → 2-layer MLP → 3-way softmax (budget levels)**

Budget levels: 49 tokens (25%), 98 tokens (50%), 196 tokens (100%)

After block 3, the CLS token representation encodes a coarse global summary of
the image. A small budget predictor MLP maps this to a distribution over three
budget levels. Easy images (simple textures, clear foreground) get 49 tokens;
ambiguous images get 98; hard images (cluttered, fine-grained) get 196.

During training: Gumbel-softmax with temperature τ=1.0 → differentiable budget
selection with straight-through gradient estimator.

During evaluation: hard argmax → discrete budget.

**What signal distinguishes easy vs hard:**

The CLS embedding after 4 blocks encodes recognition difficulty implicitly —
a confidently-categorizable image produces a CLS vector that is closer to a
class centroid in embedding space, while a hard image produces a more diffuse
embedding. The budget MLP learns to detect this signal.

The training loss reinforces this: we penalize using large budgets, so the model
learns to only request extra tokens when accuracy would otherwise degrade.

## 3. Pruning Layer

**Proposed: after block 3 (of 12 total, 0-indexed)**

This preserves 4/12 blocks of full-sequence processing (needed for attention
patterns to become meaningful) while saving compute in 8/12 blocks (67% of
remaining layers). It is the best compute/accuracy tradeoff.

**Trade-offs:**

| Prune after block | Tokens meaningful? | Compute saved | Risk |
|---|---|---|---|
| 1 | No — random-ish attention | High | High accuracy loss |
| 3 | Yes — coarse semantic structure | 67% of remaining | Low |
| 6 | Very yes — rich features | 50% of remaining | Minimal |
| 9 | Maximum — near-final features | 25% of remaining | Negligible |

Pruning too early (block 1-2): attention has not specialised; importance scores
are unreliable; accuracy degrades sharply.

Pruning too late (block 9+): most compute already spent; savings are marginal.

Block 3 is the standard choice in the literature (DynamicViT prunes at blocks
3, 6, 9; EViT prunes once at block 4). We use a single pruning point for
simplicity and interpretability.

## 4. Training Strategy

**Joint training from pretrained DeiT-tiny weights.**

The DeiT backbone is initialised from ImageNet-pretrained weights. The budget
predictor MLP and the CIFAR-100 classification head are initialised randomly.

**Loss function:**

```
L = L_CE(logits, labels) + λ * mean_budget_ratio
```

Where `mean_budget_ratio` is the average fraction of tokens used across the
batch (e.g., if all images use 98/196 tokens, this is 0.5).

λ=0.1 balances classification accuracy against compute efficiency. Too large
and the model collapses to always using the minimum budget; too small and it
always uses the maximum (no routing).

**No staged training.** Joint training from the start works because:
1. The backbone is already pretrained — it produces meaningful features immediately.
2. The budget predictor only needs to learn a simple linear separation.
3. Staged training risks the budget predictor learning on stale features.

Constant learning rate (0.0001, Adam) throughout — no scheduler. The pretrained
backbone updates slowly; the new heads train quickly.

## 5. Expected Success and Failure Cases

**Where this will succeed:**

- *Easy categories* (car, ship, airplane): large, distinctive objects filling
  most of the image. CLS attention concentrates on the object. 25-50% tokens
  will suffice → big compute savings with near-zero accuracy loss.

- *Compute efficiency*: on easy images (likely 40-60% of CIFAR-100), the model
  will use 25-50% of tokens, giving meaningful FLOPs reduction.

**Where this may fail:**

- *Fine-grained categories* (100 classes in CIFAR-100 include many similar
  animals): at 32×32 base resolution, 16×16 patches are very coarse. Token
  pruning loses detail that may be critical for fine-grained discrimination.

- *Budget collapse*: if λ is too high, the model routes everything to 25%
  budget, hurting accuracy. If too low, routing is random. λ=0.1 is our best
  estimate; may need tuning.

- *Budget MLP overfitting*: with only ~192 input features and 3 output classes,
  this is unlikely, but possible if easy/hard split is not well-defined in
  CIFAR-100.

**Sanity check built into evaluation:**

We measure per-budget accuracy breakdown — if the adaptive model genuinely
routes easy images to fewer tokens, the images assigned to the 25% budget
should have higher average confidence than those assigned to the 100% budget.
