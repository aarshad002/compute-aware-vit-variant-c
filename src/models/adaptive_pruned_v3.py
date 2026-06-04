"""
Adaptive DeiT-tiny v3: fixes max-budget collapse in v2.

Root cause of v2 collapse
--------------------------
v2's soft-blend CE always pushes the budget predictor toward budget 196 because
the pretrained backbone produces far better logits at 196 tokens than at 49 tokens.
The CE gradient magnitude >> budget cost gradient (λ=0.1 × 0.75 = 0.075 maximum),
so the predictor collapses to always choosing 196 tokens within epoch 1.

Three-part fix
--------------
1. Auxiliary CE at every budget level  (L_aux):
   The backbone receives CE gradients from ALL three token counts simultaneously,
   forcing it to learn good representations at 49, 98, and 196 tokens. Once the
   backbone handles all counts competently, logits_k are roughly comparable across
   k and the blended CE gradient is no longer systematically biased toward 196 tokens.

2. Entropy regularisation on budget probabilities  (-λ_ent · H):
   Prevents the predictor from committing to budget 196 during early epochs (before
   L_aux has brought 49-token accuracy up). The coefficient decays linearly to 0 so
   the model can commit to discrete routing later in training.

3. Stronger budget cost weight  (λ_budget = 0.5 default):
   Once CE gradients are balanced across budgets, a stronger efficiency penalty is
   needed to actually route images to cheaper budgets.

Training objective
------------------
    L = L_aux + L_blend + λ_budget · budget_cost − λ_ent · H(budget_probs)

where:
  L_aux   = (1/K) Σ_k CE(logits_k, labels)   – backbone trained at all budgets
  L_blend = CE(Σ_k p_k · logits_k, labels)    – predictor trained for accuracy
  budget_cost = Σ_k p_k · (tokens_k / 196)    – predictor pushed toward cheapness
  H       = −Σ_k p_k log p_k                  – entropy of Gumbel-softmax probs
"""

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


def _cls_attention_scores(block, x):
    """CLS-to-patch attention weights [B, N_patch], recomputed without side effects."""
    B, N, C = x.shape
    head_dim = C // block.attn.num_heads
    qkv = block.attn.qkv(block.norm1(x))
    qkv = qkv.reshape(B, N, 3, block.attn.num_heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k, _ = qkv.unbind(0)
    attn = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
    attn = attn.softmax(dim=-1)
    return attn[:, :, 0, 1:].mean(dim=1)  # [B, 196]


BUDGET_TOKENS = [49, 98, 196]   # 25%, 50%, 100%


class BudgetPredictor(nn.Module):
    """Maps CLS embedding → num_budgets logits."""

    def __init__(self, embed_dim: int, num_budgets: int = 3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, num_budgets),
        )

    def forward(self, cls_feat):
        return self.fc(cls_feat)  # [B, num_budgets]


class AdaptivePrunedDeiTv3(nn.Module):
    """
    DeiT-tiny with per-image adaptive token budget (v3 — balanced training).

    Training returns (all_logits_list, blended_logits, budget_cost, entropy)
    so the trainer can compose the full objective.  Eval returns
    (logits, mean_token_ratio) with hard argmax routing, same as v2.
    """

    def __init__(
        self,
        num_classes: int = 100,
        pretrained: bool = True,
        prune_after_block: int = 3,
        budget_tokens=None,
        gumbel_tau_init: float = 3.0,
        gumbel_tau_final: float = 0.5,
    ):
        super().__init__()
        if budget_tokens is None:
            budget_tokens = BUDGET_TOKENS
        self.budget_tokens = sorted(budget_tokens)
        self.prune_after_block = prune_after_block
        self.gumbel_temperature = float(gumbel_tau_init)  # updated by trainer

        backbone = timm.create_model('deit_tiny_patch16_224', pretrained=pretrained, num_classes=0)
        self.patch_embed = backbone.patch_embed
        self.cls_token = backbone.cls_token
        self.pos_embed = backbone.pos_embed
        self.pos_drop = backbone.pos_drop
        self.blocks = backbone.blocks
        self.norm = backbone.norm

        embed_dim = backbone.embed_dim
        self.head = nn.Linear(embed_dim, num_classes)
        self.budget_predictor = BudgetPredictor(embed_dim, len(self.budget_tokens))

    # ------------------------------------------------------------------
    def _embed(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        return self.pos_drop(x + self.pos_embed)

    def _run_blocks_until(self, x, end_block):
        for blk in self.blocks[:end_block]:
            x = blk(x)
        return x

    def _run_blocks_from(self, x, start_block):
        for blk in self.blocks[start_block:]:
            x = blk(x)
        return x

    def _prune_to_k(self, x, scores, k):
        """Keep top-k patch tokens; CLS (position 0) always kept."""
        B = x.shape[0]
        _, idx = scores.topk(k, dim=1)
        idx_sorted = idx.sort(dim=1).values + 1  # +1 for CLS offset
        cls_idx = torch.zeros(B, 1, dtype=torch.long, device=x.device)
        full_idx = torch.cat([cls_idx, idx_sorted], dim=1)  # [B, k+1]
        return x.gather(1, full_idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))

    # ------------------------------------------------------------------
    def forward(self, x):
        x = self._embed(x)
        x = self._run_blocks_until(x, self.prune_after_block)

        pivot_block = self.blocks[self.prune_after_block]
        scores = _cls_attention_scores(pivot_block, x)  # [B, 196]
        cls_feat = x[:, 0]                               # [B, D]
        budget_logits = self.budget_predictor(cls_feat)  # [B, K]

        if self.training:
            return self._forward_train(x, scores, budget_logits)
        else:
            return self._forward_eval(x, scores, budget_logits)

    def _forward_train(self, x, scores, budget_logits):
        tau = self.gumbel_temperature
        budget_probs = F.gumbel_softmax(budget_logits, tau=tau, hard=False)  # [B, K]

        all_logits = []
        for k in self.budget_tokens:
            x_k = self._prune_to_k(x, scores, k)
            x_k = self._run_blocks_from(x_k, self.prune_after_block)
            x_k = self.norm(x_k)
            all_logits.append(self.head(x_k[:, 0]))  # [B, C]

        all_logits_t = torch.stack(all_logits, dim=1)  # [B, K, C]
        blended = (budget_probs.unsqueeze(-1) * all_logits_t).sum(dim=1)  # [B, C]

        budget_ratios = torch.tensor(
            [t / 196.0 for t in self.budget_tokens], device=x.device, dtype=x.dtype
        )
        budget_cost = (budget_probs * budget_ratios).sum(dim=1).mean()

        # Entropy of budget probs — returned so trainer can add regularisation term
        entropy = -(budget_probs * (budget_probs + 1e-8).log()).sum(dim=1).mean()

        return all_logits, blended, budget_cost, entropy

    def _forward_eval(self, x, scores, budget_logits):
        """Hard argmax budget per image."""
        B = x.shape[0]
        budget_idx = budget_logits.argmax(dim=1)  # [B]

        logits = torch.zeros(B, self.head.out_features, device=x.device, dtype=x.dtype)
        total_ratio = 0.0

        for bi, k in enumerate(self.budget_tokens):
            mask = (budget_idx == bi)
            if not mask.any():
                continue
            x_g = self._prune_to_k(x[mask], scores[mask], k)
            x_g = self._run_blocks_from(x_g, self.prune_after_block)
            x_g = self.norm(x_g)
            logits[mask] = self.head(x_g[:, 0])
            total_ratio += mask.sum().item() * (k / 196.0)

        return logits, total_ratio / B

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
