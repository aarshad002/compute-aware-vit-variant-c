"""
Adaptive DeiT-tiny v2: fixes budget collapse via soft budget blending.

Root cause of v1 collapse
--------------------------
v1's _forward_train computed classification logits on max_k tokens (the
maximum budget selected anywhere in the batch), so CE loss was INDEPENDENT
of the budget predictor's choice.  The only gradient reaching the predictor
was from the budget_cost term, which always pushes toward the minimum budget.
No competing signal → 100 % collapse to budget level 0 (49 tokens) by ep 2.

Fix: soft budget blending
--------------------------
For each budget level k ∈ {49, 98, 196} compute logits_k independently.
Blend them: blended_logits = Σ p_k * logits_k  where p_k are Gumbel-softmax
probabilities (hard=False).  The CE loss on blended_logits IS differentiable
w.r.t. the budget predictor: routing an easy image to 49 tokens is rewarded
if it classifies correctly, penalised if it fails.  This creates the competing
pressure that was missing in v1.

Stabilisers
-----------
- Temperature annealing: τ decays from τ_init to τ_final so all paths get
  gradient early and the predictor commits to hard assignments later.
- λ warmup: budget penalty ramps from 0 to λ over warmup_epochs so the
  randomly-initialised head stabilises before efficiency pressure kicks in.
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


BUDGET_TOKENS = [49, 98, 196]   # 25 %, 50 %, 100 %
BUDGET_RATIOS = [t / 196.0 for t in BUDGET_TOKENS]


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


class AdaptivePrunedDeiTv2(nn.Module):
    """
    DeiT-tiny with per-image adaptive token budget (v2 — fixed training).

    Training: soft Gumbel-softmax blending across all budget levels so that
              CE loss is differentiable w.r.t. the budget predictor.
    Eval:     hard argmax → discrete budget per image (same as v1).

    The trainer is responsible for updating model.gumbel_temperature each
    epoch (temperature annealing) and passing the effective λ (λ warmup).
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
        self.gumbel_tau_init = gumbel_tau_init
        self.gumbel_tau_final = gumbel_tau_final
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
        cls_feat = x[:, 0]                              # [B, D]
        budget_logits = self.budget_predictor(cls_feat) # [B, num_budgets]

        if self.training:
            return self._forward_train(x, scores, budget_logits)
        else:
            return self._forward_eval(x, scores, budget_logits)

    def _forward_train(self, x, scores, budget_logits):
        """
        Soft budget blending: compute logits for every budget level and
        blend by Gumbel-softmax probabilities (hard=False).  This makes
        CE differentiable w.r.t. the budget predictor.
        """
        tau = self.gumbel_temperature
        budget_probs = F.gumbel_softmax(budget_logits, tau=tau, hard=False)  # [B, num_budgets]

        # Forward pass at each budget level
        all_logits = []
        for k in self.budget_tokens:
            x_k = self._prune_to_k(x, scores, k)
            x_k = self._run_blocks_from(x_k, self.prune_after_block)
            x_k = self.norm(x_k)
            all_logits.append(self.head(x_k[:, 0]))   # [B, num_classes]

        all_logits = torch.stack(all_logits, dim=1)   # [B, num_budgets, num_classes]
        blended = (budget_probs.unsqueeze(-1) * all_logits).sum(dim=1)  # [B, num_classes]

        budget_ratios = torch.tensor(
            [t / 196.0 for t in self.budget_tokens], device=x.device, dtype=x.dtype
        )
        budget_cost = (budget_probs * budget_ratios).sum(dim=1).mean()

        return blended, budget_cost

    def _forward_eval(self, x, scores, budget_logits):
        """Hard argmax budget per image. Group by budget, forward each group."""
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
