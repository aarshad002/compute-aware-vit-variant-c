import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


def _cls_attention_scores(block, x):
    """CLS-to-patch attention weights [B, N_patch], no side effects."""
    B, N, C = x.shape
    head_dim = C // block.attn.num_heads
    qkv = block.attn.qkv(block.norm1(x))
    qkv = qkv.reshape(B, N, 3, block.attn.num_heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k, _ = qkv.unbind(0)
    attn = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
    attn = attn.softmax(dim=-1)
    return attn[:, :, 0, 1:].mean(dim=1)   # [B, 196]


# Canonical budget levels (fractions of 196)
BUDGET_TOKENS = [49, 98, 196]   # 25%, 50%, 100%
BUDGET_RATIOS = [t / 196.0 for t in BUDGET_TOKENS]


class BudgetPredictor(nn.Module):
    """Maps CLS embedding → 3-way budget distribution."""

    def __init__(self, embed_dim: int, num_budgets: int = 3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, num_budgets),
        )

    def forward(self, cls_feat):
        return self.fc(cls_feat)    # [B, num_budgets] logits


class AdaptivePrunedDeiT(nn.Module):
    """
    DeiT-tiny with per-image adaptive token budget.

    After `prune_after_block`, a budget predictor decides how many patch tokens
    each image needs from {49, 98, 196}. Tokens are ranked by CLS attention;
    the top-k are kept. Different images in the same batch may use different k.

    Training: Gumbel-softmax (straight-through) for differentiable budget
              selection. Loss includes a budget cost term λ * mean_ratio.
    Eval:     Hard argmax → discrete budget per image.

    Batching strategy: images are processed per-budget group during evaluation
    (sorted by assigned budget). During training, all images use the maximum
    budget in the batch to keep tensor shapes uniform while the budget predictor
    still gets trained via the cost term.
    """

    def __init__(self, num_classes=100, pretrained=True,
                 prune_after_block: int = 3,
                 budget_tokens=None,
                 budget_cost_weight: float = 0.1,
                 gumbel_temperature: float = 1.0):
        super().__init__()
        if budget_tokens is None:
            budget_tokens = BUDGET_TOKENS
        self.budget_tokens = sorted(budget_tokens)
        self.budget_cost_weight = budget_cost_weight
        self.gumbel_temperature = gumbel_temperature
        self.prune_after_block = prune_after_block

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
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        return self.pos_drop(x + self.pos_embed)

    def _run_blocks_until(self, x, end_block):
        for block in self.blocks[:end_block]:
            x = block(x)
        return x

    def _run_blocks_from(self, x, start_block):
        for block in self.blocks[start_block:]:
            x = block(x)
        return x

    def _prune_to_k(self, x, scores, k):
        """Keep top-k patch tokens by scores; always keep CLS (position 0)."""
        B = x.shape[0]
        _, idx = scores.topk(k, dim=1)
        idx_sorted = idx.sort(dim=1).values + 1         # +1 for CLS offset
        cls_idx = torch.zeros(B, 1, dtype=torch.long, device=x.device)
        full_idx = torch.cat([cls_idx, idx_sorted], dim=1)  # [B, k+1]
        return x.gather(1, full_idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x):
        """
        Returns (logits, budget_cost) during training.
        Returns (logits, mean_budget_ratio) during eval.
        """
        x = self._embed(x)
        x = self._run_blocks_until(x, self.prune_after_block)

        # Score patch tokens
        pivot_block = self.blocks[self.prune_after_block]
        scores = _cls_attention_scores(pivot_block, x)   # [B, 196]

        cls_feat = x[:, 0]                               # [B, D]
        budget_logits = self.budget_predictor(cls_feat)  # [B, num_budgets]

        if self.training:
            return self._forward_train(x, scores, budget_logits)
        else:
            return self._forward_eval(x, scores, budget_logits)

    def _forward_train(self, x, scores, budget_logits):
        """
        Use Gumbel-softmax to select budget (straight-through).
        Process all images at the maximum selected budget to keep uniform shapes.
        Budget cost encourages using fewer tokens.
        """
        B = x.shape[0]
        budget_ratios = torch.tensor(
            [t / 196.0 for t in self.budget_tokens],
            device=x.device, dtype=x.dtype
        )

        # Gumbel-softmax: hard=True gives one-hot with STE gradient
        budget_probs = F.gumbel_softmax(budget_logits, tau=self.gumbel_temperature, hard=True)
        # [B, num_budgets] one-hot

        # Expected budget ratio per image (differentiable)
        expected_ratio = (budget_probs * budget_ratios).sum(dim=1)  # [B]
        budget_cost = expected_ratio.mean()

        # Hard budget index per image (for actual pruning)
        budget_idx = budget_probs.argmax(dim=1)  # [B]
        max_k = max(self.budget_tokens[budget_idx[i].item()] for i in range(B))

        x_pruned = self._prune_to_k(x, scores, max_k)
        x_pruned = self._run_blocks_from(x_pruned, self.prune_after_block)
        x_pruned = self.norm(x_pruned)
        logits = self.head(x_pruned[:, 0])

        return logits, budget_cost

    def _forward_eval(self, x, scores, budget_logits):
        """
        Hard argmax budget per image. Group by budget, forward each group,
        then reassemble in original order.
        """
        B = x.shape[0]
        budget_idx = budget_logits.argmax(dim=1)  # [B]

        logits = torch.zeros(B, self.head.out_features, device=x.device, dtype=x.dtype)
        total_ratio = 0.0

        for bi, k in enumerate(self.budget_tokens):
            mask = (budget_idx == bi)
            if not mask.any():
                continue
            x_group = x[mask]
            scores_group = scores[mask]
            x_pruned = self._prune_to_k(x_group, scores_group, k)
            x_pruned = self._run_blocks_from(x_pruned, self.prune_after_block)
            x_pruned = self.norm(x_pruned)
            logits[mask] = self.head(x_pruned[:, 0])
            total_ratio += mask.sum().item() * (k / 196.0)

        mean_ratio = total_ratio / B
        return logits, mean_ratio

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
