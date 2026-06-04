import timm
import torch
import torch.nn as nn


def _cls_attention_scores(block, x):
    """Return CLS-to-patch attention weights [B, N_patch] from a DeiT block (no side effects)."""
    B, N, C = x.shape
    head_dim = C // block.attn.num_heads
    qkv = block.attn.qkv(block.norm1(x))
    qkv = qkv.reshape(B, N, 3, block.attn.num_heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k, _ = qkv.unbind(0)                         # each [B, heads, N, head_dim]
    scale = head_dim ** -0.5
    attn = (q @ k.transpose(-2, -1)) * scale        # [B, heads, N, N]
    attn = attn.softmax(dim=-1)
    return attn[:, :, 0, 1:].mean(dim=1)            # [B, N_patch]


class StaticPrunedDeiT(nn.Module):
    """
    DeiT-tiny with a fixed token budget applied uniformly to every image.

    After block `prune_after_block`, the bottom (196 - keep_tokens) patch
    tokens ranked by CLS attention are discarded. The CLS token is always kept.
    """

    def __init__(self, keep_tokens: int, num_classes=100, pretrained=True,
                 prune_after_block: int = 3):
        super().__init__()
        assert 1 <= keep_tokens <= 196
        self.keep_tokens = keep_tokens
        self.prune_after_block = prune_after_block

        backbone = timm.create_model('deit_tiny_patch16_224', pretrained=pretrained, num_classes=0)
        self.patch_embed = backbone.patch_embed
        self.cls_token = backbone.cls_token
        self.pos_embed = backbone.pos_embed
        self.pos_drop = backbone.pos_drop
        self.blocks = backbone.blocks
        self.norm = backbone.norm

        embed_dim = backbone.embed_dim          # 192 for deit_tiny
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)                                 # [B, 196, 192]
        cls = self.cls_token.expand(B, -1, -1)                 # [B, 1, 192]
        x = torch.cat([cls, x], dim=1)                         # [B, 197, 192]
        x = self.pos_drop(x + self.pos_embed)

        for i, block in enumerate(self.blocks):
            if i == self.prune_after_block:
                scores = _cls_attention_scores(block, x)        # [B, 196]
                _, idx = scores.topk(self.keep_tokens, dim=1)  # [B, K]
                idx_sorted = idx.sort(dim=1).values             # preserve spatial order
                # shift by 1 to account for CLS at position 0
                patch_idx = idx_sorted + 1                      # [B, K]
                cls_idx = torch.zeros(B, 1, dtype=torch.long, device=x.device)
                full_idx = torch.cat([cls_idx, patch_idx], dim=1)  # [B, K+1]
                x = x.gather(1, full_idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
            x = block(x)

        x = self.norm(x)
        return self.head(x[:, 0])

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
