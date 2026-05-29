import timm
import torch
import torch.nn as nn


class TruncatedDeiT(nn.Module):
    """DeiT-tiny using only the first `depth` transformer blocks.

    Acts as a static pruning baseline: it has the same architecture as the
    full model up to the chosen depth, followed by a new LayerNorm and
    classification head trained specifically at that depth.
    """

    VALID_DEPTHS = (4, 8)

    def __init__(self, num_classes: int = 100, depth: int = 4, pretrained: bool = True):
        super().__init__()
        if depth not in self.VALID_DEPTHS:
            raise ValueError(f"depth must be one of {self.VALID_DEPTHS}, got {depth}")

        base = timm.create_model("deit_tiny_patch16_224", pretrained=pretrained)
        embed_dim = base.embed_dim  # 192 for deit_tiny

        self.patch_embed = base.patch_embed
        self.cls_token   = base.cls_token
        self.pos_embed   = base.pos_embed
        self.pos_drop    = base.pos_drop
        # Keep only the first `depth` blocks from the pretrained model.
        self.blocks      = nn.ModuleList(list(base.blocks)[:depth])
        # New norm and head trained from random init at this depth.
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        B = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        x   = self.pos_drop(x + self.pos_embed)

        for block in self.blocks:
            x = block(x)

        cls_out = self.norm(x[:, 0])
        return self.head(cls_out)
