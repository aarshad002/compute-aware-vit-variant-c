import timm
import torch
import torch.nn as nn


class MultiExitDeiT(nn.Module):
    """DeiT-tiny with exit classifiers after blocks 4, 8, and 12.

    All exits are trained jointly with equal-weight cross-entropy losses.
    At inference, images exit at the first layer where the maximum softmax
    probability exceeds a user-supplied confidence threshold.
    """

    # 1-indexed block numbers after which exits are placed.
    EXIT_BLOCKS = [4, 8, 12]
    NUM_EXITS   = 3

    def __init__(self, num_classes: int = 100, pretrained: bool = True):
        super().__init__()
        base = timm.create_model("deit_tiny_patch16_224", pretrained=pretrained)
        self.embed_dim = base.embed_dim  # 192

        # Backbone components (shared across all exits).
        self.patch_embed = base.patch_embed
        self.cls_token   = base.cls_token
        self.pos_embed   = base.pos_embed
        self.pos_drop    = base.pos_drop
        self.blocks      = base.blocks   # ModuleList of 12 blocks

        # Per-exit classifiers: each is an independent LayerNorm + Linear.
        # LayerNorm is necessary because the intermediate CLS token is not
        # yet normalised by the final block's norm.
        self.exit_norms = nn.ModuleList([
            nn.LayerNorm(self.embed_dim) for _ in range(self.NUM_EXITS)
        ])
        self.exit_heads = nn.ModuleList([
            nn.Linear(self.embed_dim, num_classes) for _ in range(self.NUM_EXITS)
        ])

        # Fast lookup: (1-indexed block number) → exit index
        self._exit_block_set = set(self.EXIT_BLOCKS)
        self._block_to_exit  = {b: i for i, b in enumerate(self.EXIT_BLOCKS)}

    # ------------------------------------------------------------------ #
    #  Core forward — returns all exit logits (used during training)       #
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return a list of NUM_EXITS tensors, each [B, num_classes]."""
        x = self._embed(x)
        exit_outputs = []
        exit_idx = 0
        for block_idx, block in enumerate(self.blocks):
            x = block(x)
            block_num = block_idx + 1  # 1-indexed
            if block_num in self._exit_block_set:
                cls_out = self.exit_norms[exit_idx](x[:, 0])
                exit_outputs.append(self.exit_heads[exit_idx](cls_out))
                exit_idx += 1
        return exit_outputs

    # ------------------------------------------------------------------ #
    #  Partial forward — runs only up to exit `exit_idx` (for FLOPs)      #
    # ------------------------------------------------------------------ #

    def forward_to_exit(self, x: torch.Tensor, exit_idx: int) -> torch.Tensor:
        """Run the backbone up to the specified exit, return that exit's logits."""
        x = self._embed(x)
        target_block = self.EXIT_BLOCKS[exit_idx]  # 1-indexed, e.g. 4, 8, or 12
        for block_idx, block in enumerate(self.blocks):
            x = block(x)
            if block_idx + 1 == target_block:
                break
        cls_out = self.exit_norms[exit_idx](x[:, 0])
        return self.exit_heads[exit_idx](cls_out)

    # ------------------------------------------------------------------ #
    #  Adaptive inference (single image or full batch post-hoc routing)   #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def adaptive_forward(
        self, x: torch.Tensor, threshold: float = 0.7
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run full forward, then route each sample to its exit.

        Returns:
            logits:       [B, num_classes] — logits from each sample's exit.
            assigned_exit:[B]              — which exit index each sample used.

        Note: this runs ALL blocks for ALL images (measuring expected FLOPs
        requires observing all exit confidences).  True latency speedup would
        need per-image early stopping; this gives identical accuracy results.
        """
        exit_logits = self.forward(x)  # list of [B, C]
        B = x.shape[0]
        num_classes = exit_logits[0].shape[1]

        exited       = torch.zeros(B, dtype=torch.bool, device=x.device)
        assigned     = torch.full((B,), self.NUM_EXITS - 1, dtype=torch.long, device=x.device)
        final_logits = exit_logits[-1].clone()

        for e_idx in range(self.NUM_EXITS - 1):
            max_prob = exit_logits[e_idx].softmax(dim=-1).max(dim=-1).values
            take     = (max_prob >= threshold) & ~exited
            assigned[take]     = e_idx
            final_logits[take] = exit_logits[e_idx][take]
            exited |= take

        return final_logits, assigned

    # ------------------------------------------------------------------ #
    #  Internal helper                                                      #
    # ------------------------------------------------------------------ #

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        x   = self.patch_embed(x)
        B   = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        x   = self.pos_drop(x + self.pos_embed)
        return x
