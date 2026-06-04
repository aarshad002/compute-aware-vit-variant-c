import timm
import torch.nn as nn


class DeiTBaseline(nn.Module):
    """Dense DeiT-tiny — all 196 patch tokens processed every layer."""

    def __init__(self, num_classes=100, pretrained=True):
        super().__init__()
        self.model = timm.create_model(
            'deit_tiny_patch16_224',
            pretrained=pretrained,
            num_classes=num_classes,
        )

    def forward(self, x):
        return self.model(x)

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
