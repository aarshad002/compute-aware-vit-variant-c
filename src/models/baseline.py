import timm
import torch.nn as nn


class DeiTBaseline(nn.Module):
    """Full DeiT-tiny (12 transformer blocks) fine-tuned for CIFAR-100."""

    def __init__(self, num_classes: int = 100, pretrained: bool = True):
        super().__init__()
        self.model = timm.create_model(
            "deit_tiny_patch16_224", pretrained=pretrained
        )
        # Replace the ImageNet head with a CIFAR-100 head.
        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
