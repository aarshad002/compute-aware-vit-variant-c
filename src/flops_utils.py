"""FLOPs estimation via fvcore."""
import torch
from fvcore.nn import FlopCountAnalysis


def estimate_flops(model, input_size=(1, 3, 224, 224), device='cpu'):
    """
    Returns GFLOPs for a single forward pass.
    Handles models that return tuples (adaptive model returns (logits, cost)).
    """
    model = model.eval().to(device)
    dummy = torch.zeros(input_size, device=device)

    # Wrap tuple-returning models so fvcore sees a single tensor output
    class _Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            out = self.m(x)
            return out[0] if isinstance(out, (tuple, list)) else out

    flops = FlopCountAnalysis(_Wrapper(model), dummy)
    flops.unsupported_ops_warnings(False)
    flops.uncalled_modules_warnings(False)
    return flops.total() / 1e9   # GFLOPs
