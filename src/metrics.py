import json
from pathlib import Path

import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_flops_giga(model: nn.Module, input_shape=(1, 3, 224, 224)) -> float:
    """Return GFLOPs for one forward pass using fvcore."""
    from fvcore.nn import FlopCountAnalysis

    model.eval()
    dummy = torch.zeros(input_shape)
    flops = FlopCountAnalysis(model, dummy)
    flops.unsupported_ops_warnings(False)
    flops.uncalled_modules_warnings(False)
    return flops.total() / 1e9


def save_metrics(
    path: str,
    model_name: str,
    parameters: int,
    flops_giga: float,
    best_val_acc: float,
    epoch_history: list[dict],
    **extras,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    record = {
        "model_name":   model_name,
        "parameters":   parameters,
        "flops_giga":   flops_giga,
        "best_val_acc": best_val_acc,
        "epoch_history": epoch_history,
        **extras,
    }
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"Saved metrics → {path}")
