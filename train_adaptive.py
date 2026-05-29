"""Train the multi-exit (adaptive) DeiT-tiny on CIFAR-100.

Three exit classifiers are placed after transformer blocks 4, 8, and 12.
All exits are trained jointly with equal-weight cross-entropy losses.
"""

from pathlib import Path

import torch
import torch.nn as nn

from src.data import get_cifar100_loaders
from src.metrics import count_parameters, measure_flops_giga, save_metrics
from src.models import MultiExitDeiT
from src.train import eval_adaptive_epoch, train_adaptive_epoch
from src.utils import get_device, set_seed

BATCH_SIZE   = 32
EPOCHS       = 20
LR           = 1e-4
WEIGHT_DECAY = 1e-4
SEED         = 42
NUM_CLASSES  = 100
MODEL_NAME   = "adaptive"
CKPT_DIR     = Path("checkpoints") / MODEL_NAME
RESULTS_DIR  = Path("results")     / MODEL_NAME


def _make_exit_flops_wrapper(model, exit_idx):
    """Return a nn.Module whose forward runs only up to exit_idx (for fvcore)."""
    class _ExitWrapper(nn.Module):
        def __init__(self, m, e):
            super().__init__()
            self.m = m
            self.e = e

        def forward(self, x):
            return self.m.forward_to_exit(x, self.e)

    return _ExitWrapper(model, exit_idx)


def main():
    set_seed(SEED)
    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader = get_cifar100_loaders(batch_size=BATCH_SIZE)

    model = MultiExitDeiT(num_classes=NUM_CLASSES, pretrained=True).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    params = count_parameters(model)
    # Measure FLOPs for each exit on CPU (fvcore doesn't need GPU).
    model_cpu  = MultiExitDeiT(num_classes=NUM_CLASSES, pretrained=False)
    flops_list = []
    for i in range(MultiExitDeiT.NUM_EXITS):
        wrapper    = _make_exit_flops_wrapper(model_cpu, i)
        flops_giga = measure_flops_giga(wrapper)
        flops_list.append(round(flops_giga, 4))
        print(f"FLOPs exit {i} (blocks {MultiExitDeiT.EXIT_BLOCKS[i]}): "
              f"{flops_giga:.3f} GFLOPs")
    del model_cpu

    print(f"Parameters: {params:,}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    best_val_acc  = 0.0  # tracked using the final (full-depth) exit
    epoch_history = []

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accs = train_adaptive_epoch(
            model, train_loader, optimizer, device
        )
        val_accs = eval_adaptive_epoch(model, val_loader, device)

        val_acc_final = val_accs[-1]  # exit 2 = full 12-block model
        epoch_history.append({
            "epoch":          epoch,
            "train_loss":     round(train_loss,     6),
            "train_acc_e0":   round(train_accs[0],  4),
            "train_acc_e1":   round(train_accs[1],  4),
            "train_acc_e2":   round(train_accs[2],  4),
            "val_acc_e0":     round(val_accs[0],    4),
            "val_acc_e1":     round(val_accs[1],    4),
            "val_acc_e2":     round(val_accs[2],    4),
        })
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"loss {train_loss:.4f} | "
            f"val e0 {val_accs[0]:.4f} | "
            f"val e1 {val_accs[1]:.4f} | "
            f"val e2 {val_accs[2]:.4f}"
        )

        if val_acc_final > best_val_acc:
            best_val_acc = val_acc_final
            torch.save(model.state_dict(), CKPT_DIR / "best_model.pth")
            print(f"  → saved checkpoint (val_acc_e2={best_val_acc:.4f})")

    save_metrics(
        path               = str(RESULTS_DIR / "metrics.json"),
        model_name         = MODEL_NAME,
        parameters         = params,
        flops_giga         = flops_list[-1],   # full model GFLOPs
        best_val_acc       = round(best_val_acc, 4),
        epoch_history      = epoch_history,
        flops_giga_per_exit = flops_list,
        exit_blocks        = MultiExitDeiT.EXIT_BLOCKS,
    )
    print(f"Done. Best val acc (exit 2): {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
