"""Train the full DeiT-tiny (12 blocks) baseline on CIFAR-100."""

import argparse
from pathlib import Path

import torch

from src.data import get_cifar100_loaders
from src.metrics import count_parameters, measure_flops_giga, save_metrics
from src.models import DeiTBaseline
from src.train import eval_epoch, train_one_epoch
from src.utils import get_device, set_seed

# ------------------------------------------------------------------ #
#  Hyper-parameters (fixed per CLAUDE.md)                              #
# ------------------------------------------------------------------ #
BATCH_SIZE    = 32
EPOCHS        = 20
LR            = 1e-4
WEIGHT_DECAY  = 1e-4
SEED          = 42
NUM_CLASSES   = 100
MODEL_NAME    = "baseline"
CKPT_DIR      = Path("checkpoints") / MODEL_NAME
RESULTS_DIR   = Path("results")     / MODEL_NAME


def main():
    set_seed(SEED)
    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader = get_cifar100_loaders(batch_size=BATCH_SIZE)

    model = DeiTBaseline(num_classes=NUM_CLASSES, pretrained=True).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    params     = count_parameters(model)
    flops_giga = measure_flops_giga(model)
    print(f"Parameters: {params:,}")
    print(f"FLOPs:      {flops_giga:.3f} GFLOPs")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    best_val_acc  = 0.0
    epoch_history = []

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_acc               = eval_epoch(model, val_loader, device)

        epoch_history.append({
            "epoch":      epoch,
            "train_loss": round(train_loss, 6),
            "train_acc":  round(train_acc,  4),
            "val_acc":    round(val_acc,    4),
        })
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"loss {train_loss:.4f} | "
            f"train {train_acc:.4f} | "
            f"val {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CKPT_DIR / "best_model.pth")
            print(f"  → saved checkpoint (val_acc={best_val_acc:.4f})")

    save_metrics(
        path          = str(RESULTS_DIR / "metrics.json"),
        model_name    = MODEL_NAME,
        parameters    = params,
        flops_giga    = round(flops_giga, 4),
        best_val_acc  = round(best_val_acc, 4),
        epoch_history = epoch_history,
    )
    print(f"Done. Best val acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
