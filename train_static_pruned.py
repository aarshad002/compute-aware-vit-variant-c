"""Train a truncated DeiT-tiny (first N blocks) on CIFAR-100.

Usage:
    python train_static_pruned.py --depth 4
    python train_static_pruned.py --depth 8
"""

import argparse
from pathlib import Path

import torch

from src.data import get_cifar100_loaders
from src.metrics import count_parameters, measure_flops_giga, save_metrics
from src.models import TruncatedDeiT
from src.train import eval_epoch, train_one_epoch
from src.utils import get_device, set_seed

BATCH_SIZE   = 32
EPOCHS       = 20
LR           = 1e-4
WEIGHT_DECAY = 1e-4
SEED         = 42
NUM_CLASSES  = 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, required=True, choices=[4, 8])
    args = parser.parse_args()

    model_name  = f"static_{args.depth}"
    ckpt_dir    = Path("checkpoints") / model_name
    results_dir = Path("results")     / model_name

    set_seed(SEED)
    device = get_device()
    print(f"Depth: {args.depth}  |  Device: {device}")

    train_loader, val_loader = get_cifar100_loaders(batch_size=BATCH_SIZE)

    model = TruncatedDeiT(
        num_classes=NUM_CLASSES, depth=args.depth, pretrained=True
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    params     = count_parameters(model)
    flops_giga = measure_flops_giga(model)
    print(f"Parameters: {params:,}")
    print(f"FLOPs:      {flops_giga:.3f} GFLOPs")

    ckpt_dir.mkdir(parents=True, exist_ok=True)

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
            torch.save(model.state_dict(), ckpt_dir / "best_model.pth")
            print(f"  → saved checkpoint (val_acc={best_val_acc:.4f})")

    save_metrics(
        path          = str(results_dir / "metrics.json"),
        model_name    = model_name,
        parameters    = params,
        flops_giga    = round(flops_giga, 4),
        best_val_acc  = round(best_val_acc, 4),
        epoch_history = epoch_history,
        depth         = args.depth,
    )
    print(f"Done. Best val acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
