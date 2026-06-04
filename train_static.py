"""Train static-pruned DeiT-tiny on CIFAR-100 (fixed token budget)."""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from src.data import get_cifar100_loaders
from src.models.static_pruned import StaticPrunedDeiT
from src.flops_utils import estimate_flops


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--keep_tokens', type=int, required=True,
                        help='Number of patch tokens to keep (e.g. 49, 98, 147)')
    parser.add_argument('--data_dir', default='/home/arooba/compute-aware-vit-thesis/data/')
    parser.add_argument('--out_dir', default=None)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = f'outputs/static_{args.keep_tokens}'

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}  keep_tokens={args.keep_tokens}', flush=True)

    train_loader, val_loader = get_cifar100_loaders(
        args.data_dir, args.batch_size, args.num_workers, args.seed
    )

    model = StaticPrunedDeiT(keep_tokens=args.keep_tokens, num_classes=100, pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    print(f'Parameters: {model.num_parameters:,}', flush=True)

    epoch_history = []
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.out_dir, 'best_model.pth'))

        epoch_history.append({
            'epoch': epoch,
            'train_loss': round(train_loss, 4),
            'train_acc': round(train_acc, 4),
            'val_loss': round(val_loss, 4),
            'val_acc': round(val_acc, 4),
        })
        print(
            f'Epoch {epoch:02d}/{args.epochs} | '
            f'train_loss={train_loss:.4f} train_acc={train_acc:.4f} | '
            f'val_loss={val_loss:.4f} val_acc={val_acc:.4f}',
            flush=True
        )

    flops_giga = estimate_flops(model, device=str(device))
    token_ratio = args.keep_tokens / 196.0

    metrics = {
        'model': f'deit_tiny_static_{args.keep_tokens}',
        'keep_tokens': args.keep_tokens,
        'token_ratio': round(token_ratio, 4),
        'parameters': model.num_parameters,
        'flops_giga': round(flops_giga, 4),
        'best_val_acc': round(best_val_acc, 4),
        'epoch_history': epoch_history,
    }
    with open(os.path.join(args.out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'\nBest val acc: {best_val_acc:.4f}  FLOPs: {flops_giga:.4f} G  ratio: {token_ratio:.2%}', flush=True)


if __name__ == '__main__':
    main()
