"""
Train adaptive token-pruning DeiT-tiny v2 on CIFAR-100.

Key differences from v1
-----------------------
1. Soft budget blending (core fix): CE loss is differentiable w.r.t. the budget
   predictor — all three budget-level logits are computed and blended per image.
2. Temperature annealing: Gumbel τ decays from tau_init to tau_final over training
   so that all budget paths receive gradient early and the predictor commits later.
3. λ warmup: budget penalty ramps from 0 to budget_weight over lambda_warmup_epochs
   so the randomly-initialised head stabilises before efficiency pressure kicks in.
"""
import argparse
import json
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from src.data import get_cifar100_loaders
from src.models.adaptive_pruned_v2 import AdaptivePrunedDeiTv2, BUDGET_TOKENS
from src.flops_utils import estimate_flops


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_tau(epoch, total_epochs, tau_init, tau_final):
    """Exponential temperature annealing."""
    if total_epochs <= 1:
        return tau_final
    t = (epoch - 1) / (total_epochs - 1)           # 0.0 … 1.0
    return tau_final + (tau_init - tau_final) * math.exp(-3.0 * t)


def get_lambda(epoch, budget_weight, warmup_epochs):
    """Linear λ warmup."""
    return budget_weight * min(1.0, epoch / max(1, warmup_epochs))


def train_one_epoch(model, loader, optimizer, criterion, device, lambda_eff):
    model.train()
    total_ce, total_budget, correct, total = 0.0, 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        logits, budget_cost = model(imgs)
        ce_loss = criterion(logits, labels)
        loss = ce_loss + lambda_eff * budget_cost

        loss.backward()
        optimizer.step()

        total_ce     += ce_loss.item() * imgs.size(0)
        total_budget += budget_cost.item() * imgs.size(0)
        correct      += (logits.argmax(1) == labels).sum().item()
        total        += imgs.size(0)

    return total_ce / total, correct / total, total_budget / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    total_ratio = 0.0
    budget_counts = [0] * len(model.budget_tokens)

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits, mean_ratio = model(imgs)
        loss = criterion(logits, labels)

        total_loss  += loss.item() * imgs.size(0)
        total_ratio += mean_ratio * imgs.size(0)
        correct     += (logits.argmax(1) == labels).sum().item()
        total       += imgs.size(0)

    return total_loss / total, correct / total, total_ratio / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/home/arooba/compute-aware-vit-thesis/data/')
    parser.add_argument('--out_dir', default='outputs/adaptive_v2')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--budget_weight', type=float, default=0.1,
                        help='λ on the budget cost term (after warmup)')
    parser.add_argument('--lambda_warmup_epochs', type=int, default=5,
                        help='Ramp λ from 0 to budget_weight over this many epochs')
    parser.add_argument('--tau_init', type=float, default=3.0,
                        help='Initial Gumbel temperature (high = soft/exploratory)')
    parser.add_argument('--tau_final', type=float, default=0.5,
                        help='Final Gumbel temperature (low = near hard assignment)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(
        f'Device: {device} | budget_weight={args.budget_weight} | '
        f'lambda_warmup={args.lambda_warmup_epochs} | '
        f'tau: {args.tau_init} → {args.tau_final}',
        flush=True
    )

    train_loader, val_loader = get_cifar100_loaders(
        args.data_dir, args.batch_size, args.num_workers, args.seed
    )

    model = AdaptivePrunedDeiTv2(
        num_classes=100,
        pretrained=True,
        gumbel_tau_init=args.tau_init,
        gumbel_tau_final=args.tau_final,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    print(f'Parameters: {model.num_parameters:,}', flush=True)
    print(f'Budget levels (tokens): {model.budget_tokens}', flush=True)

    epoch_history = []
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        # Update temperature and effective lambda for this epoch
        tau = get_tau(epoch, args.epochs, args.tau_init, args.tau_final)
        model.gumbel_temperature = tau
        lambda_eff = get_lambda(epoch, args.budget_weight, args.lambda_warmup_epochs)

        train_loss, train_acc, train_budget = train_one_epoch(
            model, train_loader, optimizer, criterion, device, lambda_eff
        )
        val_loss, val_acc, val_ratio = evaluate(model, val_loader, criterion, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.out_dir, 'best_model.pth'))

        epoch_history.append({
            'epoch': epoch,
            'tau': round(tau, 4),
            'lambda_eff': round(lambda_eff, 4),
            'train_loss': round(train_loss, 4),
            'train_acc': round(train_acc, 4),
            'train_budget_cost': round(train_budget, 4),
            'val_loss': round(val_loss, 4),
            'val_acc': round(val_acc, 4),
            'val_mean_token_ratio': round(val_ratio, 4),
        })
        print(
            f'Epoch {epoch:02d}/{args.epochs} | '
            f'τ={tau:.3f} λ={lambda_eff:.4f} | '
            f'train_loss={train_loss:.4f} train_acc={train_acc:.4f} '
            f'budget={train_budget:.4f} | '
            f'val_loss={val_loss:.4f} val_acc={val_acc:.4f} '
            f'token_ratio={val_ratio:.4f}',
            flush=True
        )

    # FLOPs per budget level (using static model as proxy)
    flops_per_budget = {}
    for k in model.budget_tokens:
        from src.models.static_pruned import StaticPrunedDeiT
        static = StaticPrunedDeiT(keep_tokens=k, num_classes=100, pretrained=False).to(device)
        flops_per_budget[k] = round(estimate_flops(static, device=str(device)), 4)

    # Reload best checkpoint for final eval
    model.load_state_dict(
        torch.load(os.path.join(args.out_dir, 'best_model.pth'), map_location=device)
    )
    _, best_val_acc_recheck, final_val_ratio = evaluate(model, val_loader, criterion, device)

    metrics = {
        'model': 'deit_tiny_adaptive_v2',
        'budget_tokens': model.budget_tokens,
        'budget_cost_weight': args.budget_weight,
        'lambda_warmup_epochs': args.lambda_warmup_epochs,
        'tau_init': args.tau_init,
        'tau_final': args.tau_final,
        'parameters': model.num_parameters,
        'flops_per_budget_giga': flops_per_budget,
        'final_val_mean_token_ratio': round(final_val_ratio, 4),
        'best_val_acc': round(best_val_acc, 4),
        'epoch_history': epoch_history,
    }
    with open(os.path.join(args.out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(
        f'\nBest val acc: {best_val_acc:.4f} | '
        f'Mean token ratio: {final_val_ratio:.4f} | '
        f'FLOPs per budget: {flops_per_budget}',
        flush=True
    )


if __name__ == '__main__':
    main()
