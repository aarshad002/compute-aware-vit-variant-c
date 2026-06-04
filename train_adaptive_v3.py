"""
Train adaptive token-pruning DeiT-tiny v3 on CIFAR-100.

What changed from v2 and why
-----------------------------
v2 collapsed to maximum budget (token_ratio=1.0 every epoch).
Root cause: blended CE gradient is always biased toward budget 196 because
the pretrained backbone performs much better at 196 tokens than at 49 tokens
early in training.  The budget cost term (λ=0.1 × max_diff=0.75 → 0.075) is
too weak to overcome a CE accuracy gap of 0.5–1.5 nats.

Three-part fix in v3
--------------------
1. Auxiliary CE at every budget level (L_aux):
   L_aux = (1/K) Σ_k CE(logits_k, labels)
   Gradient flows through backbone at ALL token counts, forcing the backbone
   to learn good representations at 49 tokens.  Once logits_k are comparable
   across k, the blended CE gradient is no longer biased toward budget 196.
   Note: L_aux does NOT flow through budget_predictor (budget_probs not in
   CE(logits_k, labels)), so it has zero effect on routing, only on the backbone.

2. Entropy regularisation on budget probs (-λ_ent · H):
   H = -Σ_k p_k log p_k   (entropy of Gumbel-softmax output)
   Minimising (−λ_ent · H) maximises entropy → keeps budget probs spread during
   early epochs before L_aux has had time to level the playing field.
   Coefficient decays linearly to 0 over entropy_anneal_epochs so the model
   can commit to discrete routing later.

3. Stronger budget cost weight (λ_budget=0.5 default vs v2's 0.1):
   Once the CE playing field is levelled, a larger efficiency penalty is needed
   to push the predictor toward cheap budgets.

Full objective
--------------
    L = L_aux + L_blend + λ_budget · budget_cost − λ_ent · H

All other hyperparameters unchanged from v2 (lr=1e-4, Adam, batch=32, seed=42).
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
from src.models.adaptive_pruned_v3 import AdaptivePrunedDeiTv3, BUDGET_TOKENS
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
    t = (epoch - 1) / (total_epochs - 1)
    return tau_final + (tau_init - tau_final) * math.exp(-3.0 * t)


def get_lambda_budget(epoch, budget_weight, warmup_epochs):
    """Linear λ warmup for budget cost term."""
    return budget_weight * min(1.0, epoch / max(1, warmup_epochs))


def get_lambda_entropy(epoch, ent_init, anneal_epochs):
    """Linear decay of entropy regularisation coefficient to 0."""
    if epoch >= anneal_epochs:
        return 0.0
    return ent_init * (1.0 - (epoch - 1) / anneal_epochs)


def train_one_epoch(model, loader, optimizer, criterion, device,
                    lambda_budget, lambda_entropy):
    model.train()
    total_loss = total_aux = total_blend = total_budget = 0.0
    correct = total = 0
    K = len(model.budget_tokens)

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        all_logits, blended, budget_cost, entropy = model(imgs)

        # L_aux: trains backbone at every budget level (no gradient to predictor)
        L_aux = sum(criterion(lk, labels) for lk in all_logits) / K

        # L_blend: trains predictor toward accuracy-preserving routing
        L_blend = criterion(blended, labels)

        # Budget penalty (after warmup): trains predictor toward cheaper budgets
        L_budget = lambda_budget * budget_cost

        # Entropy regularisation (early epochs): prevents collapse to max budget
        L_entropy = -lambda_entropy * entropy

        loss = L_aux + L_blend + L_budget + L_entropy
        loss.backward()
        optimizer.step()

        B = imgs.size(0)
        total_loss   += loss.item() * B
        total_aux    += L_aux.item() * B
        total_blend  += L_blend.item() * B
        total_budget += budget_cost.item() * B
        correct      += (blended.argmax(1) == labels).sum().item()
        total        += B

    N = total
    return (total_loss / N, total_aux / N, total_blend / N,
            correct / N, total_budget / N)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = total_ratio = 0.0
    correct = total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits, mean_ratio = model(imgs)
        loss = criterion(logits, labels)

        B = imgs.size(0)
        total_loss  += loss.item() * B
        total_ratio += mean_ratio * B
        correct     += (logits.argmax(1) == labels).sum().item()
        total       += B

    return total_loss / total, correct / total, total_ratio / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/home/arooba/compute-aware-vit-thesis/data/')
    parser.add_argument('--out_dir', default='outputs/adaptive_v3')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--budget_weight', type=float, default=0.5,
                        help='λ on budget cost term (after warmup)')
    parser.add_argument('--lambda_warmup_epochs', type=int, default=5)
    parser.add_argument('--ent_weight_init', type=float, default=0.5,
                        help='Initial entropy regularisation coefficient (decays to 0)')
    parser.add_argument('--entropy_anneal_epochs', type=int, default=10,
                        help='Decay entropy coefficient to 0 over this many epochs')
    parser.add_argument('--tau_init', type=float, default=3.0)
    parser.add_argument('--tau_final', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(
        f'Device: {device} | budget_weight={args.budget_weight} '
        f'λ_warmup={args.lambda_warmup_epochs} | '
        f'ent_init={args.ent_weight_init} anneal={args.entropy_anneal_epochs}ep | '
        f'tau: {args.tau_init} → {args.tau_final}',
        flush=True
    )

    train_loader, val_loader = get_cifar100_loaders(
        args.data_dir, args.batch_size, args.num_workers, args.seed
    )

    model = AdaptivePrunedDeiTv3(
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
        tau = get_tau(epoch, args.epochs, args.tau_init, args.tau_final)
        model.gumbel_temperature = tau
        lambda_budget = get_lambda_budget(epoch, args.budget_weight, args.lambda_warmup_epochs)
        lambda_entropy = get_lambda_entropy(epoch, args.ent_weight_init, args.entropy_anneal_epochs)

        train_loss, train_aux, train_blend, train_acc, train_budget = train_one_epoch(
            model, train_loader, optimizer, criterion, device, lambda_budget, lambda_entropy
        )
        val_loss, val_acc, val_ratio = evaluate(model, val_loader, criterion, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.out_dir, 'best_model.pth'))

        epoch_history.append({
            'epoch': epoch,
            'tau': round(tau, 4),
            'lambda_budget': round(lambda_budget, 4),
            'lambda_entropy': round(lambda_entropy, 4),
            'train_loss': round(train_loss, 4),
            'train_ce_aux': round(train_aux, 4),
            'train_ce_blend': round(train_blend, 4),
            'train_acc': round(train_acc, 4),
            'train_budget_cost': round(train_budget, 4),
            'val_loss': round(val_loss, 4),
            'val_acc': round(val_acc, 4),
            'val_mean_token_ratio': round(val_ratio, 4),
        })
        print(
            f'Epoch {epoch:02d}/{args.epochs} | '
            f'τ={tau:.3f} λ_b={lambda_budget:.3f} λ_e={lambda_entropy:.3f} | '
            f'loss={train_loss:.4f} aux={train_aux:.4f} blend={train_blend:.4f} '
            f'budget={train_budget:.4f} acc={train_acc:.4f} | '
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
        torch.load(os.path.join(args.out_dir, 'best_model.pth'),
                   map_location=device, weights_only=True)
    )
    _, best_val_acc_recheck, final_val_ratio = evaluate(model, val_loader, criterion, device)

    metrics = {
        'model': 'deit_tiny_adaptive_v3',
        'budget_tokens': model.budget_tokens,
        'budget_cost_weight': args.budget_weight,
        'ent_weight_init': args.ent_weight_init,
        'entropy_anneal_epochs': args.entropy_anneal_epochs,
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
