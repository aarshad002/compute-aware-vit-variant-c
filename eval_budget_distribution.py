"""
Evaluate the per-image budget distribution of the trained Adaptive DeiT-tiny v3.

Loads outputs/adaptive_v3/best_model.pth, runs the CIFAR-100 validation set
with hard-argmax budget routing, and reports what fraction of images were
assigned to each of the three token budgets (49 / 98 / 196).
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from src.data import get_cifar100_loaders
from src.models.adaptive_pruned_v3 import AdaptivePrunedDeiTv3, _cls_attention_scores


@torch.no_grad()
def eval_budget_distribution(model, loader, device):
    """Return per-budget counts and overall val accuracy."""
    model.eval()
    counts = {k: 0 for k in model.budget_tokens}
    correct = total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        B = imgs.size(0)

        # Shared prefix: patch embed + first prune_after_block blocks
        x = model._embed(imgs)
        x = model._run_blocks_until(x, model.prune_after_block)

        # Token scores and budget decision (hard argmax, same as _forward_eval)
        pivot = model.blocks[model.prune_after_block]
        scores = _cls_attention_scores(pivot, x)          # [B, 196]
        budget_logits = model.budget_predictor(x[:, 0])   # [B, K]
        budget_idx = budget_logits.argmax(dim=1)           # [B]

        # Run each budget group and accumulate counts
        logits = torch.zeros(B, model.head.out_features, device=device)
        for bi, k in enumerate(model.budget_tokens):
            mask = (budget_idx == bi)
            counts[k] += mask.sum().item()
            if not mask.any():
                continue
            x_g = model._prune_to_k(x[mask], scores[mask], k)
            x_g = model._run_blocks_from(x_g, model.prune_after_block)
            x_g = model.norm(x_g)
            logits[mask] = model.head(x_g[:, 0])

        correct += (logits.argmax(1) == labels).sum().item()
        total += B

    return counts, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',   default='/home/arooba/compute-aware-vit-thesis/data/')
    parser.add_argument('--checkpoint', default='outputs/adaptive_v3/best_model.pth')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed',       type=int, default=42)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}', flush=True)

    _, val_loader = get_cifar100_loaders(
        args.data_dir, args.batch_size, args.num_workers, args.seed
    )

    model = AdaptivePrunedDeiTv3(num_classes=100, pretrained=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    print(f'Loaded: {args.checkpoint}', flush=True)
    print(f'Budget levels (tokens): {model.budget_tokens}', flush=True)

    counts, val_acc = eval_budget_distribution(model, val_loader, device)
    total = sum(counts.values())

    print(f'\nValidation accuracy : {val_acc:.4f}  ({val_acc * 100:.2f}%)')
    print(f'\nBudget distribution over {total} validation images:')
    print(f'  {"Tokens":>8}  {"Images":>8}  {"Percent":>8}')
    print(f'  {"-" * 30}')
    for k in model.budget_tokens:
        pct = 100.0 * counts[k] / total
        print(f'  {k:>8}  {counts[k]:>8}  {pct:>7.2f}%')

    mean_tokens = sum(k * counts[k] for k in model.budget_tokens) / total
    mean_ratio  = mean_tokens / 196.0
    print(f'\n  Mean tokens / image : {mean_tokens:.1f}  (ratio = {mean_ratio:.4f})')
    print(f'  Compute savings     : {(1.0 - mean_ratio) * 100:.1f}% vs dense baseline')


if __name__ == '__main__':
    main()
