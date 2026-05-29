"""Evaluate the trained multi-exit model.

For a range of confidence thresholds, simulate per-image routing and report:
  - Top-1 accuracy
  - Fraction of images exiting at each block depth
  - Average GFLOPs per image
  - Comparison to static baselines (loaded from their metrics.json files)

Results saved to results/adaptive_eval/threshold_sweep.json.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn

from src.data import get_cifar100_loaders
from src.metrics import count_parameters, measure_flops_giga
from src.models import MultiExitDeiT
from src.utils import get_device, set_seed

BATCH_SIZE  = 32
SEED        = 42
NUM_CLASSES = 100
CKPT_PATH   = Path("checkpoints/adaptive/best_model.pth")
RESULTS_DIR = Path("results/adaptive_eval")

THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]


def _make_exit_flops_wrapper(model, exit_idx):
    class _W(nn.Module):
        def __init__(self, m, e):
            super().__init__()
            self.m = m
            self.e = e

        def forward(self, x):
            return self.m.forward_to_exit(x, self.e)

    return _W(model, exit_idx)


@torch.no_grad()
def collect_exit_logits(model, loader, device):
    """Run all validation images through the model, collect per-exit logits."""
    model.eval()
    all_exit_batches = []
    all_labels       = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        exit_logits = model(x)  # list of [B, C]
        all_exit_batches.append([l.cpu() for l in exit_logits])
        all_labels.append(y.cpu())

    num_exits    = len(all_exit_batches[0])
    exit_logits  = [
        torch.cat([b[e] for b in all_exit_batches]) for e in range(num_exits)
    ]
    labels       = torch.cat(all_labels)
    return exit_logits, labels


def simulate_routing(exit_logits, labels, threshold):
    """Route each sample to the first exit where max-prob >= threshold.

    Returns:
        accuracy       : float
        exit_fractions : list[float]  (fraction exiting at each depth)
    """
    N         = labels.shape[0]
    num_exits = len(exit_logits)

    exited       = torch.zeros(N, dtype=torch.bool)
    assigned     = torch.full((N,), num_exits - 1, dtype=torch.long)
    final_logits = exit_logits[-1].clone()

    for e_idx in range(num_exits - 1):
        max_prob = exit_logits[e_idx].softmax(dim=-1).max(dim=-1).values
        take     = (max_prob >= threshold) & ~exited
        assigned[take]     = e_idx
        final_logits[take] = exit_logits[e_idx][take]
        exited |= take

    accuracy        = (final_logits.argmax(1) == labels).float().mean().item()
    exit_fractions  = [(assigned == e).float().mean().item() for e in range(num_exits)]
    return accuracy, exit_fractions


def load_comparison_metrics():
    comparisons = {}
    for name in ("baseline", "static_4", "static_8"):
        p = Path(f"results/{name}/metrics.json")
        if p.exists():
            with open(p) as f:
                comparisons[name] = json.load(f)
    return comparisons


def main():
    set_seed(SEED)
    device = get_device()
    print(f"Device: {device}\n")

    _, val_loader = get_cifar100_loaders(batch_size=BATCH_SIZE)

    model = MultiExitDeiT(num_classes=NUM_CLASSES, pretrained=False).to(device)
    state = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(state)
    print(f"Loaded checkpoint: {CKPT_PATH}\n")

    # Measure FLOPs for each exit on CPU.
    model_cpu = MultiExitDeiT(num_classes=NUM_CLASSES, pretrained=False)
    flops_per_exit = []
    for i in range(MultiExitDeiT.NUM_EXITS):
        w = _make_exit_flops_wrapper(model_cpu, i)
        flops_per_exit.append(measure_flops_giga(w))
    del model_cpu
    full_flops = flops_per_exit[-1]
    print("GFLOPs per exit:", [f"{f:.3f}" for f in flops_per_exit])

    # Per-exit individual accuracy (no routing, each exit acting alone).
    print("\nCollecting per-image exit logits ...")
    exit_logits, labels = collect_exit_logits(model, val_loader, device)
    N = labels.shape[0]

    per_exit_acc = [
        (exit_logits[e].argmax(1) == labels).float().mean().item()
        for e in range(MultiExitDeiT.NUM_EXITS)
    ]
    for e, (acc, flops) in enumerate(zip(per_exit_acc, flops_per_exit)):
        print(f"  Exit {e} (block {MultiExitDeiT.EXIT_BLOCKS[e]:2d}): "
              f"acc={acc:.4f}, flops={flops:.3f} GFLOPs")

    # Threshold sweep.
    print(f"\n{'Threshold':>10} | {'Acc':>7} | "
          f"{'E@4':>6} | {'E@8':>6} | {'E@12':>6} | "
          f"{'GFLOPs':>8} | {'FLOPs%':>7}")
    print("-" * 70)

    sweep_results = []
    for tau in THRESHOLDS:
        acc, fracs = simulate_routing(exit_logits, labels, tau)
        avg_flops  = sum(f * gf for f, gf in zip(fracs, flops_per_exit))
        flops_pct  = avg_flops / full_flops * 100.0
        print(
            f"{tau:>10.2f} | {acc:>7.4f} | "
            f"{fracs[0]:>6.3f} | {fracs[1]:>6.3f} | {fracs[2]:>6.3f} | "
            f"{avg_flops:>8.3f} | {flops_pct:>6.1f}%"
        )
        sweep_results.append({
            "threshold":          tau,
            "accuracy":           round(acc,       4),
            "exit_fractions":     [round(f, 4) for f in fracs],
            "avg_flops_giga":     round(avg_flops, 4),
            "avg_flops_fraction": round(avg_flops / full_flops, 4),
        })

    # Load and print comparison metrics.
    comparisons = load_comparison_metrics()
    if comparisons:
        print("\n--- Static baselines (for reference) ---")
        for name, m in comparisons.items():
            print(f"  {name:12s}: acc={m['best_val_acc']:.4f}, "
                  f"flops={m['flops_giga']:.3f} GFLOPs")

    # Save results.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "flops_per_exit_giga": [round(f, 4) for f in flops_per_exit],
        "per_exit_accuracy":   [round(a, 4) for a in per_exit_acc],
        "exit_blocks":         MultiExitDeiT.EXIT_BLOCKS,
        "threshold_sweep":     sweep_results,
        "comparisons":         comparisons,
    }
    out_path = RESULTS_DIR / "threshold_sweep.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved evaluation results → {out_path}")


if __name__ == "__main__":
    main()
