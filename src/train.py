from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


# ------------------------------------------------------------------ #
#  Standard single-output model                                        #
# ------------------------------------------------------------------ #

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss   = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)
        correct    += logits.argmax(1).eq(y).sum().item()
        total      += len(y)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Returns top-1 accuracy."""
    model.eval()
    correct = 0
    total   = 0

    for x, y in tqdm(loader, desc="val  ", leave=False):
        x, y    = x.to(device), y.to(device)
        logits  = model(x)
        correct += logits.argmax(1).eq(y).sum().item()
        total   += len(y)

    return correct / total


# ------------------------------------------------------------------ #
#  Multi-exit model                                                    #
# ------------------------------------------------------------------ #

def train_adaptive_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, list[float]]:
    """Returns (avg_loss, [acc_exit_0, acc_exit_1, acc_exit_2])."""
    model.train()
    total_loss = 0.0
    num_exits  = model.NUM_EXITS
    correct    = [0] * num_exits
    total      = 0

    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        exit_logits = model(x)  # list of [B, C]

        # Equal-weight cross-entropy over all exits.
        losses = [F.cross_entropy(logits, y) for logits in exit_logits]
        loss   = sum(losses) / len(losses)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)
        for i, logits in enumerate(exit_logits):
            correct[i] += logits.argmax(1).eq(y).sum().item()
        total += len(y)

    accs = [c / total for c in correct]
    return total_loss / total, accs


@torch.no_grad()
def eval_adaptive_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> list[float]:
    """Returns per-exit top-1 accuracies [acc_exit_0, acc_exit_1, acc_exit_2]."""
    model.eval()
    num_exits = model.NUM_EXITS
    correct   = [0] * num_exits
    total     = 0

    for x, y in tqdm(loader, desc="val  ", leave=False):
        x, y = x.to(device), y.to(device)
        exit_logits = model(x)
        for i, logits in enumerate(exit_logits):
            correct[i] += logits.argmax(1).eq(y).sum().item()
        total += len(y)

    return [c / total for c in correct]
