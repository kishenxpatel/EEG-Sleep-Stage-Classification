"""Reusable training loop: early stopping, validation, history tracking.

Shared by the baseline CNN (Phase 1) and the transformer (Phase 2) so both
models are trained under identical conditions -- same optimiser, same
schedule, same stopping rule -- which is what makes comparing their F1
scores a fair comparison rather than an artifact of tuning one more than
the other.
"""

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import EARLY_STOPPING_PATIENCE, EPOCHS, LEARNING_RATE, get_device


@dataclass
class History:
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[float, float]:
    """One pass over loader. Trains if optimizer is given, else evaluates."""
    is_train = optimizer is not None
    model.train(mode=is_train)

    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if is_train:
                optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            n += x.size(0)

    return total_loss / n, correct / n


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    patience: int = EARLY_STOPPING_PATIENCE,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, History]:
    """Train with Adam + cross-entropy, early stopping on val loss.

    Restores the best (lowest val loss) weights before returning, so the
    caller always gets the checkpoint that generalised best, not whatever
    epoch training happened to stop on.
    """
    device = device or get_device()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = History()
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(epochs):
        train_loss, _ = _run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, device)

        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.val_acc.append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"epoch {epoch + 1}/{epochs}  train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        )

        if epochs_without_improvement >= patience:
            print(f"early stopping at epoch {epoch + 1} (no val improvement for {patience} epochs)")
            break

    model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: Optional[torch.device] = None
) -> Dict[str, np.ndarray]:
    """Run inference over loader. Returns predictions, true labels, and softmax probs.

    The probs are kept (not just argmax) because Phase 3's confidence
    calibration analysis needs them.
    """
    device = device or get_device()
    model = model.to(device).eval()

    all_preds, all_labels, all_probs = [], [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        all_preds.append(probs.argmax(dim=1).cpu().numpy())
        all_labels.append(y.numpy())
        all_probs.append(probs.cpu().numpy())

    return {
        "preds": np.concatenate(all_preds),
        "labels": np.concatenate(all_labels),
        "probs": np.concatenate(all_probs),
    }
