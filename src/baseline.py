"""A small 1D CNN baseline for 30-second single-channel EEG epochs."""

import torch
import torch.nn as nn

from src.config import N_CLASSES


class CNNBaseline(nn.Module):
    """Two conv1d blocks -> global average pool -> linear classifier.

    Small on purpose: this is the reference to beat, not the best possible
    baseline. Input: (batch, 1, samples). Output: (batch, N_CLASSES) logits.
    """

    def __init__(self, n_classes: int = N_CLASSES):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(32, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x).squeeze(-1)  # (batch, 32)
        return self.classifier(x)
