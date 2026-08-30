"""Patch-based transformer classifier for 30-second single-channel EEG epochs.

Divides the epoch into short patches, embeds them, adds positional
information, and runs a small transformer encoder over the resulting
token sequence. Sized to stay under a few million parameters so it trains
on Colab's free GPU tier in reasonable time -- this is deliberately not a
large model, and it is not trying to be.
"""

import math

import torch
import torch.nn as nn

from src.config import (
    DIM_FEEDFORWARD,
    DROPOUT,
    EMBED_DIM,
    N_CLASSES,
    N_HEADS,
    N_LAYERS,
    PATCH_SIZE,
    SAMPLES_PER_EPOCH,
)


class PatchEmbedding(nn.Module):
    """Projects contiguous, non-overlapping patches of the raw signal to embeddings.

    A conv1d with kernel_size == stride == patch_size is exactly a linear
    projection applied independently to each non-overlapping patch -- one
    conv layer instead of a manual unfold + linear, same result.
    """

    def __init__(self, patch_size: int = PATCH_SIZE, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.project = nn.Conv1d(
            in_channels=1, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, samples) -> (batch, embed_dim, n_patches) -> (batch, n_patches, embed_dim)
        x = self.project(x)
        return x.transpose(1, 2)


class SinusoidalPositionalEncoding(nn.Module):
    """Standard fixed (non-learned) sinusoidal positional encoding."""

    def __init__(self, embed_dim: int, max_len: int = 128):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerClassifier(nn.Module):
    """Patch embedding -> [CLS] + positional encoding -> transformer encoder -> classifier.

    Pooling choice: a learned [CLS] token, not global average pooling over
    patch tokens. With self-attention, a dedicated query token can learn to
    attend selectively to the patches that matter for a given epoch (e.g.
    weighting a transient K-complex more heavily than a flat stretch),
    whereas GAP weights every patch equally by construction. For sleep
    staging specifically, some patches within an epoch are far more
    diagnostic than others, so the extra flexibility is worth the one
    additional token.
    """

    def __init__(
        self,
        n_classes: int = N_CLASSES,
        patch_size: int = PATCH_SIZE,
        embed_dim: int = EMBED_DIM,
        n_heads: int = N_HEADS,
        n_layers: int = N_LAYERS,
        dim_feedforward: int = DIM_FEEDFORWARD,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        n_patches = SAMPLES_PER_EPOCH // patch_size

        self.patch_embed = PatchEmbedding(patch_size, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_encoding = SinusoidalPositionalEncoding(embed_dim, max_len=n_patches + 1)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, n_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, samples)
        batch_size = x.size(0)
        tokens = self.patch_embed(x)  # (batch, n_patches, embed_dim)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # (batch, n_patches + 1, embed_dim)

        tokens = self.pos_encoding(tokens)
        tokens = self.dropout(tokens)

        encoded = self.encoder(tokens)
        cls_output = self.norm(encoded[:, 0])  # take the [CLS] position
        return self.classifier(cls_output)
