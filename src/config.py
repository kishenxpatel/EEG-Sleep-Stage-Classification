"""Shared configuration: paths, device, seed, and hyperparameters."""

import random
from pathlib import Path

import numpy as np
import torch

# --- Paths ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CASSETTE_DIR = DATA_DIR / "sleep-cassette"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
FIGURES_DIR = ROOT_DIR / "figures"

# --- Signal / task constants ---
CHANNEL = "EEG Fpz-Cz"  # standard single-channel choice for Sleep-EDF
SFREQ = 100  # Hz, after resampling
EPOCH_SEC = 30  # seconds per scored epoch
SAMPLES_PER_EPOCH = SFREQ * EPOCH_SEC  # 3000
BANDPASS = (0.3, 30.0)  # Hz

# Sleep-EDF hypnogram annotation descriptions -> 5-class label ids.
# Stages 3 and 4 are merged into N3, per AASM convention (the source R&K
# scoring separates them but modern practice and this project's label set
# do not). "Movement" and "Unknown" epochs are dropped entirely (see data.py).
STAGE_TO_LABEL = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
}
LABEL_NAMES = ["W", "N1", "N2", "N3", "REM"]
N_CLASSES = len(LABEL_NAMES)

# --- Reproducibility ---
SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Seed python/numpy/torch RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Hyperparameters ---
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 30
PATCH_SIZE = 50  # samples per patch (0.5s at 100Hz) -- used by the transformer in Phase 2
EARLY_STOPPING_PATIENCE = 5
