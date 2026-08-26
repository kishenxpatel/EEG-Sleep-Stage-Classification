"""Sleep-EDF loading, preprocessing, epoch extraction, and dataset splitting.

File layout expected under DATA_DIR/sleep-cassette/ (as produced by
scripts/download_data.sh):
    SC4001E0-PSG.edf         <- raw polysomnography recording
    SC4001EC-Hypnogram.edf   <- expert-scored sleep stage annotations

Subject id is the 4-digit number after "SC" (e.g. "4001").
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import mne
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from src.config import (
    BANDPASS,
    CASSETTE_DIR,
    CHANNEL,
    EPOCH_SEC,
    SAMPLES_PER_EPOCH,
    SEED,
    SFREQ,
    STAGE_TO_LABEL,
)

mne.set_log_level("ERROR")


@dataclass
class SubjectFiles:
    subject_id: str
    psg_path: Path
    hyp_path: Path


def discover_subjects(cassette_dir: Path = CASSETTE_DIR) -> List[SubjectFiles]:
    """Pair up PSG/Hypnogram files present in cassette_dir by subject id.

    Only subjects with both files present are returned, so a partial or
    in-progress download degrades gracefully instead of crashing.
    """
    psg_files = sorted(cassette_dir.glob("SC*-PSG.edf"))
    subjects = []
    for psg_path in psg_files:
        match = re.match(r"SC(\d{4})", psg_path.name)
        if not match:
            continue
        subject_id = match.group(1)
        hyp_candidates = sorted(cassette_dir.glob(f"SC{subject_id}*-Hypnogram.edf"))
        if not hyp_candidates:
            continue
        subjects.append(SubjectFiles(subject_id, psg_path, hyp_candidates[0]))
    return subjects


def load_raw(subject: SubjectFiles) -> mne.io.Raw:
    """Load one recording, keep only CHANNEL, filter, resample, attach annotations."""
    raw = mne.io.read_raw_edf(subject.psg_path, include=[CHANNEL], preload=True)
    raw.filter(BANDPASS[0], BANDPASS[1])
    raw.resample(SFREQ)

    annotations = mne.read_annotations(subject.hyp_path)
    raw.set_annotations(annotations, emit_warning=False)

    # Sleep-EDF recordings run for many hours before/after the scored sleep
    # period (the technician started recording before lights-out). Left
    # uncropped, "Wake" would swamp the dataset far beyond its already-high
    # natural prevalence. Crop to the scored period plus a 30-minute buffer
    # on each side, matching the standard preprocessing used in the MNE
    # Sleep-EDF tutorial.
    sleep_annotations = [a for a in annotations if a["description"] in STAGE_TO_LABEL]
    if sleep_annotations:
        first_onset = sleep_annotations[0]["onset"]
        last_annotation = sleep_annotations[-1]
        last_offset = last_annotation["onset"] + last_annotation["duration"]
        buffer_sec = 30 * 60
        tmin = max(0.0, first_onset - buffer_sec)
        tmax = min(raw.times[-1], last_offset + buffer_sec)
        raw.crop(tmin=tmin, tmax=tmax)

    # Per-recording z-score normalisation. EEG amplitude scale varies
    # substantially between subjects (electrode impedance, amplifier gain),
    # which is nuisance variance, not signal -- left unnormalised, a
    # BatchNorm layer's running statistics (fit across many subjects at
    # train time) mismatch an individual subject's scale at eval time,
    # which is what caused validation loss to blow up before this was
    # added. Normalising per-recording (not per-epoch) removes that
    # cross-subject scale variance while preserving the within-recording
    # amplitude differences between stages (e.g. N3 slow waves being much
    # higher amplitude than Wake) that the model actually needs.
    raw.apply_function(_zscore, picks=CHANNEL, channel_wise=True)

    return raw


def _zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-8)


def extract_epochs_labels(raw: mne.io.Raw) -> Tuple[np.ndarray, np.ndarray]:
    """Cut raw into 30s epochs aligned with hypnogram labels.

    Returns:
        epochs: float32 array, shape (n_epochs, SAMPLES_PER_EPOCH)
        labels: int64 array, shape (n_epochs,), values in [0, 4]

    Epochs whose annotation is not one of the five scored stages
    (e.g. "Movement time", "Sleep stage ?") are dropped rather than
    guessed at.
    """
    events, event_id_map = mne.events_from_annotations(
        raw, event_id=STAGE_TO_LABEL, chunk_duration=float(EPOCH_SEC)
    )

    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id_map,
        tmin=0.0,
        tmax=EPOCH_SEC - 1.0 / SFREQ,
        baseline=None,
        preload=True,
    )

    data = epochs.get_data(picks=CHANNEL).squeeze(1).astype(np.float32)  # (n, SAMPLES_PER_EPOCH)
    labels = epochs.events[:, 2].astype(np.int64)

    assert data.shape[1] == SAMPLES_PER_EPOCH, (
        f"expected {SAMPLES_PER_EPOCH} samples/epoch, got {data.shape[1]}"
    )
    return data, labels


def load_subject_epochs(subject: SubjectFiles) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper: raw file pair -> (epochs, labels) for one subject."""
    raw = load_raw(subject)
    return extract_epochs_labels(raw)


def get_subject_split(
    subject_ids: List[str], random_state: int = SEED
) -> Tuple[List[str], List[str], List[str]]:
    """Split subject ids 60/20/20 into train/val/test.

    This split is done at the SUBJECT level, not the epoch level, which
    matters more than it looks. Consecutive epochs from the same subject
    are highly correlated (similar EEG amplitude/noise characteristics,
    slowly-varying sleep architecture), so if epochs from one subject's
    night end up in both train and test, the model can partly identify
    "which subject is this" rather than "what sleep stage is this" --
    it's leakage. An epoch-level random split would let that leakage
    inflate test F1 by a large, misleading margin. Splitting by whole
    subjects instead guarantees the test set contains genuinely unseen
    physiology.
    """
    train_ids, temp_ids = train_test_split(
        subject_ids, train_size=0.6, random_state=random_state
    )
    val_ids, test_ids = train_test_split(
        temp_ids, train_size=0.5, random_state=random_state
    )
    return train_ids, val_ids, test_ids


def build_epoch_arrays(
    subjects: List[SubjectFiles], subject_ids: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and concatenate epochs/labels for the given subject ids.

    Returns epochs, labels, and a parallel array of subject ids (as strings)
    per epoch -- the latter is needed later for per-subject error analysis.
    """
    wanted = {s.subject_id: s for s in subjects}
    all_epochs, all_labels, all_subject_ids = [], [], []
    for sid in subject_ids:
        subject = wanted[sid]
        epochs, labels = load_subject_epochs(subject)
        all_epochs.append(epochs)
        all_labels.append(labels)
        all_subject_ids.append(np.full(len(labels), sid))

    return (
        np.concatenate(all_epochs, axis=0),
        np.concatenate(all_labels, axis=0),
        np.concatenate(all_subject_ids, axis=0),
    )


class SleepEDFDataset(Dataset):
    """Yields (epoch tensor of shape (1, SAMPLES_PER_EPOCH), label) pairs."""

    def __init__(self, epochs: np.ndarray, labels: np.ndarray):
        assert len(epochs) == len(labels)
        self.epochs = epochs
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.epochs[idx]).float().unsqueeze(0)  # (1, SAMPLES_PER_EPOCH)
        y = int(self.labels[idx])
        return x, y
