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
from typing import Dict, List, Optional, Sequence, Tuple

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
    PIPELINE_VERSION,
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


WAKE_DESCRIPTION = "Sleep stage W"
CROP_BUFFER_SEC = 30 * 60


def compute_crop_window(
    descriptions: Sequence[str],
    onsets: Sequence[float],
    durations: Sequence[float],
    recording_end: float,
    buffer_sec: float = CROP_BUFFER_SEC,
) -> Optional[Tuple[float, float]]:
    """(tmin, tmax) keeping the sleep period plus a Wake buffer on each side.

    "Sleep period" runs from the first to the last *non-Wake* scored stage.
    Wake annotations must not count: Sleep-EDF hypnograms label the many
    hours of lights-on recording before and after sleep as "Sleep stage W",
    so treating W as sleep makes the first annotation the start of the
    recording and the crop a silent no-op. Returns None if the recording
    has no non-Wake stage (nothing sensible to crop to).
    """
    sleep = [
        (on, on + dur)
        for d, on, dur in zip(descriptions, onsets, durations)
        if d in STAGE_TO_LABEL and d != WAKE_DESCRIPTION
    ]
    if not sleep:
        return None
    first_onset = min(on for on, _ in sleep)
    last_offset = max(off for _, off in sleep)
    return max(0.0, first_onset - buffer_sec), min(recording_end, last_offset + buffer_sec)


def load_raw(subject: SubjectFiles) -> mne.io.Raw:
    """Load one recording, keep only CHANNEL, filter, resample, attach annotations."""
    raw = mne.io.read_raw_edf(subject.psg_path, include=[CHANNEL], preload=True)
    raw.filter(BANDPASS[0], BANDPASS[1])
    raw.resample(SFREQ)

    annotations = mne.read_annotations(subject.hyp_path)
    raw.set_annotations(annotations, emit_warning=False)

    # Sleep-EDF recordings run for many hours before/after sleep (recording
    # started before lights-out). Crop to the sleep period plus a 30-minute
    # Wake buffer each side, the standard Sleep-EDF preprocessing. See
    # compute_crop_window for why Wake annotations must not define the period.
    window = compute_crop_window(
        list(annotations.description),
        list(annotations.onset),
        list(annotations.duration),
        recording_end=raw.times[-1],
    )
    if window is not None:
        raw.crop(tmin=window[0], tmax=window[1])

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
    # mne.Epochs requires unique event_id values, but STAGE_TO_LABEL
    # intentionally maps "Sleep stage 3" and "Sleep stage 4" to the same
    # label (3, per the AASM N3/N4 merge -- see config.py). Give stage 4 a
    # scratch id MNE will accept, then remap it back after extraction.
    stage4_temp_id = max(STAGE_TO_LABEL.values()) + 1
    mne_event_id = {**STAGE_TO_LABEL, "Sleep stage 4": stage4_temp_id}

    events, event_id_map = mne.events_from_annotations(
        raw, event_id=mne_event_id, chunk_duration=float(EPOCH_SEC)
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
    labels[labels == stage4_temp_id] = STAGE_TO_LABEL["Sleep stage 4"]

    assert data.shape[1] == SAMPLES_PER_EPOCH, (
        f"expected {SAMPLES_PER_EPOCH} samples/epoch, got {data.shape[1]}"
    )
    return data, labels


def load_subject_epochs(
    subject: SubjectFiles, cache_dir: Optional[Path] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Raw file pair -> (epochs, labels) for one subject.

    If cache_dir is given, the preprocessed arrays are cached there per
    recording (the filename carries PIPELINE_VERSION, so a preprocessing
    change never silently reuses stale arrays). Point it at Drive so a
    runtime reset doesn't repeat the slow EDF loading.
    """
    cache_file = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{subject.subject_id}_{PIPELINE_VERSION}.npz"
        if cache_file.exists():
            try:
                with np.load(cache_file) as d:
                    return d["epochs"], d["labels"]
            except Exception:  # truncated write from a disconnect -> rebuild
                cache_file.unlink()

    epochs, labels = extract_epochs_labels(load_raw(subject))
    if cache_file is not None:
        tmp = cache_file.with_suffix(".tmp.npz")
        np.savez(tmp, epochs=epochs, labels=labels)
        tmp.replace(cache_file)
    return epochs, labels


def person_id(subject_id: str) -> str:
    """Person a Sleep-EDF cassette recording id belongs to.

    Ids look like SC4ssN: `ss` identifies the person and `N` the night
    (1 or 2), so "4581" and "4582" are the same person on consecutive
    nights. The first three digits are the person.
    """
    return subject_id[:3]


def get_subject_split(
    subject_ids: List[str], random_state: int = SEED
) -> Tuple[List[str], List[str], List[str]]:
    """Split recording ids 60/20/20 into train/val/test, grouped by PERSON.

    A person's two nights go to the same split. Consecutive epochs from one
    night are highly correlated, and so are two nights from the same
    person (same physiology, electrode placement and sleep habits), so a
    split by recording id lets the model partly learn "which person is
    this" instead of "what sleep stage is this" and inflates test scores.
    Splitting whole people guarantees the test set is unseen physiology.
    Returns recording ids (same type as the input), ordered as in the input.
    """
    people = sorted({person_id(s) for s in subject_ids})
    train_p, temp_p = train_test_split(people, train_size=0.6, random_state=random_state)
    val_p, test_p = train_test_split(temp_p, train_size=0.5, random_state=random_state)

    def members(group):
        group = set(group)
        return [s for s in subject_ids if person_id(s) in group]

    splits = members(train_p), members(val_p), members(test_p)
    assert_no_person_overlap(*splits)
    return splits


def assert_no_person_overlap(
    train_ids: Sequence[str], val_ids: Sequence[str], test_ids: Sequence[str]
) -> None:
    """Raise if any person appears in more than one split."""
    sets = {
        "train": {person_id(s) for s in train_ids},
        "val": {person_id(s) for s in val_ids},
        "test": {person_id(s) for s in test_ids},
    }
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = sets[a] & sets[b]
        if shared:
            raise AssertionError(
                f"person-level leakage: {len(shared)} people in both {a} and {b}: {sorted(shared)[:5]}"
            )


def build_epoch_arrays(
    subjects: List[SubjectFiles], subject_ids: List[str], cache_dir: Optional[Path] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and concatenate epochs/labels for the given subject ids.

    Returns epochs, labels, and a parallel array of subject ids (as strings)
    per epoch -- the latter is needed later for per-subject error analysis.
    """
    wanted = {s.subject_id: s for s in subjects}
    all_epochs, all_labels, all_subject_ids = [], [], []
    for sid in subject_ids:
        subject = wanted[sid]
        epochs, labels = load_subject_epochs(subject, cache_dir=cache_dir)
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
