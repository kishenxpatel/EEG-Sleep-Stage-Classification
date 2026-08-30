"""Reusable helpers for the Phase 3 error analysis.

These operate on flat (epochs-concatenated-across-subjects) arrays as
produced by `build_epoch_arrays`: `labels`/`preds`/`probs` and a parallel
`subject_ids` array giving each epoch's subject. Epochs for a given subject
are assumed contiguous and in chronological order within that block, which
is how `build_epoch_arrays` constructs them.
"""

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import f1_score


def compute_transition_mask(labels: np.ndarray, subject_ids: np.ndarray) -> np.ndarray:
    """True for epochs whose immediate predecessor or successor has a different label.

    Transitions are computed per subject so the boundary between one
    subject's last epoch and the next subject's first epoch (an artefact of
    concatenation, not a real adjacency in time) is never counted as a
    transition. An epoch is judged against the ground-truth label sequence,
    not predictions, since the question is "does this epoch sit near a real
    change in sleep stage," independent of what the model happened to output.
    """
    mask = np.zeros(len(labels), dtype=bool)
    for subject in np.unique(subject_ids):
        idx = np.where(subject_ids == subject)[0]
        subj_labels = labels[idx]
        subj_mask = np.zeros(len(subj_labels), dtype=bool)
        if len(subj_labels) > 1:
            subj_mask[:-1] |= subj_labels[:-1] != subj_labels[1:]
            subj_mask[1:] |= subj_labels[1:] != subj_labels[:-1]
        mask[idx] = subj_mask
    return mask


def per_subject_f1(
    labels: np.ndarray, preds: np.ndarray, subject_ids: np.ndarray
) -> Dict[str, float]:
    """Macro F1 computed separately within each subject's epochs."""
    scores = {}
    for subject in np.unique(subject_ids):
        idx = np.where(subject_ids == subject)[0]
        scores[subject] = f1_score(labels[idx], preds[idx], average="macro")
    return scores


def top_confusion_pairs(
    labels: np.ndarray, preds: np.ndarray, label_names: list, top_k: int = 3
) -> list:
    """The top_k (true, predicted) label pairs by off-diagonal confusion count."""
    n_classes = len(label_names)
    counts = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(labels, preds):
        counts[t, p] += 1
    np.fill_diagonal(counts, 0)

    flat_idx = np.argsort(counts, axis=None)[::-1][:top_k]
    pairs = []
    for idx in flat_idx:
        true_id, pred_id = np.unravel_index(idx, counts.shape)
        pairs.append((label_names[true_id], label_names[pred_id], int(counts[true_id, pred_id])))
    return pairs
