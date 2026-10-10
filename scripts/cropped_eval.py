"""Re-score saved predictions with long leading/trailing Wake trimmed (no retraining).

The pipeline keeps whole recordings, so ~69% of test epochs are Wake. Common
Sleep-EDF practice keeps only the sleep period plus a Wake margin on each
side. This keeps, per subject, epochs from (first non-Wake - MARGIN_MIN) to
(last non-Wake + MARGIN_MIN) and recomputes the metrics on those epochs.

Usage: python scripts/cropped_eval.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score

from src.config import LABEL_NAMES
from src.experiments import RESULTS_DIR

MARGIN_MIN = 30
MARGIN = MARGIN_MIN * 2  # 30 s epochs
SEEDS = sorted({int(p.stem.split("seed")[1].split("_")[0]) for p in RESULTS_DIR.glob("*_preds.npz")})


def crop_mask(labels, subject_ids):
    keep = np.zeros(len(labels), dtype=bool)
    for s in np.unique(subject_ids):
        idx = np.where(subject_ids == s)[0]
        sleep = np.where(labels[idx] != 0)[0]
        if len(sleep) == 0:
            continue
        lo, hi = max(0, sleep[0] - MARGIN), min(len(idx), sleep[-1] + MARGIN + 1)
        keep[idx[lo:hi]] = True
    return keep


rows = {}
for model in ["cnn", "transformer"]:
    macro, acc, per_class = [], [], []
    for seed in SEEDS:
        d = np.load(RESULTS_DIR / f"{model}_seed{seed}_preds.npz")
        keep = crop_mask(d["labels"], d["subject_ids"])
        y, p = d["labels"][keep], d["preds"][keep]
        macro.append(f1_score(y, p, average="macro"))
        acc.append(accuracy_score(y, p))
        per_class.append(f1_score(y, p, average=None, labels=list(range(5))))
    rows[model] = (np.array(macro), np.array(acc), np.array(per_class))

d0 = np.load(RESULTS_DIR / f"cnn_seed{SEEDS[0]}_preds.npz")
keep0 = crop_mask(d0["labels"], d0["subject_ids"])
print(f"kept {keep0.sum()} / {len(keep0)} test epochs ({100 * keep0.mean():.1f}%) with a {MARGIN_MIN}-min Wake margin")
print(f"Wake share after crop: {100 * (d0['labels'][keep0] == 0).mean():.1f}%  (before: {100 * (d0['labels'] == 0).mean():.1f}%)\n")

header = f"| Model | macro F1 | acc | " + " | ".join(LABEL_NAMES) + " |"
print(header)
print("|" + "---|" * (len(LABEL_NAMES) + 3))
for model, (macro, acc, pc) in rows.items():
    cells = " | ".join(f"{pc[:, i].mean():.3f} ± {pc[:, i].std(ddof=1):.3f}" for i in range(5))
    print(f"| {model} | {macro.mean():.3f} ± {macro.std(ddof=1):.3f} | {acc.mean():.3f} ± {acc.std(ddof=1):.3f} | {cells} |")

t, p = stats.ttest_rel(rows["transformer"][0], rows["cnn"][0])
print(f"\npaired t-test on cropped macro F1: t={t:.3f}, p={p:.3f}")
