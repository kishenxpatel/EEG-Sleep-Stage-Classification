"""Uncertainty on the transformer-vs-CNN gap that accounts for both noise sources.

Seed-level tests (n = 3) only capture training noise; per-recording tests treat
a person's two nights as independent. This resamples the test PEOPLE (cluster
bootstrap) and recomputes macro F1 for both models from their saved
predictions, averaged over the training seeds, giving a 95% interval for the
difference that reflects which people happened to land in the test set.
It does NOT add training-seed uncertainty beyond the 3 seeds available.

Usage: python scripts/person_bootstrap.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy import stats

from src.config import LABEL_NAMES
from src.data import person_id
from src.experiments import load_all, load_run

N_BOOT = 5000
rng = np.random.default_rng(0)
K = len(LABEL_NAMES)
SEEDS = sorted({r["seed"] for r in load_all()["cnn"]})


def person_confusions(model, seed):
    """{person: (K, K) confusion-count matrix} for one saved run."""
    d = load_run(model, seed)[0]
    people = np.array([person_id(s) for s in d["subject_ids"]])
    out = {}
    for p in np.unique(people):
        i = people == p
        cm = np.zeros((K, K))
        np.add.at(cm, (d["labels"][i], d["preds"][i]), 1)
        out[p] = cm
    return out


def macro_f1(cm):
    tp = np.diag(cm)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    return np.mean(2 * prec * rec / np.maximum(prec + rec, 1e-12))


conf = {m: [person_confusions(m, s) for s in SEEDS] for m in ["cnn", "transformer"]}
people = sorted(conf["cnn"][0])
P = len(people)
# stack: model -> (seeds, people, K, K)
stack = {m: np.array([[c[p] for p in people] for c in conf[m]]) for m in conf}


def mean_f1(m, idx):
    return np.mean([macro_f1(stack[m][s][idx].sum(0)) for s in range(len(SEEDS))])


full = np.arange(P)
point = {m: mean_f1(m, full) for m in stack}
diffs = np.empty(N_BOOT)
for b in range(N_BOOT):
    idx = rng.integers(0, P, P)
    diffs[b] = mean_f1("transformer", idx) - mean_f1("cnn", idx)

lo, hi = np.percentile(diffs, [2.5, 97.5])
print(f"{P} test people, {len(SEEDS)} seeds, {N_BOOT} bootstrap resamples of people")
print(f"macro F1 (pooled over test): CNN {point['cnn']:.3f}  Transformer {point['transformer']:.3f}")
print(f"difference (transformer - CNN): {point['transformer'] - point['cnn']:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
print(f"share of resamples where the transformer is ahead: {np.mean(diffs > 0):.3f}")

# per-person paired comparison (average the person's two nights, then compare)
pf = {m: np.array([np.mean([macro_f1(stack[m][s][i]) for s in range(len(SEEDS))]) for i in range(P)]) for m in stack}
d = pf["transformer"] - pf["cnn"]
print(f"\nper-person macro F1 (seed-mean), transformer better for {(d > 0).sum()}/{P} people")
print(f"Wilcoxon signed-rank across people: p = {stats.wilcoxon(d).pvalue:.4f}")

# per-class
print("\nper-class F1 difference (transformer - CNN), 95% bootstrap CI over people:")
for c, name in enumerate(LABEL_NAMES):
    def f1c(m, idx):
        out = []
        for s in range(len(SEEDS)):
            cm = stack[m][s][idx].sum(0)
            tp = cm[c, c]; pr = tp / max(cm[:, c].sum(), 1); rc = tp / max(cm[c].sum(), 1)
            out.append(2 * pr * rc / max(pr + rc, 1e-12))
        return np.mean(out)
    bd = np.array([f1c("transformer", i) - f1c("cnn", i) for i in (rng.integers(0, P, P) for _ in range(2000))])
    pt = f1c("transformer", full) - f1c("cnn", full)
    print(f"  {name:4s} {pt:+.3f}  [{np.percentile(bd, 2.5):+.3f}, {np.percentile(bd, 97.5):+.3f}]")
