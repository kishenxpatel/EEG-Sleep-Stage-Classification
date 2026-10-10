"""Regenerate every results figure from the saved runs in results/ (no training, no GPU).

Usage: python scripts/make_figures.py
Seed-42 runs drive the single-run figures; the F1 comparison bar chart shows
mean +/- std over all saved seeds. `example_epochs_per_stage.png` needs raw
EDF data and is produced by notebooks/01_data_baseline.ipynb instead.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score

from src.analysis import compute_transition_mask, per_subject_f1
from src.config import FIGURES_DIR, LABEL_NAMES
from src.experiments import load_all, load_run

SEED = 42
FIGURES_DIR.mkdir(exist_ok=True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, dpi=150)
    plt.close(fig)
    print("saved", name)


cnn, cnn_hist, _ = load_run("cnn", SEED)
tfm, tfm_hist, _ = load_run("transformer", SEED)
classes = list(range(len(LABEL_NAMES)))

# --- baseline training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(cnn_hist.train_loss, label="train")
ax1.plot(cnn_hist.val_loss, label="val")
ax1.set(xlabel="epoch", ylabel="loss", title="Loss")
ax1.legend()
ax2.plot(cnn_hist.val_acc, color="green")
ax2.set(xlabel="epoch", ylabel="accuracy", title="Validation accuracy")
save(fig, "baseline_training_curves.png")

# --- baseline confusion matrix
cm = confusion_matrix(cnn["labels"], cnn["preds"], labels=classes, normalize="true")
fig, ax = plt.subplots(figsize=(6, 6))
ConfusionMatrixDisplay(cm, display_labels=LABEL_NAMES).plot(ax=ax, cmap="Blues", values_format=".2f", colorbar=False)
ax.set_title("Baseline CNN: normalized confusion matrix (test set)")
save(fig, "baseline_confusion_matrix.png")

# --- baseline vs transformer: curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(cnn_hist.val_loss, label="CNN baseline")
ax1.plot(tfm_hist.val_loss, label="Transformer")
ax1.set(xlabel="epoch", ylabel="validation loss", title="Validation loss")
ax1.legend()
ax2.plot(cnn_hist.val_acc, label="CNN baseline")
ax2.plot(tfm_hist.val_acc, label="Transformer")
ax2.set(xlabel="epoch", ylabel="validation accuracy", title="Validation accuracy")
ax2.legend()
save(fig, "baseline_vs_transformer_training_curves.png")

# --- baseline vs transformer: confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
for ax, name, res in zip(axes, ["CNN baseline", "Transformer"], [cnn, tfm]):
    m = confusion_matrix(res["labels"], res["preds"], labels=classes, normalize="true")
    ConfusionMatrixDisplay(m, display_labels=LABEL_NAMES).plot(ax=ax, cmap="Blues", values_format=".2f", colorbar=False)
    ax.set_title(name)
save(fig, "baseline_vs_transformer_confusion_matrices.png")

# --- macro / per-class F1, mean +/- std over seeds
grouped = load_all()
names = ["macro F1"] + LABEL_NAMES
stats = {}
for model, runs in grouped.items():
    vals = np.array([[r["macro_f1"]] + [r["per_class_f1"][c] for c in LABEL_NAMES] for r in runs])
    stats[model] = (vals.mean(0), vals.std(0, ddof=1))
n_seeds = len(grouped["cnn"])

fig, ax = plt.subplots(figsize=(9, 4.5))
x = np.arange(len(names))
w = 0.35
for off, model, label in [(-w / 2, "cnn", "CNN baseline"), (w / 2, "transformer", "Transformer")]:
    mean, std = stats[model]
    ax.bar(x + off, mean, w, yerr=std, capsize=3, label=label)
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylabel("F1")
ax.set_title(f"Macro F1 and per-class F1 (mean ± std over {n_seeds} seeds)")
ax.legend()
save(fig, "macro_f1_comparison.png")

# --- transformer error analysis (seed 42)
labels, preds, probs, subj = tfm["labels"], tfm["preds"], tfm["probs"], tfm["subject_ids"]
cm_t = confusion_matrix(labels, preds, labels=classes, normalize="true")
trans = compute_transition_mask(labels, subj)
f1_trans = f1_score(labels[trans], preds[trans], average="macro")
f1_stable = f1_score(labels[~trans], preds[~trans], average="macro")
sf1 = sorted(per_subject_f1(labels, preds, subj).items(), key=lambda kv: kv[1])
overall = f1_score(labels, preds, average="macro")
max_prob = probs.max(axis=1)
correct = preds == labels
bins = np.linspace(0, 1, 30)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
ConfusionMatrixDisplay(cm_t, display_labels=LABEL_NAMES).plot(ax=axes[0, 0], cmap="Blues", values_format=".2f", colorbar=False)
axes[0, 0].set_title("Confusion matrix")
axes[0, 1].bar(["near transition", "stable"], [f1_trans, f1_stable], color=["crimson", "steelblue"])
axes[0, 1].set(ylabel="macro F1", title="F1: near-transition vs. stable")
axes[1, 0].bar([s for s, _ in sf1], [f for _, f in sf1], color="steelblue")
axes[1, 0].axhline(overall, color="gray", linestyle="--", linewidth=1, label="overall macro F1")
axes[1, 0].set(xlabel="subject", ylabel="macro F1", title="Per-subject macro F1")
axes[1, 0].tick_params(axis="x", labelrotation=90, labelsize=6)
axes[1, 0].legend()
axes[1, 1].hist(max_prob[correct], bins=bins, histtype="step", linewidth=1.8, label="correct", color="seagreen", density=True)
axes[1, 1].hist(max_prob[~correct], bins=bins, histtype="step", linewidth=1.8, label="incorrect", color="crimson", density=True)
axes[1, 1].set(xlabel="max softmax probability", ylabel="density", title="Confidence calibration")
axes[1, 1].legend()
fig.suptitle("Transformer error analysis: Sleep-EDF test set (seed 42)", fontsize=14)
save(fig, "error_analysis_grid.png")

# --- hypnograms for the 3 hardest subjects
order = ["W", "REM", "N1", "N2", "N3"]
pos = {LABEL_NAMES.index(n): i for i, n in enumerate(order)}
hardest = [s for s, _ in sf1[:3]]
sf1_d = dict(sf1)
fig, axes = plt.subplots(len(hardest), 1, figsize=(12, 2.5 * len(hardest)), squeeze=False)
for ax, s in zip(axes[:, 0], hardest):
    idx = np.where(subj == s)[0]
    t = np.arange(len(idx)) * 30 / 3600
    ax.step(t, [pos[l] for l in labels[idx]], where="post", label="true", color="black", linewidth=1.2)
    ax.step(t, [pos[p] for p in preds[idx]], where="post", label="predicted", color="crimson", linewidth=1.0, alpha=0.75)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.set_ylabel("stage")
    ax.set_title(f"subject {s}  (macro F1: {sf1_d[s]:.4f})")
    ax.legend(loc="upper right", fontsize=8)
axes[-1, 0].set_xlabel("time (hours)")
save(fig, "hypnogram_hardest_subjects.png")

print(f"\nseed-{SEED} transformer: macro F1 {overall:.4f}, near-transition {f1_trans:.4f}, stable {f1_stable:.4f}, hardest subjects {hardest}")
