"""Multi-seed experiment runner: trains both models per seed and saves metrics to JSON.

Each (model, seed) result is written to RESULTS_DIR as soon as it finishes,
and existing files are skipped on re-run, so a Colab disconnect only loses
the run in progress. The subject split is fixed (config.SEED); only the
training seed (weight init, batch order) varies, so seed-to-seed spread
measures training noise rather than test-set composition.
"""

import json
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from src.analysis import compute_transition_mask, per_subject_f1, top_confusion_pairs
from src.baseline import CNNBaseline
from src.config import BATCH_SIZE, LABEL_NAMES, ROOT_DIR, set_seed
from src.train import History, evaluate, train_model
from src.transformer import TransformerClassifier

RESULTS_DIR = ROOT_DIR / "results"

MODEL_FACTORIES: Dict[str, Callable] = {
    "cnn": CNNBaseline,
    "transformer": TransformerClassifier,
}


def compute_metrics(results: Dict[str, np.ndarray], test_subject_ids: np.ndarray) -> dict:
    """All headline + error-analysis numbers for one trained model's test results."""
    preds, labels, probs = results["preds"], results["labels"], results["probs"]
    class_ids = list(range(len(LABEL_NAMES)))

    per_class = f1_score(labels, preds, average=None, labels=class_ids, zero_division=0)

    transition = compute_transition_mask(labels, test_subject_ids)
    f1_transition = f1_score(labels[transition], preds[transition], average="macro") if transition.any() else None
    f1_stable = f1_score(labels[~transition], preds[~transition], average="macro") if (~transition).any() else None

    max_prob = probs.max(axis=1)
    correct = preds == labels

    return {
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "accuracy": float(accuracy_score(labels, preds)),
        "kappa": float(cohen_kappa_score(labels, preds)),
        "per_class_f1": {n: float(v) for n, v in zip(LABEL_NAMES, per_class)},
        "confusion_normalized": confusion_matrix(labels, preds, labels=class_ids, normalize="true").tolist(),
        "top_confusion_pairs": [list(p) for p in top_confusion_pairs(labels, preds, LABEL_NAMES, top_k=5)],
        "n_test_epochs": int(len(labels)),
        "n_transition_epochs": int(transition.sum()),
        "f1_near_transition": None if f1_transition is None else float(f1_transition),
        "f1_stable": None if f1_stable is None else float(f1_stable),
        "per_subject_f1": {str(k): float(v) for k, v in per_subject_f1(labels, preds, test_subject_ids).items()},
        "mean_conf_correct": float(max_prob[correct].mean()),
        "mean_conf_incorrect": float(max_prob[~correct].mean()),
    }


def run_seeds(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    test_subject_ids: np.ndarray,
    seeds: List[int],
    device=None,
    results_dir: Path = RESULTS_DIR,
) -> None:
    """Train + evaluate every model for every seed, saving results_dir/{model}_seed{N}.json."""
    results_dir.mkdir(exist_ok=True, parents=True)
    for seed in seeds:
        for name, factory in MODEL_FACTORIES.items():
            out = results_dir / f"{name}_seed{seed}.json"
            if out.exists():
                print(f"[skip] {out.name} already exists")
                continue

            print(f"\n=== {name}, seed {seed} ===")
            set_seed(seed)
            model = factory()
            n_params = sum(p.numel() for p in model.parameters())
            model, history = train_model(model, train_loader, val_loader, device=device)
            results = evaluate(model, test_loader, device=device)

            metrics = compute_metrics(results, test_subject_ids)
            metrics.update(
                model=name,
                seed=seed,
                n_params=n_params,
                epochs_trained=len(history.val_loss),
                best_val_loss=float(min(history.val_loss)),
                history={
                    "train_loss": history.train_loss,
                    "val_loss": history.val_loss,
                    "val_acc": history.val_acc,
                },
            )
            out.write_text(json.dumps(metrics, indent=2))
            np.savez_compressed(
                results_dir / f"{name}_seed{seed}_preds.npz",
                preds=results["preds"], labels=results["labels"],
                probs=results["probs"].astype(np.float16), subject_ids=test_subject_ids,
            )
            print(f"saved {out.name}  macro F1 = {metrics['macro_f1']:.4f}")


def load_run(model: str, seed: int = 42, results_dir: Path = RESULTS_DIR):
    """Reload one saved run: (results dict like train.evaluate, History, metrics dict).

    Lets notebooks 01-03 reuse the models trained once in 04 instead of
    retraining. Raises a clear error if 04 hasn't been run yet.
    """
    json_path = results_dir / f"{model}_seed{seed}.json"
    npz_path = results_dir / f"{model}_seed{seed}_preds.npz"
    if not (json_path.exists() and npz_path.exists()):
        raise FileNotFoundError(
            f"{json_path.name} / {npz_path.name} not found in {results_dir} -- "
            "run notebooks/04_multiseed_metrics.ipynb first (it trains every model once)."
        )
    metrics = json.loads(json_path.read_text())
    arrays = np.load(npz_path)
    results = {
        "preds": arrays["preds"],
        "labels": arrays["labels"],
        "probs": arrays["probs"].astype(np.float32),
        "subject_ids": arrays["subject_ids"],
    }
    history = History(**metrics["history"])
    return results, history, metrics


def load_all(results_dir: Path = RESULTS_DIR) -> Dict[str, List[dict]]:
    """Group saved per-seed metric JSONs by model name."""
    grouped: Dict[str, List[dict]] = {}
    for path in sorted(results_dir.glob("*_seed*.json")):
        m = json.loads(path.read_text())
        grouped.setdefault(m["model"], []).append(m)
    return grouped


def summarize(grouped: Dict[str, List[dict]]) -> str:
    """Markdown table of mean ± std across seeds, ready to paste into results.md."""
    def ms(vals):
        v = np.array([x for x in vals if x is not None], dtype=float)
        return f"{v.mean():.3f} ± {v.std(ddof=1) if len(v) > 1 else 0:.3f}"

    cols = ["macro F1", "acc", "kappa"] + LABEL_NAMES + ["F1 near-trans", "F1 stable"]
    lines = ["| Model | seeds | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 2)]
    for name, runs in grouped.items():
        cells = [
            ms([r["macro_f1"] for r in runs]),
            ms([r["accuracy"] for r in runs]),
            ms([r["kappa"] for r in runs]),
            *[ms([r["per_class_f1"][c] for r in runs]) for c in LABEL_NAMES],
            ms([r["f1_near_transition"] for r in runs]),
            ms([r["f1_stable"] for r in runs]),
        ]
        lines.append(f"| {name} | {len(runs)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
