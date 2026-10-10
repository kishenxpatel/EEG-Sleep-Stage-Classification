# EEG Sleep Stage Classification

Transformer vs CNN for sleep stage classification on Sleep-EDF, with a structured error analysis.

## Motivation

Sleep staging is a routine, labour-intensive task performed manually by technicians on polysomnography recordings. Automated classifiers could reduce that labour, but sleep stages are defined as much by transitions and context as by instantaneous spectral signatures — a good test bed for whether attention-based models earn their complexity over a CNN baseline.

## Methods

- **Data:** [Sleep-EDF Expanded](https://physionet.org/content/sleep-edfx/) (PhysioNet), ~200 whole-night PSG recordings, single EEG channel (Fpz-Cz), 30-second epochs, 5-class labels (W, N1, N2, N3, REM). Preprocessing: 0.3–30 Hz bandpass, resampled to 100 Hz. Split **by subject** (60/20/20), not by epoch, to avoid leakage.
- **Baseline:** a small 1D CNN (`src/baseline.py`) — two conv blocks, global average pool, linear head.
- **Transformer:** a patch-based transformer (`src/transformer.py`) — 0.5s patches, conv patch embedding, sinusoidal positional encoding, a handful of `TransformerEncoderLayer`s, classifier head. Sized to fit Colab's free GPU tier. Trained under identical conditions to the baseline (same loop, optimizer, schedule) for a fair comparison.
- **Error analysis:** confusion patterns, performance near stage transitions vs. stable periods, per-subject variability, and confidence calibration. See `results.md` and `notebooks/03_error_analysis.ipynb`.

## Results

Full Sleep-EDF cassette run (153 subjects; 91/31/31 split by subject; 3 training seeds, mean ± std):

| Model | Params | Macro F1 | Accuracy | N1 F1 |
|---|---|---|---|---|
| CNN baseline | 4k | 0.656 ± 0.020 | 0.877 ± 0.009 | 0.256 ± 0.042 |
| Transformer | 171k | 0.666 ± 0.016 | 0.882 ± 0.002 | 0.248 ± 0.062 |

The two models are statistically indistinguishable (paired t-test over seeds, p = 0.67). Both fail in the same places: N1 (recall ~0.2), N3 vs N2, REM vs N2, and near stage transitions (macro F1 ~0.49 near transitions vs ~0.69 in stable stretches). The test set is 69% Wake because recordings are not trimmed; after trimming to the sleep period ± 30 min, accuracy drops to ~0.74 while macro F1 stays ~0.65 (see `results.md`). Full breakdown and error analysis in `results.md`.

![Macro F1 comparison](figures/macro_f1_comparison.png)
![Error analysis](figures/error_analysis_grid.png)

## How to reproduce

Local development happens in this repo with CPU-only correctness checks; actual training runs execute on Google Colab's free GPU tier (see "Development workflow" below).

```bash
python -m venv .venv
.venv/Scripts/activate        # or: source .venv/bin/activate on Linux/macOS/Colab
pip install -r requirements.txt

# Full dataset (~8GB) or a small sample for local testing:
scripts/download_data.sh            # full
scripts/download_data.sh --sample   # 1 subject, for quick local checks
```

Then run `notebooks/04_multiseed_metrics.ipynb` **first**: it is the only notebook that trains models (CNN + transformer x 3 seeds), and it saves per-run metrics and predictions to `results/`. Notebooks `01_data_baseline.ipynb`, `02_transformer.ipynb` and `03_error_analysis.ipynb` then load the saved seed-42 run (seconds, no retraining). All figures can also be regenerated without a GPU:

```bash
python scripts/make_figures.py    # figures/ from results/
python scripts/cropped_eval.py    # metrics with long Wake stretches trimmed
```

### Development workflow

- **Claude Code (local):** writes/debugs `src/`, validates logic on a tiny subject sample on CPU (shapes, label alignment, no crashes — not training speed).
- **Colab (GPU):** `git clone` the repo, `pip install -r requirements.txt`, run `scripts/download_data.sh`, execute the phase's notebook on the free GPU, then commit the executed notebook + results back.

## Limitations

- Test recordings are not trimmed, so Wake is ~69% of test epochs; accuracy and W F1 are inflated relative to trimmed-recording benchmarks (macro F1 is much less affected; see `results.md`).
- The CNN (4k params) and transformer (171k) are not capacity-matched, and only 3 seeds on one fixed subject split were run.
- Single-channel EEG only (no EOG/EMG), which the clinical R&K/AASM scoring convention actually uses alongside EEG.
- The standard subject-level random split may not be the fairest possible evaluation (e.g. no stratification by age/recording site).
- No external validation on a second dataset (e.g. SHHS).
- Not a clinical-readiness claim — see `results.md` for what this project does and does not establish.

## How this was built (Claude Code–assisted)

This project was built with [Claude Code](https://claude.com/claude-code). The training runs and all reported numbers come from my own Colab runs; Claude was used for the following:

- **Mechanical and boilerplate code:** the data-loading and preprocessing pipeline, the shared training loop, the Colab workflow (train once, save metrics/predictions to `results/`, resume after disconnects, reload them in later notebooks), and the figure-regeneration scripts. Writing this by hand would have cost time that was better spent on the experiment itself, especially with only a limited Colab GPU budget.
- **Catching issues and drafting write-ups from saved metrics:** debugging pipeline problems (e.g. the per-recording normalisation failure and the N3/N4 label-merge crash), flagging evaluation caveats (the Wake-dominated test set and the unmatched CNN/transformer parameter counts), and drafting `results.md` from the saved `results/*.json` files rather than from memory.

All numbers in `results.md` can be reproduced from the saved runs with `scripts/make_figures.py` and `scripts/cropped_eval.py`.
