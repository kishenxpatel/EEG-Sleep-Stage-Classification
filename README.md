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

| Model | Macro F1 |
|---|---|
| CNN baseline | TBD (run `notebooks/01_data_baseline.ipynb`) |
| Transformer | TBD (run `notebooks/02_transformer.ipynb`) |

Full breakdown, including per-class F1 and the error analysis findings, in `results.md` once Phase 3 is complete.

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

Then run the notebooks in order: `01_data_baseline.ipynb` → `02_transformer.ipynb` → `03_error_analysis.ipynb`.

### Development workflow

- **Claude Code (local):** writes/debugs `src/`, validates logic on a tiny subject sample on CPU (shapes, label alignment, no crashes — not training speed).
- **Colab (GPU):** `git clone` the repo, `pip install -r requirements.txt`, run `scripts/download_data.sh`, execute the phase's notebook on the free GPU, then commit the executed notebook + results back.

## Limitations

- Single-channel EEG only (no EOG/EMG), which the clinical R&K/AASM scoring convention actually uses alongside EEG.
- The standard subject-level random split may not be the fairest possible evaluation (e.g. no stratification by age/recording site).
- No external validation on a second dataset (e.g. SHHS).
- Not a clinical-readiness claim — see `results.md` for what this project does and does not establish.
