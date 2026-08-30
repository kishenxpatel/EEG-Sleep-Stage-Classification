# Results

**Status:** this file is a template populated with correctness-validation numbers from a tiny 5-subject local sample (3 train / 1 val / 1 test), used to confirm the pipeline works end-to-end on CPU. It is **not** the real result. The actual numbers belong here once `01`-`03` have been run on the full ~153-subject Sleep-EDF cassette split on Colab — replace every `[FULL RUN]` marker below after that run.

## Headline comparison

| Model | Macro F1 |
|---|---|
| CNN baseline | 0.5545 (5-subject sample) / `[FULL RUN]` |
| Transformer | 0.5059 (5-subject sample) / `[FULL RUN]` |

On the local sample the transformer slightly trails the CNN baseline. Per the roadmap for this project, that is a legitimate, reportable outcome on its own — the point of this project is not to beat the baseline, it's to understand both models' failure modes. Whether the gap holds, closes, or reverses on the full dataset is an open question `[FULL RUN]` should answer.

## Error analysis findings (5-subject sample; re-run on full data before trusting magnitudes)

**Confusion patterns.** Top confusion pairs: N2→W, N2→N3, REM→N2. The model never predicted N1 at all in this run — unsurprising given N1 is ~2% of epochs and the training set here is only 3 subjects; whether this collapses on the full ~92-subject training split (where N1 has far more examples to learn from) is one of the more important things `[FULL RUN]` will tell us. The N2↔N3 confusion matches a real ambiguity in the labels themselves: those stages are separated by a slow-wave-percentage threshold, so epochs near that threshold are borderline by construction, not just for the model.

**Transition errors.** Confirmed on the sample: macro F1 near stage transitions (0.364) is meaningfully lower than in stable stretches (0.502), a 0.138 gap. This matches the standard finding in the sleep-staging literature and is expected to hold, if not sharpen, on the full dataset.

**Subject variability.** Not meaningfully assessable with a single test subject — this section only becomes informative with the full ~30-subject test split `[FULL RUN]`.

**Confidence calibration.** On the sample, mean max-softmax-probability was 0.947 for correct predictions vs. 0.728 for incorrect ones — the model is measurably less confident when wrong, a good sign for using max-probability as a rough "trust this prediction or flag for review" signal. `[FULL RUN]` should confirm whether this calibration gap holds up at scale.

See `figures/error_analysis_grid.png` for the visual summary (currently the sample-run version; regenerate after the full run).

## Honest self-assessment

**What this project establishes:** that a small (~171k parameter) patch-based transformer can approach — not exceed — a small CNN's performance on single-channel sleep staging, and that both architectures are expected to fail in similar, predictable places: near real stage transitions, and on the N1 stage specifically, which even human scorers disagree on at a high rate.

**What it does not establish:** state-of-the-art performance (no attempt was made to compete with published multi-channel, sequence-aware models); generalisation to different EEG hardware, electrode placement, or patient populations than Sleep-EDF's; or clinical readiness of either model.

**What we would extend:** multi-channel input (adding EOG to resolve the REM/N1 ambiguity that single-channel EEG structurally cannot, since that distinction clinically relies on eye movements); a sequence-of-epochs model that conditions each prediction on neighbouring epochs, which should directly address the transition-error finding above; external validation on a second dataset (e.g. SHHS) to check whether error patterns found here are Sleep-EDF-specific or general.

**What we learned in the doing:** raw EEG epochs need per-recording amplitude normalisation before hitting a BatchNorm-based CNN — without it, cross-subject differences in electrode impedance/gain caused validation loss to explode and the model to collapse to predicting the majority class (Wake) on every input, despite training loss looking completely normal. This is easy to miss because the failure looks like a training instability, not a preprocessing bug, and it would not have surfaced without deliberately validating the pipeline end-to-end on real data before treating any result as meaningful.
