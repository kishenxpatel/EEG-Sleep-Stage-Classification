# Results

> **These are the first-run (pipeline v1) results.** That run had two bugs, both since fixed in the code: the split was by recording instead of by person, and the sleep-period crop was a no-op. The v1 results are archived in `results/archive_v1_recording_split/` and analysed in `scripts/v1_postmortem.py`. A v2 re-run is pending; this file will be rewritten from it. Read the numbers below as slightly optimistic and Wake-inflated (see the two caveat sections).

Full run on Sleep-EDF Expanded (sleep-cassette): 153 recordings from 78 people, split by recording 91 / 31 / 31 (train / val / test), 84,017 test epochs. **The split is by recording, not by person; see "Person-level leakage" below.** Each model was trained with 3 seeds (42, 43, 44) on the same fixed split, so ± values reflect training noise (initialisation, batch order), not test-set choice. Raw per-run metrics are in `results/*.json`; every figure is regenerated from the saved predictions by `scripts/make_figures.py`.

## Headline comparison (mean ± std over 3 seeds)

| Model | Params | Macro F1 | Accuracy | Cohen's κ |
|---|---|---|---|---|
| CNN baseline | 4,005 | 0.656 ± 0.020 | 0.877 ± 0.009 | 0.749 ± 0.014 |
| Transformer | 171,141 | 0.666 ± 0.016 | 0.882 ± 0.002 | 0.759 ± 0.005 |

Per-class F1:

| Model | W | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|
| CNN baseline | 0.971 ± 0.003 | 0.256 ± 0.042 | 0.794 ± 0.008 | 0.669 ± 0.037 | 0.589 ± 0.014 |
| Transformer | 0.972 ± 0.001 | 0.248 ± 0.062 | 0.795 ± 0.004 | 0.700 ± 0.009 | 0.615 ± 0.019 |

**The two models are not distinguishable.** The transformer is ~0.010 ahead on macro F1, but the seed-to-seed std is 0.016–0.020 and a paired t-test over seeds gives p = 0.67 (n = 3, so low power; read this as "cannot separate", not "proven equal"). The only difference that looks consistent across seeds is a small transformer edge on N3 (0.700 vs 0.669) and REM (0.615 vs 0.589); with three seeds that is a lead worth testing, not a finding. Neither model resolves N1.

**Parameter counts are not matched.** The CNN has 4,005 parameters; the transformer has 171,141, about 43× more. Both were trained under the identical loop, optimiser and early-stopping rule, but the comparison is "small CNN vs. a much larger transformer", not capacity-matched. The transformer failing to clearly beat a CNN 43× smaller is, if anything, the more notable result.

## Wake dominates the test set (important caveat)

`src/data.py` is meant to crop each recording to the scored period plus a 30-minute buffer, but the crop is a no-op: it treats Wake annotations as "sleep" annotations, so the first one is the start of the recording and nothing is removed. Whole 20+ hour recordings are used, so **68.6% of test epochs are Wake**, mostly long stretches before and after sleep. That inflates accuracy and W F1. Re-scoring the same saved predictions after trimming each subject to the sleep period plus a 30-minute Wake margin (`scripts/v1_postmortem.py`; 43.8% of epochs kept, Wake share 28.3%):

| Model | Macro F1 | Accuracy | W | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|---|
| CNN baseline | 0.642 ± 0.019 | 0.737 ± 0.012 | 0.873 | 0.265 | 0.799 | 0.670 | 0.601 |
| Transformer | 0.655 ± 0.020 | 0.751 ± 0.006 | 0.883 | 0.258 | 0.800 | 0.703 | 0.630 |

Macro F1 barely moves (-0.014), accuracy drops 0.14, and the CNN-vs-transformer picture is unchanged (paired p = 0.62). **Quote the cropped numbers when comparing to published work**, which typically reports on trimmed recordings; the uncropped accuracy of ~0.88 is not comparable.

## Person-level leakage (important caveat)

Sleep-EDF cassette contains two nights per person (ids `SC4ssN`: `ss` = person, `N` = night), so the 153 recordings come from 78 people. `get_subject_split` splits by recording id, which means one night can be in training while the same person's other night is in the test set. In this split **16 of the 31 test recordings (52%) have the same person in training** (23 of 31 counting validation), despite the code and early docs describing the split as subject-level.

Effect on the saved predictions (`scripts/v1_postmortem.py`, mean over 3 seeds):

| Model | Macro F1, leaked-person recordings (n = 16) | Macro F1, clean recordings (n = 15) |
|---|---|---|
| CNN baseline | 0.664 | 0.644 |
| Transformer | 0.672 | 0.659 |

Leaked recordings score about 0.01–0.02 higher, the direction leakage would predict, but with 16 vs 15 recordings the difference is not significant (per-recording Welch p ≈ 0.45), so this is a hint of mild optimism, not a measured effect size. Both models are affected similarly, so the CNN-vs-transformer comparison is unlikely to change, but absolute numbers should be read as slightly optimistic. The fix is a group split by person (first three digits of the id) followed by a re-run; that was not done here.

## Error analysis

The single-run figures (`figures/error_analysis_grid.png`, `figures/hypnogram_hardest_subjects.png`) use the seed-42 transformer (macro F1 0.655). Numbers below that say "mean" are averaged over the 3 seeds.

**Confusion patterns.** Row-normalised, mean over seeds, transformer: W is nearly perfect (0.97). N2 is solid (0.86). N3 recall is 0.62, with 0.37 of N3 epochs predicted as N2. REM recall is 0.59, with 0.23 going to N2. **N1 recall is only 0.21**: 0.39 of true N1 goes to N2, 0.25 to W and 0.15 to REM. The top confusion pairs by count are consistently N1→N2, REM→N2 and N3→N2 (N2 is the "default sink"). The N3→N2 confusion matches a real ambiguity in the labels themselves: those stages are separated by a slow-wave-percentage threshold, so epochs near it are borderline by construction. N1→W and N1→REM are the expected single-channel failure: the clinical distinction leans on eye movements (EOG) and muscle tone (EMG), which this input excludes. The CNN has the same structure (N1 recall 0.23, N3→N2 0.43).

**Transition errors.** Near-transition epochs are 8.4% of the test set (7,025 epochs; an epoch is "near transition" if the epoch before or after has a different true label). Mean macro F1 near transitions is **0.494 vs 0.689 in stable stretches** for the transformer (CNN: 0.471 vs 0.680), a gap of about 0.2. This matches the standard finding in the sleep-staging literature. The hypnograms for the hardest subjects show the mechanism clearly: predictions flicker rapidly between neighbouring stages exactly where the true stage is fragmented.

**Subject variability.** Large. Seed-averaged per-subject macro F1 ranges from 0.44 to 0.76 for the transformer (median 0.63) and 0.41 to 0.78 for the CNN (median 0.60) across the 31 test subjects. The three hardest subjects (4501, 4582, 4581) have F1 ≈ 0.43–0.45 and share a pattern: long fragmented sleep with frequent N1/REM/Wake alternation, which is the regime where the model flickers.

**Confidence calibration.** Mean max-softmax probability is 0.937 for correct predictions vs 0.619 for incorrect ones (transformer, mean over seeds); the CNN is 0.909 vs 0.558. Both models are measurably less confident when wrong, so max-probability is a usable rough "flag for review" signal. The transformer is the more confident of the two overall. This is a gap in means, not a full calibration analysis (no reliability diagram or ECE was computed).

**Training behaviour.** CNN validation loss is noisy (a spike is visible in the seed-42 curves), and CNN seed 44 early-stopped after 9 epochs versus 22–24 for the other two, so the CNN's seed variance partly reflects optimisation instability. The transformer trained more smoothly (15–18 epochs across seeds).

## Honest self-assessment

**What this project establishes:** on single-channel Fpz-Cz Sleep-EDF with a recording-level split, a 171k-parameter patch transformer performs about the same as a 4k-parameter CNN (macro F1 ≈ 0.65–0.67), and both fail in the same predictable places: N1, N3 vs N2, REM vs N2, and near stage transitions. Subject-to-subject variation is much larger than the difference between the models.

**What it does not establish:** that either architecture is better (3 seeds, p ≈ 0.6); anything about capacity-matched models; state-of-the-art performance (no attempt to compete with published multi-channel, sequence-aware models); generalisation to other hardware, electrode placements or populations; or clinical readiness. The test set is a single fixed split of 31 recordings (27 people), so seed variance does not capture split variance, and the split leaks person identity (see above).

**What we would extend:** multi-channel input (adding EOG to resolve REM/N1/W ambiguity that single-channel EEG structurally cannot); a sequence-of-epochs model that conditions each prediction on neighbouring epochs, which targets the transition errors directly; an N1-specific fix (class weighting or focal loss); a person-level grouped split and cross-validation over such splits; trimming Wake in training as well as evaluation; external validation on a second dataset (e.g. SHHS).

**What we learned in the doing:** raw EEG epochs need per-recording amplitude normalisation before a BatchNorm-based CNN, otherwise cross-subject gain differences made validation loss explode and the model collapse to predicting Wake despite normal-looking training loss. Separately, an uncropped Wake-dominated test set can make a mediocre classifier look strong (0.88 accuracy vs 0.74 once trimmed), and a headline metric alone would not have shown that.
