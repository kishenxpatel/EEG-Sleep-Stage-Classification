# Results

Full run on Sleep-EDF Expanded (sleep-cassette), **pipeline v2**: 153 whole-night recordings from 78 people (two nights each), split **by person** 89 / 32 / 32 recordings (46 / 16 / 16 people) for train / val / test, each recording cropped to the sleep period plus 30 minutes of Wake either side. The test set is 32 recordings from 16 people, 40,460 epochs (Wake 33.0%, N1 10.7%, N2 35.1%, N3 7.8%, REM 13.4%). Each model was trained with 3 seeds (42, 43, 44) on the same fixed split, so ± values reflect training noise (initialisation, batch order). Raw per-run metrics are in `results/*.json`; every figure is regenerated from the saved predictions by `scripts/make_figures.py`.

An earlier run (pipeline v1) had two bugs, found and fixed before this run; see "What changed from the first run" below.

## Headline comparison (mean ± std over 3 seeds)

| Model | Params | Macro F1 | Accuracy | Cohen's κ |
|---|---|---|---|---|
| CNN baseline | 4,005 | 0.652 ± 0.034 | 0.760 ± 0.014 | 0.660 ± 0.023 |
| Transformer | 171,141 | 0.696 ± 0.013 | 0.786 ± 0.002 | 0.701 ± 0.005 |

Per-class F1:

| Model | W | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|
| CNN baseline | 0.909 ± 0.007 | 0.309 ± 0.048 | 0.803 ± 0.015 | 0.710 ± 0.006 | 0.527 ± 0.111 |
| Transformer | 0.915 ± 0.002 | 0.328 ± 0.058 | 0.823 ± 0.003 | 0.737 ± 0.020 | 0.677 ± 0.017 |

**The transformer is ahead by about 0.044 macro F1, and almost all of the gap is REM.** REM F1 is 0.677 vs 0.527 (+0.150); every other class differs by 0.01–0.03. Evidence, with what each piece does and does not cover:

- **Across seeds (training noise): consistent in sign, not significant.** The transformer is ahead in all 3 seeds (macro F1 differences +0.008, +0.034, +0.091; REM +0.075, +0.118, +0.259). A paired t-test over the 3 seeds gives p = 0.21 for macro F1 (accuracy p = 0.08, κ p = 0.09). With n = 3 this test has very little power, so it can neither confirm nor rule out a real gap.
- **Across test people (which people landed in the test set): consistent.** A cluster bootstrap over the 16 test people (`scripts/person_bootstrap.py`, 5,000 resamples) gives a macro F1 difference of **+0.044, 95% CI [+0.029, +0.063]**, with the transformer ahead for 16 of 16 people. Per-class 95% intervals exclude zero for every class, but only REM (+0.150, [+0.099, +0.200]) is large and consistent across seeds. The N1 difference (+0.019) flips sign across seeds (-0.069, +0.015, +0.111), so treat N1 as a tie; the other small gaps (W, N2, N3) are within seed noise.
- **What the bootstrap does not cover:** it holds the 3 trained models of each type fixed, so it says the gap generalises across people for *these* models, not that any CNN would lose to any transformer. Training-seed variability beyond 3 seeds is not captured.

**Parameter counts are not matched.** The CNN has 4,005 parameters, the transformer 171,141, about 43× more. Both were trained under the identical loop, optimiser and early-stopping rule, with no per-model tuning, but a REM gain from a model 43× larger cannot be attributed to attention specifically. A plausible but untested explanation is that REM is a context-dependent stage defined by sustained low-voltage mixed-frequency activity, which the CNN's small receptive field and global average pooling summarise poorly.

**Training stability differs.** Per-seed macro F1: CNN 0.687, 0.650, 0.619; transformer 0.695, 0.683, 0.710. The CNN early-stopped after 23, 16 and 10 epochs (best validation loss 0.582, 0.602, 0.628), the transformer after 18, 15 and 21 (0.599, 0.602, 0.596). In seed 42 the CNN (0.687) is within 0.008 of the transformer (0.695); the gap comes mostly from the other two CNN seeds doing worse, so a single-seed comparison could easily have shown a tie.

## Error analysis

The single-run figures (`figures/error_analysis_grid.png`, `figures/hypnogram_hardest_subjects.png`) use the seed-42 transformer (macro F1 0.695). Numbers below that say "mean" are averaged over the 3 seeds.

**Confusion patterns.** Row-normalised, mean over seeds, transformer: W recall 0.92, N2 0.89, N3 0.68, REM 0.64, **N1 only 0.28**. True N1 goes to N2 (0.38), Wake (0.20) and REM (0.14). N3 is mostly confused with N2 (0.31), and REM with N2 (0.19) and N1 (0.12). The three largest confusion pairs by count are N1→N2, REM→N2 and N3→N2 (N2 is the "default sink"). N3→N2 matches a real label ambiguity: those stages are separated by a slow-wave-percentage threshold, so epochs near it are borderline by construction. N1→W and N1→REM are the expected single-channel failure: the clinical distinction leans on eye movements (EOG) and muscle tone (EMG), which this input excludes. The CNN has the same structure but is worse where the transformer gains: N1 recall 0.26, N3 0.62 (0.37 → N2), REM 0.41 (0.35 → N2, 0.17 → N1).

**Transition errors.** An epoch is "near a transition" if the epoch before or after has a different true label; 7,240 of 40,460 test epochs (17.9%) qualify. Mean macro F1 is **0.540 near transitions vs 0.722 in stable stretches** for the transformer (CNN: 0.501 vs 0.678), a gap of about 0.18 for both models. This matches the standard finding in the sleep-staging literature. The hardest-recording hypnograms show the mechanism: predictions flicker between neighbouring stages exactly where the true stage is fragmented.

**Subject variability.** Large. Seed-averaged per-recording macro F1 ranges from 0.47 to 0.81 for the transformer (median 0.66) and 0.35 to 0.77 for the CNN (median 0.61) across the 32 test recordings. The transformer is ahead on average in 28 of 32 recordings. The three hardest recordings for the transformer (4592, 4501, 4752) have F1 around 0.47 and share a pattern: long fragmented sleep with frequent N1/REM/Wake alternation.

**Confidence calibration.** Mean max-softmax probability is 0.857 for correct predictions vs 0.620 for incorrect ones (transformer, mean over seeds); the CNN is 0.838 vs 0.592. Both models are measurably less confident when wrong, so max-probability is a usable rough "flag for review" signal. This is a gap in means, not a full calibration analysis (no reliability diagram or ECE was computed).

## What changed from the first run (pipeline v1)

The first full run (archived in `results/archive_v1_recording_split/`, analysed in `scripts/v1_postmortem.py`) had two bugs, both fixed and covered by regression tests (`tests/test_pipeline.py`):

1. **Person-level leakage.** Cassette ids look like `SC4ssN` (`ss` = person, `N` = night), so the 153 recordings come from 78 people with two nights each. The v1 split was by recording id, and 16 of the 31 v1 test recordings had the same person's other night in training. Test macro F1 on those was 0.01–0.02 higher than on the rest (not significant, n = 16 vs 15).
2. **No-op sleep crop.** The crop to the sleep period ± 30 minutes counted Wake annotations as sleep, so nothing was removed and Wake was 68.6% of test epochs, inflating accuracy (0.88 in v1 vs 0.76–0.79 now).

| | v1 (leaky, uncropped) | v2 (person split, cropped) |
|---|---|---|
| CNN macro F1 / accuracy | 0.656 / 0.877 | 0.652 / 0.760 |
| Transformer macro F1 / accuracy | 0.666 / 0.882 | 0.696 / 0.786 |
| Transformer − CNN, macro F1 | +0.010 (p = 0.67) | +0.044 (p = 0.21 over seeds; CI [+0.029, +0.063] over people) |

The v1 and v2 test sets are different samples of people (and v2 is cropped), so these are not like-for-like and the change in the CNN-vs-transformer gap should not be attributed to either bug specifically. What the comparison does show is how easily a headline can mislead: v1's "tie" and 0.88 accuracy were both artefacts of the pipeline and the single split.

## Honest self-assessment

**What this project establishes:** on single-channel Fpz-Cz Sleep-EDF with a person-level split, a 171k-parameter patch transformer scores about 0.04 macro F1 higher than a 4k-parameter CNN (0.696 vs 0.652), with nearly all of the gain in REM, and with a more stable training curve across seeds. Both models fail in the same predictable places: N1, N3 vs N2, REM vs N2, and near stage transitions. Subject-to-subject variation (per-recording F1 0.35–0.81) is much larger than the gap between the models.

**What it does not establish:** that attention is the reason for the gain (the models are not capacity-matched, and neither was tuned separately); that the gap holds across more seeds or other splits (3 seeds, one fixed split of 16 test people; the seed-level test is not significant); state-of-the-art performance (no attempt to compete with published multi-channel, sequence-aware models); generalisation to other hardware, electrode placements or populations (no external dataset); or clinical readiness.

**What we would extend:** a capacity-matched comparison (or a sweep over model sizes for both); more seeds and cross-validation over person-level splits; a sequence-of-epochs model that conditions each prediction on neighbouring epochs, which targets the transition errors directly; an N1-specific fix (class weighting or focal loss); multi-channel input (adding EOG to resolve REM/N1/Wake ambiguity that single-channel EEG structurally cannot); and external validation on a second dataset (e.g. SHHS).

**What we learned in the doing:** three things, each found by checking the data rather than trusting the code. Raw EEG epochs need per-recording amplitude normalisation before a BatchNorm-based CNN, otherwise cross-subject gain differences make validation loss explode and the model collapse to predicting Wake. A "subject-level split" is only as good as your definition of subject: this dataset has two nights per person, so the first split leaked. And a comment saying a recording is cropped is not evidence that it is: the crop was a silent no-op until someone looked at a full hypnogram.
