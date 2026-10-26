# Deployed model decision (issue #86)

Decision date: 2026-10-20. Decided from **development results only**, as required
by `docs/evaluation-protocol.md` section 7. No evaluation-partition number enters the
rule or the inputs below.

**Disclosure.** The locked report (R7) had already been read when this rule was
fixed. The rule was written down and accepted by the owner before the development
robustness numbers were computed, and it was applied mechanically. The reader should
still treat the choice as made with knowledge of the evaluation results; that is why
the evaluation outcome for the chosen model is stated in the caveats instead of
being hidden.

## Rule (accepted by the owner, 2026-10-19)

1. Candidates: the four CNNs, each seed separately, with LR as the reference.
2. Score: mean development EER on `group_development_known` over `clean`,
   `codec_opus_24k`, `noise_20db` and `noise_10db`. `silence_trim` is a shortcut
   diagnostic and is excluded.
3. Lowest score wins. A tie within 0.5 EER points goes to the lower clean development EER.
4. Report the seed spread of the winning family next to the choice. A single-seed pick
   is justified in writing.
5. Inference cost is the same for every CNN (same architecture), so it does not
   separate them.

## Inputs (development report `report_dev`, 24,844 development clips)

EER % on `group_development_known`:

| Candidate | clean | codec_opus_24k | noise_20db | noise_10db | silence_trim (excluded) | mean of the four |
| --- | --- | --- | --- | --- | --- | --- |
| lr_baseline | 15.97 | 17.27 | 29.78 | 36.48 | 25.59 | 24.87 |
| cnn_clean_seed1 | 0.63 | 4.93 | 25.20 | 52.63 | 7.64 | 20.85 |
| cnn_clean_seed2 | 0.42 | 2.90 | 24.19 | 39.16 | 8.13 | 16.67 |
| cnn_augmented_seed1 | 1.89 | 3.49 | 17.87 | 30.49 | 12.79 | 13.44 |
| cnn_augmented_seed2 | 1.73 | 5.17 | 19.86 | 34.23 | 13.03 | 15.25 |

Development human false-positive rate % / synthetic detection rate % at the frozen
threshold on the same slice:

| Candidate | clean | codec | noise_20db | noise_10db | silence_trim |
| --- | --- | --- | --- | --- | --- |
| lr_baseline | 4.98 / 70.9 | 26.96 / 87.9 | 0.04 / 0.4 | 0.00 / 0.1 | 50.78 / 91.2 |
| cnn_clean_seed1 | 1.53 / 99.7 | 82.69 / 100.0 | 0.31 / 9.6 | 0.75 / 0.8 | 27.32 / 97.8 |
| cnn_clean_seed2 | 1.81 / 99.9 | 91.44 / 100.0 | 0.00 / 0.0 | 0.00 / 0.0 | 27.71 / 97.8 |
| cnn_augmented_seed1 | 4.91 / 98.9 | 26.65 / 99.6 | 21.11 / 83.8 | 25.78 / 66.0 | 25.90 / 93.5 |
| cnn_augmented_seed2 | 1.77 / 98.3 | 8.16 / 96.2 | 52.16 / 92.0 | 85.44 / 94.7 | 29.00 / 94.1 |

Development clean ROC AUC %: lr_baseline 92.39, cnn_clean_seed1 99.97,
cnn_clean_seed2 99.99, cnn_augmented_seed1 99.68, cnn_augmented_seed2 99.73.

## Outcome

**`cnn_augmented_seed1`** has the lowest mean development EER (13.44), 1.81 points
below the runner-up (`cnn_augmented_seed2`, 15.25), so the tie band does not apply.
Its frozen threshold is -1.2519543 (candidate bundle `cnn_augmented_seed1`).

## Caveats the owner should read before relying on this

- **Seed spread.** The two augmented seeds differ by 1.8 points on the rule's score.
  This is a single-seed pick, justified by the rule only. The development margin is
  small relative to the seed-to-seed variation seen elsewhere.
- **Evaluation outcome for the chosen model (stated, not used).** In the locked report
  `cnn_augmented_seed1` has the highest clean EER among the four CNNs on unseen attacks
  (11.77% [10.28, 13.04], against 6.68-8.30% for the clean seeds and 7.35% for
  augmented seed 2), and a pooled clean human false-positive rate of 7.21% at its
  frozen threshold (development 4.91%). It is the most stable CNN under noise in
  ranking (EER 20.6% at noise_10db). The choice stands because the rule was fixed before
  it was applied to the development numbers (not before the locked report was read); the owner may still decide the model is not fit for the demo, in which case
  that is a recorded deviation, not a silent swap.
- **Thresholds do not transfer.** The deployed threshold was fixed on clean
  development audio. Under codec or noise the chosen model flags about 21-27% of
  development bona fide clips as spoof (issue #83). The demo must present the output
  as a score, not a verdict, and say that noisy or compressed audio raises false
  alarms.
- **LR is not competitive** on any development condition (mean 24.87).
- Descriptive only: development is not an independent test and the numbers are
  post-selection for the threshold and checkpoint choices.

## Not decided here

- Whether to run the pre-declared supplementary analyses on the evaluation store
  (PR #82) needs a separate explicit authorization.
- Calibration is not performed; outputs remain scores.

*Update, 2026-10-27:* the supplementary analyses were not run; the supplementary tool
was dropped from Phase 1 (deviation register in
[`architecture-and-delivery-plan.md`](architecture-and-delivery-plan.md)). The owner
kept `cnn_augmented_seed1` (2026-10-20).
