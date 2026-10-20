# G2 locked evaluation: results and findings (issue #35)

Descriptive record of the locked ASVspoof 2019 LA evaluation. Read in the order
fixed in `docs/evaluation-protocol.md` (committed before the report was
generated). This is a project evaluation, not an official leaderboard result;
scores are not calibrated probabilities. Nothing here changes a threshold,
checkpoint or analysis choice. Intervals are 95% speaker-grouped percentile
bootstrap intervals (10,000 resamples, seed 20261015, 67 evaluation speakers) and
cover test-sampling variance only.

## Provenance

| Item | Value |
| --- | --- |
| Code | commit `c4445f5083d80a91b2014fa57937eb328eb8ae5d` |
| Freeze record | `reports/evaluation_freeze_g2.json`, FROZEN sha256 `cfd1406ca977362207228aa2af8a494c45c0f3cbe1216ef9509aa3ea1d8cb4d2`; REPORTED copy sha256 `88f27bda2c317136826a5337b13cc4ac3224e948c865efa7fb120665d321fb18` (not committed here) |
| Manifest hash | `b932b859…` (role-bearing) |
| Scoring (R6) | 71,237 clips x 5 candidates x 5 conditions = 1,780,925 rows; `predictions_hash` `1ef12a29a78ec5fe59d9bb6c068a218fd69f4f9430ad70f7a51f0b06b3f3d23a`; finished 2026-10-18T14:48:17Z, 4,926 s, `--workers 30`, torch/BLAS threads 1 |
| Report (R7) | `report_hash` `2bc69920ae6c266e0de6c4db9729e75bca5ac1e0866d9d1873d4cbc351b6d0bf`; report file sha256 `c0116ce295b3c7f50a667bff79b77e3c476494b9e4cd895f36aae4421427ed2b`; scored 2026-10-18T18:22:01Z, reported 2026-10-18T20:37:09Z, 2 h 16 min, `--workers 30` |
| Host | hp-xl420gen9-01 |
| Thresholds (frozen, clean dev) | lr_baseline 1.2972940, cnn_clean_seed1 0.6495614, cnn_clean_seed2 2.6375842, cnn_augmented_seed1 -1.2519543, cnn_augmented_seed2 -1.1976757 |

Integrity check on the report (protocol section 2): 25 results over 25 distinct
pairs; every pair has 71,237 evaluation clips (7,355 bona fide, 63,882 spoof);
thresholds equal the record; the 13 attack slices A07-A19 have 4,914 spoof each;
duration bins sum to 96,081 per pair; 0 of 175 development slices differ from the
R5 rehearsal report. The two stores were byte-identical before and after R7.

## 1. Primary endpoint: clean EER on unseen attacks

Slice `group_evaluation_unseen` (A07-A15, A17, A18): 7,355 bona fide, 54,054 spoof.

| Candidate | EER % [95% CI] |
| --- | --- |
| lr_baseline | 19.93 [17.18, 22.80] |
| cnn_clean_seed1 | 8.30 [6.98, 9.50] |
| cnn_clean_seed2 | 6.68 [5.52, 8.06] |
| cnn_augmented_seed1 | 11.77 [10.28, 13.04] |
| cnn_augmented_seed2 | 7.35 [6.18, 8.38] |

The CNNs are well separated from LR. The seed spread within a family is large; the
two augmented seeds do not overlap, so clean versus augmented is not decided by
these point estimates (paired differences: PR #82).

## 2. Known algorithms (A16, A19) and per-attack, clean

Slice `group_evaluation_known_algorithm`: 7,355 bona fide, 9,828 spoof. A16 and A19
reuse the A04 and A06 algorithms; they are never pooled with the unseen group.

| Candidate | Known-algorithm EER % [95% CI] |
| --- | --- |
| lr_baseline | 28.53 [25.37, 31.66] |
| cnn_clean_seed1 | 0.67 [0.42, 0.94] |
| cnn_clean_seed2 | 1.10 [0.68, 1.46] |
| cnn_augmented_seed1 | 3.70 [2.76, 4.82] |
| cnn_augmented_seed2 | 5.12 [4.01, 6.21] |

Per-attack clean EER % (no intervals shown; see the report):

| Attack | lr_baseline | clean s1 | clean s2 | aug s1 | aug s2 |
| --- | --- | --- | --- | --- | --- |
| A07 | 14.39 | 0.35 | 1.14 | 6.80 | 2.31 |
| A08 | 7.33 | 0.59 | 0.79 | 1.12 | 1.02 |
| A09 | 3.19 | 0.00 | 0.02 | 0.04 | 0.03 |
| A10 | 17.24 | 8.36 | 8.52 | 27.09 | 10.91 |
| A11 | 14.34 | 3.00 | 0.61 | 7.90 | 0.57 |
| A12 | 25.51 | 4.25 | 2.73 | 2.71 | 1.55 |
| A13 | 21.33 | 8.65 | 6.66 | 16.26 | 9.46 |
| A14 | 13.07 | 0.98 | 2.42 | 6.06 | 1.85 |
| A15 | 13.39 | 13.88 | 11.60 | 14.47 | 8.73 |
| A16 (known) | 23.10 | 0.42 | 0.83 | 2.98 | 2.77 |
| A17 | 37.40 | 22.46 | 17.04 | 19.47 | 20.11 |
| A18 | 29.90 | 0.33 | 0.53 | 2.71 | 2.18 |
| A19 (known) | 32.74 | 0.84 | 1.30 | 4.26 | 6.50 |

The hardest unseen attacks for the CNNs are A17, A15, A10 and A13; A09 is nearly
solved by every CNN.

## 3. Pooled all-attack evaluation (clean; mixed seen and unseen, not a headline)

| Candidate | EER | ROC AUC | macro F1 | balanced acc | HFPR | SDR | tn / fp / fn / tp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lr_baseline | 21.53 | 86.57 | 49.30 | 73.82 | 3.51 | 51.16 | 7097 / 258 / 31203 / 32679 |
| cnn_clean_seed1 | 7.45 | 97.89 | 73.45 | 90.75 | 1.09 | 82.59 | 7275 / 80 / 11125 / 52757 |
| cnn_clean_seed2 | 6.18 | 98.52 | 79.41 | 92.94 | 2.38 | 88.27 | 7180 / 175 / 7495 / 56387 |
| cnn_augmented_seed1 | 10.74 | 95.76 | 76.42 | 89.78 | 7.21 | 86.78 | 6825 / 530 / 8448 / 55434 |
| cnn_augmented_seed2 | 6.94 | 98.00 | 78.23 | 92.43 | 2.45 | 87.30 | 7175 / 180 / 8110 / 55772 |

Percentages. HFPR is the human false-positive rate and SDR the synthetic detection
rate, both at the frozen threshold. The 5% development false-alarm figure is a
target, not a guarantee: augmented seed 1 is above it (7.21%).

## 4. Robustness on the unseen-attack group

EER % [95% CI] by condition (threshold-free):

| Candidate | clean | codec_opus_24k | noise_20db | noise_10db | silence_trim (diagnostic) |
| --- | --- | --- | --- | --- | --- |
| lr_baseline | 19.93 [17.18, 22.80] | 17.57 [15.38, 20.10] | 25.67 [23.22, 28.24] | 32.91 [30.72, 35.42] | 34.48 [31.68, 37.30] |
| cnn_clean_seed1 | 8.30 [6.98, 9.50] | 13.41 [11.47, 15.33] | 27.86 [25.42, 30.25] | 61.43 [58.92, 63.95] | 22.82 [21.03, 24.55] |
| cnn_clean_seed2 | 6.68 [5.52, 8.06] | 11.75 [10.14, 13.63] | 17.75 [15.67, 20.09] | 30.12 [27.35, 33.09] | 20.05 [18.17, 21.86] |
| cnn_augmented_seed1 | 11.77 [10.28, 13.04] | 13.06 [11.52, 14.63] | 19.73 [18.08, 21.47] | 20.64 [18.56, 22.72] | 23.29 [21.46, 24.94] |
| cnn_augmented_seed2 | 7.35 [6.18, 8.38] | 7.67 [6.69, 8.76] | 17.77 [15.82, 19.83] | 22.12 [19.01, 25.51] | 21.12 [19.10, 23.09] |

ROC AUC % (threshold-free):

| Candidate | clean | codec | noise_20db | noise_10db | silence_trim |
| --- | --- | --- | --- | --- | --- |
| lr_baseline | 87.98 | 90.42 | 82.06 | 73.02 | 71.46 |
| cnn_clean_seed1 | 97.52 | 94.26 | 79.19 | 34.13 | 86.26 |
| cnn_clean_seed2 | 98.26 | 95.06 | 89.43 | 76.62 | 88.48 |
| cnn_augmented_seed1 | 95.13 | 93.88 | 87.53 | 86.83 | 85.07 |
| cnn_augmented_seed2 | 97.82 | 97.29 | 89.08 | 85.14 | 87.26 |

Human false-positive rate % at the frozen threshold (threshold-bound):

| Candidate | clean | codec | noise_20db | noise_10db | silence_trim |
| --- | --- | --- | --- | --- | --- |
| lr_baseline | 3.51 | 18.60 | 0.05 | 0.04 | 39.50 |
| cnn_clean_seed1 | 1.09 | 80.71 | 0.42 | 0.98 | 27.95 |
| cnn_clean_seed2 | 2.38 | 91.95 | 0.00 | 0.00 | 32.67 |
| cnn_augmented_seed1 | 7.21 | 32.56 | 26.15 | 25.33 | 29.04 |
| cnn_augmented_seed2 | 2.45 | 12.14 | 59.58 | 86.04 | 38.60 |

Synthetic detection rate % at the frozen threshold (threshold-bound):

| Candidate | clean | codec | noise_20db | noise_10db | silence_trim |
| --- | --- | --- | --- | --- | --- |
| lr_baseline | 55.69 | 83.01 | 1.80 | 0.03 | 70.05 |
| cnn_clean_seed1 | 79.49 | 99.61 | 2.91 | 0.32 | 80.32 |
| cnn_clean_seed2 | 86.20 | 99.91 | 0.00 | 0.00 | 87.80 |
| cnn_augmented_seed1 | 84.72 | 94.01 | 84.36 | 81.61 | 80.72 |
| cnn_augmented_seed2 | 86.80 | 94.22 | 94.36 | 96.72 | 88.63 |

## 5. Development versus evaluation, clean EER %

Development numbers are post-selection (the select and threshold roles informed
choices) and cover known attacks.

| Candidate | dev known (A01-A06) | dev A04 | dev A06 | eval A16 | eval A19 | eval unseen |
| --- | --- | --- | --- | --- | --- | --- |
| lr_baseline | 15.97 | 19.48 | 29.44 | 23.10 | 32.74 | 19.93 |
| cnn_clean_seed1 | 0.63 | 0.67 | 0.27 | 0.42 | 0.84 | 8.30 |
| cnn_clean_seed2 | 0.42 | 0.59 | 0.51 | 0.83 | 1.30 | 6.68 |
| cnn_augmented_seed1 | 1.89 | 1.51 | 1.65 | 2.98 | 4.26 | 11.77 |
| cnn_augmented_seed2 | 1.73 | 2.15 | 2.51 | 2.77 | 6.50 | 7.35 |

## 6. Duration slices (development + evaluation pooled)

The report's duration slices pool both partitions (issue #85), so they are not an
evaluation result and are not read here. Sizes: lt_1s 1,184 clips, 1s_to_2s 18,848,
2s_to_4s 52,763, 4s_to_10s 23,192, gt_10s 94 (pooled, one candidate). The
evaluation-only duration table is pre-declared and produced by `eval-supplement`
(PR #82) under a separate authorization.

## Findings

1. **The CNNs outperform LR on the clean unseen, known-algorithm and pooled
   sets** (unseen attacks 6.7-11.8% vs 19.9% EER). Per attack there are two
   exceptions: LR is marginally better than cnn_clean_seed1 on A15 (13.39 vs 13.88)
   and clearly better than cnn_augmented_seed1 on A10 (17.24 vs 27.09). Known
   algorithms are far easier for the CNNs than unseen ones (0.7-5.1% vs
   6.7-11.8%).
2. **Seed variance is large.** Augmented seeds differ by 4.4 points on the primary
   endpoint with non-overlapping intervals; the clean seeds differ by 1.6 points
   with overlapping intervals. No clean-versus-augmented claim is made from point
   estimates.
3. **The frozen clean-dev thresholds do not transfer under codec or noise** while
   ranking largely survives for the augmented models (AUC 85-97%): bona fide
   flagged as spoof up to 92% (codec) and 86% (noise_10db). The clean CNNs and LR
   under noise flag almost nothing. Issue #83.
4. **Negative result:** `cnn_clean_seed1` under noise_10db has EER 61.4% and AUC
   34% (below chance); seed 2 of the same family is 30.1%. Issue #83.
5. **Augmentation helps robustness more than clean accuracy:** under noise_10db the
   augmented CNNs have EER 20.6-22.1% against 30.1-61.4% for the clean CNNs. On
   the clean condition augmented seed 1 is higher (11.77%) than both clean seeds,
   while augmented seed 2 (7.35%) is not clearly different from them.
6. **`silence_trim` raises CNN EER from 6.7-11.8% to 20.1-23.3%.** This is a
   shortcut diagnostic, consistent with partial reliance on silence cues, not
   proof, and not a robustness claim. Issue #84.
7. **LR's development numbers are optimistic** (15.97% on dev known vs 23.10 and
   32.74% on A16/A19).

## Deviations and caveats

- R6 used `--workers 30` (planned 12) and the R6 log lost its final JSON because the
  log file was replaced mid-run (inferred from file timestamps); completion rests
  on the run sidecar and the pre-R7 metadata checks (tracker log).
- A pytest run overlapped R6 on the same host; scores are independent of load.
- The report gives an interval only for slice EER. The other metrics, paired
  differences and the evaluation-only duration table are in PR #82 and have not been
  run on evaluation data (run only under an explicit authorization).
- About 675 unadjusted intervals exist in the report; no result is described as
  significant. 67 evaluation speakers are the resampling unit because source
  recordings cannot be mapped; the bona fide pool is shared across attack slices;
  candidates and conditions are scored on the same clips.
- The role-free train-manifest hash `b172e50c…` versus the role-bearing
  `b932b859…` gap (issues #52/#53) is unchanged.

## Open items

- #83 threshold transfer and noise result; #84 silence-cue follow-up (development
  data only); #85 duration slices pool partitions; #86 record the deployed model
  from development results only; #71 run-record follow-ups (lessons from R6/R7).
- PR #82 supplementary analysis: ruff clean, 24 new tests pass, full suite 504 passed
  and 1 failed (`test_config.py::test_default_config_uses_repository_local_paths`,
  caused by an exported `VOICE_DETECTOR_ARTIFACTS_DIR` in the host shell: with the
  three `VOICE_DETECTOR_*` variables unset the test file gives 2 passed). Development rehearsal ran in 2 min 30 s
  (5 workers) and the serial run gave the same report hash `e78ed516…`. Still to do:
  `/code-review`, then an explicit decision on running it on the evaluation store.
- RQ mapping (proposal research questions to evidence, with RQ2's second domain and
  RQ3 marked not run) is still to be written.
