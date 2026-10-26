# Evaluation protocol for the locked ASVspoof 2019 LA evaluation (issue #35)

Status: written before any evaluation-partition prediction was inspected. It is
committed to git before the locked report (R7) is generated, so its commit time
precedes `scored_utc` in the freeze record. It adds interpretation rules to the
machine-readable protocol in `configs/evaluation_protocol.yaml`; where the two
differ, the YAML and the freeze record define what was computed, and this
document defines how it is read and reported.

Freeze record: `reports/evaluation_freeze_g2.json`, FROZEN, sha256
`cfd1406ca977362207228aa2af8a494c45c0f3cbe1216ef9509aa3ea1d8cb4d2`, git commit
`c4445f5083d80a91b2014fa57937eb328eb8ae5d`, manifest hash `b932b859…` (the
role-bearing hash). Candidates: `lr_baseline`, `cnn_clean_seed1`,
`cnn_clean_seed2`, `cnn_augmented_seed1`, `cnn_augmented_seed2`. Conditions:
`clean`, `codec_opus_24k`, `noise_20db`, `noise_10db`, `silence_trim`.

This is a project evaluation on ASVspoof 2019 LA. It is not an official
leaderboard result and does not report min t-DCF. Outputs are scores, not
calibrated probabilities, and do not prove authenticity, identify a speaker, or
verify a spoken claim.

## 1. What is scored, and what R7 does

- R6 scored every evaluation clip (71,237) for every candidate and condition
  (5 x 5), 1,780,925 rows, with the frozen bundles and thresholds. Clip score:
  CNN `mean_window_logit`, LR `mean_window_decision_function`; decision
  `score >= threshold` in float64.
- R7 (`eval-report`) runs once. It computes the predictions hash, saves the
  record as SCORED (one-way), computes the report, writes it, then saves the
  record as REPORTED (one-way). REPORTED is final.
- Reports are never regenerated with different analysis arguments after numbers
  are seen. `--workers` is not an analysis argument: the report is byte-identical
  for any value, so a re-run after a crash may use fewer workers.

## 2. Integrity expectations (checked after REPORTED, from the report)

Expected from `docs/data.md` and the manifest, for all 25 candidate x condition
pairs:

- 71,237 evaluation clips: 7,355 bona fide, 63,882 spoof; 67 speakers.
- 4,914 spoof clips per attack A07-A19.
- Thresholds in the report equal the thresholds in the freeze record.
- Report `git_commit` is `c4445f5083d8…` and not `unknown`.
- Development slices (`attack_A01`-`A06`, `group_development_known`) equal the
  R5 dev rehearsal report slices exactly.

A violated expectation is recorded as a finding or an invalidated run; it is not
repaired by re-running.

## 3. Primary endpoint, and everything else is descriptive

Primary endpoint, per candidate: EER on the `clean` condition for slice
`group_evaluation_unseen` (evaluation attacks A07-A15, A17, A18, with the
evaluation bona fide pool), reported with its speaker-grouped 95% bootstrap
interval.

All other numbers (the other slices and conditions, secondary metrics, between
candidate comparisons) are descriptive. They are reported with their counts and,
where available, intervals, but without "significant" language and without
adjusting the story to the result. The report holds about 25 pairs x 27 slices
of intervals (about 675), unadjusted for multiplicity; a single interval
excluding a value is not evidence.

## 4. Reading order (fixed in advance)

1. Integrity (section 2), report status `reported`, record REPORTED, freeze
   identity, per-pair `n_clips`.
2. Primary endpoint (section 3), per candidate.
3. `group_evaluation_unseen` and `group_evaluation_known_algorithm` side by side
   but never pooled; then per-attack slices A07-A19 with counts. A16 and A19
   reuse the A04 and A06 algorithms, so they are known-algorithm results with
   unseen speakers.
4. Pooled all-attack evaluation numbers last, labelled "mixed seen/unseen
   algorithms; not a headline".
5. Threshold-free metrics (EER, ROC AUC) and threshold-bound metrics (macro F1,
   balanced accuracy, human false-positive rate, synthetic detection rate, and
   confusion matrices at the frozen thresholds) are read separately. Under noise
   the scores shift by several score units, so threshold-bound metrics can move
   without the ranking changing.
6. Robustness conditions (`codec_opus_24k`, `noise_20db`, `noise_10db`) against
   `clean` for the same clips and both classes.
7. `silence_trim` is a shortcut diagnostic, not a robustness claim.
8. Clean versus augmented CNN, then CNN versus LR, using the intervals and not
   point estimates; the two seeds of each CNN family are reported side by side
   and are not collapsed to a "best" seed.
9. Development-versus-evaluation comparison. Development numbers are
   post-selection (the select and threshold roles informed choices) and cover
   known attacks. The fairer control is dev A04/A06 versus evaluation A16/A19.
   The 5% false-positive target rests on a small number of threshold speakers and
   is a target, not a test guarantee.

## 5. Duration slices (option A)

The five duration slices (`duration_lt_1s`, `duration_1s_to_2s`,
`duration_2s_to_4s`, `duration_4s_to_10s`, `duration_gt_10s`) are declared with
`partition: all`. In the locked report they therefore pool the 24,844 development
clips and the 71,237 evaluation clips (96,081 per pair), including the clips that
set the thresholds.

- They are reported in R7 unchanged and labelled "development + evaluation pooled;
  not an evaluation result". They are not used to support any claim about the
  evaluation partition, including the sub-1 s policy.
- A supplementary evaluation-only duration table (same bins, same bootstrap
  specification) is pre-declared. It is computed only after REPORTED, under a
  separate explicit owner authorization, with tooling first built and tested on
  the development store. It is labelled a supplementary, post-report analysis.
- No code or slice semantics are changed before R7.

## 6. Interval coverage

What the locked report emits, stated so it is not over-read:

- Every candidate, condition and slice has class counts and confusion at the
  frozen threshold, and the metrics EER, ROC AUC, macro F1, balanced accuracy,
  human false-positive rate and synthetic detection rate where defined.
- A speaker-grouped percentile bootstrap interval (10,000 resamples, seed
  20261015, 95%) is computed for **EER of slices only**. AUC, macro F1, balanced
  accuracy, human false-positive rate and synthetic detection rate have no
  interval in the report, and there is no interval for the pooled evaluation
  partition.
- Paired differences between candidates (same resamples) are not produced by the
  report.
- The bootstrap captures test-sampling variance only: 67 speaker clusters (a
  fallback, because source recordings cannot be mapped), no training variance
  beyond two seeds per CNN family, and no uncertainty in the frozen thresholds.
- The bona fide pool is shared across attack slices, conditions are derived from
  the same clips, and all candidates are scored on the same clips, so the results
  are dependent. Correlated 2019/2021 source recordings are not independent
  evidence.

Supplementary analyses that would be pre-declared and run only after REPORTED
under a separate authorization, tested on development first:

- human false-positive rate and synthetic detection rate with intervals at the
  frozen thresholds, for the pooled evaluation partition and the unseen group;
- the evaluation-only duration table (section 5);
- a short, named list of paired EER differences (clean versus augmented CNN,
  CNN versus LR), using the same speaker resamples.

## 7. Deployment decision

The deployed model for the demo is chosen from development results and resource
cost only, before and independently of the locked report (TODO-Plan section 6).
It is not a post-test winner and is not changed after R7.

> **Amendment, 2026-10-27 (annotation only; the paragraph above is the text committed
> before R7).** The order stated above was not followed. The development-only
> selection rule was written down and accepted (2026-10-19) after the locked report
> had been read, then applied mechanically to development numbers; the choice
> (`cnn_augmented_seed1`) is therefore not a blind or pre-registered selection. It is
> disclosed in [`deployed-model-decision.md`](deployed-model-decision.md) and in the
> deviation register of the [architecture plan](architecture-and-delivery-plan.md).

## 8. Rules after R7

- No retuning, no threshold change, no calibration, no checkpoint swap, no
  retraining on the basis of locked results, and no re-run to obtain different
  numbers.
- No inspection of evaluation predictions beyond the report (error review,
  histograms, per-speaker views, failure analysis) without a separate explicit
  owner authorization with a declared specification.
- A deviation or negative result is recorded, not replaced.
- Describe results as scores. Never describe subset evaluation, externally
  pretrained models or modified protocols as official leaderboard results.

## 9. Failure handling for R7

| Event | Action |
| --- | --- |
| A refusal before SCORED (store, parity or identity mismatch) | Record is unchanged (verify it is still FROZEN). Send the output; do not touch stores or the record. |
| Crash or kill after SCORED, no report | Kill orphaned `multiprocessing.spawn` workers; move any partial `report_locked` aside without opening it; re-run the identical analysis arguments (workers may be reduced); record in the tracker log. |
| Deterministic exception in report plumbing | Stop. A fix is allowed only if no numbers were seen, definitions are unchanged, it is reviewed, it lands at a new commit, and it is recorded as a deviation (the report `git_commit` then differs from the freeze commit). |
| Complete report JSON on disk but the record is still SCORED | Re-run the identical command to finalise. Never edit the record by hand. |
| Stdout lost after a finished run | Recover `report_hash` from the record and recompute it from the report file; log it. |
| Integrity expectation violated after REPORTED | Record as a finding or an invalidated run. REPORTED is final; do not re-run. |

## 10. Known deviations recorded before R7

- R6 was run with `--workers 30` (planned 12). The worker count is not part of
  the pinned provenance and does not affect scores.
- `eval_score.log` of R6 ends at clean 50,000/71,237 with no final stdout JSON
  (cause undetermined). Completion rests on the run sidecar (`complete: true`,
  computed after the completeness validation) and on the pre-flight in the
  tracker; no evaluation predictions were opened.
- The role-free train-manifest hash `b172e50c…` versus the role-bearing
  `b932b859…`: the LR features were extracted before dev reservation roles were
  added (issues #52/#53). Scoring is unaffected.
