# Pre-tuning protocol record — declared 14 October 2026

**Status: declared before any development score existed.** No model had been
fitted, no feature cache built and no development score computed when this was
written. That is the whole point: a selection rule chosen after seeing numbers is
not a selection rule. Issue
[#31](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/31)
requires it to be committed before
[#8](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/8) runs.

The machine-readable record is [`configs/pre_tuning_protocol.yaml`](../configs/pre_tuning_protocol.yaml);
the reservation rule it depends on is [`configs/dev_reservation.yaml`](../configs/dev_reservation.yaml).
This file is the dated note that accompanies them. Where the two disagree, the
YAML is authoritative, because the code reads the YAML.

## Why this is enforced, not just documented

The architecture plan is explicit that "a YAML `evaluation_locked: true` flag
alone is not enforcement". So `src/voice_detector/pretuning.py` loads this record
and **rejects** it when a declared convention disagrees with the code that
implements it. A record that merely claims `metrics-v1` while the code computes
something else fails to load, so #8 cannot run against a stale declaration.

What is machine-checked, and how:

| Declared field | Checked against |
| --- | --- |
| `metrics_version` | `metrics.METRICS_VERSION` |
| `score_direction` | `scoring.SCORE_DIRECTION` |
| `decision_rule` | `scoring.DECISION_RULE` |
| `selection.metric`, `selection.direction` | `metrics.SELECTION_METRIC`, `metrics.SELECTION_DIRECTION`, plus a check that the metric still names a field `metrics.ClipMetrics` reports |
| `threshold.rule` | `metrics.THRESHOLD_RULE` |
| `threshold.ties_count_as` | derived by asking `scoring.decide_spoof` what it does at the threshold, not by restating its prose |
| `selection.development_role`, `threshold.development_role` | `reservation.ROLE_SELECT`, `reservation.ROLE_THRESHOLD` |
| `lr_budget.model`, `class_weight` | `pretuning.LR_MODELS`, `pretuning.LR_CLASS_WEIGHTS` |
| `lr_budget.max_candidates`, `values` | `pretuning.MAX_DECLARED_CANDIDATES`, and `values` may not exceed the declared ceiling |
| `declared_on` | an ISO calendar date, not in the future |

Each of these constants lives beside the code that implements it, so changing
the implementation breaks this record rather than silently diverging from it.
Three things remain **declarations only**, because no code yet implements them:
the tie-break criteria and their order, `aggregation.window_to_clip`, and
`lr_budget.regularization_parameter`. #8 must honour them; nothing enforces them
until it does.

## The declaration

| Item | Declared value | Implemented by |
| --- | --- | --- |
| Metric definition | `metrics-v1` — tie-grouped ROC, linear interpolation at FNR = FPR | `metrics.compute_eer` |
| Score direction | `higher_more_spoof`; synthetic is the positive class | `scoring.SCORE_DIRECTION` |
| Decision rule | spoof if `score >= threshold`, compared in float64 (ties are spoof) | `scoring.decide_spoof` |
| Selection partition | `dev_select` (`development_role = select`) | `reservation.ROLE_SELECT` |
| Selection metric | clip-level EER, lower is better, threshold-free | `evaluate_clip_scores(..., threshold=None)` |
| Tie-break | within 0.005 absolute EER: prefer the simpler system, then lower measured latency | — |
| LR budget | three `C` values `[0.01, 1.0, 100.0]`, one fixed `class_weight = balanced` | #8 |
| Window → clip | mean of the per-window LR decision function; one score per `clip_id` | #8 |
| Threshold partition | `dev_threshold` (`development_role = threshold`), opened only after candidates are fixed | `reservation.ROLE_THRESHOLD` |
| Threshold rule | lowest threshold whose bona fide FPR is at most `target_fpr = 0.05` | `metrics.threshold_at_target_fpr` |

**Scope.** Selection, aggregation and threshold are the project's general
development protocol. `lr_budget` is the MFCC + Logistic Regression baseline's
own budget. The CNN declares its own score convention and budget in
[#33](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/33),
because raw scores from different model families are not comparable.

## Stated limitations

- The 0.005 EER tie-break tolerance is an **engineering choice**, not a claim of
  statistical equivalence. It is frozen here so it cannot be widened to favour a
  preferred system later.
- Three regularization values is a **declared budget**, not a search. Extensive
  hyperparameter optimization is Phase 2.
- The threshold's 5% target holds **on the reservation it was chosen on**. With
  only the reserved speakers behind it, it is not a guarantee on other data, and
  the evaluation partition is never consulted to improve it.
- Outputs are **scores**. No calibration is fitted in this release, so nothing
  here licenses a probability or an "inconclusive" band.

## Deviation rule

Changing any value above after a development score has been computed is a
protocol deviation, not a configuration change. It is recorded as a dated row in
[`architecture-and-delivery-plan.md`](architecture-and-delivery-plan.md) with the
reason, and the affected runs are re-reported or marked invalidated. An
unfavourable result is never silently replaced.
