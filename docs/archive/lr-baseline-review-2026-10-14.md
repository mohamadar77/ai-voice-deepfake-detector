# LR baseline review — 14 October 2026

**Outcome:** the default LR mathematics and selection path are sound, but input
validation, experiment evidence and bundle publication need repairs before G1's
outputs can be accepted for downstream use. Track repairs as new issues; completed
issues remain closed (owner instruction, 14 October). The frozen proposal is unchanged.

Reviewed commit: `6b02432d5a187a0fc1bb2e0b0ccf43ada5222521`; LR implementation:
`7dedc009d83e8d30f34abb9d0bb3dacefca69524` ([PR #49](https://github.com/mohamadar77/ai-voice-deepfake-detector/pull/49)).
The supplied handoff predates that merge and the CNN pilot in PR #51. PR #49 reports
a successful DL120 extraction/training run; its artifacts were not inspected in this
review. The findings do not establish that a historical run used the wrong data.
Concurrent CNN study edits in the workspace were excluded from the reviewed scope.

## Findings and tracking

| Priority | Follow-up | Confirmed defect and consequence | Original |
| --- | --- | --- | --- |
| P1 | [#52](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/52) | `cli.py:634` reads caches without validating their split, current preprocessing/cache key or clip/label provenance. A synthetic cache declaring `evaluation` is accepted as training input. | #8; feature integration #7 |
| P1 | [#53](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/53) | `baseline.py:175` validates the record alone, not the current manifest. Changed roles are accepted; a missing catalog clip can disappear silently from both development masks. | #8 / #31 |
| P1 | [#54](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/54) | `baseline.py:262` and the CLI omit retained development predictions, SQLite `runs`, and required config/dependency/hardware/run provenance. Metrics cannot be independently reconstructed from the saved outputs. | #8 |
| P1 | [#55](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/55) | `baseline.py:313` replaces the model before writing its record. An injected record-write failure leaves a new model paired with the old threshold. Repeated saves overwrite evidence. | #8 |
| P2 | [#56](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/56) | Matching reservation restoration rewrites the original `frozen_utc` and record hash. The reported intermittent test exposes a provenance defect. | #31 |
| P2 | [#57](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/57) | Unsupported aggregation, regularization-parameter and latency-selection declarations are accepted and echoed despite executing mean/C/EER ordering. The committed defaults work correctly. | #8 |
| P2 | [#58](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/58) | CNN capacity flags use synthetic compute-only timing without the required planning margin or configured full workload. They cannot establish the complete compute-route decision. | #10 |
| P2 | [#59](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/59) | Unconditional Torch import in pilot tests aborts collection in the documented laptop fallback. | #10 |

Line numbers refer to the reviewed commit. Each issue contains evidence, acceptance
criteria, ownership, dependencies and an effort estimate. #55 concerns coherent
multi-file publication; existing #45 concerns the separate power-loss/fsync trade-off.
#52/#53 require integration checks beyond the general catalog validation in #44.

## Handoff decisions checked

- `TYPE_CHECKING` with postponed annotations and deferred audio imports preserves
  the inspected import path; no import-order regression was found.
- `bonafide=0`, `spoof=1` aligns with the actual constant
  `SCORE_DIRECTION="higher_more_spoof"` and the shared float64 threshold decision.
- Arithmetic mean of raw window margins matches the committed protocol.
- `class_weight="balanced"` is fixed across the three declared C values.
- `_eer_value()` correctly preserves EER zero. The default tolerance/smaller-C
  tie-break works; #57 covers unsupported alternative declarations.

## Verification and limits

Using Python 3.12 and the existing environment (no dependency installation):

```sh
UV_CACHE_DIR=/tmp/voice-detector-uv-cache uv run --no-sync pytest \
  tests/test_baseline.py tests/test_reserve_development_cli.py \
  tests/test_pretuning.py tests/test_metrics.py -q
```

**96 passed**, including all 12 baseline tests. Ruff passed on handoff-touched files.
The standard locked `uv run` checks could not resolve Torch 2.14.1 for Intel macOS.
Full existing-environment pytest collection failed on missing Torch (#59); excluding
the pilot produced 227 passed, 7 skipped and 9 failures, all from missing librosa in
feature tests. This is not a full-suite pass. Global Ruff found two violations in
the concurrent untracked CNN dataset file, outside this baseline review.

Additional dataset-free reproductions used genuine frozen records and mocked
fitting/output to verify #52/#53 without real model training. A 160-clip fixture
with one missing catalog row produced only 159 clips across the two role masks.
Bundle fault injection verified #55 without fitting; an advanced clock made #56
deterministic. No datasets were downloaded, no final evaluation predictions were
opened, and no implementation files were changed by the review.

## Milestone implications

- **G1, forecast 16 October:** merged #7/#8/#10/#31/#32 remain completed issues.
  New #52–#59 own the review follow-up. Correct #52/#53 before further LR fitting;
  finish reproducibility/publication/protocol work before accepting the bundle.
  Audit and preserve existing host evidence; document any reconstruction or
  invalidation instead of silently replacing prior runs.
- **G2, forecast 22 October:** #9 and #35 may progress with fixtures; accepted LR
  parity and the development rehearsal require the repaired, recorded bundle.
  #33's compute-route acceptance depends on #58. Evaluation remains locked until
  the complete #35 freeze.
- Preserve the 26 October feature freeze, 27 October G3 forecast, 1 November G4,
  4 November G5 and tentative 5 November presentation. These remain forecasts;
  this review adds blockers, not an unsupported date change. Owner focused-hour
  capacity remains unrecorded in the plan.

The proposal SHA-256 is
`b16f512499efc2f0e328a7be7d6b36cf1f103a5fdfb77c2d8e7e2367f1a09aec`.
Planning changes belong in Markdown and GitHub; no commit or push was requested.
