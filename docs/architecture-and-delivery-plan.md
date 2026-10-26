# Architecture and delivery decisions

Owner: Mohamad Abo-Ras. Planned 6 October 2026; restructured 27 October 2026.
Tentative presentation: 5 November 2026.

This is the architecture and scope addendum to the frozen final proposal
(`AI_Voice_Deepfake_Detector_Final_Project_Proposal.docx`; see "Proposal and document
precedence" below). It owns **design contracts, architecture decisions and dated
deviations**. It does not track task completion ([`../TODO-Plan.md`](../TODO-Plan.md)),
execution order ([`roadmap.md`](roadmap.md)) or the implemented command inventory
([`pipeline.md`](pipeline.md)). Repository conventions and evaluation safeguards in
[`../AGENTS.md`](../AGENTS.md) remain binding. Earlier versions of this file carried
implementation status, a task list (A0 to A9), the evidence-assistant contract, the
6 October session handoff and the consultation narrative; those were removed because
they were stale or duplicated other documents, and remain in Git history.

## Decision summary

**AI Voice Deepfake Detection: A Reproducible Study of Reliability.** Two model
families, a controlled robustness study and a working local application. Complete
pipelines are compared; the MFCC/LR versus log-Mel/CNN comparison does not isolate
architecture from feature representation.

Committed experiments: MFCC + Logistic Regression, a compact CNN trained clean, and
the same CNN trained with one declared augmentation recipe; all frozen candidates are
reported and deployment is selected from development data. No accuracy improvement is
promised.

Out of Phase 1 scope: the evidence assistant (RAG; owner no-go, 13 October 2026, #37),
MCP, extra model families, transformer audio encoders, external datasets, cloud
hosting and calibration/abstention. They are in [`research-backlog.md`](research-backlog.md)
where research-relevant, and in the deviation register below where the proposal named
them. A successful core without them is an acceptable final project.

**Implementation state (27 October 2026, verified against `src/voice_detector/` and
`tests/`).** Implemented and tested: protocol parser and audit, manifest builder,
speaker-grouped development reservation, MFCC preprocessing and feature cache, clip
metrics, LR baseline, compact log-Mel CNN training with augmentation, the
`DRAFT -> FROZEN -> SCORED -> REPORTED` evaluation harness with prediction store and
speaker-grouped bootstrap, the shared scorer, FastAPI service, Streamlit UI and Docker
packaging. Not implemented: training-fitted PCA / K-means diagnostics, the
shortcut-only baseline and EDA report, calibration, and the evidence assistant. The
completion state of each item is in the TODO plan.

## Architecture decisions

The boundaries below were planned on 6 October and are realised as follows. Reusable
code lives under `src/voice_detector/`; notebooks only import it. Python 3.12 and uv,
with no second environment or lockfile system.

| Boundary | Contract | Implementation |
| --- | --- | --- |
| Data | Parse official protocols; manifest IDs, labels, source/speaker/attack provenance, official split and development role. Local paths remain local. CSV is the canonical export; SQLite is a rebuildable audit catalog. | `protocol.py`, `manifest.py`, `reservation.py`, `hashing.py` |
| Audio and features | Validate/decode, mono 16 kHz, four-second windows, deterministic inference, MFCC / log-Mel specification. Training crop and augmentation are an explicit separate mode. | `preprocessing.py`, `features.py`, `cnn_dataset.py` |
| Training | Fit scaler, statistics and models only on training; select checkpoints on selection development; save immutable run artifacts. | `baseline.py`, `cnn_train.py`, `pretuning.py`, `cnn_pilot.py` |
| Inference | Load a trusted versioned bundle; window scores to one clip score; apply the bundle's threshold. No training imports or fitting while serving. | `scoring.py`, `cnn_scoring.py`, `runtime.py` |
| Evaluation | Verify the freeze record; score all declared candidates and conditions; persist predictions before metrics; bootstrap by group. | `evaluation.py`, `eval_scoring.py`, `eval_report.py`, `bootstrap.py`, `metrics.py`, `prediction_store.py`, `eval_supplementary.py` |
| Robustness | Deterministic noise/compression from a frozen spec, applied to both classes. | `augmentation.py`, `configs/robustness_spec.yaml` |
| API | Typed upload/error contract, byte/decode/concurrency bounds, startup bundle verification, cleanup, health/readiness, metadata-only logs. | `serving/` (`app.py`, `service.py`, `decode.py`, `limits.py`, `loader.py`, `observability.py`, `schema.py`, `settings.py`) |
| UI | Thin Streamlit client of the API; spectrogram, score, measured limitations. Never loads a model itself. | `ui.py` (calls the API over HTTP) |
| Containers | One image, separate API/UI services; CPU inference, bundle mounted read-only, temporary uploads. | `Dockerfile`, `compose.yaml` |

### Data and bundle contracts

Manifest: `clip_id`, `source_id` when available, speaker/attack IDs, official partition, development role, label, duration, sample rate, checksum, provenance and local path resolution. Never silently invent source identity; record unknown provenance and the weaker grouping fallback. Partition assignment precedes every derived window/feature/transformation. Schema detail: [`data.md`](data.md).

Versioned model bundle: schema/model/preprocessing versions, model parameters, training-fitted transforms, feature spec, window/aggregation policy, label mapping, score direction, operating threshold, manifest/config/Git identifiers, dependency versions and checksums. Only trusted locally produced bundles are loaded; user uploads cannot supply model artifacts. A checksum detects mismatch but does not make an untrusted serialized object safe.

Cache key: audio checksum, original partition/clip ID, preprocessing hash, fitted-statistics hash, transformation spec and seed. Separate training and deterministic inference caches. Do not persist one randomly augmented feature per clip and accidentally reuse it as every training epoch's augmentation.

Run record: unique run ID, configuration, seed, dependency versions, Git commit and dirty-state disclosure, manifest hash, preprocessing, model version, runtime, hardware, predictions and metrics. Large artifacts stay ignored; commit small approved summaries and schemas without restricted paths.

API contract (as implemented; field-level detail in [`pipeline.md`](pipeline.md)): `POST /predict`, `GET /health`, `GET /ready`, `GET /model`. A response returns `score`, `score_direction`, `threshold`, `decision` (`flagged_spoof_like` or `not_flagged`), versions, hashes, `request_id`, decoded duration and processing time. Synthetic is the positive class; larger scores mean more synthetic. There is no `label`, `confidence`, probability or "inconclusive" output (calibration is not implemented). Scores from different model families are not comparable probabilities. The UI's API URL is configurable: localhost for local processes, the service hostname for containers.

### Evaluation state and leakage controls

1. Audit metadata and establish official partitions. Use training/development for exploratory feature/model choices. Evaluation integrity checks may validate membership/decodability with predeclared rules, but evaluation label distributions or failure patterns must not guide model changes.
2. Reserve development speakers before experiments. `dev_select` chooses features/configurations/checkpoints/deployment. `dev_threshold` is untouched by those choices and sets operating thresholds after candidates are fixed. Proposed reservation: approximately 25% of development speakers, adjusted for the available speaker count and class/attack coverage before freezing membership. Document counts and deterministic membership; do not force an infeasible percentage.
3. Predeclare selection metric and tie-break before tuning. Proposed default: clip-level EER on `dev_select`; within 0.005 absolute EER (0.5 percentage points), prefer the simpler system, then lower measured latency. This tolerance is an engineering choice, not proof of statistical equivalence. Freeze it before tuning. Class weighting, early stopping, duration policy and all hyperparameters live in YAML, not notebook defaults.
4. Set each frozen candidate's threshold on `dev_threshold` to target human false-positive rate at most 5%, with exact tie/comparison behavior and sample counts. If the reservation cannot support a useful operating estimate, disclose the limitation; do not consult evaluation to fix it. Calibration is deferred.
5. Freeze models/seeds, hashes, membership, preprocessing/aggregation, score direction, thresholds, metrics/slices, transformations and bootstrap method before evaluation predictions are inspected. Include all selected seeds and the deployment decision, not only a post-test winner.
6. Implement explicit `DRAFT → FROZEN → SCORED → REPORTED` records. Score the locked test using the shared scorer; write predictions, then metrics, and reject silent result overwrite. A YAML `evaluation_locked: true` flag alone is not enforcement. The implementation must validate a complete freeze record and refuse mismatched artifacts. Local files do not provide tamper-proof security; immutability is an auditable workflow.
7. After inspection, report errors and limitations without fitting or selecting again. A genuine implementation defect requires a documented invalidated run and transparent deviation; never silently replace an unfavorable result.

Required results: EER, macro F1, balanced accuracy, ROC AUC, class counts, confusion matrices, human false-positive rate and synthetic detection rate at the frozen threshold; unfamiliar-attack results and slice sample counts. Pair robustness changes by underlying recording. Bootstrap repeated observations by original source where available and disclose fallback/dependence limitations. Missing-class slices receive undefined metrics with counts rather than invented values. Test EER is descriptive and never selects the deployed threshold.

An attack-only slice normally contains synthetic clips: report detection rate and counts. For attack-specific EER/AUC, predeclare a bona fide comparison pool and disclose its reuse across attacks. Such slices are not independent datasets. Three first-seed bundles across four conditions produce 12 model–condition combinations; extra paired seeds expand that matrix. Keep each bundle's threshold unchanged across conditions.

Diagnostics contract (**not implemented**; open as #36): fit PCA for a two-dimensional visualization and K-means in scaled training MFCC space. A proposed training-only search over K in {3, 5, 8}, with a capped seeded silhouette sample, keeps the task bounded. Final values and sample cap are recorded before fitting; neither cluster IDs nor visual separation establish authenticity.

### Duration and robustness decisions

Mono/16 kHz/four-second windows are proposal choices. Window stride, tail handling and short-clip policy were frozen as option D (11 October 2026; pooled MFCC statistics over real samples only, sub-frame tails folded, a clip under four seconds is one window over real audio; `mfcc-v1` became `mfcc-v2`). The CNN uses the log-Mel spec `mel-v1` in `configs/cnn_study.yaml` without unjustified zero-padding. LR and CNN each use their own frozen score convention (mean window decision function / mean window logit).

The service accepts WAV/FLAC of 1 to 30 seconds and at most 20 MiB. That range is an input policy, not evidence of validated performance across it; clips under 1 s (1,277 in the LA manifest, per #14) were scored by the benchmark but not validated separately and are rejected by the service (#14, closed). API validation and benchmark exclusion rules are distinct: no difficult evaluation clip was silently removed because the demo rejects it.

Robustness: clean, one compression setting (Opus 24 kbps), two noise levels (20 and 10 dB SNR) on the same source clips for every condition, with parameters and seeds in `configs/robustness_spec.yaml`; training augmentation uses training data only; plus an evaluation-only silence-trim diagnostic that is not a fifth robustness condition and not a candidate.

## Timeline and resource envelope

Dates are planning targets for one owner, not guarantees; the presentation date is tentative. The agreed planning envelope was **110 to 150 focused person-hours** (about 140 to 190 including 25% contingency), excluding unattended compute. That envelope was an estimate, not measured feasibility, and the owner's actual weekly hours were never recorded (open item in the TODO plan). The 25% contingency was spent by 13 October.

| Target | Gate | Deliverable |
| --- | --- | --- |
| 6-7 October | G0 | Environment, data terms/access, manifest schema, hardware/storage |
| 8-12 October | G1 | Audited data, shared preprocessing, reproducible LR development report |
| 13-18 October | G2 | Clean/augmented CNN, development selection, reserved thresholds, all-candidate freeze |
| 19-25 October | G3 | Locked metrics, paired robustness, failure report, real-model demo |
| 26 October | Feature freeze | No new model families, datasets or extensions |
| 27 October-1 November | G4 | Fresh-checkout and container verification, performance results, model card, slides |
| 2-4 November | G5 | Rehearsed <=12-minute presentation, fallback and backup |
| 5 November | Presentation | Tentative |

Re-forecast dates and current priorities are in [`roadmap.md`](roadmap.md). Tuning ceiling that was applied: three LR regularization values with one fixed class-imbalance policy; one compact CNN architecture with four full runs (clean/augmented x two seeds), at most 20 epochs, early stopping on development clip EER. Clean and augmented runs share architecture, seed and intended budget; actual completed epochs are in the run records, and equal ceilings do not guarantee equal exposure, so the comparison is a practical pipeline comparison, not an isolated augmentation effect. If a full-dataset run had not fit, a documented subset rule would have been frozen before looking at scores; that was not needed.

## Proposal and document precedence

The final project proposal — `AI_Voice_Deepfake_Detector_Final_Project_Proposal.docx` (the original) — is the authoritative, immutable source for project scope, commitments and acceptance criteria. It is frozen and signed off (see the sign-off below) and must not be edited, regenerated or replaced. `AI_Voice_Deepfake_Detector_Final_Project_Proposal_Revised.docx` is a non-selected historical candidate with no governing authority. This dated Markdown addendum is subordinate: it records dated implementation decisions, module contracts, schedule and explicit deviations. Where any statement in this addendum — or in any other planning document — conflicts with the final proposal, the proposal governs unless an explicit, dated deviation recorded here changes it (naming the affected commitment, reason, owner decision, date and acceptance impact). Neither `.docx` is ever edited.

Canonical authority for scope, commitments and acceptance: the final proposal (the original `...Proposal.docx`). Dated implementation decisions, deviations and deadlines: this file. Completion status: root TODO. Execution order: roadmap. Implemented flow: pipeline. Ownership of every document: [`README.md`](README.md). Syllabus mapping: [`course-coverage.md`](course-coverage.md), where optional examples are not commitments.

**Correction — 2026-10-11.** An earlier edit this day wrongly named `_Revised.docx` as the authoritative proposal and attributed a "Phase 1–5" execution sequence to it (that structure belongs to the Revised candidate, not the frozen original). That designation is withdrawn. Per the owner decision and a second architecture/tech-lead consultation (Codex `gpt-6-astra`, read-only), the original `...Proposal.docx` is the sole frozen proposal; see the sign-off below. This also supersedes the prior "the original Word proposal remains historical scientific background / superseded by this addendum" language.

### Final proposal sign-off — 2026-10-11

- **Owner:** Mohamad Abo-Ras
- **Status:** ADOPTED by owner (2026-10-11).
- **Architecture/tech-lead recommendation:** APPROVE the original proposal (Candidate A), via Codex `gpt-6-astra` read-only consultation.

**Final proposal (frozen, immutable):** `docs/AI_Voice_Deepfake_Detector_Final_Project_Proposal.docx`
**SHA-256 (immutability pin):** `b16f512499efc2f0e328a7be7d6b36cf1f103a5fdfb77c2d8e7e2367f1a09aec`

This exact file is the sole FINAL, FROZEN, IMMUTABLE proposal and the source of truth for the approved proposal baseline. Its bytes must never be edited, regenerated or replaced; verify the SHA-256 at submission/release. `docs/AI_Voice_Deepfake_Detector_Final_Project_Proposal_Revised.docx` (SHA-256 `128c132debafa395650cc7b89cb28c8a75cd6e02d5434fa371d5a5c3528d0a94`) is a non-selected historical candidate with no governing authority. Both `.docx` files remain unchanged.

All future corrections, clarifications, schedule revisions, scope decisions and acceptance changes belong exclusively in Markdown (`docs/architecture-and-delivery-plan.md`, `docs/roadmap.md`, `TODO-Plan.md`), never in either `.docx`. This architecture plan holds the canonical dated decision record; each substantive departure names the affected proposal commitment, reason, owner decision, date and acceptance impact. Absent a recorded change, the proposal baseline governs. `AGENTS.md` safeguards remain binding.

Initial reconciliation recorded with this sign-off: retain the full ASVspoof 2019 LA core; defer transformer embeddings and external transfer datasets (ASVspoof 2021 DF, WaveFake) for this release, with rationale; keep the evidence assistant optional and gated on full core acceptance. The original's dated October–November milestones are preserved as the baseline plan; current forecasts and slippage are tracked in the roadmap and TODO.

This sign-off freezes the proposal only. It does not certify environment readiness, completed experiments, achieved performance or completion of delivery gates.

### Revised-candidate retirement — 2026-10-11

`AI_Voice_Deepfake_Detector_Final_Project_Proposal_Revised.docx` (SHA-256 `128c132debafa395650cc7b89cb28c8a75cd6e02d5434fa371d5a5c3528d0a94`) was safely dropped from the repository on 2026-10-11 after its unique details were synced into the planning docs, so no information is lost. Per the Codex architecture/tech-lead consultation (consult #2, "content to preserve from the non-chosen file"), the following were folded in before removal:

- **Report-template freeze** added to the pre-inspection freeze inventory (TODO §7).
- **Fresh-checkout reproduction** expanded from baseline-only to data preparation + baseline + **CNN** + report (TODO §10, A9 above), consistent with the final-acceptance checklist.
- **"Failure-analysis dashboard"** intent recorded as satisfied by the failure report plus the Streamlit evidence view — no separate dashboard deliverable (TODO §8).
- **RAG engineering** vocabulary (enriched metadata; structured, schema-validated output) added to the optional evidence-assistant plan (TODO §11).

The frozen proposal (the original) remains the authoritative baseline; these are implementation elaborations recorded in Markdown, not changes to the `.docx`. The retired file's bytes are identified by the SHA-256 above should it ever need to be matched again.

## Dated decisions — 13 October 2026 (architecture review)

Full rationale: [archive/architecture-review-2026-10-13.md](archive/architecture-review-2026-10-13.md) (historical). The proposal `.docx` was re-verified unchanged (SHA-256 `b16f5124…a09aec`).

### Deviation register (proposal commitments not delivered in this release)

The 11 October sign-off deferred these items "with rationale" but did not record the fields this plan requires. They are recorded here.

| Proposal commitment (where) | Status | Reason | Owner decision | Acceptance impact |
| --- | --- | --- | --- | --- |
| ASVspoof 2021 DF locked transfer test. Appears in research question 2 ("a second evaluation domain"), the data-partition table and the "Transfer and codec shift" experiment. | Deferred | CPU-only host and the schedule. The proposal's own risk rule says to defer 2021 transfer before delaying the main report. | Sign-off, 2026-10-11 | The report must state that RQ2's second-domain part was not run. RQ2 is answered only for unseen attacks A07–A19 plus compression/noise. In Phase 2 it leads the [research backlog](research-backlog.md). |
| WaveFake external test ("if time permits") | Deferred | Conditional in the proposal; no capacity | Sign-off, 2026-10-11 | Listed as not run |
| Frozen wav2vec 2.0 embeddings. Appears in research question 3 ("if resources allow") and the "next enhancement" row. | Deferred | The resource gate fails: CPU-only host and schedule | Sign-off, 2026-10-11 | RQ3 is reported as not run; no transfer-learning claim |
| Calibration (reliability plot, Brier score) and the "inconclusive" band | Not implemented | The proposal makes both conditional ("if calibration is implemented") | Architecture agreement, 2026-10-06 | Outputs are labeled scores; no inconclusive output |
| Cloud deployment ("attempt one small cloud deployment after the local demo is reliable") | Deferred | Conditional in the proposal; local container only | Architecture agreement, 2026-10-06 | No hosted claim |
| RAG evidence assistant and MCP interface ("optional extension") | RAG: **no-go**, owner decision 2026-10-13 (#37). MCP: deferred. | Optional in the proposal | Architecture agreement, 2026-10-06 | None. Course coverage stays conceptual. |
| Deployed-model selection order (evaluation protocol section 7: choose the deployed bundle from development results before and independently of the locked report) | Deviation, disclosed | The development-only selection rule was written and accepted on 2026-10-19 after the locked report (R7) had been read; it was applied mechanically to development numbers (`docs/deployed-model-decision.md`, #86). Swapping the model afterwards would be selection on test data. | Owner, 2026-10-20: keep `cnn_augmented_seed1` | Described everywhere as a "development-only rule", never "pre-declared" or "blind". The chosen model has the highest clean unseen-attack EER of the four CNNs (11.77%), reported as a finding. |
| Supplementary eval-report analysis tool (PR #82, `eval-supplement`) not run on evaluation data | Dropped from Phase 1 | Code merged but never reviewed (#87) or authorized for execution on locked evaluation predictions. Architect recommendation (2026-10-20): drop if not reviewed and authorized by 2026-10-24. Deadline passed without review. | Agent, 2026-10-25: drop per architect deadline; owner approved "go with your recommendations" on 2026-10-20 | No supplementary per-attack or failure-mode tables beyond what the locked report already contains. The tool remains in the codebase for Phase 2 use after review. |

### Plan amendments adopted (owner-requested review)

- **G1 narrowed to its critical path:** option D (#4) → features (#7) → development reservation + pre-tuning record ([#31](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/31)) and metrics ([#32](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/32)) → LR dev report (#8) → CNN pilot (#10). EDA (#6), scorer parity (#9) and container smoke (#11) move to a **G2 parallel lane**, done while CNN training runs unattended. #31 and #32 **block** #8, because LR tuning and threshold selection must not touch development data before the reservation is frozen.
- **Forecast:** G1 16 Oct, G2 freeze 22 Oct, G3 27 Oct. Feature freeze (26 Oct), G4 (1 Nov), G5 (4 Nov) and the presentation (5 Nov) are held. The 25% contingency is spent. The cut list is in the archived review.
- **New G2 work items:**
  - CNN study, including the input/padding policy, decided before training ([#33](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/33));
  - robustness + silence-shortcut transformation spec ([#34](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/34));
  - evaluation protocol + freeze record ([#35](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/35));
  - PCA/K-means ([#36](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/36)).
- **Silence-shortcut diagnostic.** This implements the proposal's risk commitment "inspect silence and duration effects" as a controlled test, not an EDA plot. It has two parts:
  - a shortcut-only baseline on development data (#6);
  - a paired, evaluation-only silence-trimmed condition, declared before the freeze and scored for every frozen bundle (#34).

  It is a diagnostic condition alongside the four robustness conditions, not a fifth robustness condition, and not a new candidate.
- **Predeclared analysis details (#35):**
  - duration bins <1, 1–<2, 2–<4, 4–10 and >10 s, with per-class counts;
  - bootstrap resampling by `speaker_id` when no source mapping exists;
  - the report maps each proposal research question to evidence or to an explicit "not run";
  - published B01/B02 challenge baselines are cited as context only, never as reproduced results.
- **Demo set:** the owner's consented recordings plus owner-generated synthetic clips from a modern TTS/voice-clone tool whose terms allow it. Labeled illustrative with n reported. A miss is presented as a measured limitation.
- **Container on the presentation machine** is checked during G2/G3 (#11), not first at G5.

### Owner decisions — 13 October 2026 (two-phase strategy)

Recorded from the owner's direction after the review. The project's direction is
unchanged; the immediate focus moves from planning to experiments, evidence and
delivery.

- **Two phases.** Phase 1 is the graduation release (5 Nov). It optimizes for:
  1. a working end-to-end system;
  2. correct methodology;
  3. reproducible experiments;
  4. measured results and failure analysis;
  5. demo and presentation;
  6. delivery within time and compute.

  Experimental correctness is never traded for the deadline. Phase 2 is post-graduation research, tracked separately in [research-backlog.md](research-backlog.md) and the GitHub milestone *Phase 2: post-graduation research (backlog)*. Phase 1 keeps interfaces extensible and archives evidence, but builds no research-only extensions.
- **S1 evidence assistant: no-go for this release** ([#37](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/37), closed). RAG/MCP are off the research track. Cross-domain transfer (2021 DF) leads the Phase 2 backlog, matching the proposal's own priority order. This supersedes the S1 contract below for this release.
- **Moved to Phase 2:** [#38](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/38) (min t-DCF) and [#39](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/39) (matched-input control). UI polish is cut to the minimum demo view: score, spectrogram, measured summary, limitations.
- **Phase 1 / Phase 2 decision table** (canonical copy in [research-backlog.md](research-backlog.md)):
  - keep (Phase 1): MFCC + LR, CNN, the silence-shortcut investigation, REST API + simple UI, containerization;
  - **mandatory**: development/test separation and unseen-attack evaluation;
  - defer: advanced pretrained models, the RAG assistant, extensive hyperparameter optimization, and additional research datasets (unless already committed by the proposal and feasible).

  Defer research *features*, not the architecture's ability to support them. Datasets enter only as canonical-schema manifests through dataset-specific builders; model interfaces stay behind `Scorer` and the versioned bundle; preprocessing is versioned per feature type; every run is tracked in JSON plus the SQLite `runs` table.
- **Backlog pull gate.** With good progress, the owner may pull a backlog item into Phase 1. Every one of these conditions must hold:
  1. P0–P2 are done and P3 is on forecast;
  2. new candidates, metrics or conditions are declared before the G2 freeze, while scoring-only additions on frozen bundles get their own freeze record;
  3. the item finishes before the 26 Oct feature freeze;
  4. Phase 1 results are never changed;
  5. the item is timeboxed, with a stop rule.

  Each pull is recorded here before work starts. Ordered candidates: second CNN seed pair, #38 min t-DCF, 2021 DF scoring-only transfer (committed by the proposal), #39.
- **No advanced or pretrained models in Phase 1.** A reliable, thoroughly evaluated classical model takes priority over a pretrained model with weak evaluation.
- **Methodology claims require artifacts.** Leakage tests on the real manifest, reproducible re-runs, per-attack results and offline/API parity come before any "rigorous" claim in the report or slides. See the review, "Design is not evidence".
- **Known versus unseen attacks (correction).** ASVspoof 2019 LA evaluation attacks A16 and A19 reuse the A04 and A06 algorithms from training (Wang et al., 2020; verify against the challenge PDFs on the host). The locked report therefore has three slices:
  - development A01–A06: known algorithms, unseen speakers;
  - evaluation A16/A19: known algorithms, unseen speakers;
  - evaluation A07–A15, A17, A18: unseen algorithms and unseen speakers; this is the headline "unseen" result.

  Each slice is reported per attack and pooled. Recorded in #35 and `docs/data.md`.
- **Metrics.** Per-class precision and recall are added to the required set (#32). Precision depends on class prevalence (about 1:8.8), so it is reported with the prevalence it was measured at and is never presented as transferable.
- **Develop locally, test and run on the target.** Code is developed on the owner's laptop and pushed. CI runs the full suite on Linux x86_64. The DL120 Fedora host (44 threads, 32 GB, 270 GB) is the **target and demo environment**: the full suite on each pushed slice, data jobs, training, locked evaluation and the demo. A slice is done only when local checks, CI and the host suite pass on the same commit; code is never patched on the host. Disk budget and the loop are in `docs/environment.md`. The host is reached **remotely over SSH**, including for the demo. Rules:
  - long jobs run detached and resumable;
  - services bind to localhost and are reached through an SSH tunnel;
  - to keep the G5 "no network dependence" criterion, the presentation image, frozen bundle and demo clips are exported to the laptop's Docker as a verified offline fallback, with the backup recording as the last resort.

  See `docs/environment.md`, "Access".
- **Working mode.** Issue by issue, with tested code and verifiable outputs; no further broad architecture reviews in Phase 1. The owner picks the review gate (`/consult`, implement then `/code-review`, or none) for each slice.

### External GPU compute protocol (owner-run Kaggle / Colab) — not used

The CPU host was sufficient for all four CNN runs, so this protocol was never invoked; it is kept as the rule for any future external run.

The DL120 CPU host stays the default. The #10 pilot decides whether a GPU is needed. Use the GPU route if a forecast CNN run exceeds the 12 device-hour cap, or if the run matrix exceeds 48 device-hours. The owner runs the notebooks by hand. Any external run must:

1. **Never receive evaluation audio or evaluation features.** Upload only the training and development partitions, or their features, as a *private* dataset, with the ODC-By attribution notice. Locked evaluation scoring runs only on the host, through the shared scorer, after the freeze.
2. Check out a **pinned commit**. Install with `uv sync --frozen --all-groups` under Python 3.12 (`uv python install 3.12`; the locked Linux x86_64 CUDA `torch` matches these runtimes). A platform-forced deviation is recorded, never silently patched.
3. Read every setting from the versioned YAML config. The notebook stays a thin launcher that imports package code.
4. Record GPU model, driver/CUDA, torch version, seeds and the determinism settings used. GPU kernels may be nondeterministic, so record the tolerance.
5. Pull back the checkpoint, run record and learning curves, with SHA-256 checksums. **Recompute the development predictions on the host** with the shared scorer, and check them against the notebook's predictions within the recorded tolerance before the bundle can be frozen.

### Dated decisions — 14 October 2026 (#31 reservation and pre-tuning record)

- **Development reservation frozen as a rule** (`configs/dev_reservation.yaml`,
  schema `dev-reservation-v1`, assignment `sha256-rank-v1`). About 25% of the 20
  development speakers are reserved for `threshold`, the rest go to `select`,
  rounded half up and clamped so neither role is empty. A speaker's rank is
  `sha256("<seed>|<speaker_id>")` under a dedicated reservation seed, kept
  separate from `project.seed` so a training-seed change cannot move a protocol
  boundary. Chosen over a PRNG shuffle because the shuffling algorithm is not
  guaranteed stable across library versions, while SHA-256 is fixed by its
  specification. Implements leakage control 2.
- **Coverage gates the freeze.** Both roles must hold both classes and all of
  A01-A06, and every reserved speaker must contribute at least one bona fide
  clip. A failure refuses the freeze and reports the evidence; the *fraction* is
  then adjusted, never the rule, and the reason is recorded in the frozen record.
- **The frozen record is the authoritative membership**
  (`reports/dev_reservation_frozen.json`): speaker IDs only, with per-role counts
  by label and attack, per-speaker counts, the manifest hash before and after,
  and the seed, method and commit. No clip IDs, paths or audio. The manifest's
  `development_role` column is the git-ignored working copy, and
  `reservation.verify_frozen_reservation` lets a later slice prove it read the
  manifest the record was frozen against.
- **The freeze changes the manifest hash by design**, so it runs on the host
  **before** the #7 feature extraction (the hazard recorded on #7/#31).
- **Pre-tuning protocol record declared before any development score existed**
  (`configs/pre_tuning_protocol.yaml`, dated note in
  [pre-tuning-protocol.md](pre-tuning-protocol.md)): clip-level EER on
  `dev_select`; tie-break within 0.005 absolute EER, then the simpler system,
  then lower measured latency; three LR `C` values `[0.01, 1.0, 100.0]` with one
  fixed `class_weight = balanced`; mean per-window LR decision function
  aggregated per `clip_id`; threshold at a 5% bona fide false-positive rate on
  `dev_threshold`. Implements leakage controls 3 and 4.
- **The record is enforced, not decorative.** `src/voice_detector/pretuning.py`
  rejects a record that disagrees with the constant in the module implementing
  each declared convention — `metrics.METRICS_VERSION`,
  `metrics.SELECTION_METRIC`, `metrics.SELECTION_DIRECTION`,
  `metrics.THRESHOLD_RULE`, `scoring.SCORE_DIRECTION`, `scoring.DECISION_RULE`
  and the reservation role names — and derives the tie rule by asking
  `scoring.decide_spoof` what it does at the threshold rather than restating it.
  The candidate budget is bounded by `pretuning.MAX_DECLARED_CANDIDATES`, so it
  cannot grow into a hyperparameter search. This answers the plan's own rule that
  a configuration flag is not enforcement. The tie-break order,
  `window_to_clip` and `regularization_parameter` are still declarations only, as
  nothing implements them yet; #8 must honour them. Later changes are protocol
  deviations with dated rows.
- **Both freeze guards are independent, and the committed record outranks the
  manifest** (added 14 Oct after the slice review). The manifest is git-ignored
  and hours to rebuild, so a rebuilt manifest carries no roles to disagree with:
  `reserve-development` reconciles against `reports/dev_reservation_frozen.json`
  first and the manifest's role column second, and either disagreement refuses
  without `--force`. Loading a record re-derives its membership from the seed and
  fraction it records and requires the roles to be disjoint, so a hand-edited
  record is rejected although its hashes still agree. A clamped
  `threshold_fraction` that rounding cannot explain is a coverage failure rather
  than a silently different reservation, and the post-write self-check now runs
  against the staged catalog before the first `os.replace`.
- **Disclosed limitation:** with 5 reserved speakers the operating threshold
  rests on few speakers. Its achieved false-positive rate holds on this
  reservation only; the evaluation partition is never consulted to improve it.
- **Process deviation.** The #31 core reached `main` as a WIP commit (PR #42)
  without tests, config or CLI. Owner decision, 14 Oct: leave it and finish
  forward rather than revert, with the completing PR reviewed over the whole
  slice range so the merged core is still reviewed. One PR per issue remains the
  standard; this is the recorded exception.

### Hosts used (added 27 October 2026)

The 13 October decision names the DL120 host as the target and demo environment. The record shows two Linux x86_64 CPU hosts were used: `hpe-dl120gen10-01` (G0 environment gate, data acquisition and checksums) and `hp-xl420gen9-01` (locked scoring R6 and report R7, serving verification, the latency benchmark, container build and parity; per TODO-Plan the four CNN runs). Which machine presents on 5 November is not recorded consistently; see [`environment.md`](environment.md).

### Resolved since the 14 October decisions

- #14 (sub-1 s gate): the service rejects clips under 1 s with `duration_out_of_range`; closed.
- #34 (codec, SNR and trim parameters): frozen in `configs/robustness_spec.yaml` before the freeze.
- The 13 October rule that every run is tracked in JSON plus a SQLite `runs` table is only half implemented: run records are JSON files in each bundle, and no `runs` table exists in `src/` (open as #54).
- 14 October review follow-ups ([archived review](archive/lr-baseline-review-2026-10-14.md)): the owner decision stands that completed issues stay closed and repairs get new linked issues. Issues #52 to #59 are still open on GitHub and sit in the Phase 2 milestone (demo-first scope cut). The role-free versus role-bearing manifest hash gap (#52/#53) is disclosed in [`evaluation-protocol.md`](evaluation-protocol.md) section 10.

## Evidence-assistant stretch (S1)

No-go for this release (owner, 13 October 2026, #37). The earlier bounded contract (TF-IDF/BM25 retrieval over approved documents, cited generation only if evaluable, 10 + 10 development and held-out questions, no agents, MCP or recording memory, and an assistant that can never change thresholds or classify audio) is preserved in Git history. RAG and MCP are not on the research track either ([`research-backlog.md`](research-backlog.md)). [`course-coverage.md`](course-coverage.md) treats them as conceptual coverage only.
