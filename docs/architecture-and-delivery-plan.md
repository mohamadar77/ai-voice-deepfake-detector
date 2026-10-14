# Technical architecture and delivery decisions

Owner: Mohamad Abo-Ras. Planning date: 6 October 2026. Tentative presentation: 5 November 2026.

This document is the current architecture and scope addendum to the frozen final proposal (`AI_Voice_Deepfake_Detector_Final_Project_Proposal.docx`; see the **Final proposal sign-off** under "Proposal and document precedence" below). It defines the release decisions and acceptance evidence; [TODO-Plan.md](../TODO-Plan.md) tracks implementation completion. [Course coverage](proposal-review-and-course-coverage.md) is an educational appendix, not a menu of promised features. Repository conventions and evaluation safeguards in [AGENTS.md](../AGENTS.md) remain binding.

## Decision and current evidence

**AI Voice Deepfake Detection: A Reproducible Study of Reliability.** Deliver two model families, a controlled robustness study and a working local application. Compare complete pipelines; do not claim the MFCC/LR versus log-Mel/CNN comparison isolates architecture from feature representation.

Committed experiments: MFCC + Logistic Regression, compact CNN trained clean, and the same CNN trained with one declared augmentation recipe. Report all frozen candidates; select deployment using development only. Add bounded training-fitted PCA/K-means diagnostics. No accuracy improvement is promised.

The only selected stretch was an evaluated evidence assistant over approved documents and frozen reports; it is a no-go for this release (owner, 13 October 2026, #37). Extra model families, transformer audio encoders, external datasets, cloud hosting, calibration/abstention, agent orchestration and MCP are deferred beyond this release. A successful core without the assistant is an acceptable final project.

Code inspection currently shows a configuration loader and smoke CLI, tests for setup/configuration, dependency/configuration files and CI scaffolding. It does not show a dataset pipeline, detector, prediction API or evaluation results. Historical setup checkboxes are retained as prior records, not re-certified by this architecture review. Environment validation must be repeated at the first gate: a prior command in this session could not locate Python 3.12 through uv.

[Environment notes](environment.md) record an M4 Pro, 48 GB memory and roughly 540 GiB free on 4 October; they also record unavailable MPS in that execution environment. These are historical observations. Recheck storage, Python, CPU/MPS and container CPU compatibility before assigning a compute budget.

## Architecture decisions

| Boundary | Planned responsibility and contract | Evidence before acceptance |
| --- | --- | --- |
| `data/` package | Parse official protocols; manifest IDs, labels, source/speaker/attack provenance, official split and development role. Local paths remain local. CSV is the canonical export; SQLite is a rebuildable audit catalog. | Tiny fixtures catch duplicate IDs, missing audio, split crossing and unknown labels. Recorded provenance gaps. |
| `audio/` and `features/` | Validate/decode, mono 16 kHz, four-second windows, deterministic inference, MFCC/log-Mel specification. Training crop/augmentation is an explicit separate mode. | Shapes/finite values, padding/tail handling, repeated-run equality and train-only fitting tests. |
| `training/` | Fit scaler/statistics/models only on training; select checkpoints on selection development; save immutable run artifacts. | Learning curves, bounded runs, exact configs/seeds, no threshold-reservation access during selection. |
| `inference/` | Load a trusted versioned bundle; compute window scores and one original-clip score; apply the bundle's threshold. No training imports or fitting during serving. | Offline/API parity on fixtures and permitted development clips, compatible versions/hashes. |
| `evaluation/` | Verify freeze record; score all declared candidates and conditions; persist predictions before metrics; bootstrap original-source groups. | Freeze/hash mismatch refuses scoring; known-score metric tests; one output per clip/condition/model. |
| `robustness/` | Deterministic noise/compression construction from a frozen spec, applied equally to both classes. | Same source membership and seeds, bounded clipping, transformation failure accounting. |
| `api/` | Typed upload/error contract, byte/decode/concurrency bounds, startup bundle load, cleanup, health/readiness and metadata logs. | Valid/corrupt/oversize/silent/unavailable/cancelled request tests and cleanup checks. |
| `app/` | Thin Streamlit client of the API; spectrogram, score and measured limitations. Never independently load the model. | Actual HTTP upload path and removal of UI-held audio after processing/session cleanup. |
| Local containers | One image, separate API/UI services; CPU inference, bundle mounted read-only, temporary uploads. | Clean build/start, readiness, inference parity and measured cold/warm latency. |

Names are planned module boundaries, not packages already implemented. Keep reusable implementation under `src/voice_detector/`; notebooks only import it. Use Python 3.12 and uv, with no second environment or lockfile system.

### Data and bundle contracts

Manifest: `clip_id`, `source_id` when available, speaker/attack IDs, official partition, development role, label, duration, sample rate, checksum, provenance and local path resolution. Never silently invent source identity; record unknown provenance and the weaker grouping fallback. Partition assignment precedes every derived window/feature/transformation.

Versioned model bundle: schema/model/preprocessing versions, model parameters, training-fitted transforms, feature spec, window/aggregation policy, label mapping, score direction, operating threshold, manifest/config/Git identifiers, dependency versions and checksums. Only trusted locally produced bundles are loaded; user uploads cannot supply model artifacts. A checksum detects mismatch but does not make an untrusted serialized object safe.

Cache key: audio checksum, original partition/clip ID, preprocessing hash, fitted-statistics hash, transformation spec and seed. Separate training and deterministic inference caches. Do not persist one randomly augmented feature per clip and accidentally reuse it as every training epoch's augmentation.

Run record: unique run ID, configuration, seed, dependency versions, Git commit and dirty-state disclosure, manifest hash, preprocessing, model version, runtime, hardware, predictions and metrics. Large artifacts stay ignored; commit small approved summaries and schemas without restricted paths.

API: `POST /predict`, `GET /health`, `GET /ready`. Return score, direction, threshold, label, versions, request ID, decoded duration and processing time. Synthetic is the positive class; larger scores mean more synthetic. Scores from different model families are not numerically comparable probabilities. No unvalidated confidence/probability or inconclusive output. Configure the UI's API URL: localhost for local processes, the API service hostname for separate containers.

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

For diagnostics, fit PCA for a two-dimensional visualization and K-means in scaled training MFCC space. A proposed training-only search over K in {3, 5, 8}, with a capped seeded silhouette sample, keeps the task bounded. Final values and sample cap are recorded before fitting; neither cluster IDs nor visual separation establish authenticity.

### Duration and robustness decisions to freeze

Mono/16 kHz/four-second windows are existing proposal choices. Window stride, tail inclusion, short-clip padding, training crop sampling and aggregation space (for example mean raw window score) still require a single explicit spec and tests. LR and CNN must use their own frozen score conventions consistently. Decide treatment of long clips so extra windows do not silently change per-original-clip training weights.

Accepting 1–30 seconds/20 MB is a target, not evidence of validated performance across that range. Predeclare duration bins and report counts. Narrow the supported scope if data does not support it. API validation and benchmark exclusion rules are distinct: do not silently delete difficult evaluation clips because a demo rejects silence.

Freeze clean + one compression setting + two noise levels, the same source clips for every condition, noise/SNR construction, codec/bitrate, amplitude/clipping policy and seeds. Training augmentation uses training only. Define transformation failure handling before evaluation; report exclusions/failures by condition instead of comparing silently different cohorts.

## Timeline and resource gates

Dates are planning targets for one owner, not a guarantee or measured runtime. The presentation date is tentative in the syllabus. At the first gate record available focused hours, current storage and measured pilot costs. The agreed planning envelope is **110–150 focused person-hours**, or approximately **140–190 hours including 25% contingency**, excluding unattended compute. This assumes familiarity with Python/ML, established libraries and timely data access, and roughly 32–43 hours/week including contingency over the remaining month. At 15–20 hours/week, the full scope is not a credible commitment without help or a scope/date change. These are estimates, not measured feasibility. Bound data/compute before experiments; renegotiate the schedule if the required core still does not fit.

| Target | Gate and deliverable | Stop/go decision |
| --- | --- | --- |
| 6–7 October | G0: revalidate environment, data terms/access, manifest schema, hardware/storage and personal-hour budget | No expensive acquisition/training until access and capacity are recorded. |
| 8–12 October | G1: audited data, shared preprocessing, reproducible LR development report; thin API/UI path using LR or explicitly labeled test fixtures | Demonstrate parity and cleanup early. CNN timing pilot estimates complete runs and evaluation cost. |
| 13–18 October | G2: clean/augmented CNN comparison, train-fitted diagnostics, selected deployment, reserved thresholds and freeze record | Declare all candidates and transformation specs before final testing. Missing compute capacity means a predeclared subset, not adaptive test selection. |
| 19–25 October | G3: locked metrics, paired robustness, failure report, working real-model demo, initial container/cleanup verification | Full core acceptance is needed before a stretch starts. Given the deadline, the assistant may reasonably remain deferred. |
| 26 October | Feature freeze | No new model families, datasets or extensions; repairs and verification only. |
| 27 October–1 November | G4: fresh-checkout/container verification, performance results, model card and slides | Repeatability and presentation readiness take priority over polish. |
| 2–4 November | G5: rehearsed <=12-minute presentation, permitted samples, recorded backup/local fallback | Demonstration works on presentation hardware without network dependence. |
| 5 November | Tentative presentation | Explain results, failures, limits and lessons learned. |

Proposed effort per phase: G0 8–12 hours; G1 24–32; G2 24–32; G3 24–32; G4 20–26; G5 10–14. The total is 110–148 hours, rounded to 110–150 before contingency.

Agreed initial tuning ceiling, to confirm at G0/G1: three LR regularization settings and one fixed imbalance policy; one compact CNN architecture with at most four full runs (clean/augmented × two seeds), proposed maximum 20 epochs and 12 device-hours/run. The second seed pair requires a pilot forecast within 48 device-hours total plus the calendar reserve; decide from capacity, not the first seed's quality. Select checkpoints on development and report all declared seeds. Record unmeasured training variability if using one seed. Set patience and disk caps in YAML. Forecast extraction, validation, compression and final scoring separately, using at least 1.5 times pilot time/storage estimates for planning.

Match clean/augmented architecture, seed and intended training budget. Record actual completed epochs, optimizer steps and clip exposure as well as ceilings and early-stop rules. Equal ceilings alone do not ensure equal exposure. If early stopping or a runtime cap leaves unmatched exposure, report the practical pipeline comparison and limitation; do not claim an isolated augmentation effect or call an unfinished run a completed ablation.

If the full dataset cannot fit, freeze a documented subset rule before looking at model scores; preserve official partition boundaries and disclose the study as a subset. Do not substitute extra external benchmarks or weaker leakage rules for missing capacity.

## Agreed implementation TODOs

All items below remain open; documenting a design is not implementing it. Detailed existing checkboxes remain in the root plan.

- [ ] **A0 — feasibility record (G0):** repeat uv/Python smoke checks; confirm terms/storage/hours; measure hardware and write dataset/run-budget decisions. Owner: Mohamad.
- [ ] **A1 — data contract (G0/G1):** implement metadata parser and fixtures, canonical manifest, SQL audit, checksums/provenance, official and reserved-development membership. Depends on A0.
- [ ] **A2 — shared preprocessing (G1):** specify YAML feature/window/padding/aggregation settings, implement decoding/features, train-only transforms, versioned cache and parity tests. Depends on A1.
- [ ] **A3 — baseline and bundle (G1):** LR tuning within budget, development metrics, trusted bundle loader/scorer, reproduction command and run record. Depends on A2.
- [ ] **A4 — early application (G1):** thin API/UI, typed errors, bounded decode/concurrency, health/readiness, cleanup and local CPU container smoke. Depends on A3; fixture-only UI work may start earlier and must be labeled.
- [ ] **A5 — CNN study (G2):** timed pilot, explicit loss/weighting/optimizer, clean and augmented training, learning curves and conditional paired second seeds. Depends on A2/A3 and capacity gate.
- [ ] **A6 — diagnostic and freeze (G2):** training-fitted PCA/K-means with small training-only K search; development deployment selection, reserved thresholds, full freeze record and refusal tests. Depends on A3/A5.
- [ ] **A7 — locked evaluation (G3):** predictions for all declared models/conditions, required metrics/slices, source-group uncertainty, paired differences and failure report. Depends on A6.
- [ ] **A8 — release reliability (G3/G4):** shared scorer/API parity, measured input-range behavior, container CPU parity, invalid/cancelled request cleanup, cold/warm latency/memory and no-audio logs. Depends on A4/A7.
- [ ] **A9 — reproducibility and presentation (G4/G5):** fresh-checkout reproduction of data preparation, baseline training, CNN training and the report (not baseline only), checksummed bundle instructions, model card, approved summaries, course-evidence appendix, 12-minute slides and backup recording. Depends on A7/A8.
- [ ] **S1 — stretch go/no-go (before 26 October):** only if A0–A8 and core reproduction/model-card evidence pass with presentation contingency remaining, authorize the bounded evidence-assistant experiment; otherwise record deferred. No automatic extension from being listed here.

Issue mapping (13 October 2026):

| Item | GitHub issues |
| --- | --- |
| A1 | #5 (closed), #31 |
| A2 | #4, #7 |
| A3 | #32, #8, #9 |
| A4 | #9, #11 |
| A5 | #10, #33 |
| A6 | #34, #35, #36 |
| A7 | #35, plus G3 issues minted just-in-time |
| S1 | #37 |

Test plan: fast fixtures for partition/provenance invariants; finite deterministic features and padded-window aggregation; freeze mismatches; known-score/one-class metric cases; paired resampling; API parity and error/cleanup paths. Core CI stays independent of datasets, network, GPUs and trained checkpoints. Implementation changes require `uv run ruff check .` and relevant `uv run pytest`; run the full fast suite when practical.

## Selected stretch contract

> **Superseded for this release (13 October 2026):** the owner decided S1 is a no-go (#37). The contract below is kept as the historical record and as course-coverage context. Phase 2 work is in [research-backlog.md](research-backlog.md).

Allocate **zero committed hours** to this stretch. A 16–24-hour pilot is a provisional estimate only if full core acceptance (including Docker/reliability) finishes early enough to complete it before 26 October with release contingency preserved. The current schedule does not provide that window; likely outcome is deferral. Start with approved documentation and already-frozen results, a versioned corpus, TF-IDF/BM25 retrieval comparison and a source-cited answer interface. Add LLM generation only if its configuration/access/cost and evaluation fit the remaining budget; lexical or templated answers alone do not count as RAG. Semantic retrieval is another optional part of this same bounded experiment, not mandatory infrastructure. No agent orchestration, MCP or persistent recording memory.

Author separate development and held-out questions before tuning (for example 10 + 10). Include numeric, answerable, unanswerable and prompt-injection cases. Freeze corpus/prompt/index versions before final question testing. Report recall@k, citation and numeric correctness, unsupported claims, abstention, latency and usage with counts. This is pilot evidence. The assistant cannot change model thresholds, run training, or substitute its own audio authenticity decision. If the gate fails, retain the course topics as conceptual coverage and future work.

## Proposal and document precedence

The final project proposal — `AI_Voice_Deepfake_Detector_Final_Project_Proposal.docx` (the original) — is the authoritative, immutable source for project scope, commitments and acceptance criteria. It is frozen and signed off (see the sign-off below) and must not be edited, regenerated or replaced. `AI_Voice_Deepfake_Detector_Final_Project_Proposal_Revised.docx` is a non-selected historical candidate with no governing authority. This dated Markdown addendum is subordinate: it records dated implementation decisions, module contracts, schedule and explicit deviations. Where any statement in this addendum — or in any other planning document — conflicts with the final proposal, the proposal governs unless an explicit, dated deviation recorded here changes it (naming the affected commitment, reason, owner decision, date and acceptance impact). Neither `.docx` is ever edited.

Canonical authority for scope, commitments and acceptance: the final proposal (the original `...Proposal.docx`). Dated implementation decisions, deviations and deadlines: this file. Completion status: root TODO. Concise execution order: roadmap. Data flow: pipeline. Educational mapping: course-coverage review, with optional examples treated as deferred unless explicitly selected here.

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

Full rationale: [architecture-review-2026-10-13.md](architecture-review-2026-10-13.md). The proposal `.docx` was re-verified unchanged (SHA-256 `b16f5124…a09aec`).

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

### Plan amendments adopted (owner-requested review)

- **G1 narrowed to its critical path:** option D (#4) → features (#7) → development reservation + pre-tuning record ([#31](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/31)) and metrics ([#32](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/32)) → LR dev report (#8) → CNN pilot (#10). EDA (#6), scorer parity (#9) and container smoke (#11) move to a **G2 parallel lane**, done while CNN training runs unattended. #31 and #32 **block** #8, because LR tuning and threshold selection must not touch development data before the reservation is frozen.
- **Forecast:** G1 16 Oct, G2 freeze 22 Oct, G3 27 Oct. Feature freeze (26 Oct), G4 (1 Nov), G5 (4 Nov) and the presentation (5 Nov) are held. The 25% contingency is spent. The cut list is in the review and the roadmap.
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

### External GPU compute protocol (owner-run Kaggle / Colab)

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

### Still pending owner decision

- [#14](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/14): sub-1 s duration gate, before the G2 freeze.
- [#34](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/34): concrete codec, SNR and trim parameters, before the G2 freeze.

## Architecture consultation

A new technical-architecture reviewer, Meitner (`01a11256-2313-7ae3-9800-bcebeb0f1e1e`), used `gpt-6-astra` with `xhigh` reasoning. The lead agent independently audited code/status/documentation and sent a counterproposal covering module contracts, freeze discipline, timeline, early API integration and the late stretch gate.

Discussion and resolution:

- Lead proposed one canonical plan, early baseline API integration, versioned shared inference, all-candidate freeze and a likely-deferred stretch; the architect agreed and added an early Linux/Docker smoke test.
- Architect challenged the initial 80–120-hour estimate. Lead accepted 110–150 focused hours plus 25% contingency and the need to record actual owner availability at G0.
- Architect proposed a bounded CNN run matrix, explicit freeze states, fitted-statistics cache hashes and source-group uncertainty. Lead accepted them and added attack-only slice handling and actual training-exposure accounting.
- Lead asked for final objections after these changes. Architect answered that there were no remaining material objections, clarifying that equal epoch ceilings/patience alone do not guarantee equal completed exposure. That clarification is included above.

The user additionally requested existing session `01a11238-7c72-72c2-96f2-ee77ba12524e`. Direct agent messaging returned “agent not found”; the CLI queue returned “no rollout found.” No additional session was spawned in its place. Its review is **pending/unavailable**, and no three-way agreement is claimed. The plan is ready to start at G0 on the agreement of the lead and Meitner; compute/time feasibility and the unavailable participant's feedback are not represented as confirmed.

## Immediate work order

1. Revalidate the existing Python 3.12/uv setup, smoke command and current machine capacity; record focused hours/week. No detector results are needed for this step.
2. Write `docs/data.md` and a versioned manifest schema plus tiny protocol fixtures; define development speaker roles and nullable provenance.
3. Implement/test the metadata audit and partition checks before full feature extraction. Record access/storage permission and pilot budget.
4. Implement one shared deterministic decoding/window/feature path and its parity tests, then the LR bundle and thin API/UI/container path.

Implementation, data acquisition, training and external changes are separate future work; this consultation updated planning documents only.

## Session handoff — 6 October 2026

Resume from this state in the next Codex session:

- Project: `ai-voice-deepfake-detector`.
- Current conversation: project proposal, technical architecture and G0 feasibility preparation.
- Current core decision: ASVspoof 2019 LA; MFCC + Logistic Regression; clean and augmented compact CNN; locked evaluation; paired noise/compression robustness; PCA/K-means diagnostics; FastAPI/Streamlit; local Docker; reproducibility and observability.
- Current stretch: evidence assistant only if the complete core passes early; zero committed hours and likely deferred.
- Completed today: architecture plan, data contract, manifest validator, fixture tests, aligned TODO/roadmap/pipeline/README/project-management documents.
- No dataset audio, trained model, checkpoint, final prediction or accuracy result exists.
- Open blocker 1: current environment is macOS `x86_64` with Python 3.14; project requires Python 3.12 and the locked PyTorch resolution has no compatible x86_64 macOS wheel. Python 3.12.13 was installed in `/tmp/voice-detector-python`, but locked sync stopped at the PyTorch platform mismatch.
- Open blocker 2: ASVspoof 2019 LA audio and official protocol files are not present. Do not download or train until access, terms, storage and provenance are recorded.
- Next first action: revalidate the intended Apple Silicon/ARM64 machine or choose a compatible documented environment, then run the locked setup checks.
- Next data action: obtain/provide ASVspoof 2019 LA audio and train/development/evaluation protocols; preserve official partitions.
- Next coding action after the environment gate: implement protocol fixtures and the manifest audit, then shared deterministic preprocessing and the Logistic Regression bundle.
- Existing requested architecture session `01a11265-c108-7150-930c-ea044f6bbbbf` was unavailable through the agent interface; do not claim its review was received.

Continue by reading this handoff, `docs/data.md`, `TODO-Plan.md` and `AGENTS.md` before taking actions. Treat this section as a checkpoint, not as evidence that the listed future work is complete.
