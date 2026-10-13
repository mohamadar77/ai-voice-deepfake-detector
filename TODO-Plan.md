# AI Voice Deepfake Detector TODO Plan

Owner: Mohamad Abo-Ras  
Created: 4 October 2026  
Target presentation: 5 November 2026, subject to the course schedule

Build and evaluate a reproducible detector for human versus synthesized or voice converted speech. Compare an MFCC feature baseline with a Convolutional Neural Network (CNN), study unfamiliar attacks and audio distortions, and demonstrate a working application.

This checklist follows the original proposal and the [6 October architecture addendum](docs/architecture-and-delivery-plan.md). Existing checked setup items are historical records; runtime/environment readiness must be revalidated at G0. Check an item only when its output exists and its completion criteria have been met. Record experiment findings even when a model fails to improve.

## How to use this plan

- **Proposal:** `docs/AI_Voice_Deepfake_Detector_Final_Project_Proposal.docx` is the final, frozen, immutable proposal and source of truth for scope/commitments/acceptance (SHA-256 `b16f5124…`; signed off 2026-10-11 — see the sign-off in `docs/architecture-and-delivery-plan.md`). `_Revised.docx` is a non-selected historical candidate. `docs/architecture-and-delivery-plan.md` is the current scope, contracts, budget and schedule addendum, subordinate to the frozen proposal.
- **Execution:** this file is the source of truth for implementation status. Check an item only when its output exists and its completion criteria are met.
- **Resuming implementation:** start from the [Phase 1 resume brief](docs/claude-session-handoff-2026-10-13.md). It gives the reading order, standing rules, environment, open-PR merge order and the work order.
- **Roadmap:** `docs/roadmap.md` contains the prioritized Phase 1 critical path, safely deferrable work and stop/go rules.
- **Two phases (owner, 13 October 2026):** Phase 1 = this graduation release (5 November). Phase 2 = post-graduation research, kept in the separate [research backlog](docs/research-backlog.md). Phase 1 builds no research-only extensions, but keeps datasets (canonical-schema manifests), model interfaces, preprocessing versions and experiment tracking modular so Phase 2 can plug in. With good progress, the owner may pull backlog items through the [pull gate](docs/research-backlog.md#pulling-backlog-items-into-phase-1-progress-gate). Extensive hyperparameter optimization is deferred.
- **Core first:** data audit, MFCC baseline, CNN, locked evaluation, robustness, failure analysis, API, demo, container, reproducibility and presentation.
- **Stretch:** the evaluated evidence assistant was the only selected stretch. It is a **no-go for this release** (owner, 13 October 2026; [#37](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/37)).
- **Deferred beyond release (Phase 2):** ASVspoof 2021 DF, WaveFake/In-the-Wild, transformer/SSL/LSTM/extra model families, matched-input controls, min t-DCF, cloud, calibration/abstention, agents and MCP. The deviation register is in the architecture plan. These are not release TODOs.

Use GitHub Issues for individual tasks and a board with Backlog, Ready, In progress and Done. Keep one main implementation task in progress at a time. Each issue should include its expected output, dependencies and definition of done. Keep short decision notes when scope changes.

## Phase 1 critical path — tracked in GitHub

This file stays the master plan and status source of truth; GitHub carries only the
work in flight. **Gate milestones:** G0 (closed) … G5 exist as GitHub milestones, plus
*Phase 2: post-graduation research (backlog)*. Issues are minted just-in-time and link
back to the sections here; sections are not mirrored issue-for-issue. Priorities and
rationale: [roadmap](docs/roadmap.md) and the
[13 October architecture review](docs/architecture-review-2026-10-13.md).

**Board:** GitHub Projects board [*AI Voice Deepfake Detector — Delivery*](https://github.com/users/mohamadar77/projects/3)
(columns Todo · In Progress · Done) tracks these issues; one item is In Progress at a time.
The owner picks the review gate for each slice before implementation starts.

| Pri | Issue | Slice | Milestone | TODO § |
| --- | --- | --- | --- | --- |
| ✅ | [#4](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/4) / PR #30 | Option D windowing (`mfcc-v2`): **done**, merged 14 Oct; host 73 tests green on `a987ec9` | G1 | §4 |
| P0 | [#31](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/31) | **Speaker-grouped development reservation + pre-tuning record** (blocks #8) | G1 | §2/§5 |
| ✅ | [#32](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/32) | **Clip-level metrics module**: **done**, merged via PR #41 on 14 Oct (`/code-review` fixes applied; CI and the host's 111-test suite green on `8f1f6c8`) | G1 | §5/§7 |
| P1 | [#7](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/7) | MFCC feature extraction + cache keyed to manifest/config | G1 | §4 |
| P1 | [#8](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/8) | MFCC + LR baseline: development report, bundle, run record | G1 | §5 |
| P1 | [#10](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/10) | CNN timing pilot (decides CPU host vs Kaggle/Colab) | G1 | §4/§6 |
| P2 | [#35](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/35) | Evaluation protocol + freeze record; **rehearse end-to-end on `dev_select`** | G2 | §7 |
| P2 | [#34](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/34) | Robustness + **silence-shortcut** transformation spec | G2 | §8 |
| P2 | [#6](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/6) | EDA, exclusions, **shortcut-only baseline** | G2 | §3 |
| P3 | [#33](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/33) | Compact log-Mel CNN: input policy, clean + augmented | G2 | §6 |
| P3 | [#36](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/36) | Training-fitted PCA / K-means diagnostics | G2 | §8 |
| P3 | [#14](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/14) | **Decision:** sub-1 s duration gate (before freeze) | G2 | §7/§9 |
| P3‖ | [#9](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/9) | Concrete scorer + offline/API parity | G2 lane | §9 |
| P3‖ | [#11](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/11) | CPU container smoke, incl. presentation machine | G2 lane | §10 |
| — | [#5](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/5) | Manifest builder (closed) | G1 | §3 |

**Dependency order:** PR #30 (#4) → #7; freeze #31 membership on the host **before** the #7 extraction, because roles change the manifest hash; #31 + #32 → #8 → #9; #8 → #35 rehearsal;
#10 → #33; #34 + #36 + #33 + thresholds → #35 `FROZEN` → G3 locked scoring. #6, #9 and
#11 run in the G2 parallel lane while the CNN trains unattended.

**Moved to Phase 2:** [#38](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/38) (min t-DCF), [#39](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/39) (matched-input control). **Parked:**
CI/DevEx hardening #15–#29 until core ML is done.

## Milestones

| Target dates (baseline) | Forecast (13 Oct) | Milestone | Completion evidence |
| --- | --- | --- | --- |
| 6–7 October | Closed 11 Oct | G0: feasibility and data contract | Revalidated environment, terms/access, capacity/hours and manifest/split contracts |
| 8–12 October | **16 October** | G1: data and baseline (critical path) | Features, development reservation, metrics, LR development report and bundle, CNN timing pilot. Early-app items moved to the G2 lane. |
| 13–18 October | **22 October** | G2: CNN and protocol freeze | Clean/augmented CNN, train-fit diagnostics, development selection, reserved thresholds and all-candidate freeze |
| 19–25 October | **27 October** | G3: evaluation and complete demo | Locked paired results, uncertainty/failures, model card draft and real-model UI |
| 26 October | 26 October (held) | Feature freeze | No new models, datasets or extensions |
| 27 October–1 November | 1 November (held) | G4: release and slides | Full container/reproduction/performance checks and presentation draft |
| 2–4 November | 4 November (held) | G5: rehearsal and backup | Fresh-start demonstration, backup recording and <=12-minute rehearsal |
| 5 November | 5 November | Tentative presentation | Findings, working demonstration and limitations |

Budget: estimated 110–150 focused hours, approximately 140–190 with 25% contingency, plus unattended compute. Record actual owner availability at G0; the date is not a feasibility guarantee. **13 October:** G1 slipped about 4 days and the 25% contingency is spent. Owner focused hours per week are still unrecorded and must be recorded now. The cut list is in the roadmap/review.

## Start now

The G0 checklist that used to live here is complete (see §1–§2 and `docs/data.md`). Current next actions, in order:

1. ~~Merge PR #30 (option D)~~ (done 14 Oct). Record the owner's focused hours per week.
2. ~~[#32](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/32) metrics module~~ (done 14 Oct, PR #41). [#31](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/31) development reservation (code locally; membership frozen on the host), **before any tuning**.
3. [#7](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/7) features → [#8](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/8) LR development report: the first real evidence.
4. [#10](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/10) CNN pilot: decide between the CPU host and Kaggle/Colab under the external-compute protocol.
5. Start [#35](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/35) early and rehearse the full freeze → score → report pipeline on `dev_select` with the LR bundle.

## 1. Repository and environment

- [x] Create or clone `ai-voice-deepfake-detector` under the intended GitHub account.
- [x] Add the proposal under `docs/` and this checklist at the repository root.
- [x] Write a README covering the problem, current status, scope and planned deliverables.
- [x] Create `src/voice_detector/`, `configs/`, `notebooks/`, `app/`, `tests/`, `reports/`, `docs/` and `.github/workflows/`.
- [x] Create local `data/` and `artifacts/` directories and ignore their generated contents.
- [x] Ignore secrets, `.env`, audio datasets, feature caches, checkpoints and local databases. Commit schemas and small approved summaries instead.
- [x] Select a supported Python version and isolated environment; verify package compatibility on the development machine.
- [x] Record and lock dependencies for NumPy, Pandas, plotting, audio processing, scikit-learn, PyTorch, FastAPI and Streamlit as needed.
- [x] Set configurable dataset and artifact paths; provide an example configuration without credentials.
- [x] Record available CPU, memory, storage and accelerator access.
- [x] Create milestone Issues and a Project board. **2026-10-11:** GitHub milestones G0–G5 created (G0 closed) and the G1 issue set (#5–#11, plus decision #4) opened and linked under "Current slice". **2026-10-13:** Projects board [*AI Voice Deepfake Detector — Delivery* (project #3)](https://github.com/users/mohamadar77/projects/3) added (Todo / In Progress / Done) and linked above.

**Done when:** a fresh checkout can create the environment, import the package and run a small smoke command. The first configuration and setup instructions are committed.

## 2. Data access and protocol

Depends on: repository and environment.

- [x] Obtain ASVspoof 2019 logical access (LA) data and its official countermeasure protocols. **Acquired 2026-10-11** on the DL120 host: LA extracted to `data/raw/LA`; ODC-By 1.0 terms recorded; the three CM protocols audit to the published counts. See [`docs/data.md`](docs/data.md).
- [x] Record the initial G0 environment/storage check in `docs/data.md`: no local audio, about 520 GiB available, shell Python 3.14.6, uv cache rerouted successfully.
- [x] Resolve the G0 runtime blocker: intended Python 3.12/ARM64 environment versus current x86_64 PyTorch wheel resolution. Do not change the lockfile without a recorded platform decision. **Resolved 2026-10-11:** reserved a Linux `x86_64` host (HPE DL120 Gen10, Fedora 44, glibc 2.43, Xeon Gold 6152/44 cores, 30 GB RAM, CPU-only); `uv sync --frozen --all-groups`, smoke, ruff and pytest all pass there. No lockfile change. See [`docs/environment.md`](docs/environment.md).
- [x] Record source URLs, release identifiers, terms, archive checksums and download instructions in `docs/data.md`. **Done 2026-10-11:** Edinburgh DataShare handle, ODC-By 1.0 license, the Todisco et al. 2019 citation, and SHA-256 of `dataset.zip`/`LA.zip` recorded. The exact mirror download URL is pending owner confirmation; the checksums are the authoritative anchor.
- [x] Verify storage requirements, extraction success and missing files. **Done 2026-10-11:** host capacity verified (preflight / [`docs/environment.md`](docs/environment.md)); the LA archive extracted cleanly with the full documented directory structure. Per-clip file existence against the manifest is checked by the manifest builder (§3).
- [ ] Preserve official training, development and evaluation boundaries.
- [x] Define the label mapping explicitly: bona fide human speech versus spoofed/synthetic speech. **Done:** `label` ∈ {`bonafide`, `spoof`} is mapped directly from the protocol KEY column and enforced by `src/voice_detector/protocol.py`; documented in [`docs/data.md`](docs/data.md).
- [ ] Reserve a speaker grouped part of development data for thresholds and optional calibration before tuning starts ([#31](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/31)).
- [ ] Document which development portion supports model selection and which supports operating thresholds ([#31](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/31)).
- [ ] Record primary-dataset source-group provenance where available and disclose missing mappings. The manifest builder records no 2019 LA source grouping, so the bootstrap falls back to `speaker_id` (#35). Cross-collection work is in Phase 2.

**Done when:** data sources and permitted uses are documented; partitions and label meanings are unambiguous; the primary dataset can be read locally.

## 3. Manifest, cleaning and EDA

Depends on: data access.

- [x] Draft the canonical manifest fields and provenance rules in `docs/data.md`.
- [x] Add dataset-free manifest row/duplicate validation in `src/voice_detector/manifest.py` with focused tests.
- [x] Parse the official CM protocol into structured entries (speaker, clip, attack, label, split) with a metadata-only audit and fixtures in `src/voice_detector/protocol.py`, exposed as `voice-detector protocol-audit`. Reads no audio; runs before the dataset is present.
- [x] Implement a manifest builder with clip ID, path, source ID, speaker ID, attack ID, label, split, duration, sample rate, checksum and provenance where available (decodes audio; extends the protocol entries above). **Done:** #5 / PR #13; host run 2026-10-13 matched the published counts (121,461 clips; manifest hash `b172e50c…`).
- [x] Export a versioned CSV manifest and SQLite catalog with a documented schema. **Done:** #5 (CSV + SQLite + `catalog_meta`).
- [ ] Verify that manifest IDs and paths are unique and referenced files exist.
- [ ] Audit exact duplicates, speaker overlap across official partitions and mapped original source overlap.
- [ ] Define the response to each overlap: preserve official protocol reporting, add a clearly labeled clean subset where appropriate, and disclose unresolved limitations.
- [ ] Keep all windows, transformed copies and derived features assigned to their original recording and partition.
- [ ] Inspect class counts, attack counts, duration distributions, sample rates, channels, silence and clipping ([#6](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/6)).
- [ ] Audit whether duration, padding, silence or file processing differs systematically between classes ([#6](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/6)). Measure per-clip leading/trailing non-speech with the same frozen rule as the [#34](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/34) trim condition; use train/development only.
- [ ] Shortcut-only baseline: LR on duration + leading/trailing silence only, fit on train and reported as an EER on `dev_select` ([#6](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/6)). Report it next to the real models.
- [ ] Create SQL queries for partition counts, label balance and experiment metadata.
- [ ] Save EDA figures and an exclusion report with reasons.
- [ ] Freeze a documented stratified subset before inspecting model scores if full processing is infeasible.

**Done when:** manifest checks pass, exclusions are traceable and EDA identifies the main data limitations. Record the manifest hash used for experiments.

## 4. Shared audio preprocessing

Depends on: audited manifest.

- [ ] Decode consistently to mono 16 kHz and validate decoded audio.
- [ ] Implement four second training windows with a documented crop and padding policy.
- [ ] Implement deterministic evaluation windows and aggregation to one score per original clip.
- [ ] Extract MFCC summary features for the baseline.
- [ ] Generate log Mel spectrograms for the CNN; record feature settings in configuration. Decide the CNN short-clip/padding policy before training: no unjustified zero-padding, because it would bring back the shortcut that option D removed ([#33](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/33)).
- [ ] Fit feature scalers and normalization statistics on training data only.
- [ ] Cache features using identifiers tied to the manifest and preprocessing configuration.
- [ ] Check feature shapes, finite values, deterministic evaluation behavior and train/serve consistency.
- [ ] Record preparation time, cache size and a small training pilot runtime.

**Done when:** the baseline and CNN consume the intended representations of the same permitted clips, and the serving path reproduces evaluation preprocessing.

## 5. MFCC baseline

Depends on: preprocessing.

- [ ] Train Logistic Regression using training features only.
- [ ] Use a small declared development tuning budget and document class imbalance handling. Selection metric, tie-break, LR grid, aggregation and threshold rule are frozen in the pre-tuning record ([#31](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/31)). Tune on `dev_select` only.
- [ ] Save the fitted scaler, classifier, configuration and development predictions.
- [ ] Compute development macro F1, balanced accuracy, ROC AUC, EER, per-class precision/recall and a confusion matrix with the shared metrics module ([#32](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/32)). Report precision with the prevalence it was measured at.
- [ ] Select the operating threshold on `dev_threshold` only, after the configuration is fixed. Report both human false alarms and synthetic speech detection.
- [ ] Record seed, dependency versions, Git commit, manifest hash and runtime as a JSON run record, and insert it into a SQLite `runs` table.
- [ ] Add a command that reproduces baseline training and development evaluation.

**Done when:** a reproducible reference model and development report exist. Final evaluation remains unused for tuning.

## 6. CNN model

Depends on: working baseline and compute pilot.

- [ ] Implement a compact spectrogram CNN with a small Multilayer Perceptron (MLP) classification head ([#33](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/33)).
- [ ] Configure one compact architecture, loss, weighting, optimizer, early stopping and matched clean/augmented runs; record actual epochs/steps/clip exposure.
- [ ] Verify the training loop on a small subset before a full run.
- [ ] Train the main model using a declared development tuning budget.
- [ ] Save learning curves, checkpoint, configuration and development predictions.
- [ ] Compare baseline and CNN on the same development protocol.
- [ ] Run a second seed if the compute budget permits; otherwise document the unmeasured training variability.
- [ ] If the #10 pilot exceeds the CPU caps, train on owner-run Kaggle/Colab under the external-compute protocol (architecture plan). Training/development data only; development predictions recomputed on the host for parity before freezing.
- [ ] Select the deployed model using development results and practical resource requirements.

**Done when:** the CNN is reproducible and its development performance and inference cost are compared with the baseline. A stronger baseline is a valid outcome.

## 7. Freeze and run final evaluation

Depends on: model development complete.

- [ ] Write `docs/evaluation-protocol.md` before final test predictions are inspected ([#35](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/35)).
- [ ] Rehearse the whole freeze → score → metrics → report pipeline on `dev_select` with the LR bundle before the real freeze. No evaluation data is used.
- [ ] Freeze LR, clean CNN and augmented CNN bundles (plus every declared seed), score direction, thresholds, membership, conditions, metrics, slices and the report template/outline before inspection.
- [ ] Implement DRAFT/FROZEN/SCORED/REPORTED records, hash validation and refusal of silent result overwrite; a YAML Boolean is insufficient.
- [ ] Verify no final test data was used to fit scalers, select checkpoints, calibrate outputs or choose thresholds.
- [ ] Generate clip level predictions on the locked ASVspoof 2019 LA evaluation set.
- [ ] Report EER, macro F1, balanced accuracy, ROC AUC, class counts and confusion matrices.
- [ ] Report human false positive rate and synthetic detection rate at the development chosen threshold. Treat the proposed 5% development false alarm target as a target, not a test guarantee.
- [ ] Report unfamiliar attack detection rates/counts; predeclare and disclose a bona fide comparison pool when calculating attack-specific EER/AUC.
- [ ] Report **known versus unseen attacks** as separate slices:
  - development A01–A06: known algorithms, unseen speakers;
  - evaluation A16/A19: these reuse the A04/A06 algorithms;
  - evaluation A07–A15, A17, A18: unseen algorithms; this is the headline unseen result, never pooled with A16/A19.
- [ ] Predeclare duration bins <1, 1–<2, 2–<4, 4–10 and >10 s with per-class counts. Handle sub-1 s clips per #14.
- [ ] Compute 95% bootstrap intervals grouped by original source recording. Where no source mapping exists, group by `speaker_id` (67 evaluation speakers). Paired differences use the same resamples. Disclose the speaker dependence and the small `dev_threshold` speaker count.
- [ ] Save prediction files and structured metrics; document any deviations from the original protocol.
- [ ] Label evaluation on a subset and externally pretrained models accurately; do not claim an official leaderboard result.
- [ ] Map each proposal research question to its evidence, or to an explicit "not run": RQ2's second domain and RQ3 are deferred (deviation register).

**Done when:** results trace to frozen inputs and model versions, and the report answers the primary model comparison and unfamiliar attack questions. Final test findings do not feed further tuning.

## 8. Robustness and error analysis

Depends on: an evaluation pipeline; specify experiments before final scores.

- [ ] Define one training augmentation recipe and compare it with clean training under the same protocol. Declare any overlap with the test conditions; label overlapping gains as matched-condition.
- [ ] Define clean audio, one compression setting and two noise levels for robustness tests ([#34](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/34)). Pin or record the codec tool version, which sits outside `uv.lock`. Use generated noise with per-clip seeds; no new noise dataset.
- [ ] Add the **silence-trim shortcut diagnostic**: a paired, evaluation-only condition with a frozen trim rule and a minimum-length fallback, scored for every frozen bundle and reported next to the robustness ablation ([#34](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/34)).
- [ ] Apply each test transformation consistently to both classes and the same original clips.
- [ ] Keep transformation parameters and random seeds in configuration.
- [ ] Report paired changes in detection errors and false alarms under each condition.
- [ ] Review false positives and false negatives by attack (known/unseen), codec, duration, silence and source.
- [ ] Fit PCA and K-means on training features; cap the training-only K search and silhouette sample; project/assign evaluation examples without refitting ([#36](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/36)).
- [ ] Save diagnostic plots and a small set of licensed or consented failure examples.
- [ ] Describe dataset shortcuts, failure patterns, what improved and what remained unresolved.
- [ ] The proposal's "failure-analysis dashboard" is satisfied by this failure report plus the Streamlit evaluation-summary/evidence view (§9); no separate dashboard application is required.

**Done when:** the report contains a controlled robustness comparison and evidence supported failure analysis. Feature plots are presented as diagnostics.

## 9. API and demonstration interface

Start a thin vertical slice after the baseline; finish after model selection.

- [ ] Implement a shared clip scorer with trusted versioned bundle and fitted-statistics/config hashes; check offline/API and CPU-container parity.
- [ ] Build POST /predict, GET /health and GET /ready with typed responses; configure the UI API URL for localhost versus container service hostname.
- [ ] Validate WAV/FLAC, at most 20 MB and target 1–30-second decoded duration; evaluate duration coverage or explicitly narrow supported claims. Bound decoding and concurrency.
- [ ] Handle corrupt files, unsupported formats, empty audio, silence and unavailable models clearly.
- [ ] Return predicted class, model score, model version and processing time.
- [ ] Build a Streamlit upload interface and spectrogram view. Polish is cut to score, spectrogram, measured-summary table and limitations (owner, 13 October).
- [ ] Show the chosen model's measured evaluation summary and limitations.
- [ ] Label uncalibrated outputs as scores; add probability or inconclusive behavior only after the corresponding optional evaluation is complete.
- [ ] Delete temporary audio after success, errors and cancellation; clear UI-held buffers and verify cleanup.
- [ ] Choose a small reproducible set of permitted demo recordings: the owner's consented own-voice clips, plus synthetic clips the owner generates with a modern TTS/voice-clone tool whose terms allow it. Record the tool, version and terms. Label the set illustrative with n reported; a miss is shown as a measured limitation.
- [ ] Test the full upload → preprocessing → inference → display flow.

**Done when:** valid inputs produce an assessment and informative errors handle invalid inputs. The demonstrated model matches the evaluated version.

## 10. Reliability, packaging and documentation

Depends on: a working application.

- [ ] Log request ID, model version, audio duration, status and latency without raw recordings or secrets.
- [ ] Long host jobs (extraction, CNN training, locked scoring) run detached (`tmux`/`systemd-run --user`) with file logs and are resumable (per-epoch checkpoints with optimizer/RNG state; atomic cache writes). API/UI bind to localhost and are reached via an SSH tunnel.
- [ ] Measure p50/p95 latency, failure rate and memory on named hardware using a declared input mix and request count.
- [ ] Add focused tests for feature shapes, split leakage, inference preprocessing and API error handling.
- [ ] Configure lightweight GitHub Actions checks that do not require downloading the full dataset or training models.
- [ ] Package API and UI in containers and document one startup command.
- [ ] Verify a clean local container startup and health check, in the presentation configuration on the DL120 host, which is the presentation machine (owner, 13 Oct), with image size and cold start recorded. Rehearse the venue access path (on-site, or an SSH tunnel to localhost-bound services) ([#11](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/11)).
- [ ] Write the model card: intended use, data, language scope, metrics, known failures and limitations.
- [ ] Document training, evaluation, artifacts, result reproduction and temporary audio handling.
- [ ] Provide artifact retrieval instructions and checksums where redistribution is permitted.
- [ ] Archive frozen bundles, freeze records, per-clip predictions, run records and the manifest, with SHA-256 checksums, in a retained location outside Git. Phase 2 comparisons depend on them.
- [ ] Verify fresh-checkout reproduction of data preparation, baseline training, CNN training and the report (not baseline only).

**Done when:** another developer can follow the setup and reproduction instructions, and the application has measured reliability and a model card.

## 11. Selected stretch: evidence assistant

**Decided 13 October 2026: no-go for this release** (owner; [#37](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/37) closed). The items below are kept as the historical contract and course-coverage context; they are not release TODOs. RAG/MCP are off the Phase 2 research track (see the [research backlog](docs/research-backlog.md)).

- [ ] Record the explicit go/no-go decision and remaining time.
- [ ] Curate approved documentation/model card/frozen results; version document IDs, chunks, enriched metadata and index.
- [ ] Compare TF-IDF/BM25 with separate development and held-out questions; semantic retrieval is optional.
- [ ] Implement cited, structured and schema-validated generated answers only if model access, cost and evaluation fit; templated search alone is not RAG.
- [ ] Evaluate retrieval recall, citation/numeric correctness, unsupported claims, abstention, latency and usage, including unanswerable/injection cases.
- [ ] Keep audio decisions and frozen thresholds in the detector. Never pass raw uploads into the evidence workflow.

## 12. Deferred work

ASVspoof 2021 DF/WaveFake, frozen audio transformers, LSTM/extra classifiers, cloud deployment, calibration/abstention, agents and MCP are future work, not acceptance requirements or automatic next tasks. The course appendix explains their relevance and reasons for deferral. **Phase 2:** the prioritized post-graduation research backlog, and what Phase 1 must preserve for it, is in [docs/research-backlog.md](docs/research-backlog.md). The deviation register is in the architecture plan.

## 13. Presentation and release

- [ ] Summarize the problem, practical motivation and relevant prior approaches. Cite the published ASVspoof 2019 B01 (CQCC-GMM) and B02 (LFCC-GMM) evaluation results as context only, never as reproduced results.
- [ ] Present actual data sizes, properties, cleaning and leakage controls.
- [ ] Explain baseline and CNN design choices and experiment controls.
- [ ] Show model comparisons, known-versus-unseen attack results, the silence/duration shortcut finding, robustness findings, failure examples and limitations.
- [ ] Demonstrate a valid recording and an invalid input; include an inconclusive example only if that feature was evaluated.
- [ ] Describe lessons learned and future work based on measured results.
- [ ] Prepare a presentation of at most 12 minutes; show ideas, plots and behavior rather than code.
- [ ] Allocate approximately 1 minute motivation, 2 data, 2 methods, 3 results, 2 demo, 1 conclusions and 1 buffer.
- [ ] Freeze features and organize release artifacts.
- [ ] Test demonstration startup and prepared samples on the presentation machine.
- [ ] Prepare a short backup recording, saved plots and a local fallback. The demo runs remotely over an SSH tunnel to the host, so the local fallback is the presentation image, frozen bundle and demo clips loaded on the laptop's Docker (`docker save`/`load`, `linux/amd64`), verified offline with score parity in G4.
- [ ] Rehearse and time the presentation.
- [ ] Tag the final release and record its commit, model version and manifest hash.

## Final acceptance checklist

- [ ] Real data and cleaning are documented and auditable.
- [ ] Baseline and CNN comparisons are reproducible.
- [ ] Locked unfamiliar attack evaluation and a controlled robustness study are complete.
- [ ] Missing transfer or optional experiments are explicitly disclosed.
- [ ] Failure analysis supports the conclusions; measured results replace assumptions.
- [ ] The application runs and handles valid and invalid inputs.
- [ ] Reliability and latency measurements identify hardware and test conditions.
- [ ] Model card, setup guide, configs and result reports are available.
- [ ] Only completed and evaluated features are described as implemented.
- [ ] The presentation and backup demonstration are ready.

## Decisions and blockers

| Date | Decision or blocker | Impact | Next action |
| --- | --- | --- | --- |
| 6 October 2026 | Lead/Meitner architecture agreement | One core, early API/container path, all-candidate freeze, zero-hour stretch | Execute G0; see architecture addendum |
| 6 October 2026 | Requested existing reviewer unavailable through agent and CLI APIs | No three-way approval claimed | Incorporate feedback if the session becomes reachable |
| 6 October 2026 | Session handoff recorded | Resume from G0 environment/data blockers and the architecture addendum | Read `docs/architecture-and-delivery-plan.md` handoff before continuing |
| 11 October 2026 | Final proposal selected and signed off | Original `...Proposal.docx` frozen as sole source of truth (SHA-256 `b16f5124…`); `_Revised.docx` non-authoritative | All future changes recorded in Markdown only; see architecture-plan sign-off |
| 11 October 2026 | Windowing tail/padding policy frozen — **option D**: pool MFCCs over real (non-padded) samples only; fold sub-frame tails; `<4 s` clip = one window over real audio | Removes zero-pad contamination of pooled MFCC stats (a potential class shortcut); unblocks [#7](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/7)/[#8](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/8); bumps `mfcc-v1`→`mfcc-v2` | Implement in #7 with tests (see [#4](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/4)); CNN padding decided separately at G2 |
| 13 October 2026 | Option D **implemented** ahead of #7 in its own PR (closes [#4](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/4)): pure `window_spans` policy in `preprocessing.py`, `clip_feature_matrix` pools real samples only, `window_waveform` retired, `mfcc-v1`→`mfcc-v2` (code constant `PREPROCESSING_VERSION`, config asserted to match) | Keeps the value-changing policy on a small, pure, exhaustively-tested surface and off the feature-cache I/O review; feature values change under v2 | Feature cache ([#7](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/7)) builds on the frozen v2 next |
| 13 October 2026 | Architecture review ([review](docs/architecture-review-2026-10-13.md)): the development reservation was unscheduled before LR tuning; the evaluation harness had no slot; G1 slipped about 4 days with contingency spent; owner hours unrecorded | G1 narrowed to its critical path; new [#31](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/31)–[#36](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/36); #6/#9/#11 moved to the G2 parallel lane; milestones re-dated (G1 16 Oct, G2 22 Oct, G3 27 Oct); deviation register added to the architecture plan | Execute the Phase 1 critical path ([roadmap](docs/roadmap.md)) |
| 13 October 2026 | Owner direction: **two-phase strategy**. Phase 1 = graduation; Phase 2 = separate research backlog. S1 RAG **no-go**; #38/#39 → Phase 2; UI polish cut; no advanced/pretrained models in Phase 1; methodology claims only with artifacts; work issue by issue | Scope reduced to the core plus the silence/duration shortcut and known/unseen attack analysis; owner-run Kaggle/Colab GPU is available under the external-compute protocol | Owner picks the review gate per slice; start P0 ([#32](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/32), [#31](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/31)) |
| 13 October 2026 | Correction: evaluation attacks A16/A19 reuse the A04/A06 algorithms; they are not unseen | The headline "unseen" result excludes them; they become a known-algorithm control on unseen speakers | Encode the three slices in [#35](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/35); `docs/data.md` corrected |

## First actions

- [x] Revalidate runtime and hardware/storage (G0; DL120 host, `docs/environment.md`).
- [ ] Record the owner's focused hours/week. Still open; this blocks forecast validation.
- [x] Verify ASVspoof 2019 LA access/terms and write the data contract (`docs/data.md`).
- [x] Implement protocol fixtures, manifest schema and metadata-only audit (`protocol.py`, `manifest.py`).
- [ ] Implement/test deterministic preprocessing and pilot capacity.
- [ ] Train/record the baseline development result when the implementation task authorizes training; connect the same bundle to an early API/UI path.

## Project references

- [Final frozen project proposal (source of truth)](docs/AI_Voice_Deepfake_Detector_Final_Project_Proposal.docx)
- [Current architecture, discussion and delivery decisions](docs/architecture-and-delivery-plan.md)
- [Implementation roadmap and Phase 1 critical path](docs/roadmap.md)
- [Architecture review, 13 October 2026](docs/architecture-review-2026-10-13.md)
- [Post-graduation research backlog (Phase 2)](docs/research-backlog.md)
- [Project management](docs/project-management.md)
- [Course coverage appendix](docs/proposal-review-and-course-coverage.md)
