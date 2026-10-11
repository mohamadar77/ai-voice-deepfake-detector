# AI Voice Deepfake Detector TODO Plan

Owner: Mohamad Abo-Ras  
Created: 4 October 2026  
Target presentation: 5 November 2026, subject to the course schedule

Build and evaluate a reproducible detector for human versus synthesized or voice converted speech. Compare an MFCC feature baseline with a Convolutional Neural Network (CNN), study unfamiliar attacks and audio distortions, and demonstrate a working application.

This checklist follows the original proposal and the [6 October architecture addendum](docs/architecture-and-delivery-plan.md). Existing checked setup items are historical records; runtime/environment readiness must be revalidated at G0. Check an item only when its output exists and its completion criteria have been met. Record experiment findings even when a model fails to improve.

## How to use this plan

- **Proposal:** `docs/AI_Voice_Deepfake_Detector_Final_Project_Proposal.docx` is the final, frozen, immutable proposal and source of truth for scope/commitments/acceptance (SHA-256 `b16f5124…`; signed off 2026-10-11 — see the sign-off in `docs/architecture-and-delivery-plan.md`). `_Revised.docx` is a non-selected historical candidate. `docs/architecture-and-delivery-plan.md` is the current scope, contracts, budget and schedule addendum, subordinate to the frozen proposal.
- **Execution:** this file is the source of truth for implementation status. Check an item only when its output exists and its completion criteria are met.
- **Roadmap:** `docs/roadmap.md` contains the current vertical slice, stop/go gates and the next concrete commands.
- **Core first:** data audit, MFCC baseline, CNN, locked evaluation, robustness, failure analysis, API, demo, container, reproducibility and presentation.
- **Only selected stretch:** the evaluated evidence assistant, with zero committed hours and a strict early-core-completion gate before 26 October.
- **Deferred beyond release:** ASVspoof 2021 DF, WaveFake, transformer/LSTM/extra model families, cloud, calibration/abstention, agents and MCP. Their educational coverage is conceptual; they are not release TODOs.

Use GitHub Issues for individual tasks and a board with Backlog, Ready, In progress and Done. Keep one main implementation task in progress at a time. Each issue should include its expected output, dependencies and definition of done. Keep short decision notes when scope changes.

## Current slice (G1) — tracked in GitHub

This file stays the master plan and status source of truth; GitHub carries only the
work in flight. **Gate milestones:** G0 (closed) … G5 exist as GitHub milestones. The
active-slice issues below are minted just-in-time and link back to the sections here;
sections are not mirrored issue-for-issue.

| Issue | Slice | TODO § |
| --- | --- | --- |
| [#5](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/5) | Manifest builder over LA audio + versioned CSV/SQLite export | §3 |
| [#6](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/6) | EDA and exclusion report on the LA manifest | §3 |
| [#7](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/7) | MFCC feature extraction + cache keyed to manifest/config | §4 |
| [#8](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/8) | MFCC + Logistic Regression baseline bundle | §5 |
| [#9](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/9) | Concrete scorer behind serving API + offline/API parity | §9 |
| [#10](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/10) | CNN timing pilot on CPU (no full training) | §4/§6 |
| [#11](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/11) | CPU Docker container smoke for API + UI | §10 |
| [#4](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/4) | **Decision (blocker):** freeze windowing tail/padding policy | §4 |

**Dependency order:** #5 → (#7 after #4) → #8 → #9; #6 after #5; #10 after feature
settings; #11 after #9. #4 (`needs-decision`) blocks #7 and #8 — freeze it before the
MFCC features lock.

## Milestones

| Target dates | Milestone | Completion evidence |
| --- | --- | --- |
| 6–7 October | G0: feasibility and data contract | Revalidated environment, terms/access, capacity/hours and manifest/split contracts |
| 8–12 October | G1: data, baseline and early application | Audit/features, LR bundle, HTTP upload path, cleanup, CNN timing pilot and CPU container smoke |
| 13–18 October | G2: CNN and protocol freeze | Clean/augmented CNN, train-fit diagnostics, development selection, reserved thresholds and all-candidate freeze |
| 19–25 October | G3: evaluation and complete demo | Locked paired results, uncertainty/failures, model card draft and real-model UI |
| 26 October | Feature freeze | No new models, datasets or extensions |
| 27 October–1 November | G4: release and slides | Full container/reproduction/performance checks and presentation draft |
| 2–4 November | G5: rehearsal and backup | Fresh-start demonstration, backup recording and <=12-minute rehearsal |
| 5 November | Tentative presentation | Findings, working demonstration and limitations |

Budget: estimated 110–150 focused hours, approximately 140–190 with 25% contingency, plus unattended compute. Record actual owner availability at G0; the date is not a feasibility guarantee.

## Start now

The repository has setup scaffolding and historical setup records. Revalidate the runtime before the next implementation slice:

1. Revalidate Python 3.12/uv and record current hours/compute/storage; confirm ASVspoof 2019 LA access, terms and local storage without downloading audio into Git.
2. Record the source and protocol details in `docs/data.md` and finalize the manifest schema.
3. Implement a metadata-only manifest audit and run it on the official protocol files.
4. Run a tiny, declared preprocessing and timing pilot; do not inspect evaluation labels or train a model yet.
5. Decide whether the full 2019 LA run fits the available disk, memory and compute. Record the decision and any subset rule before model experiments.

The first coding target is therefore the data contract and manifest audit, not the CNN. The immediate stop/go gate is: official partitions are readable, labels are unambiguous, source overlap checks are defined, and a four-second preprocessing pilot completes deterministically.

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
- [x] Create milestone Issues and a Project board. **2026-10-11:** GitHub milestones G0–G5 created (G0 closed) and the G1 issue set (#5–#11, plus decision #4) opened and linked under "Current slice". A Projects *board* is still to be added — the current `gh` token lacks the `project` scope; add it in the GitHub UI, or run `gh auth refresh -s project` and re-attempt.

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
- [ ] Reserve a speaker grouped part of development data for thresholds and optional calibration before tuning starts.
- [ ] Document which development portion supports model selection and which supports operating thresholds.
- [ ] Record primary-dataset source-group provenance where available and disclose missing mappings. Cross-collection work is deferred.

**Done when:** data sources and permitted uses are documented; partitions and label meanings are unambiguous; the primary dataset can be read locally.

## 3. Manifest, cleaning and EDA

Depends on: data access.

- [x] Draft the canonical manifest fields and provenance rules in `docs/data.md`.
- [x] Add dataset-free manifest row/duplicate validation in `src/voice_detector/manifest.py` with focused tests.
- [x] Parse the official CM protocol into structured entries (speaker, clip, attack, label, split) with a metadata-only audit and fixtures in `src/voice_detector/protocol.py`, exposed as `voice-detector protocol-audit`. Reads no audio; runs before the dataset is present.
- [ ] Implement a manifest builder with clip ID, path, source ID, speaker ID, attack ID, label, split, duration, sample rate, checksum and provenance where available (decodes audio; extends the protocol entries above).
- [ ] Export a versioned CSV manifest and SQLite catalog with a documented schema.
- [ ] Verify that manifest IDs and paths are unique and referenced files exist.
- [ ] Audit exact duplicates, speaker overlap across official partitions and mapped original source overlap.
- [ ] Define the response to each overlap: preserve official protocol reporting, add a clearly labeled clean subset where appropriate, and disclose unresolved limitations.
- [ ] Keep all windows, transformed copies and derived features assigned to their original recording and partition.
- [ ] Inspect class counts, attack counts, duration distributions, sample rates, channels, silence and clipping.
- [ ] Audit whether duration, padding, silence or file processing differs systematically between classes.
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
- [ ] Generate log Mel spectrograms for the CNN; record feature settings in configuration.
- [ ] Fit feature scalers and normalization statistics on training data only.
- [ ] Cache features using identifiers tied to the manifest and preprocessing configuration.
- [ ] Check feature shapes, finite values, deterministic evaluation behavior and train/serve consistency.
- [ ] Record preparation time, cache size and a small training pilot runtime.

**Done when:** the baseline and CNN consume the intended representations of the same permitted clips, and the serving path reproduces evaluation preprocessing.

## 5. MFCC baseline

Depends on: preprocessing.

- [ ] Train Logistic Regression using training features only.
- [ ] Use a small declared development tuning budget and document class imbalance handling.
- [ ] Save the fitted scaler, classifier, configuration and development predictions.
- [ ] Compute development macro F1, balanced accuracy, ROC AUC, EER and a confusion matrix.
- [ ] Select a development operating threshold and report both human false alarms and synthetic speech detection.
- [ ] Record seed, dependency versions, Git commit, manifest hash and runtime.
- [ ] Add a command that reproduces baseline training and development evaluation.

**Done when:** a reproducible reference model and development report exist. Final evaluation remains unused for tuning.

## 6. CNN model

Depends on: working baseline and compute pilot.

- [ ] Implement a compact spectrogram CNN with a small Multilayer Perceptron (MLP) classification head.
- [ ] Configure one compact architecture, loss, weighting, optimizer, early stopping and matched clean/augmented runs; record actual epochs/steps/clip exposure.
- [ ] Verify the training loop on a small subset before a full run.
- [ ] Train the main model using a declared development tuning budget.
- [ ] Save learning curves, checkpoint, configuration and development predictions.
- [ ] Compare baseline and CNN on the same development protocol.
- [ ] Run a second seed if the compute budget permits; otherwise document the unmeasured training variability.
- [ ] Select the deployed model using development results and practical resource requirements.

**Done when:** the CNN is reproducible and its development performance and inference cost are compared with the baseline. A stronger baseline is a valid outcome.

## 7. Freeze and run final evaluation

Depends on: model development complete.

- [ ] Write `docs/evaluation-protocol.md` before final test predictions are inspected.
- [ ] Freeze LR, clean CNN and augmented CNN bundles (plus every declared seed), score direction, thresholds, membership, conditions, metrics, slices and the report template/outline before inspection.
- [ ] Implement DRAFT/FROZEN/SCORED/REPORTED records, hash validation and refusal of silent result overwrite; a YAML Boolean is insufficient.
- [ ] Verify no final test data was used to fit scalers, select checkpoints, calibrate outputs or choose thresholds.
- [ ] Generate clip level predictions on the locked ASVspoof 2019 LA evaluation set.
- [ ] Report EER, macro F1, balanced accuracy, ROC AUC, class counts and confusion matrices.
- [ ] Report human false positive rate and synthetic detection rate at the development chosen threshold. Treat the proposed 5% development false alarm target as a target, not a test guarantee.
- [ ] Report unfamiliar attack detection rates/counts; predeclare and disclose a bona fide comparison pool when calculating attack-specific EER/AUC.
- [ ] Compute 95% bootstrap intervals grouped by original source recording; disclose limits from speaker dependence where relevant.
- [ ] Save prediction files and structured metrics; document any deviations from the original protocol.
- [ ] Label evaluation on a subset and externally pretrained models accurately; do not claim an official leaderboard result.

**Done when:** results trace to frozen inputs and model versions, and the report answers the primary model comparison and unfamiliar attack questions. Final test findings do not feed further tuning.

## 8. Robustness and error analysis

Depends on: an evaluation pipeline; specify experiments before final scores.

- [ ] Define one training augmentation recipe and compare it with clean training under the same protocol.
- [ ] Define clean audio, one compression setting and two noise levels for robustness tests.
- [ ] Apply each test transformation consistently to both classes and the same original clips.
- [ ] Keep transformation parameters and random seeds in configuration.
- [ ] Report paired changes in detection errors and false alarms under each condition.
- [ ] Review false positives and false negatives by attack, codec, duration and source.
- [ ] Fit PCA and K-means on training features; cap the training-only K search and silhouette sample; project/assign evaluation examples without refitting.
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
- [ ] Build a Streamlit upload interface and spectrogram view.
- [ ] Show the chosen model's measured evaluation summary and limitations.
- [ ] Label uncalibrated outputs as scores; add probability or inconclusive behavior only after the corresponding optional evaluation is complete.
- [ ] Delete temporary audio after success, errors and cancellation; clear UI-held buffers and verify cleanup.
- [ ] Choose a small reproducible set of permitted demo recordings.
- [ ] Test the full upload → preprocessing → inference → display flow.

**Done when:** valid inputs produce an assessment and informative errors handle invalid inputs. The demonstrated model matches the evaluated version.

## 10. Reliability, packaging and documentation

Depends on: a working application.

- [ ] Log request ID, model version, audio duration, status and latency without raw recordings or secrets.
- [ ] Measure p50/p95 latency, failure rate and memory on named hardware using a declared input mix and request count.
- [ ] Add focused tests for feature shapes, split leakage, inference preprocessing and API error handling.
- [ ] Configure lightweight GitHub Actions checks that do not require downloading the full dataset or training models.
- [ ] Package API and UI in containers and document one startup command.
- [ ] Verify a clean local container startup and health check.
- [ ] Write the model card: intended use, data, language scope, metrics, known failures and limitations.
- [ ] Document training, evaluation, artifacts, result reproduction and temporary audio handling.
- [ ] Provide artifact retrieval instructions and checksums where redistribution is permitted.
- [ ] Verify fresh-checkout reproduction of data preparation, baseline training, CNN training and the report (not baseline only).

**Done when:** another developer can follow the setup and reproduction instructions, and the application has measured reliability and a model card.

## 11. Selected stretch: evidence assistant

Zero committed hours. Start only after full core acceptance, including Docker/reliability, reproduction and model-card evidence, early enough to finish a provisional 16–24-hour pilot before 26 October while preserving release contingency. The current schedule likely leaves it deferred.

- [ ] Record the explicit go/no-go decision and remaining time.
- [ ] Curate approved documentation/model card/frozen results; version document IDs, chunks, enriched metadata and index.
- [ ] Compare TF-IDF/BM25 with separate development and held-out questions; semantic retrieval is optional.
- [ ] Implement cited, structured and schema-validated generated answers only if model access, cost and evaluation fit; templated search alone is not RAG.
- [ ] Evaluate retrieval recall, citation/numeric correctness, unsupported claims, abstention, latency and usage, including unanswerable/injection cases.
- [ ] Keep audio decisions and frozen thresholds in the detector. Never pass raw uploads into the evidence workflow.

## 12. Deferred work

ASVspoof 2021 DF/WaveFake, frozen audio transformers, LSTM/extra classifiers, cloud deployment, calibration/abstention, agents and MCP are future work, not acceptance requirements or automatic next tasks. The course appendix explains their relevance and reasons for deferral.

## 13. Presentation and release

- [ ] Summarize the problem, practical motivation and relevant prior approaches.
- [ ] Present actual data sizes, properties, cleaning and leakage controls.
- [ ] Explain baseline and CNN design choices and experiment controls.
- [ ] Show model comparisons, robustness findings, failure examples and limitations.
- [ ] Demonstrate a valid recording and an invalid input; include an inconclusive example only if that feature was evaluated.
- [ ] Describe lessons learned and future work based on measured results.
- [ ] Prepare a presentation of at most 12 minutes; show ideas, plots and behavior rather than code.
- [ ] Allocate approximately 1 minute motivation, 2 data, 2 methods, 3 results, 2 demo, 1 conclusions and 1 buffer.
- [ ] Freeze features and organize release artifacts.
- [ ] Test demonstration startup and prepared samples on the presentation machine.
- [ ] Prepare a short backup recording, saved plots and a local fallback.
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

## First actions

- [ ] Revalidate runtime, hardware/storage and focused hours/week (G0).
- [ ] Verify ASVspoof 2019 LA access/terms and write the data contract.
- [ ] Implement protocol fixtures, manifest schema and metadata-only audit.
- [ ] Implement/test deterministic preprocessing and pilot capacity.
- [ ] Train/record the baseline development result when the implementation task authorizes training; connect the same bundle to an early API/UI path.

## Project references

- [Final frozen project proposal (source of truth)](docs/AI_Voice_Deepfake_Detector_Final_Project_Proposal.docx)
- [Current architecture, discussion and delivery decisions](docs/architecture-and-delivery-plan.md)
- [Implementation roadmap](docs/roadmap.md)
- [Project management](docs/project-management.md)
- [Course coverage appendix](docs/proposal-review-and-course-coverage.md)
