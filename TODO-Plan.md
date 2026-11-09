# AI Voice Deepfake Detector TODO Plan

Owner: Mohamad Abo-Ras  
Created: 4 October 2026  
Presented: 5 November 2026 (owner, recorded 2026-11-06). Release not tagged  
Last reconciled with the repository: 6 November 2026 (post-presentation status reconciliation; see `reports/documentation-reconciliation-step2a.md`). Previous reconciliation: 27 October 2026 (documentation cleanup; see [docs/README.md](docs/README.md))

Build and evaluate a reproducible detector for human versus synthesized or voice converted speech. Compare an MFCC feature baseline with a Convolutional Neural Network (CNN), study unfamiliar attacks and audio distortions, and demonstrate a working application.

This checklist follows the original proposal and the [6 October architecture addendum](docs/architecture-and-delivery-plan.md). Existing checked setup items are historical records; runtime/environment readiness must be revalidated at G0. Check an item only when its output exists and its completion criteria have been met. Record experiment findings even when a model fails to improve.

## How to use this plan

- **Proposal:** `docs/AI_Voice_Deepfake_Detector_Final_Project_Proposal.docx` is the final, frozen, immutable proposal and source of truth for scope/commitments/acceptance (SHA-256 `b16f5124…`; signed off 2026-10-11 — see the sign-off in `docs/architecture-and-delivery-plan.md`). The `_Revised.docx` candidate was non-selected, is not in the repository and was never committed; it is identified only by its SHA-256 in the architecture plan. `docs/architecture-and-delivery-plan.md` is the current scope, contracts, budget and schedule addendum, subordinate to the frozen proposal.
- **Execution:** this file is the source of truth for implementation status. Check an item only when its output exists and its completion criteria are met.
- **Documentation:** [docs/README.md](docs/README.md) is the index and says which document is authoritative for what. This file is the only task tracker.
- **Roadmap:** `docs/roadmap.md` holds the milestone sequence, the remaining execution order and the stop/go rules.
- **Two phases (owner, 13 October 2026):** Phase 1 = this graduation release (5 November). Phase 2 = post-graduation research, kept in the separate [research backlog](docs/research-backlog.md). Phase 1 builds no research-only extensions, but keeps datasets (canonical-schema manifests), model interfaces, preprocessing versions and experiment tracking modular so Phase 2 can plug in. The [pull gate](docs/research-backlog.md#pulling-backlog-items-into-phase-1-progress-gate-closed) that allowed backlog items to be pulled into Phase 1 is **closed** (2026-11-06): both of its deadlines are spent and the presentation has been delivered, so nothing more can be pulled in. Extensive hyperparameter optimization is deferred.
- **Core first:** data audit, MFCC baseline, CNN, locked evaluation, robustness, failure analysis, API, demo, container, reproducibility and presentation.
- **Stretch:** the evaluated evidence assistant was the only selected stretch. It is a **no-go for this release** (owner, 13 October 2026; [#37](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/37)). The shortcut-only baseline and EDA report (#6) and the PCA / K-means diagnostics (#36) are open, off the critical path, and tracked below.
- **Deferred beyond release (Phase 2):** ASVspoof 2021 DF, WaveFake/In-the-Wild, transformer/SSL/LSTM/extra model families, matched-input controls, min t-DCF, cloud, calibration/abstention, agents and MCP. The deviation register is in the architecture plan. These are not release TODOs.

Use GitHub Issues for individual tasks and a board with Backlog, Ready, In progress and Done. Keep one main implementation task in progress at a time. Each issue should include its expected output, dependencies and definition of done. Keep short decision notes when scope changes.

## Phase 1 critical path — tracked in GitHub

This file stays the master plan and status source of truth; GitHub carries only the
work in flight. **Gate milestones:** G0 (closed) … G5 exist as GitHub milestones, plus
*Phase 2: post-graduation research (backlog)*. Issues are minted just-in-time and link
back to the sections here; sections are not mirrored issue-for-issue. Priorities and
rationale: [roadmap](docs/roadmap.md) and the
[13 October architecture review](docs/archive/architecture-review-2026-10-13.md) (historical).

**Board:** GitHub Projects board [*AI Voice Deepfake Detector — Delivery*](https://github.com/users/mohamadar77/projects/3)
(columns Todo · In Progress · Done) tracks these issues; one item is In Progress at a time.
The owner picks the review gate for each slice before implementation starts.

Status of the issues behind this plan, as of **6 November 2026** (checked against
`gh issue list` and the GitHub milestones). Priorities from 14 October are no longer
shown; what is still owed is in the [roadmap](docs/roadmap.md). The G4/G5 issues
(#107-#112) were minted on 26 October and were missing from this table until the
6 November reconciliation.

| Status | Issue | Slice | Milestone | TODO § |
| --- | --- | --- | --- | --- |
| Done | [#4](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/4) / PR #30 | Option D windowing (`mfcc-v2`) | G1 | §4 |
| Done | [#31](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/31) | Speaker-grouped development reservation + pre-tuning record; frozen record committed (`81c3cf7`) | G1 | §2/§5 |
| Done | [#32](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/32) | Clip-level metrics module (PR #41) | G1 | §5/§7 |
| Done | [#5](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/5), [#7](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/7), [#8](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/8), [#10](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/10) | Manifest builder, MFCC cache, LR baseline, CNN timing pilot | G1 | §3-§6 |
| Done | [#33](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/33), [#34](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/34), [#35](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/35), [#14](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/14) | CNN study, robustness + silence-trim spec, evaluation protocol and freeze lifecycle, sub-1 s gate | G2 | §6-§9 |
| Done | [#9](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/9), [#11](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/11) | Concrete scorer + offline/API parity; CPU container smoke (on `Linux VM`, not yet the presentation configuration) | G2 lane | §9/§10 |
| Done | [#56](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/56), [#68](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/68) | `reserve-development` keeps a matching committed frozen record byte-for-byte; fixes the flaky reservation test (bug fix after the freeze; PR #119) | G1 / test | §2 |
| Open | [#6](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/6) | EDA, exclusions, shortcut-only baseline | G2 | §3 |
| Open | [#36](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/36) | Training-fitted PCA / K-means diagnostics | G2 | §8 |
| Open | [#107](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/107) | Fresh-checkout reproduction of data prep, baseline, CNN and report | G4 | §10 |
| Open | [#108](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/108) | Decide and record the presentation machine; verify container + venue path there | G4 | §10 |
| Open | [#109](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/109) | Record the demo set (consented own voice + synthetic clips) with tool, version and terms | G4 | §9/§13 |
| Open | [#110](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/110) | Finalize the model card or keep it DRAFT | G4 | §10 |
| Open | [#111](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/111) | Rehearsal, offline fallback, backup recording and release tag | G5 | §13 |
| Open | [#112](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/112) | Record the exact `dataset.zip` mirror download URL in `docs/data.md` | G4 | §2 |
| Open (findings) | [#83](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/83), [#84](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/84), [#103](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/103) | Threshold non-transfer, silence-cue reliance, own-voice false alarm (anecdotal) | n/a | §8/§13 |
| Open (review) | [#89](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/89) | `/code-review` of the merged serving slice | n/a | §9 |
| Phase 2 | [#52](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/52)-[#55](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/55), [#57](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/57)-[#59](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/59) | LR review follow-ups (input/protocol guards, SQLite run entry, bundle publication, capacity forecast, test collection) | Phase 2 | §5 |
| Phase 2 | [#38](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/38), [#39](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/39) | min t-DCF, matched-input control | Phase 2 | n/a |
| Phase 2 | [#118](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/118) | `reserve-development`: audit trail for `--force` overrides, and what `reservation_forced` means | Phase 2 | §2 |
| Parked | [#15](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/15)-[#29](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/29) | CI / DevEx hardening until core ML is done | n/a | §10 |

The 14 October LR review ([archived](docs/archive/lr-baseline-review-2026-10-14.md))
decided that completed issues stay closed and repairs get new linked issues. The
G1 repair issues #52-#55 and #57-#59 were moved to the Phase 2 milestone and are still open (#56 is fixed separately, see the decisions table); none
changes a locked result.

## Milestones

| Target dates (baseline) | Status (6 Nov) | Milestone | Completion evidence |
| --- | --- | --- | --- |
| 6-7 October | Closed 11 Oct | G0: feasibility and data contract | Revalidated environment, terms/access, capacity/hours and manifest/split contracts |
| 8-12 October | Done (G1 repairs #52-#55, #57-#59 moved to Phase 2; #56 fixed separately) | G1: data and baseline | Features, development reservation, metrics, LR development report and bundle, CNN timing pilot |
| 13-18 October | Done 18 Oct, except #6 and #36 | G2: CNN and protocol freeze | Clean/augmented CNN, development selection, reserved thresholds and all-candidate freeze |
| 19-25 October | Locked report REPORTED 18 Oct; serving, UI, Docker, benchmark and parity merged; model card still draft | G3: evaluation and complete demo | Locked paired results, uncertainty/failures, model card draft and real-model UI |
| 26 October | Passed | Feature freeze | No new models, datasets or extensions |
| 27 October-1 November | **Not delivered.** All five issues (#107-#110, #112) open and untouched since 26 Oct; GitHub milestone *G4* shows 5 open / 0 closed | G4: release and slides | Deck and speaker notes exist and were corrected on 27 Oct and 3 Nov. Container/reproduction/performance checks not run |
| 2-4 November | **Not recorded.** #111 open; GitHub milestone *G5* shows 1 open / 0 closed | G5: rehearsal and backup | No rehearsal, backup recording or offline-fallback verification is recorded anywhere in the repository |
| 5 November | **Delivered** (owner, recorded 2026-11-06) | Presentation | Owner statement only; no repository artifact records the event (no tag, no commit after 3 Nov). Delivered with G4/G5 open |
| after 5 November | **Release undecided** | Release and sign-off | No tag (`git tag -l` empty); the sign-off record below is unfilled and three acceptance lines are unmet without a waiver (#111). Was four until the 5 November demo closed the valid/invalid-input line |

Budget: estimated 110–150 focused hours, approximately 140–190 with 25% contingency, plus unattended compute. Record actual owner availability at G0; the date is not a feasibility guarantee. **13 October:** G1 slipped about 4 days and the 25% contingency is spent. Owner focused hours per week are still unrecorded and must be recorded now. The cut list is in the roadmap/review.

## Next actions (6 November 2026)

The presentation has been delivered, so these are no longer schedule items; they are
what is still owed to close Phase 1. Order and rationale are in the
[roadmap](docs/roadmap.md); each action maps to the open items in the sections below.

1. **Owner decision: the release.** Tag it and fill in the sign-off record at the end
   of this file, or record a dated waiver per unmet acceptance line. Nothing else
   closes the project while this is undecided (#111, §13).
2. Record what the presentation actually used - which machine presented, whether the
   container or the offline fallback ran, and the demo set with tool, version and
   terms. This is now a record of a past event, and the easiest thing to lose by
   waiting (#108, #109; §9, §10, §13).
3. Fresh-checkout reproduction following [`docs/reproduction.md`](docs/reproduction.md)
   (#107, §10). Still the largest unmet acceptance claim.
4. Resolve the model card's open items or keep it a draft (#110, §10).
5. Record the `dataset.zip` mirror URL (#112, §2).

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
- [x] Resolve the G0 runtime blocker: intended Python 3.12/ARM64 environment versus current x86_64 PyTorch wheel resolution. Do not change the lockfile without a recorded platform decision. **Resolved 2026-10-11:** reserved a Linux `x86_64` host (Big Linux VM with Fedora 44, glibc 2.43, Xeon Gold 6152/44 cores, 30 GB RAM, CPU-only); `uv sync --frozen --all-groups`, smoke, ruff and pytest all pass there. No lockfile change. See [`docs/environment.md`](docs/environment.md).
- [x] Record source URLs, release identifiers, terms, archive checksums and download instructions in `docs/data.md`. **Done 2026-10-11:** Edinburgh DataShare handle, ODC-By 1.0 license, the Todisco et al. 2019 citation, and SHA-256 of `dataset.zip`/`LA.zip` recorded. The checksums are the authoritative anchor. The exact mirror download URL is tracked as a separate open item below.
- [x] Verify storage requirements, extraction success and missing files. **Done 2026-10-11:** host capacity verified (preflight / [`docs/environment.md`](docs/environment.md)); the LA archive extracted cleanly with the full documented directory structure. Per-clip file existence against the manifest is checked by the manifest builder (§3).
- [x] Preserve official training, development and evaluation boundaries. Enforced by the evaluation protocol, partition-aware preprocessing, reservation system, and the locked freeze/score pipeline.
- [x] Define the label mapping explicitly: bona fide human speech versus spoofed/synthetic speech. **Done:** `label` ∈ {`bonafide`, `spoof`} is mapped directly from the protocol KEY column and enforced by `src/voice_detector/protocol.py`; documented in [`docs/data.md`](docs/data.md).
- [x] Reserve a speaker grouped part of development data for thresholds and optional calibration before tuning starts ([#31](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/31)). **Rule, tooling and tests done 2026-10-14** (`configs/dev_reservation.yaml`, `src/voice_detector/reservation.py`, `voice-detector reserve-development`); the membership freeze was run on the host and `reports/dev_reservation_frozen.json` is committed (`81c3cf7`). Calibration is deferred (deviation register).
- [x] Document which development portion supports model selection and which supports operating thresholds ([#31](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/31)). **Done 2026-10-14:** `docs/data.md`, "Development reservation" (`select` for model choices, `threshold` for the operating point only) and [`docs/pre-tuning-protocol.md`](docs/pre-tuning-protocol.md).
- [ ] Record the exact download URL of the `dataset.zip` mirror in `docs/data.md` (owner). Checked 2026-10-27: it is not recorded anywhere in the repository, so it stays open.
- [x] Record primary-dataset source-group provenance where available and disclose missing mappings. **Done:** the LA release has no source-recording mapping, so `source_id` is null and the bootstrap falls back to `speaker_id` (67 evaluation speakers); disclosed in `docs/data.md`, `docs/evaluation-protocol.md` section 6 and `docs/evaluation-results-g2.md`. Cross-collection work is in Phase 2.

**Done when:** data sources and permitted uses are documented; partitions and label meanings are unambiguous; the primary dataset can be read locally.

## 3. Manifest, cleaning and EDA

Depends on: data access.

- [x] Draft the canonical manifest fields and provenance rules in `docs/data.md`.
- [x] Add dataset-free manifest row/duplicate validation in `src/voice_detector/manifest.py` with focused tests.
- [x] Parse the official CM protocol into structured entries (speaker, clip, attack, label, split) with a metadata-only audit and fixtures in `src/voice_detector/protocol.py`, exposed as `voice-detector protocol-audit`. Reads no audio; runs before the dataset is present.
- [x] Implement a manifest builder with clip ID, path, source ID, speaker ID, attack ID, label, split, duration, sample rate, checksum and provenance where available (decodes audio; extends the protocol entries above). **Done:** #5 / PR #13; host run 2026-10-13 matched the published counts (121,461 clips; manifest hash `b172e50c…`).
- [x] Export a versioned CSV manifest and SQLite catalog with a documented schema. **Done:** #5 (CSV + SQLite + `catalog_meta`).
- [x] Verify that manifest IDs and paths are unique and referenced files exist. (`manifest.py` rejects duplicate `clip_id`; `build-manifest` refuses missing audio without `--allow-missing` and matches the published 121,461 clips; a full SHA-256 pass on 2026-10-18 confirmed all manifest files exist, #77.)
- [ ] Audit exact duplicates, speaker overlap across official partitions and mapped original source overlap. Partly covered: speaker disjointness across splits is asserted at manifest build (`assert_speakers_disjoint_across_splits`) and no source mapping exists (`docs/data.md`). An exact-duplicate-audio (checksum) audit has not been done.
- [ ] Define the response to each overlap: preserve official protocol reporting, add a clearly labeled clean subset where appropriate, and disclose unresolved limitations.
- [x] Keep all windows, transformed copies and derived features assigned to their original recording and partition. Enforced by preprocessing (`window_spans`, `clip_id` tracking) and the evaluation pipeline (split-before-transform, per-clip aggregation).
- [ ] Inspect class counts, attack counts, duration distributions, sample rates, channels, silence and clipping ([#6](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/6)).
- [ ] Audit whether duration, padding, silence or file processing differs systematically between classes ([#6](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/6)). Measure per-clip leading/trailing non-speech with the same frozen rule as the [#34](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/34) trim condition; use train/development only.
- [ ] Shortcut-only baseline: LR on duration + leading/trailing silence only, fit on train and reported as an EER on `dev_select` ([#6](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/6)). Report it next to the real models.
- [ ] Create SQL queries for partition counts, label balance and experiment metadata.
- [ ] Save EDA figures and an exclusion report with reasons.
- [x] Freeze a documented stratified subset before inspecting model scores if full processing is infeasible. **N/A:** the full 121,461-clip dataset was processed, so no subset was needed.

**Done when:** manifest checks pass, exclusions are traceable and EDA identifies the main data limitations. Record the manifest hash used for experiments.

## 4. Shared audio preprocessing

Depends on: audited manifest.

- [x] Decode consistently to mono 16 kHz and validate decoded audio. (`AudioConfig`, `load_waveform` in `preprocessing.py`; serving path uses the same code.)
- [x] Implement four second training windows with a documented crop and padding policy. (`window_spans` in `preprocessing.py`; CNN config in `cnn_study.yaml`; padding policy in #33.)
- [x] Implement deterministic evaluation windows and aggregation to one score per original clip. (Scoring modules: mean of window logits → one `clip_score` per `clip_id`; used in eval-score and serving.)
- [x] Extract MFCC summary features for the baseline. (`features.py`; LR scored on the locked evaluation, 2026-10-18.)
- [x] Generate log Mel spectrograms for the CNN; record feature settings in configuration. Decide the CNN short-clip/padding policy before training: no unjustified zero-padding, because it would bring back the shortcut that option D removed ([#33](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/33)). (Used by all four CNN runs; shared with serving via `segment_to_spectrogram`, 2026-10-20.)
- [x] Fit feature scalers and normalization statistics on training data only. (`cnn_norm_stats.pt` fitted on train partition; MFCC scaler fitted on train; verified in run records.)
- [x] Cache features using identifiers tied to the manifest and preprocessing configuration. (MFCC cache keyed by `mfcc-v2` + manifest hash; CNN spectrograms generated on the fly.)
- [x] Check feature shapes, finite values, deterministic evaluation behavior and train/serve consistency. (Parity tests: `test_serving_parity_host.py`, container parity script; eval and serving share `segment_to_spectrogram`.)
- [x] Record preparation time, cache size and a small training pilot runtime. (#10 pilot: `cnn_pilot.py` + `cnn_pilot_record.json`; CNN training runtimes in run records.)

**Done when:** the baseline and CNN consume the intended representations of the same permitted clips, and the serving path reproduces evaluation preprocessing.

## 5. MFCC baseline

Depends on: preprocessing.

**14 October review:** implementation is merged and host execution is reported in
PR #49. Follow-ups #52–#55 and #57 (input/protocol guards, retained evidence, bundle
publication) were moved to the Phase 2 milestone and remain open; the original #8 stays
closed. The LR bundle was used unchanged in the locked evaluation. Details:
[review](docs/archive/lr-baseline-review-2026-10-14.md) (historical).

- [x] Train Logistic Regression using training features only. (PR #49; host execution reported.)
- [x] Use a small declared development tuning budget and document class imbalance handling. Selection metric, tie-break, LR grid, aggregation and threshold rule are frozen in the pre-tuning record ([#31](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/31)). Tune on `dev_select` only. **The record exists and is enforced** (`configs/pre_tuning_protocol.yaml`, loaded and validated by `src/voice_detector/pretuning.py`; dated note in [`docs/pre-tuning-protocol.md`](docs/pre-tuning-protocol.md)): three `C` values `[0.01, 1.0, 100.0]` under a declared `max_candidates: 3` ceiling, `class_weight=balanced`, clip EER selection, mean window decision function, threshold at 5% bona fide FPR. The loader checks each declared convention against the constant in the module that implements it, and derives the tie rule by asking `scoring.decide_spoof` what it does at the threshold; the merged LR code now implements the committed mean/C/simplicity path. Alternative aggregation/parameter/latency declarations still need refusal checks (#57); see the 14 October review. The current YAML defaults execute correctly.
- [x] Save the fitted scaler, classifier, configuration and development predictions. (Bundle in `artifacts/lr_baseline/`; run record + predictions.)
- [x] Compute development macro F1, balanced accuracy, ROC AUC, EER, per-class precision/recall and a confusion matrix with the shared metrics module ([#32](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/32)). Report precision with the prevalence it was measured at. (In `docs/evaluation-results-g2.md`.)
- [x] Select the operating threshold on `dev_threshold` only, after the configuration is fixed. Report both human false alarms and synthetic speech detection. (Threshold at 5% bona fide FPR on reserved speakers; eval-results-g2.)
- [x] Record seed, dependency versions, Git commit, manifest hash and runtime as a JSON run record. (LR run record in the bundle.)
- [ ] Insert each run record into a SQLite `runs` table. Not implemented: no `runs` table exists in `src/` (checked 2026-10-27); tracked as #54 (Phase 2). The earlier version of this item was ticked in error.
- [x] Add a command that reproduces baseline training and development evaluation. (`voice-detector train-lr` trains, selects, sets the threshold and writes the development report; `lr-train` and `lr-dev-report` never existed.)

**Done when:** a reproducible reference model and development report exist. Final evaluation remains unused for tuning.

## 6. CNN model

Depends on: working baseline and compute pilot.

- [x] Implement a compact spectrogram CNN with a small Multilayer Perceptron (MLP) classification head ([#33](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/33)). (`CompactMelCNN`, `cnn_train.py`.)
- [x] Configure one compact architecture, loss, weighting, optimizer, early stopping and matched clean/augmented runs; record actual epochs/steps/clip exposure. (`configs/cnn_study.yaml`: cross-entropy, balanced weights, Adam, patience 5, max 20 epochs.)
- [x] Verify the training loop on a small subset before a full run. (#10 pilot, `cnn_pilot.py`.)
- [x] Train the main model using a declared development tuning budget. (Four runs, clean/augmented × seeds 20261004/20261005; `docs/evaluation-results-g2.md`.)
- [x] Save learning curves, checkpoint, configuration and development predictions. (Per-seed checkpoints, run records, dev predictions in `artifacts/`.)
- [x] Compare baseline and CNN on the same development protocol. (Eval-results-g2: LR vs 4 CNN variants on identical dev and eval protocol.)
- [x] Run a second seed if the compute budget permits; otherwise document the unmeasured training variability. (Both seeds run and reported.)
- [x] If the #10 pilot exceeds the CPU caps, train on owner-run Kaggle/Colab under the external-compute protocol (architecture plan). Training/development data only; development predictions recomputed on the host for parity before freezing. **N/A:** CPU host was sufficient; all four runs completed on Linux VM.
- [x] Select the deployed model using development results and practical resource requirements. (`docs/deployed-model-decision.md`, 2026-10-20: `cnn_augmented_seed1` by a development-only rule that was written down after the locked report had been read, so the selection is not blind; issue #86)

**Done when:** the CNN is reproducible and its development performance and inference cost are compared with the baseline. A stronger baseline is a valid outcome.

## 7. Freeze and run final evaluation

Depends on: model development complete.

- [x] Write `docs/evaluation-protocol.md` before final test predictions are inspected ([#35](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/35)).
- [x] Rehearse the whole freeze → score → metrics → report pipeline on `dev_select` with the LR bundle before the real freeze. No evaluation data is used.
- [x] Freeze LR, clean CNN and augmented CNN bundles (plus every declared seed), score direction, thresholds, membership, conditions, metrics, slices and the report template/outline before inspection.
- [x] Implement DRAFT/FROZEN/SCORED/REPORTED records, hash validation and refusal of silent result overwrite; a YAML Boolean is insufficient.
- [x] Verify no final test data was used to fit scalers, select checkpoints, calibrate outputs or choose thresholds. (Evaluation protocol enforced by freeze/score pipeline; all fitting on train, selection on dev_select, thresholds on dev_threshold only.)
- [x] Generate clip level predictions on the locked ASVspoof 2019 LA evaluation set. (R6, 2026-10-18; `docs/evaluation-results-g2.md`)
- [x] Report EER, macro F1, balanced accuracy, ROC AUC, class counts and confusion matrices. (R7)
- [x] Report human false positive rate and synthetic detection rate at the development chosen threshold. Treat the proposed 5% development false alarm target as a target, not a test guarantee.
- [x] Report unfamiliar attack detection rates/counts; predeclare and disclose a bona fide comparison pool when calculating attack-specific EER/AUC.
- [x] Report **known versus unseen attacks** as separate slices:
  - development A01–A06: known algorithms, unseen speakers;
  - evaluation A16/A19: these reuse the A04/A06 algorithms;
  - evaluation A07–A15, A17, A18: unseen algorithms; this is the headline unseen result, never pooled with A16/A19.
- [x] Predeclare duration bins <1, 1–<2, 2–<4, 4–10 and >10 s with per-class counts. Handle sub-1 s clips per #14.
- [ ] Compute 95% bootstrap intervals grouped by original source recording. Where no source mapping exists, group by `speaker_id` (67 evaluation speakers). Paired differences use the same resamples. Disclose the speaker dependence and the small `dev_threshold` speaker count. Partly done: R7 gives slice-EER intervals only. The other metrics' intervals and paired differences were in the PR #82 supplementary tool, which was dropped from Phase 1 (deviation register), so this item stays open unless the owner authorizes `eval-supplement` on the evaluation store with a declared spec.
- [x] Save prediction files and structured metrics; document any deviations from the original protocol.
- [x] Label evaluation on a subset and externally pretrained models accurately; do not claim an official leaderboard result.
- [x] Map each proposal research question to its evidence, or to an explicit "not run": RQ2's second domain and RQ3 are deferred (deviation register). **2026-10-20:** "Research-question mapping" section added to `docs/evaluation-results-g2.md` (PR #94).

**Done when:** results trace to frozen inputs and model versions, and the report answers the primary model comparison and unfamiliar attack questions. Final test findings do not feed further tuning.

## 8. Robustness and error analysis

Depends on: an evaluation pipeline; specify experiments before final scores.

- [x] Define one training augmentation recipe and compare it with clean training under the same protocol. Declare any overlap with the test conditions; label overlapping gains as matched-condition. (Opus 24k + noise 10-30 dB; overlap with codec_24k and noise conditions declared as matched-condition in eval-results-g2 and slide 8.)
- [x] Define clean audio, one compression setting and two noise levels for robustness tests ([#34](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/34)). Pin or record the codec tool version, which sits outside `uv.lock`. Use generated noise with per-clip seeds; no new noise dataset. (`robustness_spec.yaml`: clean, codec_24k, noise_20dB, noise_10dB; ffmpeg/opus version recorded in run records.)
- [x] Add the **silence-trim shortcut diagnostic**: a paired, evaluation-only condition with a frozen trim rule and a minimum-length fallback, scored for every frozen bundle and reported next to the robustness ablation ([#34](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/34)). (Silence-trim condition in eval-results-g2; CNN EER rises from 7-12% to 20-23%.)
- [x] Apply each test transformation consistently to both classes and the same original clips. (Same clips, both classes, frozen seeds; verified in eval-results-g2.)
- [x] Keep transformation parameters and random seeds in configuration. (`robustness_spec.yaml` with per-condition seeds.)
- [x] Report paired changes in detection errors and false alarms under each condition. (Eval-results-g2: per-condition EER, FPR, detection rates with CIs.)
- [x] Review false positives and false negatives by attack (known/unseen), codec, duration, silence and source. (Per-attack EER tables in eval-results-g2; A09 nearly solved; A17 is hardest for LR and three of four CNNs, A10 for `cnn_augmented_seed1`.)
- [ ] Fit PCA and K-means on training features; cap the training-only K search and silhouette sample; project/assign evaluation examples without refitting ([#36](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/36)). No PCA or K-means code exists (checked 2026-10-27). The frozen proposal commits to these views; if they are not run, record the deviation (owner) and say so in the final materials.
- [ ] Save diagnostic plots and a small set of licensed or consented failure examples.
- [x] Describe dataset shortcuts, failure patterns, what improved and what remained unresolved. (Eval-results-g2: silence shortcut finding, augmentation benefit, seed variance, threshold non-transfer, per-attack difficulty spread.)
- [x] The proposal's "failure-analysis dashboard" is satisfied by this failure report plus the Streamlit evaluation-summary/evidence view (§9); no separate dashboard application is required. (Streamlit UI shows model info, limitations; eval-results-g2 is the failure report.)

**Done when:** the report contains a controlled robustness comparison and evidence supported failure analysis. Feature plots are presented as diagnostics.

## 9. API and demonstration interface

Start a thin vertical slice after the baseline; finish after model selection.

- [x] Implement a shared clip scorer with trusted versioned bundle and fitted-statistics/config hashes; check offline/API and CPU-container parity. (ScoringService + build_scorer; container parity: 198 clips, max_abs_diff 0.0, 0 flips.)
- [x] Build POST /predict, GET /health and GET /ready with typed responses; configure the UI API URL for localhost versus container service hostname. (FastAPI app with typed Pydantic responses; compose.yaml service hostname.)
- [x] Validate WAV/FLAC, at most 20 MB and target 1–30-second decoded duration; evaluate duration coverage or explicitly narrow supported claims. Bound decoding and concurrency. (Validation in serving; duration_out_of_range rejection for <1 s clips.)
- [x] Handle corrupt files, unsupported formats, empty audio, silence and unavailable models clearly. (Structured error responses; tested in docker_smoke.)
- [x] Return the decision at the frozen threshold (`decision`, not a verdict), model score, model version and processing time. (ScoreOutcome.response: score, decision, threshold, model_id, model_version, bundle_hash, n_windows, duration_seconds.)
- [x] Build a Streamlit upload interface and spectrogram view. Polish is cut to score, spectrogram, measured-summary table and limitations (owner, 13 October). (Streamlit UI with upload, score display, spectrogram, limitations notice.)
- [x] Show the chosen model's measured evaluation summary and limitations. (GET /model endpoint: model info with limitations.)
- [x] Label uncalibrated outputs as scores; add probability or inconclusive behavior only after the corresponding optional evaluation is complete. (Response field is `score`; limitations notice states uncalibrated; calibration deferred.)
- [x] Delete temporary audio after success, errors and cancellation; clear UI-held buffers and verify cleanup. (Temporary audio not retained; no logging of raw audio.)
- [ ] Choose a small reproducible set of permitted demo recordings: the owner's consented own-voice clips, plus synthetic clips the owner generates with a modern TTS/voice-clone tool whose terms allow it. Record the tool, version and terms. Label the set illustrative with n reported; a miss is shown as a measured limitation. **Status 2026-11-06 (owner record of the 5 November demo):** the set is now recorded in [`docs/data.md`](docs/data.md), "Demonstration clips" - the owner's own consented recording as the bona fide side, text-to-speech clips from **Ondoku3** (<https://ondoku3.com>, a web service with no version identifier) as the spoof side, and the `scripts/generate_demo_clips.py` inputs for the duration, silence and format rejections. Anecdotally the Ondoku3 clips were flagged spoof-like, the correct direction; the clip count, the scores and the own-voice clip's outcome were not recorded, so "n reported" is still unmet. Ondoku3's terms were checked and cited on 2026-11-06 ([`docs/data.md`](docs/data.md), "Ondoku3 terms"): a research presentation is an expressly permitted use, and no Business-Plan purpose is engaged. The clips came from a **free** Ondoku3 account (owner, 2026-11-06), so the credit condition applied, and the delivered deck carried no credit notice (verified by search). `Voice by Ondoku3.com` was added to slide 10 on 2026-11-06 for any reuse; it was not shown on 5 November. **The item stays open on two points, both owner-only:** the clip count and the own-voice clip's score, and who authored the prompt text entered into Ondoku3 ([#109](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/109)). Two earlier own-voice phone recordings were both flagged spoof-like (#103, anecdotal).
- [x] Test the full upload → preprocessing → inference → display flow. (docker_smoke.sh + container parity on Linux VM; demo_backup.py synthetic clips scored end-to-end.)

**Done when:** valid inputs produce an assessment and informative errors handle invalid inputs. The demonstrated model matches the evaluated version.

## 10. Reliability, packaging and documentation

Depends on: a working application.

- [x] Log request ID, model version, audio duration, status and latency without raw recordings or secrets. (FastAPI structured logging with request_id; no raw audio retained.)
- [x] Long host jobs (extraction, CNN training, locked scoring) run detached (`tmux`/`systemd-run --user`) with file logs and are resumable (per-epoch checkpoints with optimizer/RNG state; atomic cache writes). API/UI bind to localhost and are reached via an SSH tunnel. (All host jobs ran in tmux; CNN checkpoints per epoch; localhost binding in compose.yaml.)
- [x] Measure p50/p95 latency, failure rate and memory on named hardware using a declared input mix and request count. **2026-10-20:** benchmark on Linux VM (Xeon E5-2650L v3 @ 1.80 GHz), 250 rounds × 5 clip types [unreconciled: the committed `configs/latency_benchmark.yaml` declares `rounds_per_clip: 50`, i.e. 250 timed requests in total, which is also what the model card says; check the benchmark output on the host and correct one of them], 0 failures, p50 19.4 ms, p95 107.8 ms overall, peak RSS 1016 MiB. Config: `configs/latency_benchmark.yaml`; script: `scripts/latency_benchmark.py`.
- [x] Add focused tests for feature shapes, split leakage, inference preprocessing and API error handling. (Test suite: preprocessing tests, serving parity, API error cases in docker_smoke; 111+ tests on host.)
- [x] Tests for two invariants named in `AGENTS.md`. The bootstrap resamples whole speakers, never single clips, and scores both candidates of a paired comparison on identical resamples (`tests/test_bootstrap.py::TestResamplingUnitIsTheSpeaker`). The LR scaler is fitted on training rows only, and scoring other data does not refit it (`tests/test_baseline.py::TestScalerIsFittedOnTrainingRowsOnly`). Added 2026-10-27; both files pass locally, and each new test fails against a deliberately broken copy of the code (clip-level resampling, unpaired resamples, fitting on development features, refitting while scoring). No behavior change.
- [ ] The same train-only check for the CNN: `cnn_norm_stats.pt` is computed from training rows only. `tests/test_cnn_dataset.py` covers `compute_norm_stats` shapes only, and a stronger test needs `torch`, which this Mac cannot install, so none was added. Add it where `torch` installs (host or CI); no behavior change.
- [x] Configure lightweight GitHub Actions checks that do not require downloading the full dataset or training models. (`.github/workflows/ci.yml`: on push and pull request it runs `uv sync --frozen --all-groups`, `ruff check` and `pytest`, all dataset-free. Hardening #15–#29 stays parked.)
- [x] Package API and UI in containers and document one startup command. One image, two services: `VOICE_DETECTOR_BUNDLE_DIR=<bundle> docker compose up --wait` (see `compose.yaml`; smoke: `scripts/docker_smoke.sh`).
- [ ] Verify a clean local container startup and health check, in the presentation configuration on the DL120 host, which is the presentation machine (owner, 13 Oct), with image size and cold start recorded. Rehearse the venue access path (on-site, or an SSH tunnel to localhost-bound services) ([#11](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/11)). **Partial, 2026-10-20 (Linux VM, not yet the DL120 presentation configuration):** build ~25 min, image 4.82 GB (CPU-only torch wheels via `[tool.uv.sources]`; was 19.97 GB with CUDA wheels), cold start 39 s, warmup 28.6 s + 71.8 ms, API 612 MiB / UI 49 MiB, `/ready` healthy, read-only root, `serve --check` bundle hash verified. Venue access path not yet rehearsed. Which machine presents is unresolved (see `docs/environment.md`).
- [x] Write the model card: intended use, data, language scope, metrics, known failures and limitations. See `docs/model-card.md`, which is **still marked draft**; its status line lists the five open items (serving code review #89, fresh-checkout reproduction, demo provenance, real-device audio, bundle redistribution and eval-only durations).
- [x] Document training, evaluation, artifacts, result reproduction and temporary audio handling. See `docs/reproduction.md` (commands verified against the CLI help on 2026-10-27; the procedure itself has not been re-run).
- [ ] Provide artifact retrieval instructions and checksums where redistribution is permitted.
- [x] Archive frozen bundles, freeze records, per-clip predictions, run records and the manifest, with SHA-256 checksums, in a retained location outside Git. Phase 2 comparisons depend on them. **2026-10-20:** `scripts/archive_artifacts.sh` wrote `/root/archive_g2_20261020` on Linux VM (80 files, 572 MB, `SHA256SUMS`, copy verified against source, FROZEN/REPORTED/report hashes equal the published values). Off-host copy verified on the owner's Mac; external device backup confirmed by owner 2026-10-20.
- [ ] Verify fresh-checkout reproduction of data preparation, baseline training, CNN training and the report (not baseline only), following `docs/reproduction.md`, and correct that guide where it diverges. Not done; no record of any such run exists.

**Done when:** another developer can follow the setup and reproduction instructions, and the application has measured reliability and a model card.

## 11. Selected stretch: evidence assistant

**Decided 13 October 2026: no-go for this release** (owner; [#37](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/37) closed). The items below are kept as the historical contract and course-coverage context; they are not release TODOs. RAG/MCP are off the Phase 2 research track (see the [research backlog](docs/research-backlog.md)).

- Record the explicit go/no-go decision and remaining time.
- Curate approved documentation/model card/frozen results; version document IDs, chunks, enriched metadata and index.
- Compare TF-IDF/BM25 with separate development and held-out questions; semantic retrieval is optional.
- Implement cited, structured and schema-validated generated answers only if model access, cost and evaluation fit; templated search alone is not RAG.
- Evaluate retrieval recall, citation/numeric correctness, unsupported claims, abstention, latency and usage, including unanswerable/injection cases.
- Keep audio decisions and frozen thresholds in the detector. Never pass raw uploads into the evidence workflow.

## 12. Deferred work

ASVspoof 2021 DF/WaveFake, frozen audio transformers, LSTM/extra classifiers, cloud deployment, calibration/abstention, agents and MCP are future work, not acceptance requirements or automatic next tasks. The course appendix explains their relevance and reasons for deferral. **Phase 2:** the prioritized post-graduation research backlog, and what Phase 1 must preserve for it, is in [docs/research-backlog.md](docs/research-backlog.md). The deviation register is in the architecture plan.

## 13. Presentation and release

- [x] Summarize the problem, practical motivation and relevant prior approaches. Cite the published ASVspoof 2019 B01 (CQCC-GMM) and B02 (LFCC-GMM) evaluation results as context only, never as reproduced results. (Slides 1-3 in outline.)
- [x] Present actual data sizes, properties, cleaning and leakage controls. (Slides 3-4 in outline: 121,461 clips, partition walls, speaker reservation.)
- [x] Explain baseline and CNN design choices and experiment controls. (Slides 5-6 in outline.)
- [x] Show model comparisons, known-versus-unseen attack results, the silence/duration shortcut finding, robustness findings, failure examples and limitations. (Slides 7-9 in outline.)
- [x] Demonstrate a valid recording and an invalid input; include an inconclusive example only if that feature was evaluated. **Done 2026-11-05** (owner record, 2026-11-06): the service ran in Docker on the presenting laptop and answered uploads live. Valid audio (own voice, and Ondoku3 text-to-speech) and the invalid inputs were both demonstrated - the too-short and digital-silence WAVs through the Streamlit uploader, the MP3 rejection through `curl`, as the 3 November correction intended. No inconclusive example was shown, correctly: no calibration or abstention exists. Provenance in [`docs/data.md`](docs/data.md); the laptop's container was never parity-checked against the evaluation build ([#108](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/108)).
- [x] Describe lessons learned and future work based on measured results. (Slide 12 in outline: 6 findings + Phase 2 list.)
- [x] Prepare a presentation of at most 12 minutes; show ideas, plots and behavior rather than code. **2026-10-20:** 12-slide outline in `docs/presentation-outline.md` with speaker notes and time allocations; the 12-slide HTML deck `docs/presentation.html` exists (slides v0, PR #105) and was synchronized with the outline on 2026-10-27, then corrected again on 2026-11-03 (PR #122/#123: slide 10's invalid-input beat called for an MP3 upload the Streamlit uploader cannot accept; the beat now uses `too_short.wav` and `digital_silence.wav` through the UI with MP3 rejection shown via `curl`). The deck was delivered on 2026-11-05. **Changed after delivery, 2026-11-06:** slide 10 gained the `Voice by Ondoku3.com` credit that Ondoku3's free tier requires and that the delivered deck did not carry, so the file now differs from what the audience saw in that one respect ([`docs/data.md`](docs/data.md), "Ondoku3 terms"). Timing was never rehearsed on the record (#111).
- [x] Allocate approximately 1 minute motivation, 2 data, 2 methods, 3 results, 2 demo, 1 conclusions and 1 buffer. See outline.
- [ ] Freeze features and organize release artifacts. Features were frozen on 26 October; the release artifacts were never organized or tagged (#111).
- [x] Test demonstration startup and prepared samples on the presentation machine. **Done 2026-11-05** (owner record, 2026-11-06): the presenting laptop ran the image and the prepared clips locally and served live requests, so startup and samples were exercised on the machine that presented. Not a substitute for the clean-startup measurement G4 asked for - image size, cold start and container parity were never recorded there ([#108](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/108)), and the earlier figures are from `Linux VM`.
- [ ] Prepare a short backup recording, saved plots and a local fallback. The demo runs remotely over an SSH tunnel to the host, so the local fallback is the presentation image, frozen bundle and demo clips loaded on the laptop's Docker (`docker save`/`load`, `linux/amd64`), verified offline with score parity in G4. **Partial:** `scripts/demo_backup.py` generates pre-scored fallback JSON; needs host run.
- [x] Check every number in the deck, outline, model card and root README against `docs/evaluation-results-g2.md` and the committed freeze record. Done 2026-10-27: the pooled table re-derives exactly from its own confusion counts; the five thresholds, the deployed bundle hash (also the `configs/serving.yaml` pin), the freeze-record file hash `cfd1406c…`, manifest-hash and commit prefixes all match; the selection-rule means recompute. The sweep found and fixed three deck errors (CNN block 3 has no max-pool, so its output is 64×16×100; the augmentation probabilities are nested, about 25% of clips per transform per epoch, not 50%; A17 is not the hardest attack for `cnn_augmented_seed1`, A10 is) and replaced "roughly in half" with the measured 41–66% reduction.
- [ ] Numbers that cannot be checked from the repository. These were to be confirmed against the host outputs **before** the presentation; the talk was given on 5 November without a recorded confirmation, so they remain unverified from the repository and should be confirmed before any release or publication reuses them: the per-input-type latency figures, p99, peak RSS, cold start, container memory, image size and package count (`reports/latency_benchmark.json` and the Docker logs are not committed); the threshold set's 574 bona fide clips and 4.88% achieved FPR and the development EER table (run records on the host); the latency request count (see §10); the locked report JSON itself (hash `c0116ce2…`, on the host and in the archive).
- [ ] Rehearse and time the presentation, at least twice. No rehearsal is recorded; the presentation was delivered on 5 November without one on the record (#111).
- [ ] Release hygiene sweep at the tag: no secrets, `.env`, local absolute paths, raw or private audio, checkpoints or databases in Git. Partial check 2026-10-27 on tracked files: no audio, checkpoint, database, `.env` or `.DS_Store` files; the one local absolute path in this file was removed; `scripts/demo_setup.sh` and `scripts/setup_linux-vm.sh` still contain host-side `/root/...` paths, which are host paths, not personal data. Repeat at the tag.
- [ ] Tag the final release and fill in the sign-off record at the end of the Final acceptance checklist (commit, model version, manifest hash and the other listed fields).

## Final acceptance checklist

- [ ] Real data and cleaning are documented and auditable.
- [ ] Baseline and CNN comparisons are reproducible.
- [x] Locked unfamiliar attack evaluation and a controlled robustness study are complete. (R7 REPORTED; 5 conditions × 5 bundles; per-attack tables.)
- [x] Missing transfer or optional experiments are explicitly disclosed. (Deviation register: 2021 DF, wav2vec2, calibration, cloud all disclosed.)
- [x] Failure analysis supports the conclusions; measured results replace assumptions. (Silence shortcut, seed variance, threshold non-transfer, per-attack difficulty.)
- [x] The application runs and handles valid and invalid inputs. (Demonstrated live on 5 November 2026 on the presenting laptop's Docker: real responses to valid audio, and the duration, silence and format rejections; owner record 2026-11-06, [`docs/environment.md`](docs/environment.md). Behaviour only - the scores that container returned were never parity-checked against the evaluation build, [#108](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/108).)
- [x] Reliability and latency measurements identify hardware and test conditions. (Benchmark: Linux VM, 250 rounds, named hardware/config; model card §6.)
- [x] Model card, setup guide, configs and result reports are available. (`docs/model-card.md` is still a draft with listed open items; `docs/reproduction.md`, `configs/`, `docs/evaluation-results-g2.md`.)
- [x] Only completed and evaluated features are described as implemented. (Deviations disclosed; no claim of calibration, cloud, or 2021 DF.)
- [ ] The presentation and backup demonstration are ready. **2026-11-06:** the presentation was delivered on 5 November; the backup demonstration, offline fallback and rehearsal are not recorded (#111), so this line is unmet and needs either evidence or a waiver.

Any acceptance line that is waived rather than met needs a reason, its impact and an owner/date recorded in the decisions table below; an unmet line with no waiver blocks sign-off.

### Release sign-off record

Filled in when the release is tagged. Values already established are shown with their source; the rest stay pending until verified.

**Status 2026-11-06: not filled in, and the release is not tagged.** The presentation
was delivered on 5 November 2026 (owner) without a release. The waiver rule below is
therefore live: three acceptance lines are unmet with no waiver recorded, which by that
rule blocks sign-off (it was four until the owner's 5 November demo record closed the
valid/invalid-input line on 2026-11-06). Closing this requires an owner decision, not more work by an
agent - either tag the release and complete the fields, or record a dated waiver per
unmet line with its reason, impact and owner/date.

| Field | Value |
| --- | --- |
| Release commit / tag | pending |
| Manifest hash | role-free `b172e50c…`; role-bearing `b932b859…` (the one used by the locked evaluation); confirm the release-time value |
| Selected model and checkpoint hash | `cnn_augmented_seed1`, bundle hash `999bd10a831fcdff43829fbc197c3d4991491153fcead7cf594c1a971e3fe8a0` (`docs/model-card.md`) |
| Preprocessing version | `mel-v1` (CNN), `mfcc-v2` (LR) |
| Operating threshold and selection rule | -1.251954287290573; 5% bona fide FPR on 5 reserved development speakers (`docs/model-card.md`, `docs/pre-tuning-protocol.md`) |
| Primary evaluation report | `docs/evaluation-results-g2.md`; report file sha256 `c0116ce2…` |
| Robustness report | same document, section 4 |
| Test / CI command and result | pending (`uv run ruff check .` and `uv run pytest` on the release commit, host and CI) |
| Demo startup command and result | pending (`docs/reproduction.md` sections 8-9 on the presentation setup) |
| Unresolved acceptance items | pending (must be empty or waived) |
| Approved scope limitations | the deviation register in `docs/architecture-and-delivery-plan.md` |
| Sign-off date | pending |

## Decisions and blockers

| Date | Decision or blocker | Impact | Next action |
| --- | --- | --- | --- |
| 6 October 2026 | Lead/Meitner architecture agreement | One core, early API/container path, all-candidate freeze, zero-hour stretch | Execute G0; see architecture addendum |
| 6 October 2026 | Requested existing reviewer unavailable through agent and CLI APIs | No three-way approval claimed | Incorporate feedback if the session becomes reachable |
| 6 October 2026 | Session handoff recorded | Resume from G0 environment/data blockers and the architecture addendum | Superseded: the handoff section was removed from the architecture plan on 27 October 2026; use `docs/README.md` and `docs/roadmap.md` |
| 11 October 2026 | Final proposal selected and signed off | Original `...Proposal.docx` frozen as sole source of truth (SHA-256 `b16f5124…`); `_Revised.docx` non-authoritative | All future changes recorded in Markdown only; see architecture-plan sign-off |
| 11 October 2026 | Windowing tail/padding policy frozen — **option D**: pool MFCCs over real (non-padded) samples only; fold sub-frame tails; `<4 s` clip = one window over real audio | Removes zero-pad contamination of pooled MFCC stats (a potential class shortcut); unblocks [#7](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/7)/[#8](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/8); bumps `mfcc-v1`→`mfcc-v2` | Implement in #7 with tests (see [#4](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/4)); CNN padding decided separately at G2 |
| 13 October 2026 | Option D **implemented** ahead of #7 in its own PR (closes [#4](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/4)): pure `window_spans` policy in `preprocessing.py`, `clip_feature_matrix` pools real samples only, `window_waveform` retired, `mfcc-v1`→`mfcc-v2` (code constant `PREPROCESSING_VERSION`, config asserted to match) | Keeps the value-changing policy on a small, pure, exhaustively-tested surface and off the feature-cache I/O review; feature values change under v2 | Feature cache ([#7](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/7)) builds on the frozen v2 next |
| 13 October 2026 | Architecture review ([review](docs/archive/architecture-review-2026-10-13.md)): the development reservation was unscheduled before LR tuning; the evaluation harness had no slot; G1 slipped about 4 days with contingency spent; owner hours unrecorded | G1 narrowed to its critical path; new [#31](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/31)–[#36](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/36); #6/#9/#11 moved to the G2 parallel lane; milestones re-dated (G1 16 Oct, G2 22 Oct, G3 27 Oct); deviation register added to the architecture plan | Execute the Phase 1 critical path ([roadmap](docs/roadmap.md)) |
| 13 October 2026 | Owner direction: **two-phase strategy**. Phase 1 = graduation; Phase 2 = separate research backlog. S1 RAG **no-go**; #38/#39 → Phase 2; UI polish cut; no advanced/pretrained models in Phase 1; methodology claims only with artifacts; work issue by issue | Scope reduced to the core plus the silence/duration shortcut and known/unseen attack analysis; owner-run Kaggle/Colab GPU is available under the external-compute protocol | Owner picks the review gate per slice; start P0 ([#32](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/32), [#31](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/31)) |
| 13 October 2026 | Correction: evaluation attacks A16/A19 reuse the A04/A06 algorithms; they are not unseen | The headline "unseen" result excludes them; they become a known-algorithm control on unseen speakers | Encode the three slices in [#35](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/35); `docs/data.md` corrected |
| 14 October 2026 | **Development reservation rule frozen** ([#31](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/31)): speaker-grouped `select`/`threshold` split, ~25% of the 20 development speakers reserved, ranked by `sha256("<seed>\|<speaker_id>")` under a dedicated reservation seed (`dev-reservation-v1` / `sha256-rank-v1`); coverage (both classes, A01-A06, bona fide per reserved speaker) must hold or the freeze is refused | Membership reproduces independently of PRNG/library drift; the role-bearing manifest gets a new hash by design, so a cache or run record cannot cite a moved reservation. Freeze runs on the host **before** the [#7](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/7) extraction | Owner runs `voice-detector reserve-development` on the host and commits `reports/dev_reservation_frozen.json` |
| 14 October 2026 | **Pre-tuning protocol record declared** before any development score existed ([#31](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/31)): clip EER on `dev_select`, 0.005 tie-break tolerance then simpler system then latency, three LR `C` values with one fixed class-imbalance policy, mean window decision function aggregated per `clip_id`, threshold at 5% bona fide FPR on `dev_threshold` | `pretuning.py` rejects a record that disagrees with `metrics-v1`, the selection metric, the threshold rule, the score direction or the decision rule, so the declaration cannot drift from the code; [#8](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/8) is unblocked on the protocol side | Later changes are protocol deviations with dated rows, not config edits |
| 14 October 2026 | Review of the whole #31 slice (`/code-review`, Opus 5 reviewer) returned **request-changes**: the re-freeze guard read only the git-ignored manifest, so a rebuild plus an edited fraction moved membership silently at `rc=0`; `verify_frozen_reservation` accepted a speaker under both roles, order-dependently; `apply_development_roles` returned `[]` for a one-pass iterable; the catalog drift check was skippable and the freeze then stamped provenance it could not vouch for | All 14 confirmed findings fixed on owner approval: the committed record is now reconciled before the manifest, roles must be disjoint, a record is re-derived from its own seed and fraction, a clamped fraction fails coverage, a half-written column refuses cleanly instead of raising, the self-check runs before the first `os.replace`, and the pre-tuning loader's conventions were bound to the modules that implement them | Guards are in place before the host freeze writes the authoritative record, which is the only window in which they are free to add |
| 14 October 2026 | Process deviation: the #31 core (`reservation.py`, `manifest.py`) reached `main` as a WIP commit via PR #42, without tests, config or CLI | Owner decision (14 Oct): leave it and finish forward rather than revert; the completing PR is reviewed over the whole slice range, not only its own delta, so the merged core is still reviewed | One PR per issue remains the standard; this is recorded as the exception |
| 18 October 2026 | Locked evaluation scored and reported (R6/R7): FROZEN record committed as `reports/evaluation_freeze_g2.json`; record REPORTED on the host | Locked results are final; no retuning, threshold change, checkpoint swap or extra look at evaluation predictions without explicit owner authorization and a declared spec | None; report in `docs/evaluation-results-g2.md` |
| 20 October 2026 | Deployed model recorded: `cnn_augmented_seed1` by a development-only rule accepted 2026-10-19, **after the locked report had been read** (#86). Owner kept it despite the highest clean unseen-attack EER of the four CNNs | The selection is disclosed as non-blind everywhere it is discussed; the demo must present a score, not a verdict | `docs/deployed-model-decision.md` |
| 25 October 2026 | Supplementary eval-report analysis (PR #82, `eval-supplement`) dropped from Phase 1 (architect deadline 24 Oct passed without review/authorization) | No evaluation-only duration table or paired differences; the tool stays in the code for Phase 2 | Listed in the deviation register |
| 26 October 2026 | Feature freeze | No new models, datasets or extensions; analysis text, honest-framing, presentation and bug fixes only | G4 verification and G5 rehearsal (roadmap) |
| 27 October 2026 | Documentation consolidated: `docs/README.md` index with one authority per topic; stale handoffs, status snapshot, project-management and dev-reservation host runbook removed; three historical records archived; this file kept as the only tracker | Removes contradictions (e.g. the reservation was committed in `81c3cf7`, the host runbook said it was not) | Open items above are unchanged except the status updates noted inline |
| 27 October 2026 | `reserve-development` no longer rewrites a committed frozen record that matches (#56, #68): a matching re-run keeps it byte-for-byte; a record naming a different manifest is refused unless `--force` | Removes a provenance hazard (a routine re-run changed the authoritative record's freeze time and commit) and the flaky test. A bug fix after the feature freeze; no change to membership, thresholds or any locked result | Owner chose implement then `/code-review`; merge after review |
| 5 November 2026 | **Presentation delivered** (owner, recorded 2026-11-06). Given with the G4 verification set (#107-#110, #112) and the G5 rehearsal/fallback/tag item (#111) still open; no repository artifact records the event | Phase 1 is presented but not released: no tag, sign-off record unfilled, three acceptance lines unmet without a waiver (four before the demo record of 2026-11-06) | Owner decides the release: tag and complete the sign-off record, or record a dated waiver per unmet line |
| 6 November 2026 | **Demo record captured from the owner:** the 5 November demo ran in Docker on the presenting laptop (not the DL120, not an SSH tunnel to `Linux VM`), served uploads live, and showed the own-voice clip, Ondoku3 text-to-speech clips and the three invalid-input rejections. Ondoku3 clips were flagged spoof-like (anecdotal) | Two §13 items and one acceptance line close on evidence; unmet acceptance lines drop from four to three. Provenance recorded in `docs/data.md`; the presenting host recorded in `docs/environment.md`. Two gaps remain: Ondoku3's terms, and the own-voice clip's outcome and clip counts ([#109](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/109)). The laptop container was never parity-checked ([#108](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/108)) | Owner supplies the terms and the clip outcomes; release decision unchanged |
| 6 November 2026 | **Ondoku3 terms checked** against the service's own licence article and Terms of Use (read 2026-11-06). Use in a research presentation is expressly permitted and no Business-Plan purpose is engaged; a free account obliges a visible credit notice wherever the audio is played, and copyright in the generated audio follows the input text | Demo-clip provenance is now substantively documented in `docs/data.md`. A possible compliance gap is disclosed rather than assumed away: if a free account produced the clips and no credit was shown on 5 November, the condition was unmet | Owner states free vs paid account and whether credit was displayed; add the credit line to the deck and any reuse. Not legal advice; the owner accepted the terms ([#109](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/109)) |
| 6 November 2026 | **Free Ondoku3 account confirmed** (owner), so the free-tier credit condition applied to the 5 November demo. Search of the delivered deck and outline: no Ondoku credit was present | A disclosed, closed-out compliance gap rather than a hidden one. `Voice by Ondoku3.com` added to slide 10 and to the outline's slide-10 notes on 2026-11-06 for any reuse; the deck file now differs from the delivered version in that one respect, which is recorded. Whether credit was given aloud on the day is not recorded and is not claimed | Keep the line while these clips are used; it also covers any recording or re-showing. Owner still to supply clip counts, the own-voice score and the prompt-text author ([#109](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/109)) |
| 6 November 2026 | Post-presentation status reconciliation (Step 2A). The 27 October issue table was missing the entire G4/G5 set (#107-#112) and #118; status documents still described the presentation as pending | Status layer now matches the evidence: presentation recorded as delivered, G4/G5 recorded as not delivered, expired dated mechanisms closed. No locked result, threshold, config, code or test touched | Report in `reports/documentation-reconciliation-step2a.md`; outstanding owner decisions listed there |

## First actions

- [x] Revalidate runtime and hardware/storage (G0; DL120 host, `docs/environment.md`).
- [ ] Record the owner's focused hours/week. Still open; this blocks forecast validation.
- [x] Verify ASVspoof 2019 LA access/terms and write the data contract (`docs/data.md`).
- [x] Implement protocol fixtures, manifest schema and metadata-only audit (`protocol.py`, `manifest.py`).
- [x] Implement/test deterministic preprocessing and pilot capacity. (Preprocessing tests; CNN pilot #10; capacity verified.)
- [x] Train/record the baseline development result when the implementation task authorizes training; connect the same bundle to an early API/UI path. (LR baseline PR #49; CNN bundles; serving via ScoringService.)

## Project references

- [Documentation index](docs/README.md)
- [Final frozen project proposal (source of truth)](docs/AI_Voice_Deepfake_Detector_Final_Project_Proposal.docx)
- [Architecture, design contracts and deviation register](docs/architecture-and-delivery-plan.md)
- [Roadmap](docs/roadmap.md)
- [Post-graduation research backlog (Phase 2)](docs/research-backlog.md)
- [Course coverage](docs/course-coverage.md)
- [Archived historical records](docs/archive/)
