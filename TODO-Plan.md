# AI Voice Deepfake Detector TODO Plan

Owner: Mohamad Abo-Ras  
Created: 4 October 2026  
Target presentation: 5 November 2026, subject to the course schedule

Build and evaluate a reproducible detector for human versus synthesized or voice converted speech. Compare an MFCC feature baseline with a Convolutional Neural Network (CNN), study unfamiliar attacks and audio distortions, and demonstrate a working application.

This checklist follows the current project proposal. Every checkbox starts unchecked because implementation progress has not been verified. Check an item only when its output exists and its completion criteria have been met. Record experiment findings even when a model fails to improve.

## How to use this plan

- **Proposal:** `docs/AI_Voice_Deepfake_Detector_Final_Project_Proposal.docx` explains the question, scope, method and acceptance criteria.
- **Execution:** this file is the source of truth for implementation status. Check an item only when its output exists and its completion criteria are met.
- **Roadmap:** `docs/roadmap.md` contains the current vertical slice, stop/go gates and the next concrete commands.
- **Core first:** data audit, MFCC baseline, CNN, locked evaluation, robustness, failure analysis, API, demo, container, reproducibility and presentation.
- **Deferred:** ASVspoof 2021 DF transfer, frozen transformer embeddings, cloud deployment, WaveFake, calibration, RAG and MCP. Start these only after the core acceptance criteria pass.

Use GitHub Issues for individual tasks and a board with Backlog, Ready, In progress and Done. Keep one main implementation task in progress at a time. Each issue should include its expected output, dependencies and definition of done. Keep short decision notes when scope changes.

## Milestones

| Target dates | Milestone | Completion evidence |
| --- | --- | --- |
| 5–7 October | Feasibility and data contract | Environment smoke check, dataset access decision, storage estimate and manifest schema |
| 8–12 October | Data audit and baseline | Audited manifest, EDA, preprocessing checks, MFCC baseline and compute pilot |
| 13–19 October | CNN and protocol | CNN comparison, development threshold, locked evaluation protocol and extension decision |
| 20–26 October | Evaluation and demo | Locked results, robustness study, failure analysis and working upload flow |
| 27 October–1 November | Release and slides | Feature freeze, container verification, reliability measurements and presentation draft |
| 2–4 November | Rehearsal and backup | Reproducible startup, tested demonstration and rehearsal within 12 minutes |
| 5 November | Presentation | Working demonstration, findings and limitations |

## Start now

The repository setup is complete. The next implementation slice is intentionally small and produces a decision before any expensive training:

1. Confirm ASVspoof 2019 LA access, terms and local storage without downloading audio into Git.
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
- [ ] Create milestone Issues and a Project board.

**Done when:** a fresh checkout can create the environment, import the package and run a small smoke command. The first configuration and setup instructions are committed.

## 2. Data access and protocol

Depends on: repository and environment.

- [ ] Obtain ASVspoof 2019 logical access (LA) data and its official countermeasure protocols.
- [ ] Record source URLs, release identifiers, terms, archive checksums and download instructions in `docs/data.md`.
- [ ] Verify storage requirements, extraction success and missing files.
- [ ] Preserve official training, development and evaluation boundaries.
- [ ] Define the label mapping explicitly: bona fide human speech versus spoofed/synthetic speech.
- [ ] Reserve a speaker grouped part of development data for thresholds and optional calibration before tuning starts.
- [ ] Document which development portion supports model selection and which supports operating thresholds.
- [ ] Prepare ASVspoof 2021 DF access and released evaluation metadata if resources permit. Use it for evaluation, not supervised training.
- [ ] Obtain the published source file maps to audit overlap between collections.

**Done when:** data sources and permitted uses are documented; partitions and label meanings are unambiguous; the primary dataset can be read locally.

## 3. Manifest, cleaning and EDA

Depends on: data access.

- [ ] Implement a manifest builder with clip ID, path, source ID, speaker ID, attack ID, label, split, duration, sample rate, checksum and provenance where available.
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
- [ ] Configure training batches, loss, class weighting, optimizer, early stopping and checkpoint selection.
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
- [ ] Freeze candidate configurations, checkpoints, score direction, operating thresholds, test membership and metrics.
- [ ] Verify no final test data was used to fit scalers, select checkpoints, calibrate outputs or choose thresholds.
- [ ] Generate clip level predictions on the locked ASVspoof 2019 LA evaluation set.
- [ ] Report EER, macro F1, balanced accuracy, ROC AUC, class counts and confusion matrices.
- [ ] Report human false positive rate and synthetic detection rate at the development chosen threshold. Treat the proposed 5% development false alarm target as a target, not a test guarantee.
- [ ] Report unfamiliar attack groups separately, with sample counts.
- [ ] Compute 95% bootstrap intervals grouped by original source recording; disclose limits from speaker dependence where relevant.
- [ ] Run frozen models on ASVspoof 2021 DF if accessible; report codec, source and published evaluation subset slices.
- [ ] Identify overlap between 2019 and 2021 evaluation recordings and avoid presenting correlated results as independent evidence.
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
- [ ] Fit PCA and clustering on training features; project or assign evaluation examples without refitting on them.
- [ ] Save diagnostic plots and a small set of licensed or consented failure examples.
- [ ] Describe dataset shortcuts, failure patterns, what improved and what remained unresolved.

**Done when:** the report contains a controlled robustness comparison and evidence supported failure analysis. Feature plots are presented as diagnostics.

## 9. API and demonstration interface

Start a thin vertical slice after the baseline; finish after model selection.

- [ ] Implement a shared inference module that loads a versioned model bundle.
- [ ] Build FastAPI prediction and health endpoints with typed responses.
- [ ] Validate WAV and FLAC files, decoded duration of 1–30 seconds and a maximum upload size of 20 MB.
- [ ] Handle corrupt files, unsupported formats, empty audio, silence and unavailable models clearly.
- [ ] Return predicted class, model score, model version and processing time.
- [ ] Build a Streamlit upload interface and spectrogram view.
- [ ] Show the chosen model's measured evaluation summary and limitations.
- [ ] Label uncalibrated outputs as scores; add probability or inconclusive behavior only after the corresponding optional evaluation is complete.
- [ ] Delete temporary audio after successful and failed requests; verify cleanup.
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
- [ ] Verify baseline reproduction from a fresh checkout.

**Done when:** another developer can follow the setup and reproduction instructions, and the application has measured reliability and a model card.

## 11. Next enhancements

Choose only after the core pipeline works and the measured budget allows it.

### Frozen transformer audio representations

- [ ] Time a small wav2vec 2.0 feature extraction run and estimate full cost.
- [ ] Freeze the encoder, pool its representations and train a linear or small MLP classifier.
- [ ] Use the same split and tuning discipline as the core models.
- [ ] Cache embeddings and record external pretraining provenance.
- [ ] Compare detection performance, latency, memory and extraction cost.
- [ ] Clearly distinguish this research setting from the original challenge training restrictions.

### Cloud deployment

- [ ] Select one hosting target after confirming access, budget and resource needs.
- [ ] Deploy the evaluated model and the same containerized inference path.
- [ ] Verify health, input limits, temporary file cleanup and operational logs.
- [ ] Measure hosted latency and verify the demo URL.
- [ ] Keep a tested local demonstration fallback.

## 12. Optional extensions

### WaveFake external evaluation

- [ ] Verify dataset terms, generator metadata and compatible preprocessing.
- [ ] Evaluate frozen models without fitting or tuning on this data.
- [ ] Report limited reference speaker and language diversity.
- [ ] Report external results separately from the primary benchmark.

### Calibration and abstention

- [ ] Fit calibration only on the reserved development portion.
- [ ] Evaluate calibration using a documented validation procedure; report a reliability plot and Brier score with split provenance.
- [ ] Define an uncertainty band on development data and freeze it before final testing.
- [ ] Report decision coverage and errors among decided clips alongside overall model results.
- [ ] Explain that unfamiliar audio may still receive a confident wrong assessment.

### Grounded evidence assistant

- [ ] Curate dataset documentation, the model card and approved result reports.
- [ ] Version source identifiers, chunking settings and the document index.
- [ ] Compare BM25 with document embedding retrieval on 15–20 labeled questions.
- [ ] Add source cited answers with numeric claims sourced from experiment records.
- [ ] Measure retrieval recall at k, citation correctness, unsupported claims and unanswerable handling.
- [ ] Keep authenticity assessment in the audio inference path.

### Agent and MCP integration

- [ ] Expose `inspect_audio`, `predict_audio` and `get_model_report` through FastMCP.
- [ ] Reuse validated inputs, bounded execution and structured inference results.
- [ ] Implement one bounded agent workflow only if it improves the evidence assistant or demo.
- [ ] Verify tool errors, source traceability and timeout handling.

**Extension rule:** start with the transformer, then cloud deployment, then the evidence assistant. Add MCP only for a useful integration. Stop adding features on 26 October.

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
| — | Record decisions here as implementation progresses | — | — |

## First actions

- [ ] Initialize the repository and commit the proposal and this plan.
- [ ] Create the environment and verify audio decoding.
- [ ] Verify ASVspoof 2019 LA access and available storage.
- [ ] Build the first dataset manifest and EDA report.
- [ ] Train and record the MFCC baseline development result.

## Project references

- [Project proposal](AI_Voice_Deepfake_Detector_Final_Project_Proposal.docx)
- [GitHub starting plan and references](GitHub_Starting_Plan_and_References.md)
- [Syllabus coverage notes](Syllabus_Coverage_for_Data_and_Modeling.md)

The proposal, supplied syllabus and graduation brief define this plan. Companion file links assume these files are placed together; adjust paths when adding them to the repository.
