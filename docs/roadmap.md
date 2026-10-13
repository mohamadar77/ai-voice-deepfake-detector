# Implementation roadmap

This file gives the execution order. Scope, contracts and dated decisions are in the
[architecture and delivery plan](architecture-and-delivery-plan.md), and
[TODO-Plan.md](../TODO-Plan.md) tracks completion. The
[final frozen proposal](AI_Voice_Deepfake_Detector_Final_Project_Proposal.docx)
(signed off 2026-10-11) is the immutable source of truth. The dates below are
forecasts against its baseline milestones; they can slip without changing the
proposal. Rationale for the current order:
[architecture review, 13 October 2026](architecture-review-2026-10-13.md).

**The question Phase 1 answers:** *Can we build a reliable AI-generated voice
detector, and scientifically demonstrate where it succeeds, where it fails, and why?*

## Current position (13 October 2026)

**Done:**
- G0 environment and data gate (DL120 CPU host; ASVspoof 2019 LA acquired and
  checksummed, ODC-By).
- CM protocol parser and audit.
- LA manifest builder, matched exactly to the published counts (121,461 clips;
  manifest hash `b172e50c…`).
- Shared MFCC preprocessing with the frozen option D windowing (PR #30, `mfcc-v2`).
- FastAPI/Streamlit serving skeleton behind a neutral `Scorer`.

**Not done:** no trained model, no development metrics, no evaluation harness, no
container. The plan is more mature than the evidence, so the focus is now
experiments, evidence and delivery.

## Two phases

- **Phase 1, graduation release (now to 5 Nov).** Working end-to-end system,
  correct methodology, reproducible experiments, measured results and failure
  analysis, demo, presentation. Correctness is never traded for the deadline.
- **Phase 2, post-graduation research.** Kept in a separate
  [research backlog](research-backlog.md). Not started in Phase 1.

## Phase 1 critical path (prioritized)

The owner picks the review gate for each slice before implementation starts.

| Pri | Step | Issues | Where it runs | Why it is in this position |
| --- | --- | --- | --- | --- |
| P0 | **Protect the evaluation before any tuning.** Merge option D; freeze the development reservation and the pre-tuning record; build the metrics module (EER, ROC AUC, macro F1, balanced accuracy, per-class precision/recall, FPR/detection at a threshold) with known-score tests. | PR #30 (#4), #31, #32 | #32 dataset-free (local); #31 code local, membership run on the host | Tuning on unreserved development data cannot be undone. |
| P1 | **First real evidence (G1 exit, forecast 16 Oct).** MFCC features and cache; LR baseline development report (tuned on `dev_select`, thresholded on `dev_threshold`, SQLite run record); CNN timing pilot. | #7, #8, #10 | Host | Turns the plan into numbers. The pilot decides between the CPU host and owner-run Kaggle/Colab. |
| P2 | **The evaluation pipeline, early.** Freeze records, prediction store and grouped bootstrap, then a **full rehearsal on `dev_select` with the LR bundle** (no evaluation data). Transform spec, including the silence-trim diagnostic, also rehearsed on development data for mechanics only; parameters are predeclared, never tuned to results. Shortcut-only baseline (duration + silence) on development data. | #35, #34, #6 | Host | Proves that freeze → score → metrics → report works before it matters. The shortcut results are the standout failure analysis. |
| P3 | **Models and freeze (G2, forecast 22 Oct).** CNN clean and augmented, with the input policy decided first; PCA/K-means fitted on training data with K frozen; thresholds on `dev_threshold`; all-candidate `FROZEN` record. Decide #14. | #33, #36, #35, #14 | Host CPU, or Kaggle/Colab per the external-compute protocol | The freeze must cover every candidate, condition and slice. |
| P3‖ | **Parallel lane while the CNN trains.** Scorer + offline/API parity; container smoke in the presentation configuration on the DL120 host, which is also the presentation machine; demo set: consented own-voice recordings plus a modern TTS/voice clone (illustrative). | #9, #11 | Local + host + presentation machine | Fills unattended compute time. The demo with modern voices is a planned honest-failure moment. |
| P4 | **Locked evidence (G3, forecast 27 Oct).** Score every bundle × {clean, codec, noise ×2} plus the silence-trim diagnostic. Report known vs unseen attacks (dev A01–A06 / eval A16, A19 / eval A07–A15, A17, A18), per attack and pooled, with duration bins, speaker-grouped bootstrap and paired differences. Then the failure report, real-model demo and model-card draft. | #35, plus G3 issues minted just-in-time | Host only | Feature freeze on 26 Oct is held; 27 Oct is for finishing analysis and report text only. |
| P5 | **Release (G4, 1 Nov) and rehearsal (G5, 4 Nov).** Fresh-checkout reproduction (data preparation, LR, CNN, report); p50/p95/memory on named hardware; archive frozen evidence with checksums for Phase 2; model card; slides; ≤12-minute rehearsal; backup recording. | G4/G5 issues minted just-in-time | Host + presentation machine | Repeatability and presentation readiness before polish. |

## Safely deferrable work

| Item | Disposition |
| --- | --- |
| S1 evidence assistant (RAG), MCP | **No-go for this release** (owner, 13 Oct; #37 closed) |
| #38 min t-DCF; #39 matched-input LR control | Phase 2 milestone |
| ASVspoof 2021 DF, In-the-Wild/WaveFake, wav2vec 2.0/SSL front-ends, calibration/abstention, cloud | Phase 2 backlog, recorded in the deviation register |
| UI polish | Cut to score, spectrogram, measured-summary table and limitations |
| Extensive hyperparameter optimization | Phase 2. Phase 1 keeps the declared budget: three LR settings and one CNN architecture. |
| CI/DevEx hardening (#15–#29) | Parked until core ML is done; optional Phase 1 tail |
| Second CNN seed pair | Conditional on the #10 forecast. GPU availability makes it more likely, not mandatory. |

**Pulling backlog items back in.** With good progress, the owner may pull a backlog item into Phase 1 through the [pull gate](research-backlog.md#pulling-backlog-items-into-phase-1-progress-gate): critical path ahead, freeze rules respected, done before 26 Oct, Phase 1 results unchanged, timeboxed. Candidates in order: second CNN seed pair, #38, 2021 DF scoring-only transfer, #39.

**Never cut:**
- the development reservation, freeze record and locked evaluation;
- the four robustness conditions and the silence diagnostic;
- known/unseen attack slices;
- the failure report, API/UI/container and model card;
- rehearsal and the backup recording.

## Stop/go rules

- The forecast assumes the plan's 32–43 focused hours per week. **The owner's actual
  hours are still unrecorded**; record them, and revise scope or dates if they do
  not support the forecast.
- The 25% contingency is spent. Apply the cut list in the
  [review](architecture-review-2026-10-13.md#re-baselined-schedule) before slipping
  G4/G5.
- Freeze LR, clean CNN and augmented CNN (all declared seeds), threshold rules,
  transformations and slices before final testing. A configuration flag alone does
  not enforce this.
- A baseline win, a weak unseen-attack result or a silence-shortcut collapse is a
  valid finding. The deployed model is selected on development data only.
- Evaluation audio and features never leave the host. External GPUs see training
  and development data only.
- No new models, datasets or extensions after 26 October.

## Definition of done

Phase 1 is done when all of the following are reproduced and delivered:
- the manifest, shared preprocessing and the LR/CNN study;
- the locked metrics with known/unseen slices, the robustness study, the silence
  diagnostic and the failure report;
- the API/UI and CPU container running on the presentation machine;
- verified privacy, reliability and artifact provenance;
- the model card and the rehearsed presentation.

Use the architecture plan's A0–A9 items and the root checklist as evidence.
