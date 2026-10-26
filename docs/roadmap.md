# Roadmap

Sequence and milestones only. Whether an individual task is done is recorded in
[`../TODO-Plan.md`](../TODO-Plan.md), the single status authority; this file
deliberately keeps no checklist of its own. Scope, contracts and dated deviations:
[`architecture-and-delivery-plan.md`](architecture-and-delivery-plan.md). The
[frozen proposal](AI_Voice_Deepfake_Detector_Final_Project_Proposal.docx) is the
source of truth for commitments. Dates below are schedule targets, not guarantees.

**The question Phase 1 answers:** *Can we build a reliable AI-generated voice
detector, and scientifically demonstrate where it succeeds, where it fails, and why?*

## Two phases

- **Phase 1, graduation release (presentation 5 November 2026).** A working
  end-to-end system, correct methodology, reproducible experiments, measured
  results and failure analysis, a demo and a presentation. Correctness is never
  traded for the deadline.
- **Phase 2, post-graduation research.** Everything in
  [`research-backlog.md`](research-backlog.md). Not started in Phase 1.

## Where the project stands (27 October 2026)

The feature freeze (26 October) has passed: no new models, datasets or extensions.
Milestones G0 to G3 are complete in substance:

| Milestone | Outcome | Evidence |
| --- | --- | --- |
| G0 feasibility and data contract | Passed 11 Oct | [`data.md`](data.md), [`environment.md`](environment.md) |
| G1 data and baseline | Manifest, development reservation, metrics, MFCC cache, LR baseline, CNN timing pilot | [`data.md`](data.md), [`pre-tuning-protocol.md`](pre-tuning-protocol.md) |
| G2 CNN and protocol freeze | Four CNN runs (clean and augmented, two seeds), robustness spec with silence-trim diagnostic, all-candidate FROZEN record | `reports/evaluation_freeze_g2.json` |
| G3 locked evaluation and demo | Locked scoring and report REPORTED 18 Oct; deployed model recorded 20 Oct; serving API, Streamlit UI, Docker image, latency benchmark and container parity; model card draft | [`evaluation-results-g2.md`](evaluation-results-g2.md), [`deployed-model-decision.md`](deployed-model-decision.md), [`model-card.md`](model-card.md) |

Two items planned for G2 were not delivered and are still open in the TODO plan:
the EDA / exclusion report with the shortcut-only baseline (#6) and the
training-fitted PCA / K-means diagnostics (#36). They are not on the critical path
(see below). Locked results are final; nothing in the remaining sequence may change
a threshold, checkpoint or locked number ([`evaluation-protocol.md`](evaluation-protocol.md)
section 8).

## Remaining sequence

Order matters; each step feeds the next.

1. **G4, release and verification (to 1 November).**
   1. Fresh-checkout reproduction of data preparation, baseline, CNN training and
      the report on a clean host, following [`reproduction.md`](reproduction.md);
      correct the guide where it diverges and record the outcome.
   2. Decide and record which machine presents (see the open host question in
      [`environment.md`](environment.md)), then run the clean container startup, health
      check, image size and cold start there, and rehearse the venue access path.
   3. Choose the demo set (consented own-voice clips and owner-generated synthetic
      clips with tool, version and terms recorded) and prepare the invalid-input
      cases. The own-voice false-alarm observation (#103) is presented as a measured
      limitation, not hidden.
   4. Finalize the model card: close the open items listed in its status line, or
      leave it marked draft.
   5. Freeze the presentation artifacts: slides and speaker notes consistent with
      the model card and locked results.
2. **G5, rehearsal and backup (2 to 4 November).** Timed rehearsal within 12
   minutes; the offline fallback (presentation image, frozen bundle and demo clips
   loaded in Docker on the presenting laptop, verified offline with score parity)
   and a backup recording; final fresh-start demonstration.
3. **Release (by 4 November).** Tag the release and record its commit, model
   version and manifest hash.
4. **Presentation (5 November, tentative).**

## Not on the critical path

Done only if time remains after step 1 and only as documentation or analysis of
existing results, never as new model work:

- EDA figures, exclusion report and the shortcut-only baseline (#6), which would
  strengthen the silence-shortcut finding (#84) with a development-only control.
- PCA / K-means diagnostics (#36). The frozen proposal commits to training-fitted
  PCA and clustering views, and the deviation register does not yet list them as
  undelivered. If they are not run, the owner must record that as a deviation and the
  final materials must say they were not run.
- Release-artifact retrieval instructions, which depend on a redistribution
  decision for the model bundle.
- CI hardening (#15 to #29), parked until the core work finishes.

## Deferred beyond Phase 1

ASVspoof 2021 DF transfer, In-the-Wild / WaveFake, frozen transformer embeddings,
LSTM and extra model families, matched-input controls (#39), min t-DCF (#38),
calibration and abstention, cloud deployment, extensive hyperparameter search, and
the RAG evidence assistant (no-go, #37) and MCP. See
[`research-backlog.md`](research-backlog.md) and the deviation register.

## Stop/go rules

- No new models, datasets or extensions after the feature freeze. Allowed: analysis
  text, honest-framing updates, presentation material, bug fixes in existing code.
- The locked evaluation is REPORTED: no retuning, threshold change, checkpoint swap,
  re-run or additional look at evaluation predictions without the owner's explicit
  authorization and a declared specification.
- A baseline win, a weak unseen-attack result, a below-chance condition or a
  silence-shortcut collapse is a valid finding. Report it.
- Evaluation audio and features never leave the host.
- If the remaining time cannot cover step 1, cut from the end of the
  "not on the critical path" list, never from the rehearsal, the fallback demo or
  the model-card honesty items.

## Definition of done

Phase 1 is done when the acceptance checklist at the end of
[`../TODO-Plan.md`](../TODO-Plan.md) is satisfied or each unmet line is disclosed as
a deviation, and the presentation has been rehearsed.
