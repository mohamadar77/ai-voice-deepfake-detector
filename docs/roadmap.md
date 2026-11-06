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

- **Phase 1, graduation release (presented 5 November 2026).** A working
  end-to-end system, correct methodology, reproducible experiments, measured
  results and failure analysis, a demo and a presentation. Correctness is never
  traded for the deadline. The presentation was delivered; the release is not
  closed (see below).
- **Phase 2, post-graduation research.** Everything in
  [`research-backlog.md`](research-backlog.md). Not started in Phase 1.

## Where the project stands (6 November 2026)

**Presented, not released.** The presentation was delivered on **5 November 2026**
(owner, recorded 2026-11-06; no repository artifact records the event — there is no
release tag and no commit after 3 November). The feature freeze (26 October) has
passed: no new models, datasets or extensions. Milestones G0 to G3 are complete in
substance; G4 and G5 are not, and the talk was given with their issues still open:

| Milestone | Outcome | Evidence |
| --- | --- | --- |
| G0 feasibility and data contract | Passed 11 Oct | [`data.md`](data.md), [`environment.md`](environment.md) |
| G1 data and baseline | Manifest, development reservation, metrics, MFCC cache, LR baseline, CNN timing pilot | [`data.md`](data.md), [`pre-tuning-protocol.md`](pre-tuning-protocol.md) |
| G2 CNN and protocol freeze | Four CNN runs (clean and augmented, two seeds), robustness spec with silence-trim diagnostic, all-candidate FROZEN record | `reports/evaluation_freeze_g2.json` |
| G3 locked evaluation and demo | Locked scoring and report REPORTED 18 Oct; deployed model recorded 20 Oct; serving API, Streamlit UI, Docker image, latency benchmark and container parity; model card draft | [`evaluation-results-g2.md`](evaluation-results-g2.md), [`deployed-model-decision.md`](deployed-model-decision.md), [`model-card.md`](model-card.md) |
| G4 release and verification | **Not delivered.** All five issues ([#107](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/107)-[#110](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/110), [#112](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/112)) open and untouched since 26 Oct | GitHub milestone *G4: release and slides*: 5 open, 0 closed |
| G5 rehearsal and backup | **Not recorded.** [#111](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/111) (rehearsal, fallback, backup recording, release tag) still open | GitHub milestone *G5: rehearsal and backup*: 1 open, 0 closed |
| Presentation | **Delivered 5 Nov 2026** | Owner statement, recorded 2026-11-06; not corroborated by a repository artifact |

Two items planned for G2 were not delivered and are still open in the TODO plan:
the EDA / exclusion report with the shortcut-only baseline (#6) and the
training-fitted PCA / K-means diagnostics (#36). Locked results are final; nothing
in the remaining work may change a threshold, checkpoint or locked number
([`evaluation-protocol.md`](evaluation-protocol.md) section 8).

The open release decision is the owner's and is recorded nowhere yet: either tag the
release and fill in the sign-off record at the end of
[`../TODO-Plan.md`](../TODO-Plan.md), or disclose each unmet acceptance line as a
dated waiver. That file's own rule — an unmet line with no waiver blocks sign-off —
still applies after the presentation.

## Remaining work to close Phase 1

The original dated sequence (G4 to 1 November, G5 to 4 November, release by
4 November, presentation 5 November) is spent: the presentation happened and the
rest did not. What follows is no longer a schedule but a list of what is still
owed, in the order that makes each item cheapest. The first item is a decision, not
a task, and it gates the others.

1. **Decide the release (owner).** Tag the release and fill in the sign-off record
   at the end of [`../TODO-Plan.md`](../TODO-Plan.md), or record a dated waiver for
   each unmet acceptance line with its reason and impact. Nothing below can close
   the project while this is undecided ([#111](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/111)).
2. **Record what the presentation actually used** ([#108](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/108),
   [#109](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/109)) - **mostly done 2026-11-06.** The owner's record: the demo ran in
   Docker on the presenting laptop and served uploads live; the clips were the owner's
   own recording, Ondoku3 text-to-speech clips and the three invalid-input rejections.
   Host in [`environment.md`](environment.md), provenance in [`data.md`](data.md).
   Ondoku3's terms were checked on 2026-11-06: a research presentation is expressly
   permitted, but the free account the clips came from obliges a visible credit notice
   wherever the audio plays. The delivered deck had none; `Voice by Ondoku3.com` was
   added to slide 10 afterwards, and the gap is disclosed rather than papered over.
   **Gaps remain, both owner-only:** the clip counts and what the own-voice clip scored,
   and who wrote the prompt text. A further technical one: the laptop's container was
   never parity-checked against the `linux/amd64` evaluation build (#108).
3. **Fresh-checkout reproduction** ([#107](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/107))
   of data preparation, baseline, CNN training and the report on a clean host,
   following [`reproduction.md`](reproduction.md); correct the guide where it
   diverges and record the outcome. It remains the largest unmet acceptance claim.
4. **Model card disposition** ([#110](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/110)):
   close the open items listed in its status line, or keep it marked DRAFT with
   that status disclosed in the release record.
5. **Dataset mirror URL** ([#112](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/112))
   in [`data.md`](data.md).

## Optional, and never required for sign-off

Documentation or analysis of existing results only, never new model work. None of
these is a release blocker; each is either disclosed as undelivered or left open:

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
- The presentation has happened, so the old cut rule (drop optional work before
  the rehearsal, the fallback demo or the model-card honesty items) no longer has
  anything to protect. What replaces it: nothing still owed may be closed by
  assertion. An item is either evidenced, or disclosed as a dated waiver.

## Definition of done

Phase 1 is done when the acceptance checklist at the end of
[`../TODO-Plan.md`](../TODO-Plan.md) is satisfied or each unmet line is disclosed as
a deviation, and the presentation has been rehearsed.

**Not met as of 6 November 2026.** The presentation was delivered, but three
acceptance lines are unmet with no waiver, the sign-off record is unfilled, and no
rehearsal is recorded (#111). Phase 1 is therefore presented but not done, and the
gap is the release decision above, not further experiments.
