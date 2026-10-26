# Documentation index

Each active document below is the single authority for the subject named next to
it. When two documents disagree, the owner listed here wins and the other one is
wrong. Task completion lives only in [`../TODO-Plan.md`](../TODO-Plan.md); no
document in this directory keeps a second checklist.

## Project overview and proposal

| Document | Authoritative for |
| --- | --- |
| [`../README.md`](../README.md) | Public overview, supported use, quick start |
| [`AI_Voice_Deepfake_Detector_Final_Project_Proposal.docx`](AI_Voice_Deepfake_Detector_Final_Project_Proposal.docx) | The frozen proposal: scope, commitments, acceptance. Never edited (SHA-256 recorded in the architecture plan) |
| [`../TODO-Plan.md`](../TODO-Plan.md) | Task completion and execution status (single source of truth) |
| [`roadmap.md`](roadmap.md) | Milestone sequence and remaining execution priorities |

## Architecture and implementation

| Document | Authoritative for |
| --- | --- |
| [`architecture-and-delivery-plan.md`](architecture-and-delivery-plan.md) | Design contracts, architecture decisions, dated deviations from the proposal |
| [`pipeline.md`](pipeline.md) | The implemented offline and serving pipeline and its command inventory |

## Setup and reproduction

| Document | Authoritative for |
| --- | --- |
| [`reproduction.md`](reproduction.md) | Verified commands, order, prerequisites and expected outputs, from a fresh checkout to the demo |
| [`environment.md`](environment.md) | Supported platforms, the hosts used, development loop, disk budget |

## Data

| Document | Authoritative for |
| --- | --- |
| [`data.md`](data.md) | ASVspoof 2019 LA provenance, licence, checksums, manifest schema, development reservation |

## Evaluation methodology and results

| Document | Authoritative for |
| --- | --- |
| [`pre-tuning-protocol.md`](pre-tuning-protocol.md) | Selection metric, tuning budget and threshold rule declared before any development score existed |
| [`evaluation-protocol.md`](evaluation-protocol.md) | Locked-evaluation methodology and interpretation rules (written before the report was generated) |
| [`evaluation-results-g2.md`](evaluation-results-g2.md) | The locked REPORTED results, failure analysis and research-question mapping |

## Model selection and model card

| Document | Authoritative for |
| --- | --- |
| [`deployed-model-decision.md`](deployed-model-decision.md) | Why `cnn_augmented_seed1` is deployed, and the selection-order caveat |
| [`model-card.md`](model-card.md) | Intended use, measured performance, failures and limitations of the deployed model (still marked draft; open items are listed in its status line) |

## Presentation and course coverage

| Document | Authoritative for |
| --- | --- |
| [`presentation.html`](presentation.html) | The delivered 12-slide deck |
| [`presentation-outline.md`](presentation-outline.md) | Speaker notes and timing for that deck |
| [`course-coverage.md`](course-coverage.md) | Syllabus topic to project evidence, with implemented / design-only / not-implemented status |

## Future research

| Document | Authoritative for |
| --- | --- |
| [`research-backlog.md`](research-backlog.md) | Everything outside the committed Phase 1 scope, including the Phase 1 to Phase 2 pull gate |

## Historical archive

[`archive/`](archive/) holds point-in-time records kept for provenance:

- [`archive/architecture-review-2026-10-13.md`](archive/architecture-review-2026-10-13.md): the architecture review behind the two-phase plan.
- [`archive/dl120-verification-tracker.md`](archive/dl120-verification-tracker.md): the host runbook and log for the 5-candidate freeze, locked scoring and report.
- [`archive/lr-baseline-review-2026-10-14.md`](archive/lr-baseline-review-2026-10-14.md): the review of the LR baseline slice.

Archived documents are historical. They are not edited to match the present, so
they may describe work as pending that has since finished and name scripts or
flags that have changed. Read current decisions and status from the active
documents above, never from the archive.
