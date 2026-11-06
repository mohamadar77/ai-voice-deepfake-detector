# Documentation index

Each active document below is the single authority for the subject named next to
it. When two documents disagree, the owner listed here wins and the other one is
wrong. Task completion lives only in [`../TODO-Plan.md`](../TODO-Plan.md); no
document in this directory keeps a second checklist.

Two kinds of authority exist, and prose is the weaker one. Where a document says so
explicitly, a **machine-readable record under `../configs/` or `../reports/` wins
over the prose that describes it** - because the code reads the record, not the
prose. Those records are listed under "Machine-readable authorities" below. Anything
not listed in this file is not an authority; see "Not authoritative" at the end.

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
| [`data.md`](data.md) | ASVspoof 2019 LA provenance, licence, checksums, manifest schema, development reservation; and the provenance of the presentation's demonstration clips, which are illustrative and never measured on |

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

## Machine-readable authorities

Where the prose and the record disagree, these win, and the documents that depend on
them say so in their own text. They are committed; the large artifacts they refer to
are not.

| Record | Authoritative for | Prose that defers to it |
| --- | --- | --- |
| [`../configs/pre_tuning_protocol.yaml`](../configs/pre_tuning_protocol.yaml) | The declared selection metric, tie-break, aggregation, threshold rule and LR budget, as loaded and checked by `pretuning.py` | [`pre-tuning-protocol.md`](pre-tuning-protocol.md): "Where the two disagree, the YAML is authoritative, because the code reads the YAML" |
| [`../configs/evaluation_protocol.yaml`](../configs/evaluation_protocol.yaml) | What the locked evaluation actually computed: candidates, conditions, slices, bootstrap specification | [`evaluation-protocol.md`](evaluation-protocol.md): "where the two differ, the YAML and the freeze record define what was computed" |
| [`../reports/evaluation_freeze_g2.json`](../reports/evaluation_freeze_g2.json) | The frozen evaluation identity: state, candidates, bundle hashes, frozen thresholds, manifest hash, git commit. The committed copy is the **FROZEN** copy (sha256 `cfd1406c…`); the REPORTED copy lives on the host | [`evaluation-protocol.md`](evaluation-protocol.md), [`evaluation-results-g2.md`](evaluation-results-g2.md), [`model-card.md`](model-card.md) |
| [`../reports/dev_reservation_frozen.json`](../reports/dev_reservation_frozen.json) | Frozen membership of the speaker-grouped `select` / `threshold` development split (15 / 5 speakers) | [`data.md`](data.md), "Development reservation" |
| [`../configs/dev_reservation.yaml`](../configs/dev_reservation.yaml) | The reservation rule the frozen record was derived from | [`data.md`](data.md), [`pre-tuning-protocol.md`](pre-tuning-protocol.md) |
| [`../configs/robustness_spec.yaml`](../configs/robustness_spec.yaml) | Robustness condition parameters and seeds, the silence-trim diagnostic, and the CNN training-augmentation recipe | [`pipeline.md`](pipeline.md), [`model-card.md`](model-card.md) |
| [`../configs/serving.yaml`](../configs/serving.yaml) | The deployed bundle hash and exact threshold the service refuses to start without | [`pipeline.md`](pipeline.md), [`deployed-model-decision.md`](deployed-model-decision.md) |
| [`../configs/deployed_model_summary.yaml`](../configs/deployed_model_summary.yaml) | The measured numbers the Streamlit UI displays, copied one-way from [`evaluation-results-g2.md`](evaluation-results-g2.md) and pinned by `tests/test_ui.py` | [`model-card.md`](model-card.md) |
| [`../configs/cnn_study.yaml`](../configs/cnn_study.yaml) | The `mel-v1` feature contract, CNN architecture/budget and declared seeds | [`model-card.md`](model-card.md), [`architecture-and-delivery-plan.md`](architecture-and-delivery-plan.md) |

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

## Not authoritative

Material that may appear in or beside this directory and is **not** a source of
truth for anything. Nothing here may be cited as project status, scope or design.

| Item | What it is |
| --- | --- |
| `research/INCREMENTAL_UPGRADES_AND_ROBUSTNESS_HARDENING.md` | A superseded external planning input (26 October 2026). Its genuinely missing items were migrated in PR #113, which recorded the decision not to adopt it; it was written without access to this repository, carries an expired schedule of its own, and prescribes duplicate filenames that must not be created. Kept for reference only, by the owner's decision of 6 November 2026; this does not adopt it. |
| `prompts/` | Agent work orders. They describe work someone intends to have done, not the state of the system, and their design statements are proposals until migrated into the documents above. Kept for reference only. |
| `../reports/documentation-audit-step1.md`, `../reports/documentation-reconciliation-step2a.md` | Dated audit and reconciliation records. They report on the documentation at a point in time; the documents they examine remain the authorities. |
| Any other untracked working notes the owner keeps under this directory | Working material the owner has not published. Leave untracked and untouched. |
