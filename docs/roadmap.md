# Implementation roadmap

Current scope, dates, resource budgets and discussion outcomes are in the [architecture and delivery plan](architecture-and-delivery-plan.md). [TODO-Plan.md](../TODO-Plan.md) tracks completion. This file provides the execution order. The [final frozen proposal](AI_Voice_Deepfake_Detector_Final_Project_Proposal.docx) (signed off 2026-10-11; see the architecture-plan sign-off) is the immutable source of truth; the dates below are current forecasts against its baseline milestones and may slip without changing the proposal.

## Current position

Configuration/CLI/test scaffolding exists; historical environment setup is documented but must be revalidated at G0. No detector implementation or measured results were found during the architecture review. Dataset acquisition and training are not part of this documentation task.

## Critical path

| Gate / target | Output | Evidence required to continue |
| --- | --- | --- |
| G0 — 6–7 October | Environment, capacity/hours, data terms and manifest/development-role contracts | Python 3.12/uv smoke passes; storage/access and owner-hour budget recorded |
| G1 — 8–12 October | Audited data, shared preprocessing, LR bundle, thin API/UI and CPU-container smoke | Train-only fitting, deterministic clip scoring, HTTP parity and cleanup; timed CNN pilot |
| G2 — 13–18 October | Clean/augmented CNN, bounded PCA/K-means, selection and thresholds, all-candidate freeze | Reserved development untouched by selection; complete hashes/configs/membership/transformations frozen |
| G3 — 19–25 October | Locked comparison, paired robustness, source-group uncertainty, failure report and real-model demo | All declared candidates/conditions reported; no post-test selection |
| Freeze — 26 October | Product feature freeze | No new extensions or model families |
| G4 — 27 October–1 November | Release, performance/reproduction verification, model card and slides | Local CPU container works with trusted bundle; measured limitations recorded |
| G5 — 2–4 November | Fresh-start rehearsal and backup | <=12-minute presentation and local demonstration fallback |
| 5 November | Tentative presentation | Working demo and honest measured conclusions |

## First work session

1. Revalidate uv/Python and the setup smoke command; record focused hours/week and current hardware/storage.
2. Write `docs/data.md`, the manifest schema and tiny protocol fixtures.
3. Define official partition and development-speaker roles plus nullable source provenance.
4. Implement metadata-only audit tests before large downloads/features or training.
5. Record access and measured pilot budgets before expensive model work.

## Stop/go rules

- The 110–150-hour core estimate (about 140–190 with contingency) is conditional. If owner availability or compute cannot support it, revise scope/date before promising the release.
- If access is blocked, implement fixtures/contracts; do not invent results.
- If full processing is too expensive, freeze a documented subset before score inspection, preserving official partitions and labeling subset results.
- Freeze LR, clean CNN and augmented CNN, all declared seeds, threshold rules and transformations before final testing. A configuration flag alone does not enforce this.
- A baseline win or weak unseen-attack result is valid research; deployed model selection stays on development.
- The evidence assistant is the only selected stretch, starts only after full core acceptance and must fit before 26 October with release reserve. Otherwise defer it.
- External datasets, additional model families, cloud, calibration/abstention, agent orchestration and MCP are outside this release.

## Definition of done

Reproduce the manifest, shared preprocessing, baseline/CNN study, locked metrics/robustness and failure report; run the API/UI and CPU containers; verify privacy, reliability and artifact provenance; deliver the model card and rehearsed presentation. Use the architecture plan's A0–A9 acceptance TODOs and the root checklist as evidence.
