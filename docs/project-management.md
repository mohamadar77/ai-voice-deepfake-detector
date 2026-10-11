# Project management setup

The repository documentation identifies the GitHub remote as `mohamadar77/ai-voice-deepfake-detector`. Creating remote issues/boards is a separate action requiring authenticated access; none were created by the architecture consultation.

## Sources of truth

- [Final frozen proposal](AI_Voice_Deepfake_Detector_Final_Project_Proposal.docx): the authoritative, immutable project baseline for scope, commitments and acceptance (SHA-256 `b16f5124…`; signed off 2026-10-11 — see the sign-off in the architecture plan). `_Revised.docx` is a non-authoritative historical candidate.
- [Architecture and delivery plan](architecture-and-delivery-plan.md): scope, technical contracts, estimates, gates, consultation decisions and the canonical dated decision/deviation record (subordinate to the frozen proposal).
- [Root TODO](../TODO-Plan.md): completion status and evidence.
- [Roadmap](roadmap.md): implementation order.
- [Course coverage](proposal-review-and-course-coverage.md): evidence/conceptual/deferred mapping.

## Board and work units

Use Backlog, Ready, In progress and Done. Keep one main implementation issue active. Map future issues to A0–A9 in the architecture plan; each states owner, dependency, expected artifact, acceptance test and estimate. Do not close an issue because its design was documented.

## Milestones

| Dates, 2026 | Milestone |
| --- | --- |
| 6–7 October | G0: environment/capacity and data contract |
| 8–12 October | G1: data/preprocessing, baseline, API/UI and container pilot |
| 13–18 October | G2: clean/augmented CNN and all-candidate protocol freeze |
| 19–25 October | G3: locked evaluation, robustness, failure report and demo |
| 26 October | Product feature freeze |
| 27 October–1 November | G4: release/reproduction/performance verification and slides |
| 2–4 November | G5: rehearsal and backup |
| 5 November | Tentative presentation |

Budget assumption: 110–150 focused hours for core; approximately 140–190 including contingency, excluding unattended compute. Validate actual owner availability at G0. Optional evidence assistant: zero committed hours, likely deferred.

## Decision log discipline

Record protocol and scope decisions before affected experiments, preserve negative results and do not overwrite final runs silently. Link evidence when changing a checkbox. The final frozen proposal is the original `AI_Voice_Deepfake_Detector_Final_Project_Proposal.docx` (SHA-256 `b16f5124…`; see the sign-off in the architecture plan): the authoritative, immutable source for scope and commitments. It must not be edited, and any needed change is recorded in the roadmap, TODO or architecture addendum instead. _(Corrected 2026-10-11: earlier text here calling the proposal "background"/"superseded by the architecture addendum," and a same-day edit that wrongly named `_Revised.docx` as authoritative, are both withdrawn; `_Revised.docx` is a non-authoritative historical candidate.)_ Requested third-reviewer feedback is pending because the supplied session could not be reached.
