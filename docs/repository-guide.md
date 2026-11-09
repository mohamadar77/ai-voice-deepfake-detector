# GitHub repository guide

This document records repository-level conventions and recommended GitHub settings.
It does not define model architecture, research scope, evaluation methodology or task
completion. Those remain with the authorities listed in [`README.md`](README.md).

## Repository identity

Verified against the connected repository on 9 November 2026:

| Setting | Current value / status |
| --- | --- |
| Repository | [`mohamadar77/ai-voice-deepfake-detector`](https://github.com/mohamadar77/ai-voice-deepfake-detector) |
| Visibility | Public |
| Default branch | `main` |
| Code owners | `.github/CODEOWNERS` assigns all paths to `@mohamadar77` |
| Continuous integration | `.github/workflows/ci.yml` runs on push and pull request |
| Python | 3.12 |
| Environment / lock | `uv`, `uv.lock` |
| Project board | [AI Voice Deepfake Detector — Delivery](https://github.com/users/mohamadar77/projects/3) |

Suggested GitHub **About** description:

> Reproducible research and an end-to-end demo for detecting synthesized and voice-converted speech, with locked evaluation, robustness analysis, FastAPI, Streamlit and Docker.

Suggested topics (review and add only those that accurately describe the maintained code):

`deepfake-detection` `audio-deepfake-detection` `speech-processing` `asvspoof`
`pytorch` `scikit-learn` `fastapi` `streamlit` `reproducible-research` `python`

Keep the current public visibility unless the owner explicitly chooses otherwise. Before
changing the About text or topics, verify they still match the project's current status
and do not imply a released or production-ready detector.

## Sources of truth and precedence

The repository already has a deliberate authority model. Preserve it; do not create
competing project plans or duplicate evaluation reports.

| Concern | Authority |
| --- | --- |
| Frozen project scope, commitments and acceptance | `docs/AI_Voice_Deepfake_Detector_Final_Project_Proposal.docx` (immutable) |
| Design contracts, architecture decisions and dated deviations | `docs/architecture-and-delivery-plan.md` |
| Which document owns each subject; machine-readable authority map | `docs/README.md` |
| Task completion and execution status | `TODO-Plan.md` (the only task tracker) |
| Milestone sequence, execution priorities and stop/go rules | `docs/roadmap.md` |
| Post-graduation research ideas | `docs/research-backlog.md` |
| Agent and code conventions, data safety and evaluation safeguards | `AGENTS.md` |
| Current implemented commands | `docs/pipeline.md` |
| Verified reproduction steps | `docs/reproduction.md` |
| Locked evaluation methodology and results | `docs/evaluation-protocol.md`, `docs/evaluation-results-g2.md`, and the machine-readable records named by `docs/README.md` |
| Deployed model limitations and intended use | `docs/model-card.md` (currently DRAFT) |
| Repository contribution and review process | `CONTRIBUTING.md` and this guide |

When documents disagree, follow the precedence explicitly recorded in `docs/README.md`.
Do not silently reconcile contradictions by editing frozen records or historical
archives. Update the document that owns the subject and then update the index if its
authority map changes.

## Branches and pull requests

Use short-lived branches from `main`:

- `feat/<description>` for a scoped feature that is explicitly in scope;
- `fix/<description>` for a defect;
- `docs/<description>` for documentation;
- `research/<description>` for an owner-approved research task;
- `chore/<description>` for repository maintenance.

Open a pull request for reviewable changes. Keep PRs focused, link the related issue,
and complete `.github/PULL_REQUEST_TEMPLATE.md`. A PR must state what was tested and
what was not. The CI workflow is the required automated baseline; a passing CI run does
not replace review of methodology, data rights, privacy or scientific claims.

The repository currently has one listed code owner. Protect `main` against force-push
and deletion. Require the `checks` CI job to pass before merge once branch protection is
configured and the check is visible on a test PR. Do **not** require one approving review
from an independent reviewer while no independent reviewer is configured; that would
block the solo maintainer from merging their own changes. The owner may add independent
review requirements when another qualified reviewer is available.

Prefer squash merging for focused changes to keep the main history readable. Do not
force-push shared branches. Delete short-lived branches after merge where appropriate.

## Issues, milestones and project board

The project already uses the GitHub Project board
[*AI Voice Deepfake Detector — Delivery*](https://github.com/users/mohamadar77/projects/3)
and milestones G0–G5 plus a Phase 2 research backlog. Check the current board and
milestones before creating anything; do not recreate them.

Use issues for discrete, actionable work. Each implementation issue should state the
expected output, dependencies, acceptance evidence and the owning section of
`TODO-Plan.md`. Mint issues just in time rather than mirroring every checklist line.
Keep one main implementation task In Progress at a time, consistent with the plan.

Suggested label vocabulary (check existing labels before adding; do not create
near-duplicates):

| Label | Use |
| --- | --- |
| `bug` | Reproducible defect in existing behavior |
| `documentation` | Documentation or procedure correction |
| `reproducibility` | Environment, run-record or clean-checkout reproduction work |
| `evaluation-integrity` | Split, protocol, leakage, threshold or result-integrity concern |
| `experiment` | Owner-approved experiment within the active scope |
| `research-backlog` | Phase 2 proposal; not authorization to implement |
| `release-blocker` | Evidence-backed item that blocks the owner's release/sign-off decision |

GitHub Issues are the work-in-flight coordination surface. They do not replace
`TODO-Plan.md` as the task-completion authority.

## Continuous integration

The existing `.github/workflows/ci.yml` workflow is the baseline. It:

1. runs for pushes and pull requests;
2. checks out the repository;
3. installs Python 3.12 and the locked dependencies with `uv sync --frozen --all-groups`;
4. runs `uv run ruff check .`;
5. runs `uv run pytest`.

Keep routine CI dataset-free, credential-free and CPU-only. Do not download ASVspoof
or other large datasets in public CI, expose evaluation audio, require private model
bundles, or add GPU-dependent jobs to the default test workflow. Keep opt-in host tests
clearly marked and documented. Dependency updates must preserve `uv.lock`, the explicit
PyTorch CPU index configuration and a passing CI run.

Do not add broad CI/CD or automatic deployment as part of a documentation-only change.
If CI behavior needs to change, make a focused issue/PR and explain the effect on the
reproducible environment.

## Release process

The repository records Phase 1 as **presented but not released**. The presentation date
is 5 November 2026; the project status and remaining acceptance items live in
`README.md`, `docs/roadmap.md` and `TODO-Plan.md`. This governance guide does not close
any task, authorize a waiver or create a release.

Before tagging a release:

1. the owner must make the release decision recorded in `TODO-Plan.md`;
2. each acceptance criterion must have evidence or an explicit dated waiver, following
   the plan's own rules;
3. the model card must remain clearly marked DRAFT unless its blockers are resolved;
4. the final tag and GitHub Release notes must identify the exact commit and link the
   relevant results, model card and reproduction instructions;
5. claims must distinguish the project evaluation from an official ASVspoof result and
   disclose the known model limitations.

Do not create or imply a release tag as part of repository setup.

## Licensing and third-party material

No root-level software `LICENSE` file was present in the inspected repository. Do not
add one or assume a license without the owner's explicit choice. The ODC Attribution
license described in `docs/data.md` applies to the ASVspoof database material; it does
not automatically license this repository's source code, trained weights, generated
audio, presentation material or other third-party dependencies. Review and document
those rights separately before redistribution.

## Security, privacy and observability

Follow [`SECURITY.md`](../SECURITY.md) for private vulnerability reports. Keep secrets
out of Git and CI. Do not add raw audio, private recordings, user uploads, checkpoints,
feature caches or local databases to issues, PRs, logs or CI artifacts. Logs should be
metadata-only and must not expose audio content or sensitive local paths.

Treat model outputs as uncalibrated scores. Do not add product wording or GitHub badges
that imply certified authenticity, speaker verification, benchmark leadership or
production readiness.

## Initial repository-maintenance checklist

This list is repository configuration guidance, not a second project task tracker.

- [ ] Review the GitHub About description and topics.
- [ ] Protect `main` from deletion and force-push.
- [ ] Require the existing `checks` workflow before merging pull requests.
- [ ] Decide whether GitHub private vulnerability reporting is enabled.
- [ ] Review existing labels and align them with the vocabulary above without duplicates.
- [ ] Keep the repository's public visibility only while that remains the owner's intent.
- [ ] Decide the source-code license separately; do not infer it from the dataset license.
