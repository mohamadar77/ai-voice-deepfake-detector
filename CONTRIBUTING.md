# Contributing

Thank you for helping improve the AI Voice Deepfake Detector. This repository is a
reproducible research and engineering project. Correct methodology, traceable evidence,
data provenance and honest limitations take precedence over a higher score or a larger
feature set.

## Before starting

1. Read [`AGENTS.md`](AGENTS.md) for binding repository-wide engineering, data and
evaluation constraints.
2. Read [`docs/README.md`](docs/README.md) to find the authoritative document for the
   topic you intend to change.
3. Read [`TODO-Plan.md`](TODO-Plan.md) for task status and the existing GitHub issue /
   milestone references. Do not create a parallel task checklist.
4. Check [`docs/roadmap.md`](docs/roadmap.md) and
   [`docs/research-backlog.md`](docs/research-backlog.md) before proposing scope.

The presentation was delivered on 5 November 2026, but the repository records Phase 1
as presented and not released. Do not describe the project as released or signed off
unless the owner has completed the release decision and the evidence in `TODO-Plan.md`
has been updated.

## Scope and research integrity

- Phase 1's feature freeze has passed. Do not add models, datasets, new evaluation
  conditions, cloud services, calibration, abstention, agents or MCP as incidental
  changes. New research ideas belong in a research-proposal issue and the Phase 2
  backlog unless the owner explicitly reopens scope.
- Do not inspect locked evaluation predictions or use evaluation labels to make model,
  feature, threshold, checkpoint, preprocessing or hyperparameter decisions.
- Do not change frozen protocol/configuration/records merely to make results look
  better. Any authorized protocol change must be explicit, dated, explained and
  traceable; preserve prior results and describe invalidated runs.
- Report negative results and limitations. Do not call project results official
  ASVspoof leaderboard results.
- Do not claim an uncalibrated score is a probability, proof of authenticity, speaker
  verification, or evidence that spoken claims are true.

## Development environment

Use Python 3.12 and `uv`. Do not introduce another environment manager or lockfile.

```bash
uv sync --frozen --all-groups
uv run ruff check .
uv run pytest
```

For a dataset-free sanity check:

```bash
uv run voice-detector smoke --config configs/default.yaml
```

Use the target-platform and fresh-checkout instructions in
[`docs/environment.md`](docs/environment.md) and
[`docs/reproduction.md`](docs/reproduction.md). Do not assume the complete pipeline has
been freshly reproduced unless that run is recorded in the project evidence.

## Code and experiment conventions

- Reusable implementation belongs in `src/voice_detector/`; notebooks should call
  package code rather than define a second implementation.
- Put experiment and application choices in versioned YAML under `configs/`.
- Keep training, evaluation and serving boundaries intact. The API and UI must use the
  shared inference path; do not create a separate UI scoring implementation.
- Keep behavior deterministic where possible; pass and record seeds explicitly.
- Add focused, dataset-free tests for normal behavior, invalid inputs and failure paths.
- Record configuration, seed, dependencies, Git commit, manifest hash, preprocessing,
  model identity, runtime, hardware, predictions and metrics for experiments.
- Keep generated models, audio, features, SQLite files, prediction stores and large
  outputs out of Git.

## Data, privacy and licensing

- Never commit ASVspoof audio, third-party audio, personal voice recordings, uploaded
  audio, feature caches, model checkpoints, databases, credentials or `.env` files.
- Do not attach real or licensed audio to GitHub issues or pull requests. Share only
  sanitized text logs and small, permitted metadata.
- Follow [`docs/data.md`](docs/data.md) for provenance, checksums, terms and attribution.
  The dataset license does not automatically grant rights to redistribute audio,
  generated recordings, checkpoints or this repository's source code.
- Uploaded audio must remain temporary. Do not log raw audio or persist it after
  success or failure.

## Branch and commit workflow

Use a short-lived branch from `main`:

```text
feat/<short-description>
fix/<short-description>
docs/<short-description>
research/<short-description>
chore/<short-description>
```

Use conventional, imperative commit subjects where practical, for example:

```text
docs: add contributor and repository guidance
fix: reject malformed audio before decoding
research: document a proposed unseen-attack evaluation
```

Keep a pull request focused. Link the existing issue where applicable. Do not bundle
unrelated refactors, model changes and documentation edits.

## Pull request expectations

A PR should explain:

- what changed and why, including the intended user or research outcome;
- the relevant issue, milestone or approved owner decision;
- tests and commands actually run, plus anything not run and why;
- whether data, model, serving, evaluation or reproducibility behavior changed;
- whether any result, threshold, protocol, frozen artifact or source-of-truth document
  changed, and the authorization/evidence for that change.

Do not state that tests passed unless they were run. Do not include private paths,
credentials, raw voice samples or generated artifacts in the diff.

## Documentation and status

- Update the document that owns the subject, then update `docs/README.md` if its
  authority map changes.
- `TODO-Plan.md` is the only task-completion tracker. Update it only when the evidence
  and completion criteria exist; GitHub issues are for discrete work coordination.
- `docs/roadmap.md` owns milestone sequence, not task-by-task completion.
- The frozen proposal is immutable. Record compatible implementation decisions in the
  architecture addendum rather than editing the proposal.
- Historical archive documents are point-in-time records; do not rewrite them to match
  current status.

## Review

The repository currently lists `@mohamadar77` as sole code owner. A review from the
maintainer is valuable, but do not configure a required independent approval until an
independent reviewer is available. See [`docs/repository-guide.md`](docs/repository-guide.md)
for the recommended GitHub branch rules and current repository setup.
