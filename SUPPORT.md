# Support

This is an independently maintained research and engineering project. It does not
provide a hosted detection service, guaranteed response time or production support.

## Where to ask

- **Usage and setup:** start with [`README.md`](README.md),
  [`docs/reproduction.md`](docs/reproduction.md) and
  [`docs/environment.md`](docs/environment.md).
- **A reproducible defect:** open a Bug Report issue using the repository template.
  Include the commit, Python/OS details, command and sanitized logs.
- **Documentation problem:** open a Documentation issue and identify the exact file
  and heading.
- **New research direction:** use the Research Proposal template. Ideas outside the
  frozen Phase 1 scope belong in [`docs/research-backlog.md`](docs/research-backlog.md)
  and require an explicit owner decision before implementation.
- **Security concern:** follow [`SECURITY.md`](SECURITY.md); do not report it publicly.

## Before opening an issue

Check [`docs/README.md`](docs/README.md) for the authoritative document on the topic,
search existing issues, and inspect `TODO-Plan.md` for the current status. Do not upload
ASVspoof data, personal voice recordings, third-party audio, model checkpoints,
credentials or local environment files.

## Limitations

Model results are dataset- and condition-specific. The deployed model card is marked
DRAFT and documents measured limitations. Outputs are uncalibrated scores, not
probabilities or proof that a recording is genuine. This project is not a speaker
verifier and must not be used as the sole basis for hiring, lending, law-enforcement,
medical, legal or other consequential decisions.
