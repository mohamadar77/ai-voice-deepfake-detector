---
name: Bug report
about: Report a reproducible defect in existing behavior
labels: bug
---

## Summary

Describe the defect and its user-visible or research impact.

## Reproduction

List the smallest steps or command that reproduces it. Do not include audio files.

## Expected behavior

What did you expect to happen?

## Actual behavior

What happened instead? Include sanitized error messages or logs.

## Environment

- Commit / branch:
- OS and architecture:
- Python version:
- `uv` version:
- CPU / accelerator (if relevant):
- Execution mode (local / Docker / CI):

## Relevant command and configuration

Paste the command and the relevant non-secret configuration. Redact local paths,
tokens and personal information.

## Checks already run

- [ ] `uv run ruff check .`
- [ ] Relevant `uv run pytest ...` selection
- [ ] Full `uv run pytest` (when practical)

## Data and privacy check

- [ ] I have not attached ASVspoof audio, third-party audio, personal voice recordings,
      model checkpoints, feature caches, databases, credentials or `.env` files.
- [ ] Logs and configuration are sanitized.
