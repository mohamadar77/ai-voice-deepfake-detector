## Summary

Describe what changed and why. State the intended user, engineering or research outcome.

## Related work

Link the issue, milestone or explicit owner decision. If none applies, explain why.

## Change classification

- [ ] Code
- [ ] Tests
- [ ] Documentation
- [ ] Configuration / experiment protocol
- [ ] Data / model artifact handling
- [ ] CI / repository maintenance

## Validation

List the commands actually run and summarize their results. Do not claim checks passed
unless they ran on this commit.

```text
uv run ruff check .
uv run pytest
```

Other checks / omitted checks and reason:

## Research and evaluation integrity

- [ ] No locked evaluation labels or predictions were used for model or protocol choices.
- [ ] Train / development / evaluation boundaries remain intact.
- [ ] No threshold, score direction, checkpoint, frozen protocol or evaluation result was
      changed without explicit authorization and a dated record.
- [ ] Negative results and limitations are preserved rather than hidden.
- [ ] Claims do not imply official leaderboard status, calibrated probability, authenticity
      proof or speaker verification.
- [ ] If this is documentation-only, I have not changed model behavior or result values.

## Data, privacy and reproducibility

- [ ] No audio, personal recordings, dataset files, feature caches, checkpoints, prediction
      stores, databases, secrets or `.env` files are included.
- [ ] No raw audio or sensitive content is written to logs or CI artifacts.
- [ ] New experiment choices are recorded in versioned configuration and use explicit seeds.
- [ ] Documentation and the authority index are updated where needed.
- [ ] `TODO-Plan.md` was updated only if completion evidence exists; no duplicate task list
      was introduced.

## Reviewer notes

List known limitations, follow-up work and any action required before release.
