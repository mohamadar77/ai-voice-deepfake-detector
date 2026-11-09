---
name: Research proposal (Phase 2)
about: Propose a future research question; this is not authorization to implement it
labels: research-backlog
---

> Phase 1's feature freeze has passed. Proposals outside its committed scope belong to
> Phase 2 and require an explicit owner decision before implementation. Do not inspect
> locked evaluation predictions or use evaluation data to develop this proposal.

## Research question

State one falsifiable question.

## Hypothesis and motivation

What do you expect to learn, and why is the question useful?

## Proposed data

List dataset/version, acquisition route, terms, attribution, provenance, consent and
whether any personal or third-party voice data is involved. Do not attach audio.

## Leakage and evaluation plan

Explain how speaker/source separation, partition boundaries, development-only choices,
locked evaluation and repeated-recording dependence will be handled.

## Baselines and controls

Name the minimum baseline, relevant controls and failure analysis.

## Metrics and decision rule

Specify clip-level metrics, operating point, uncertainty and sample-count reporting.
State what result would disconfirm the hypothesis.

## Compute and reproducibility

Estimate compute/storage requirements and specify seeds, configs, run records and
artifact handling.

## Scope and approval

- [ ] This is proposed for Phase 2, not silently pulled into Phase 1.
- [ ] I have checked `docs/research-backlog.md` and existing issues for duplication.
- [ ] I understand that opening this issue does not authorize implementation or use of
      locked evaluation data.
