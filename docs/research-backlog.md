# Post-graduation research backlog (Phase 2)

Owner: Mohamad Abo-Ras. Created 13 October 2026.

Phase 1 is the graduation release. It was **presented on 5 November 2026** and
delivers a complete, scientifically credible, reproducible AI application with a
demonstration. The release itself is not closed: see
[`roadmap.md`](roadmap.md) for what is still owed and
[`../TODO-Plan.md`](../TODO-Plan.md) for the sign-off record. Phase 2 does not wait
on that closure, but it must not consume it either: the Phase 1 evidence stays
frozen whatever the owner decides about the tag.

Phase 2 extends that validated foundation into a stronger, more generalizable
detection system, with experiments suitable for potential publication. This file is
the **separate backlog** for Phase 2. Nothing here is a Phase 1 task unless it is
pulled through the gate below.

**Phase 1 / Phase 2 decision table** (owner, 13 October 2026):

| Area | Decision |
| --- | --- |
| MFCC + Logistic Regression | Keep (Phase 1) |
| CNN | Keep (Phase 1) |
| Development/test separation | **Mandatory** |
| Unseen-attack evaluation | **Mandatory** |
| Silence-shortcut investigation | Keep (Phase 1) |
| REST API + simple UI | Keep (Phase 1) |
| Containerization | Keep (Phase 1) |
| Advanced pretrained models (wav2vec 2.0 and others) | Defer to Phase 2 |
| Additional research datasets | Defer, unless already committed by the proposal *and* feasible (see the pull gate) |
| RAG evidence assistant | Defer (no-go for this release, #37) |
| Extensive hyperparameter optimization | Defer. Phase 1 keeps the small declared tuning budget (three LR settings; one CNN architecture) |

**Defer the research features, not the architecture's ability to support them.**
Phase 1 keeps datasets, model interfaces, preprocessing and experiment tracking
modular (next section).

GitHub milestone: *Phase 2: post-graduation research (backlog)*. Issues are minted
there only when an item is picked up; #38 and #39 already live there.

## What Phase 1 must preserve for Phase 2

These are Phase 1 obligations, cheap now and expensive to recreate later. They are
small design rules, not extra features:

- **Modular datasets.** A dataset enters the system only as a manifest with the
  canonical schema (`docs/data.md`): clip, speaker, source, attack, label, split and
  provenance. Dataset-specific parsing (today the LA protocol and audio layout) stays
  inside its own builder. Features, training, evaluation and serving consume the
  manifest, never raw dataset layouts. Adding 2021 DF or In-the-Wild later should
  mean writing a new builder, not touching the pipeline.
- **Experiment tracking.** Every run writes a JSON run record: config, seed, commit,
  manifest hash, preprocessing version, runtime, hardware and the paths/hashes of
  predictions and metrics. The planned SQLite `runs` table that Phase 2 comparisons
  would query is not implemented (#54), so Phase 2 starts from the JSON records and
  the retained archive.
- **Versioned preprocessing.** `PREPROCESSING_VERSION` / config versions per feature
  type (`mfcc-v2`, the log-Mel spec in #33). New front-ends get new versions; the
  existing ones are never mutated.

- **Model-agnostic interfaces.** The `Scorer` protocol, the versioned bundle format
  and `PREPROCESSING_VERSION` let a new front-end or classifier plug into the same
  locked evaluation harness without forking it.
- **Archived evidence.** Keep the frozen bundles, freeze records, per-clip
  predictions (clip × condition × model), run records and the manifest, with
  SHA-256 checksums in a retained location outside Git (TODO §10). Phase 2
  comparisons need the exact Phase 1 numbers, not re-runs.
- **The evaluation harness (#35) as a reusable component.** New test sets
  (2021 DF, In-the-Wild) are new *conditions/datasets* in the same
  `DRAFT → FROZEN → SCORED → REPORTED` workflow, with their own freeze records.
- **Negative results and deviations** stay recorded in Phase 1 documents. Phase 2
  never silently replaces them.

## Pulling backlog items into Phase 1 (progress gate): CLOSED

> **Closed 2026-11-06.** This gate can no longer admit anything. Both of its hard
> deadlines are spent (declaration before the 22 October G2 freeze; completion
> before the 26 October feature freeze), and the presentation it was protecting was
> delivered on 5 November 2026. Nothing may now be pulled into Phase 1: the locked
> evaluation is REPORTED and the remaining Phase 1 work is release closure, not new
> evidence. The conditions and candidate ranking below are kept unchanged as the
> historical record of how the gate was meant to work, and as the starting priority
> order for Phase 2 — items 1, 3 and 4 of the ranking are now Phase 2 Gate 1 and
> Gate 2 work. Links to this section from [`../TODO-Plan.md`](../TODO-Plan.md) and
> [`roadmap.md`](roadmap.md) should be read as pointing at a closed mechanism.

The owner may pull a backlog item into Phase 1 if progress is good. A pull is
allowed only when **all** of the following hold, and it is recorded in the
architecture plan decision log before work starts:

1. **The critical path is ahead.** Every P0–P2 item in the [roadmap](roadmap.md) is
   done, P3 is on its forecast, and the pull does not consume the G4/G5 reserve.
2. **It fits the freeze rules.**
   - Anything that adds a *candidate model, metric or test condition* to the locked
     evaluation must be declared before the G2 freeze (forecast 22 Oct).
   - A *scoring-only* addition on already-frozen bundles, such as a new external
     test set, may come after the freeze, under its own freeze record, and is
     reported separately.
3. **It finishes before the 26 October feature freeze.** After that, there are no
   new models, datasets or extensions; that is a proposal commitment.
4. **It never changes Phase 1 results.** Frozen bundles, thresholds and locked
   results stay as reported. A pulled item adds evidence; it does not replace
   evidence.
5. **It has a bounded estimate and a stop rule.** If it is not done by its
   timebox, it returns to the backlog and is disclosed as incomplete.

**Pull candidates, in order:**

| Order | Item | Why first | Freeze constraint |
| --- | --- | --- | --- |
| 1 | Second CNN seed pair (variance) | Already in the Phase 1 plan as conditional; Kaggle/Colab makes it cheap | Declared seeds are frozen at G2 |
| 2 | #38 min t-DCF | 2–3 h; literature comparability | Declare the metric before the G2 freeze |
| 3 | 2021 DF scoring-only transfer (backlog item 1) | Committed by the proposal (RQ2); "already committed and feasible" | Its own freeze record after G2; done by 26 Oct; scored on the host only |
| 4 | #39 matched-input control | Strengthens the feature-versus-architecture claim | Must be a declared candidate by the G2 freeze |

Advanced pretrained models, new non-proposal datasets and calibration are **not**
pull candidates for Phase 1.

## Phase 2 research gates (proposed, 6 November 2026 — not yet approved)

Gates, not dates. Each one states what must be true before it starts and what it has
to produce to be finished; none of them promises a schedule. **Status: proposed.**
They were drafted from the Phase 2 readiness work order and need the owner's
approval before they govern anything; until then the prioritized backlog below is
still the operative scope and these gates are a structuring proposal over it.

The gates do not change the Phase 1 boundary: the locked 2019 LA evaluation stays
REPORTED and immutable, and no Phase 2 experiment may tune, select, threshold or
re-inspect against it. A new test set is a new declared protocol with its own freeze
record, never an extension of the Phase 1 one.

### Gate 0 — Foundation

**Prerequisites:** Phase 1 evidence preserved and locatable (frozen bundles, freeze
records, per-clip predictions, run records, manifest, with checksums, per "What
Phase 1 must preserve for Phase 2" above).
**Work:** dataset-agnostic ingestion and provenance contracts; run identity and
artifact integrity; the evaluation safeguards carried over as reusable components
rather than LA-specific code. The proposed foundation contracts are recorded in
[`architecture-and-delivery-plan.md`](architecture-and-delivery-plan.md) under
"Proposed Phase 2 foundation contracts".
**Exit:** a clean Phase 2 workspace can reproduce and inspect the Phase 1 evidence,
and can ingest a new dataset, without any risk to a Phase 1 artifact. No training
and no dataset download is needed to pass this gate.

### Gate 1 — Cross-domain generalization

**Prerequisites:** Gate 0 passed; the transfer protocol declared before scoring.
**Work:** backlog item 1 (ASVspoof 2021 DF transfer, scoring-only on the frozen
Phase 1 bundles) with the 2019/2021 source-overlap audit; optionally item 2
(In-the-Wild / WaveFake) if the owner approves the data and terms separately.
**Exit:** a protocol-compliant, traceable account of transfer performance **and its
limitations**, with overlapping and no-overlap subsets reported separately where the
organizers' source maps allow, and described as a transfer test rather than an
independent one wherever sources overlap. A collapse is a valid result.

### Gate 2 — Model and robustness research

**Prerequisites:** Gate 1 reported, so a transfer baseline exists to compare against.
**Work:** backlog items 4 (shortcut-robust training), 6 (matched-input controls,
#39), 12 (stronger baselines and ablations), 3 (SSL front-ends) and 9 (additional
seeds), each under its own declared protocol and freeze.
**Exit:** fair, repeatable comparisons under the new protocols, with equal budgets
per model family, uncertainty reported, and compute and inference cost stated
alongside accuracy. Any comparison that cannot state its budget is not finished.

### Gate 3 — Reliability and publication readiness

**Prerequisites:** Gates 1 and 2 reported.
**Work:** backlog items 5 (calibration and abstention, only if a probability output
is an explicit objective), 13 (paired statistical analysis with multiple-comparison
control), 11 (modern generators), 7 (min t-DCF, #38), and a literature and novelty
review at that time.
**Exit:** a defensible research package and an evidence-based decision about
publication. **Publication is not guaranteed and novelty is not assumed**; the
"Publication framing" section below stays a draft until this gate says otherwise.

## Backlog, in priority order

| # | Research item | Question it answers | Preconditions and rules |
| --- | --- | --- | --- |
| 1 | **Cross-domain transfer: ASVspoof 2021 DF** (scoring-only on frozen Phase 1 bundles) | Proposal RQ2, second domain: does a 2019-LA-trained detector survive new vocoders and codecs? | The proposal already names it. Predeclare the subset before scoring. Audit 2019/2021 source overlap with the organizers' metadata; correlated sources are not independent evidence. Report the official DF EER per codec/vocoder slice. Report the full set and, when the organizers' source maps allow, an overlapping and a no-overlap subset as separate results; keep the complete protocol rather than dropping or balancing examples, and describe the result as a transfer test, not an independent one, wherever sources overlap. If access or time fails, record it as not run (as RQ2 already is). Phase 1 bundles stay frozen. References, checked 2026-10-27: [challenge repository and scoring scripts](https://github.com/asvspoof-challenge/2021) and [DF dataset card](https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/ASVspoof2021_DF) (both reachable); the [official 2021 release page](https://www.asvspoof.org/index2021.html) returned HTTP 503 on that date and should be rechecked. |
| 2 | **Real-world generalization: In-the-Wild** (Müller et al., 2022) and/or **WaveFake** | Does benchmark performance predict performance on real public-figure deepfakes? | A new dataset needs owner approval and a terms check. Label it an external test, not population-wide. WaveFake has limited speaker diversity (noted in the proposal). |
| 3 | **SSL front-ends** (frozen wav2vec 2.0 / XLS-R / WavLM embeddings + linear or MLP head) | Proposal RQ3: the accuracy versus inference-cost tradeoff against MFCC+LR and the CNN | External pretraining puts the results in a separate setting from the challenge rules; label them accordingly. Cache embeddings. GPU via Kaggle/Colab under the external-compute protocol. |
| 4 | **Shortcut-robust training** (silence-trimmed or VAD-cropped training, duration-balanced sampling) | Can the Phase 1 silence/duration shortcut be removed without collapsing detection? | Builds directly on the #6/#34 findings. A new candidate needs its own freeze. |
| 5 | **Calibration and abstention** (fitted on reserved development data; reliability plot, Brier score, coverage-versus-error for an "inconclusive" band) | Can the score become a usable probability, and does abstention catch unfamiliar audio? | The proposal gives the rules: fit on the reserved development portion only, and report coverage with errors among decided clips. |
| 6 | **Matched-input controls** (#39: LR on pooled log-Mel; MFCC + MLP) | How much of any CNN gain is representation versus architecture? | Same development budget and frozen protocol. |
| 7 | **Additional metrics for literature comparison** (#38: min t-DCF with organizer ASV scores) | Direct comparability with published ASVspoof 2019 baselines | Amend the `docs/data.md` ASV-score exclusion: tandem metric only. |
| 8 | **Extensive hyperparameter optimization** (bounded search with a predeclared budget, on `dev_select` only) | Is the Phase 1 ranking robust to tuning effort? | Equal budgets per model family, or the comparison is not fair. Never on evaluation data. |
| 9 | **Second and third seeds / variance study**, if Phase 1 ran a single seed | How large is training variability relative to the differences between models? | Paired by seed. Report the spread, not the best seed. |
| 10 | **Cross-language generalization** | Does a detector trained on English VCTK-derived speech transfer to other languages? | Needs multilingual bona fide and spoof data with known provenance and terms. The proposal excluded multilingual claims for Phase 1. |
| 11 | **Modern generators** (current open TTS/voice-clone systems; a curated, consented or licensed set) | Do 2019-era detectors survive current generators? The Phase 1 demo gives only an illustrative hint. | Generation provenance (tool, version, terms) is recorded. It is test-only and never mixed into training without its own protocol. |
| 12 | **Stronger baselines and controlled ablations** (LFCC/CQCC front-ends, LCNN/RawNet-style, AASIST-class; one-factor-at-a-time ablations) | Where do gains come from, under one locked protocol? | Same harness and freeze workflow. Ablations are predeclared. |
| 13 | **Statistical significance and extensive benchmarking** (paired, grouped bootstrap or permutation tests across models/conditions; multiple-comparison control) | Are the differences real? | Uses the archived Phase 1 predictions plus new runs. Report effect sizes with intervals. |
| 14 | **Hosted deployment** (one small cloud instance) | Does the same versioned pipeline run hosted, with health and latency checks? | The proposal's condition: only after the local demo is reliable and access and cost are known. Engineering, not research. |

## Explicitly not research-track

- **RAG evidence assistant / MCP interface.** These are course-coverage concepts. They
  were a no-go for Phase 1 (#37) and are not part of the publication track. Revisit
  only as an engineering demo.
- **Live call monitoring, speaker identification, generator attribution, video and
  multilingual claims.** These are outside the proposal's release scope and need
  separate ethics, data and terms review.

## Publication framing (draft)

Phase 2 ends in research paper preparation. The working thesis, carried over from Phase 1: *Can we build a reliable
AI-generated voice detector, and scientifically demonstrate where it succeeds, where
it fails, and why?*

A publishable contribution most likely combines:
- the Phase 1 shortcut analysis;
- items 1–2 (transfer collapse or survival across domains);
- item 4 (whether shortcut-robust training fixes it).

All of it would run under the locked, all-candidate evaluation protocol, with
grouped uncertainty. Target venues and novelty must be checked against the
literature at the time; this is not a claim of novelty.
