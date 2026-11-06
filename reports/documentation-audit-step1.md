# Documentation audit — Step 1 (inventory and consistency)

**Audit date:** 2026-11-06
**Repository state audited:** branch `main` at `d900088`, plus two untracked files
(`docs/prompts/`, `docs/research/`).
**Scope:** read-only inventory, classification, duplication and consistency audit of
all project documentation. No file was modified, moved, renamed or deleted; this
report is the only file created.
**Phase 1 outcome (established from the owner, 2026-11-06):** the presentation
**was delivered on 5 November 2026**; the release was **not tagged**. This is
recorded here because it is not derivable from the repository (see D.5), and it
resolves the open question the first draft of this report could not answer.

**Method:** every document listed in section B was read in full or, for the four
longest, read by section. Claims were checked against `src/voice_detector/`,
`configs/`, `reports/`, `tests/`, `git log`, and the GitHub issue list
(`gh issue list`, read-only). Mechanical checks were scripted: relative Markdown
links, heading anchors, and backticked file references. Findings are marked
**[verified]** when checked against an artifact or command output and
**[unverified]** when they rest on documentation alone.

---

## A. Executive summary

**Overall health: good on evidence, stale on status.**

This is an unusually disciplined documentation set for a solo project. A
consolidation pass on 27 October 2026 (recorded in `TODO-Plan.md`'s decisions
table) created `docs/README.md` as an explicit one-authority-per-topic index,
moved three point-in-time records into `docs/archive/`, and removed earlier
handoff, status-snapshot and project-management duplicates. The result is that the
duplication problems the audit brief anticipated (`TODO-Plan.md` vs
`docs/roadmap.md`, architecture plan vs architecture review, protocol vs
pre-tuning protocol) have **already been resolved by explicit delegation** rather
than left as overlaps. Each of those pairs now carries a header stating what it
does *not* own.

The **evidence layer is accurate and independently verifiable**. Every
cross-document number I could recompute or re-hash matched:

| Check | Result |
| --- | --- |
| Frozen proposal SHA-256 vs the pin in the architecture plan | matches `b16f5124…a09aec` **[verified]** |
| `reports/evaluation_freeze_g2.json` SHA-256 vs the pin in 4 documents | matches `cfd1406c…cb4d2`; record `state: FROZEN` **[verified]** |
| Five frozen thresholds in the freeze record vs results doc, model card, serving config | all five match to full float64 **[verified]** |
| Deployed bundle hash `999bd10a…` vs `configs/serving.yaml` pin and model card | matches **[verified]** |
| Manifest hash `b932b859…` (role-bearing) vs freeze record | matches **[verified]** |
| `configs/deployed_model_summary.yaml` (UI summary) vs `docs/evaluation-results-g2.md` | all 20+ values match **[verified]** |
| Dataset counts (121,461 / 12,483 bona fide / 108,978 spoof / 107 speakers) across `data.md`, model card, deck | internally consistent and additive **[verified]** |
| Development reservation 15 `select` / 5 `threshold` speakers | matches `reports/dev_reservation_frozen.json` **[verified]** |
| CNN parameter count "~23.7 K" | recomputed from `cnn_pilot.py` layer definitions: 23,650 **[verified]** |
| Architecture description (pool after blocks 1 and 2 only) | matches `CompactMelCNN` **[verified]** |
| `mel-v1` spec (n_fft 400, hop 160, 64 mels, 20–8000 Hz) | matches `configs/cnn_study.yaml` **[verified]** |
| Augmentation probabilities (nested 0.5 / 0.5 / 0.5, SNR 10–30 dB) | matches `configs/robustness_spec.yaml` **[verified]** |
| "No PCA / K-means code exists" (#36) | confirmed: no match in `src/` or `tests/` **[verified]** |
| "No SQLite `runs` table exists" (#54) | confirmed: only `predictions`, `metadata`, `clips`, `catalog_meta` **[verified]** |
| Relative Markdown links across all 27 `.md` files | 0 broken **[verified]** |
| Heading anchors in cross-document links | 0 broken **[verified]** |

The problems are concentrated in the **status layer**, not the evidence layer:

1. Every active status document is dated **27 October 2026** and describes a
   future that has passed. The target presentation date (5 November) is behind
   us, no release tag exists, and the G4/G5 closure issues are all still open.
2. `TODO-Plan.md` is declared the single status authority but **omits the entire
   G4/G5 issue set** (#107–#112, #118) that was minted on 26 October — the day
   before its last reconciliation.
3. Two untracked documents (`docs/research/INCREMENTAL_UPGRADES_AND_ROBUSTNESS_HARDENING.md`,
   `docs/prompts/PHASE2_READINESS_AND_FOUNDATION_PROMPT.md`) have appeared inside
   `docs/`. They are **competing planning authorities with their own expired
   schedules and their own file-naming scheme**, and committing the first one was
   explicitly decided against in PR #113.

None of these threaten the locked evaluation record, which remains intact and
correctly pinned.

### The five most important findings

1. **The whole status layer is past its own horizon (Critical).** `README.md`
   still says Phase 1 "is in its verification and rehearsal stage" with a
   "tentative presentation 5 November 2026"; `docs/roadmap.md` is headed "Where
   the project stands (27 October 2026)" with a remaining sequence whose last
   dated step is 5 November. Today is 6 November. `git tag -l` is empty, so the
   release was never tagged, and the sign-off record in `TODO-Plan.md` is still
   `pending` in 7 of 13 fields. **[verified]** Nothing in the repository records
   whether the presentation happened; the owner confirms it **was delivered on
   5 November**, and that the release remains **untagged** — so the documents
   understate what was achieved and overstate what remains.
2. **The single tracker is missing the remaining critical path (Critical).**
   `TODO-Plan.md`'s issue table claims to be "as of 27 October 2026 (checked
   against `gh issue list`)", yet issues #107 (fresh-checkout reproduction), #108
   (presentation machine), #109 (demo set provenance), #110 (model card),
   #111 (rehearsal, fallback, release tag) and #112 (dataset mirror URL) — all
   created 2026-10-26 and all still **OPEN** — appear nowhere in it. These are
   exactly the five "Next actions" the same file lists in prose. 54 issues are
   open in total; the table names about a dozen. **[verified]**
3. **Two untracked documents reintroduce the competing-tracker problem the
   project already rejected (High).** PR #113 (merged 27 Oct) reviewed
   `INCREMENTAL_UPGRADES_AND_ROBUSTNESS_HARDENING.md`, migrated the handful of
   genuinely missing items, and recorded the decision not to add it to the
   repository because "its proposed file names and 'update checkmarks in this
   file' rule would reintroduce a competing tracker". The file is now sitting in
   `docs/research/` as an unlabeled 689-line "Implementation-ready execution
   plan" with a Phase 0–6 schedule for 26 October–4 November — a third timeline
   alongside G0–G5 and the TODO milestones, all of it expired. **[verified]**
4. **The documentation health claim in `docs/README.md` is now incomplete
   (High).** It opens "Each active document below is the single authority for the
   subject named next to it", but it does not list the two new files, and it
   never mentions `configs/` or `reports/` — even though `pre-tuning-protocol.md`
   says "Where the two disagree, the YAML is authoritative" and
   `evaluation-protocol.md` says "the YAML and the freeze record define what was
   computed". The real authority graph has machine-readable nodes the index does
   not show. **[verified]**
5. **The evidence layer is sound and should be left alone (High, positive).**
   Fifteen independent cross-checks (table above) all passed, including re-hashing
   the frozen proposal and the freeze record. Documents that could not be verified
   say so themselves: `reproduction.md` carries a verification-status banner,
   `model-card.md` is marked DRAFT with five numbered open items, and
   `TODO-Plan.md` §13 lists the host-only numbers that cannot be checked from the
   repository. Step 2 should not touch any of this; the fixes needed are status
   fixes.

---

## B. Complete documentation inventory

`A` = active, `H` = historical/archive, `P` = execution prompt, `D` = planning
document, `E` = experimental evidence. Class numbers refer to section E's
classification scheme (1 core, 2 architecture, 3 research/methodology,
4 evaluation evidence, 5 execution plans/prompts, 6 presentation, 7 archive).

| # | Path | Lines | Last change | Kind | Class | Purpose and main contents | Audience | Relevance today |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `README.md` | 148 | 2026-10-27 | A | 1 | Public overview, status paragraph, core scope, repo layout, setup, config, evaluation safeguards, demo commands, deliverables, data notice | Newcomer, instructor | **Stale status**; everything else current |
| 2 | `TODO-Plan.md` | 390 | 2026-10-27 | A/D | 1 | Declared single status authority: 13 numbered sections of checkboxes (116 done, 30 open), issue status table, milestone table, next actions, final acceptance checklist, release sign-off record, dated decisions table, references | Owner, agents | **Authoritative but stale and incomplete** (finding 2) |
| 3 | `AGENTS.md` | 52 | 2026-10-04 | A | 1 | Binding repository conventions: scope priorities, uv/Python 3.12, code organization, data privacy, non-negotiable evaluation constraints, reproducibility records | Contributors, agents | Current; the only doc unchanged since 4 Oct and it has not needed changing |
| 4 | `docs/README.md` | 77 | 2026-10-27 | A | 1 | Documentation index; names the single authority per topic and the tie-break rule; archive caveat | All readers | Current but **incomplete** (finding 4) |
| 5 | `docs/roadmap.md` | 110 | 2026-10-27 | A/D | 1 | Milestone sequence only (explicitly keeps no checklist): two-phase framing, G0–G3 outcomes table, remaining G4/G5 sequence, not-on-critical-path list, deferred list, stop/go rules, definition of done | Owner | **Stale**: "Where the project stands (27 October 2026)"; its remaining sequence is in the past |
| 6 | `docs/architecture-and-delivery-plan.md` | 332 | 2026-10-27 | A/D | 2 | Addendum to the frozen proposal: decision summary, implementation state, module-boundary contract table, data/bundle/cache/run-record contracts, leakage controls, duration/robustness decisions, timeline envelope, proposal precedence and sign-off, **deviation register**, 13/14 Oct dated decisions, hosts, resolved items, S1 retirement | Owner, reviewer, instructor | Current; the deviation register is load-bearing |
| 7 | `docs/pipeline.md` | 74 | 2026-10-27 | A | 2 | What is implemented: offline Mermaid flow, serving Mermaid flow, response fields, endpoints, local packaging, CLI command inventory table | Developer | Current and accurate against `cli.py` |
| 8 | `docs/data.md` | 278 | 2026-10-27 | A | 3 | Dataset identity, source/release, ODC-By licence and attribution duties, integrity verification (checksums, protocol counts), local layout, exclusions and transformations, primary data contract and manifest schema, development reservation rule, G0 exit evidence, artifact policy, superseded 6 Oct history | Owner, reviewer | Current; one open item (mirror URL, #112) disclosed inline |
| 9 | `docs/environment.md` | 217 | **2026-11-03** | A | 2 | Which host produced which measurement, unresolved presentation machine, DL120 spec and capacity, second host `Linux VM` with the live-REPORTED-record warning, superseded planning-machine history, local storage, compatibility gate | Owner, agents on hosts | Current; **most recently maintained document** |
| 10 | `docs/reproduction.md` | 301 | 2026-10-27 | A | 3 | Fresh-checkout command sequence in 12 sections (environment → manifest → reservation → features → LR → CNN → freeze/score/report → serving → Docker → benchmarks → demo → archive), plus temporary-audio and reproducibility-record notes | Reproducer | Current, with an explicit "not re-run to write this guide" banner |
| 11 | `docs/pre-tuning-protocol.md` | 92 | 2026-10-27 | A/E | 4 | The 14 Oct pre-declaration: what is machine-checked and against which code constant, the declaration table, stated limitations, deviation rule, 27 Oct annotation | Reviewer, instructor | Current; immutable by design |
| 12 | `docs/evaluation-protocol.md` | 187 | 2026-10-27 | A/E | 4 | Locked-evaluation methodology written before R7: what R6/R7 do, integrity expectations, primary endpoint, fixed reading order, duration-slice caveat, interval coverage, deployment-decision clause plus its 27 Oct amendment, post-R7 rules, failure handling, known pre-R7 deviations | Reviewer, instructor | Current; immutable by design |
| 13 | `docs/evaluation-results-g2.md` | 258 | 2026-10-27 | A/E | 4 | The locked REPORTED results: provenance table, primary endpoint, known-algorithm and per-attack tables, pooled table, robustness matrices (EER/AUC/HFPR/SDR), dev-vs-eval table, duration caveat, 7 findings, deviations, RQ1–RQ3 mapping, open items | Reviewer, instructor | Current; **the results authority**; immutable |
| 14 | `docs/deployed-model-decision.md` | 89 | 2026-10-27 | A/E | 4 | Why `cnn_augmented_seed1` is deployed: non-blind-selection disclosure, the rule, development input tables, outcome, five caveats, what is not decided, 27 Oct update | Reviewer, instructor | Current; carries the most important honesty caveat |
| 15 | `docs/model-card.md` | 207 | 2026-10-27 | A/E | 4 | Model card, **marked DRAFT** with 5 numbered blockers: what it is, intended/not-intended use, data table, measured performance, known failures (#83/#84/#103), serving and resource measurements, provenance, ethics | Instructor, any user | Current; draft status is honest and accurate |
| 16 | `docs/course-coverage.md` | 67 | 2026-10-27 | A | 3 | Syllabus-topic → project-evidence map with Implemented / Partial / Design only / Not implemented / Out of scope, claim boundaries, deferred coverage | Instructor | Current; conservative and matches the code |
| 17 | `docs/research-backlog.md` | 145 | 2026-10-27 | A/D | 3 | Phase 2 scope authority: Phase 1/2 decision table, what Phase 1 must preserve, **Phase 1 pull gate**, 14-item prioritized backlog, explicitly-not-research-track, publication framing | Owner | Current except the pull gate, which is now moot |
| 18 | `docs/presentation-outline.md` | 262 | **2026-11-03** | A | 6 | Speaker notes and timings for the 12 slides, backup materials; states the HTML wins on conflict | Owner | Current; synced with the deck |
| 19 | `docs/presentation.html` | 976 | **2026-11-03** | A | 6 | The delivered 12-slide deck (self-contained HTML) | Audience | Current; numbers match the results doc |
| 20 | `docs/AI_Voice_Deepfake_Detector_Final_Project_Proposal.docx` | — (49,638 B) | frozen | A | 1 | The frozen, immutable proposal: scope, commitments, acceptance, RQ1–RQ3 | Instructor, owner | **Immutable source of truth**; SHA-256 re-verified |
| 21 | `docs/archive/architecture-review-2026-10-13.md` | 221 | 2026-10-27 | H | 7 | The 13 Oct senior-architect review that produced the two-phase plan: verdict, keep list, "design is not evidence", under-impress risks, plan findings, re-baselined schedule, changes made, open owner inputs, two-phase addendum | Provenance | Historical; correctly archived |
| 22 | `docs/archive/dl120-verification-tracker.md` | 251 | 2026-10-27 | H | 7 | Host runbook and log for the 5-candidate freeze, locked scoring and report; setup blocks, step tracker, R7 runbook, failure handling, log | Provenance | Historical; last updated 2026-10-17 |
| 23 | `docs/archive/lr-baseline-review-2026-10-14.md` | 86 | 2026-10-27 | H | 7 | The LR-baseline slice review: findings/tracking table, handoff decisions checked, verification limits, milestone implications | Provenance | Historical |
| 24 | `docs/research/INCREMENTAL_UPGRADES_AND_ROBUSTNESS_HARDENING.md` | 689 | **untracked** | D/P | 5 | Externally-prepared "implementation-ready execution plan" dated 26 Oct: executive decision, P0/P1/P2 scope rules, high-level and low-level design contracts, end-to-end pipeline, **Phase 0–6 execution plan for 26 Oct–4 Nov**, acceptance criteria, prescribed file names | Owner / an agent | **Expired and unlabeled**; see findings 3 and section C.3 |
| 25 | `docs/prompts/PHASE2_READINESS_AND_FOUNDATION_PROMPT.md` | 503 | **untracked** | P | 5 | Agent task prompt to prepare the repo for Phase 2: operating rules, source-of-truth audit, Phase 1 closure (naming #107–#112), Phase 2 planning, multi-dataset ingestion design, evaluation identity, experiment tracking, integrity follow-ups, first experiment, Gates 0–3, testing requirements, documentation decisions, execution boundaries, deliverables | An agent | **Active intent, not yet acted on**; see section C.4 |
| 26 | `app/README.md` | 8 | 2026-10-27 | A | 1 | Explains the directory is intentionally empty and points to `serving/` and `ui.py` | Developer | Current |
| 27 | `artifacts/README.md` | 5 | — | A | 1 | What belongs in the ignored artifacts directory | Developer | Current |
| 28 | `data/README.md` | 5 | — | A | 1 | What belongs in the ignored data directory; no third-party audio | Developer | Current |
| 29 | `notebooks/README.md` | 3 | — | A | 1 | Notebooks import package code | Developer | Current (directory is empty) |
| 30 | `reports/README.md` | 3 | — | A | 1 | Commit small summaries here, not large artifacts | Developer | Current |

**Committed machine-readable records that documents cite as authority** (not
prose, but part of the authority graph and omitted from `docs/README.md`):
`reports/evaluation_freeze_g2.json` (FROZEN, 11,457 B),
`reports/dev_reservation_frozen.json` (7,503 B), and `configs/*.yaml`
(11 files, including `evaluation_protocol.yaml`, `pre_tuning_protocol.yaml`,
`dev_reservation.yaml`, `robustness_spec.yaml`, `serving.yaml`,
`deployed_model_summary.yaml`).

**Filename-vs-content note (the brief asked not to trust filenames):** two names
mislead. `docs/evaluation-results-g2.md` is named for milestone G2 but contains
the **final locked G3 results** and is the project's results authority.
`docs/archive/dl120-verification-tracker.md` is named for the DL120 host but the
runs it tracks mostly executed on the *second* host, `Linux VM`.

---

## C. Duplication and overlap analysis

I checked each pair named in the brief for **actual** duplication — the same fact
asserted independently in two places, where one can drift from the other — rather
than related subject matter.

### C.1 `TODO-Plan.md` vs `docs/roadmap.md` — **no duplication; keep both**

Already resolved by explicit delegation, and the delegation is stated in both
directions. `roadmap.md` opens: "Sequence and milestones only. Whether an
individual task is done is recorded in `../TODO-Plan.md`, the single status
authority; this file deliberately keeps no checklist of its own."
`TODO-Plan.md` reciprocates: "**Roadmap:** `docs/roadmap.md` holds the milestone
sequence, the remaining execution order and the stop/go rules."

The one real overlap is **milestone dates**, which appear in three places:
`TODO-Plan.md`'s milestone table (G0 … presentation), `roadmap.md`'s G0–G3
outcomes table plus remaining sequence, and the architecture plan's timeline
table. **[verified]** The three agree today (G0 11 Oct, G1 8–12, G2 13–18, G3
19–25, freeze 26, G4 27 Oct–1 Nov, G5 2–4 Nov, presentation 5 Nov), so this is a
latent drift risk rather than a live conflict. Authoritative: **`roadmap.md`** for
sequence, **`TODO-Plan.md`** for per-item completion. Recommendation: have the
architecture plan's timeline table point at the roadmap instead of restating
dates.

### C.2 `architecture-and-delivery-plan.md` vs the architecture review — **no duplication; keep both**

`docs/archive/architecture-review-2026-10-13.md` is a dated review document; the
decisions it produced were migrated into the architecture plan's "Dated decisions
— 13 October 2026" section, and the plan links to the review as "(historical)".
The plan's own header records that stale content (A0–A9 task list, the S1
contract, the 6 Oct handoff, consultation narrative) was deliberately removed and
left in Git history. `docs/README.md`'s archive note tells readers not to take
current status from the archive. This is correct archival practice. Authoritative:
**the architecture plan**.

Same verdict for `docs/archive/lr-baseline-review-2026-10-14.md` (its decisions
live in the plan's "Resolved since the 14 October decisions" and in
`TODO-Plan.md`'s decisions table) and for
`docs/archive/dl120-verification-tracker.md` (its results live in
`evaluation-results-g2.md`).

### C.3 `INCREMENTAL_UPGRADES_AND_ROBUSTNESS_HARDENING.md` vs the planning documents — **real duplication; the file should not become an authority**

This is the most consequential overlap in the repository. The untracked
`docs/research/` copy duplicates, with different wording and sometimes different
prescriptions:

| Content in the hardening plan | Already owned by | Conflict risk |
| --- | --- | --- |
| §2 P0/P1/P2 scope rules | `roadmap.md` stop/go rules + `research-backlog.md` | Two priority vocabularies |
| §3.3 "Recommended repository organization" | the actual tree + `architecture-and-delivery-plan.md` module table | Prescribes a different layout |
| §4.1–§4.11 design contracts (manifest, preprocessing, model, metrics, locked evaluation, robustness, serving, observability, SQLite) | `architecture-and-delivery-plan.md` + `pipeline.md` + `data.md` | Second contract text that can drift |
| §4.5 "commit `EVALUATION_PROTOCOL.md`" | `docs/evaluation-protocol.md` exists | Prescribes a duplicate under a new name |
| §4.6 primary evaluation + 2021 DF transfer | `evaluation-protocol.md`, `research-backlog.md` item 1 | — |
| §5.3 non-negotiable invariants | `AGENTS.md` "Non-negotiable evaluation constraints" | Two copies of binding rules |
| §6 Phase 0–6 execution plan, 26 Oct–4 Nov, with deliverables and exit gates | `roadmap.md` G4/G5 + `TODO-Plan.md` | **A third expired timeline** |
| §6 "`reports/phase_inventory.md`" with implemented/needs-hardening/missing status | `TODO-Plan.md` | A second status tracker |
| AC-table acceptance criteria | `TODO-Plan.md` final acceptance checklist | A second acceptance list |

**[verified]** The hardening plan references eight paths that do not exist and
would duplicate existing documents if created: `DATASETS.md`,
`EVALUATION_PROTOCOL.md`, `MODEL_CARD.md`, `REPRODUCIBILITY.md`,
`reports/data_audit.md`, `reports/phase_inventory.md`,
`reports/evaluation/model_comparison.csv`, `reports/evaluation/per_attack.csv`.

**[verified]** The plan itself concedes its boundary: "The supplied GitHub `docs/`
URL could not be retrieved in this review environment. Therefore, items below are
**required verification points**, not claims that a particular implementation is
absent."

**[verified]** PR #113 ("docs: integrate missing items from the hardening plan",
merged 27 Oct) already did the reconciliation: it migrated the release sign-off
record and waiver rule, the release hygiene sweep, "rehearse at least twice",
the bootstrap/train-only test item, the roadmap sign-off pointer, and three
`research-backlog.md` additions — and recorded that the plan is "**not** archived
or added to the repo" for exactly the reasons above.

**Recommendation:** the hardening plan is **historical input, already consumed**.
Keep it out of the authority graph. Either leave it untracked where it is, or — if
it must be retained in-repo for provenance — place it under `docs/archive/` with a
header that states it is a superseded external input, that PR #113 reconciled it,
and that its file names and checklists are not to be created. Do **not** promote
any of its prescribed filenames.

### C.4 `PHASE2_READINESS_AND_FOUNDATION_PROMPT.md` vs project execution documentation — **not duplication; it is an un-filed work order**

This one is different in kind. It is an *instruction to an agent*, not a
description of the project, and it is internally well-behaved: its §11 explicitly
says "Keep the current documentation ownership model", lists the existing
lower-case documents as the authorities, and says "Do not create generic
duplicates such as `DATASETS.md`, `EVALUATION_PROTOCOL.md`,
`REPRODUCIBILITY.md`, or `MODEL_CARD.md`", and "The hardening plan itself need not
be copied into `docs/` … If it is preserved, label it as historical input, not an
active status authority." Its §3 also already identifies the staleness this audit
found independently: "The target presentation date was 5 November 2026, which has
now passed. Reconcile the actual outcome; do not assume the presentation or
release did or did not happen."

So it does not conflict with the documentation model — it **endorses** it. The
problem is placement and status: a 503-line execution prompt sitting untracked in
`docs/` alongside the authorities, with no header saying it is a pending work
order rather than a description of the system. It also contains forward-looking
design decisions (multi-dataset manifest identity, dataset-aware freeze records,
JSON-vs-SQLite run-record ownership, Gates 0–3) that **are not yet reflected in
`research-backlog.md`** — so right now it is the only place some Phase 2 design
intent is written down, which is itself a single-point-of-failure.

**Recommendation:** treat it as the **Step 2 input**, not as documentation. If it
stays in the repo, it belongs in a clearly-labelled prompts location outside the
authority index, or its durable content (Gates 0–3, the architecture decisions)
should be migrated into `docs/research-backlog.md` and the architecture plan — the
same migrate-then-retire treatment PR #113 gave the hardening plan.

### C.5 `evaluation-protocol.md` vs `pre-tuning-protocol.md` — **no duplication; keep both**

These cover different, non-overlapping decision points and both are dated
pre-declarations that must not be rewritten:

- `pre-tuning-protocol.md` (declared **14 Oct**, before any development score
  existed): the *development* protocol — selection partition and metric,
  tie-break, window→clip aggregation, threshold rule, and the LR tuning budget.
- `evaluation-protocol.md` (committed **before R7**): the *locked evaluation*
  protocol — what R6/R7 do, integrity expectations, primary endpoint, reading
  order, interval coverage, post-REPORTED rules, failure handling.

The only shared facts are score direction, decision rule and the 5%-FPR threshold
rule, and both documents delegate to the same code constants and YAML rather than
restating them ("where the two differ, the YAML and the freeze record define what
was computed"). Authoritative: **`pre-tuning-protocol.md`** for development
selection/thresholds, **`evaluation-protocol.md`** for the locked evaluation.

### C.6 `evaluation-results-g2.md` vs other reported results — **controlled replication; keep the one-way copy rule**

The locked numbers appear in five places. This *is* replication, but it is
deliberate, one-directional and partly test-enforced:

| Location | Role | Guard |
| --- | --- | --- |
| `docs/evaluation-results-g2.md` | **authority** | immutable; REPORTED |
| `docs/model-card.md` | subset, for the deployed model | header: "Every number below is copied from a committed source … Nothing here is a new measurement" |
| `docs/deployed-model-decision.md` | development inputs + the one stated evaluation outcome | explicitly "stated, not used" |
| `configs/deployed_model_summary.yaml` | the numbers the Streamlit UI shows | comment: "Values are copied from that document, never recomputed. `tests/test_ui.py` checks that each number appears in the source row" |
| `docs/presentation.html` + `presentation-outline.md` | deck figures | TODO-Plan §13 records a 27 Oct number sweep that found and fixed three deck errors |

**[verified]** I re-checked all four copies against the authority and found **no
discrepancies**. The replication is acceptable because the copy direction is
declared and one copy is test-pinned. Recommendation: keep it, and keep the
`deployed_model_summary.yaml` test pattern; do not add a sixth copy.

### C.7 `research-backlog.md` vs future roadmap items — **minor overlap, one stale mechanism**

`roadmap.md`'s "Deferred beyond Phase 1" paragraph and the architecture plan's
deviation register both list the same deferred items as `research-backlog.md`
(2021 DF, In-the-Wild/WaveFake, transformers, LSTM, #38/#39, calibration, cloud,
RAG/MCP). The three serve different purposes — pointer, acceptance-impact record,
prioritized research scope — and all three point at the backlog, so this is
acceptable pointer duplication.

The real issue is the **pull gate** (`research-backlog.md` §"Pulling backlog items
into Phase 1"), referenced from `TODO-Plan.md` and `roadmap.md`. Its conditions
are "finishes before the 26 October feature freeze" and "declared before the G2
freeze (forecast 22 Oct)" — both dates are in the past, so the gate can no longer
admit anything, but it is still written in the present tense as a live mechanism
with a ranked candidate table. **[verified]** Authoritative:
**`research-backlog.md`** for Phase 2 scope; the gate section needs closing.

---

## D. Consistency issues

Grouped by the brief's categories. Severity in brackets.

### D.1 Conflicting model metrics — **none found**

**[verified]** Zero conflicts. Every metric I cross-checked between
`evaluation-results-g2.md`, `model-card.md`, `deployed-model-decision.md`,
`configs/deployed_model_summary.yaml` and `docs/presentation.html` agreed, down to
confusion-matrix counts (6,825 / 530 / 8,448 / 55,434) and interval bounds
([10.28, 13.04]). The deck's dataset figures (121,461 / 12,483 / 108,978 / 107)
reconcile exactly with `data.md`'s per-partition table. The deck omits the pooled
threshold-bound metrics rather than restating them, which is the safer choice.

### D.2 Dataset descriptions and split definitions — **consistent**

**[verified]** `data.md`, `model-card.md` §3, `evaluation-protocol.md` §2 and the
deck all use the same partition table (train 2,580/22,800/A01–A06/20 speakers;
dev 2,548/22,296/A01–A06/20; eval 7,355/63,882/A07–A19/67). The A16/A19
known-algorithm correction (13 Oct) is applied uniformly in `data.md`,
`evaluation-protocol.md` §4, `evaluation-results-g2.md` §2 and `model-card.md` §3.
The 838 extra FLAC files in the LA directory are disclosed in `data.md` and match
the archived tracker's note and open issue #77.

### D.3 Outdated model architecture descriptions — **one clarity problem** [Low]

**[verified]** The architecture prose is correct: 3 conv blocks (16/32/64, 3×3,
batch norm, ReLU), max-pool after blocks 1 and 2 only, adaptive average pool,
linear head to 2 classes, 23,650 parameters. All of it matches `CompactMelCNN`.

The problem is provenance, not accuracy: the **deployed** architecture is defined
in `src/voice_detector/cnn_pilot.py`, a module whose own docstring says "Minimal
log-Mel CNN for the timing pilot … Deliberately compact to get a lower bound on
per-step cost; the full study may widen." `model-card.md` §1 and
`course-coverage.md` both cite `cnn_pilot.py` as the source for the production
model's parameter count. Open issue #72 already names "CompactMelCNN location" as
a follow-up. A reader auditing the deployed model is sent to a file labelled as a
pilot. Documentation-only fix: say explicitly that the study reused the pilot's
architecture unchanged.

Minor: `model-card.md` §1 gives the input as 64-bin log-Mel over 4-second windows
and the deck shows `1×64×401`, while `CompactMelCNN.__init__` defaults to
`n_frames=399`. **[verified]** Immaterial — the adaptive pool makes the frame
count irrelevant to parameters — but the two numbers differ.

### D.4 Inconsistent preprocessing specifications — **none found; one missing pointer** [Low]

**[verified]** `mfcc-v2` (option D windowing) and `mel-v1` are described
identically in `architecture-and-delivery-plan.md`, `pipeline.md`,
`model-card.md`, `reproduction.md` and the deck, and `mel-v1`'s values match
`configs/cnn_study.yaml`. The augmentation recipe in `model-card.md` §1 (nested
0.5 probabilities → ~25% of clips per transform per epoch, ~62.5% left clean;
Opus 24 kbps; AWGN 10–30 dB SNR) matches `configs/robustness_spec.yaml`
`training_augmentation` exactly — including the correction recorded in the 27 Oct
number sweep.

Missing pointer: `model-card.md`'s augmentation row cites
`configs/cnn_study.yaml` (which holds the seeds) but the augmentation parameters
actually live in `configs/robustness_spec.yaml`. The card never names that file,
so a reader cannot find the spec it is describing.

### D.5 Conflicting project milestones — **three agreeing copies, all expired** [Critical]

**[verified]** No disagreement between the three milestone tables (see C.1), but
all three are expired, and nothing in the repository records what happened after
3 November. The owner confirms (2026-11-06) that the **presentation was delivered
on 5 November 2026** and the **release was not tagged**; the "Reality" column
below reflects that.

| Document | Says | Reality on 2026-11-06 |
| --- | --- | --- |
| `README.md` status | "Phase 1 … is in its verification and rehearsal stage"; "tentative presentation 5 November 2026" | **Presentation delivered 5 Nov** (owner); the stage description is now wrong, and "tentative" no longer applies |
| `roadmap.md` | "Where the project stands (27 October 2026)"; G4 "to 1 November"; G5 "2 to 4 November"; "Release (by 4 November)"; "Presentation (5 November, tentative)" | Presentation step **done**; "Release (by 4 November)" **not done** — no tag |
| `TODO-Plan.md` milestones | G4 "In progress"; G5 "Pending"; presentation "Pending" | Presentation row is **wrong** (it happened); G4/G5 completion per item still unrecorded |
| `TODO-Plan.md` sign-off record | 7 of 13 fields `pending`, including release commit/tag, test result, demo result, unresolved items, sign-off date | `git tag -l` empty — **no tag exists**, confirmed by the owner, so these stay pending until the release is tagged or the line is waived |
| `architecture-and-delivery-plan.md` timeline | same dates | expired |

The last substantive commits are dated **2026-11-03** (`72603f1` slide-10
invalid-input fix and `demo_setup.sh` rebuild skip; `d900088` the host
`git pull` warning), i.e. work consistent with G4/G5 preparation continuing past
1 November — but no document records that.

### D.6 Outdated implementation status — **one serious gap, several small ones**

**(a) The single tracker omits the G4/G5 issues [Critical].** **[verified] via
`gh issue list`.** `TODO-Plan.md`'s table is headed "Status of the issues behind
this plan, as of 27 October 2026 (checked against `gh issue list`)" and lists
#4–#59, #68, #83, #84, #89, #103 and the parked #15–#29. It does not contain:

| Issue | Created | State | Title |
| --- | --- | --- | --- |
| #107 | 2026-10-26 | OPEN | G4: fresh-checkout reproduction of data prep, baseline, CNN and report |
| #108 | 2026-10-26 | OPEN | G4: decide and record the presentation machine; verify container + venue path |
| #109 | 2026-10-26 | OPEN | G4: record the demo set (consented own voice + synthetic clips) with tool/version/terms |
| #110 | 2026-10-26 | OPEN | G4: finalize the model card or keep it DRAFT |
| #111 | 2026-10-26 | OPEN | G5: rehearsal, offline fallback, backup recording and release tag |
| #112 | 2026-10-26 | OPEN | Record the exact download URL of the ASVspoof `dataset.zip` mirror |
| #118 | 2026-10-26 | OPEN | `reserve-development`: audit trail for `--force` overrides |

These are precisely the five items `TODO-Plan.md` lists in prose as "Next actions
(27 October 2026)" and that `roadmap.md` lists as the remaining sequence — so the
*work* is tracked, but the issue↔section mapping the table exists to provide is
missing for all of it. A further 21 open issues (#43–#46, #53, #55, #58, #59,
#63, #64, #66, #67, #70–#79) appear nowhere in the table either; the plan does say
"sections are not mirrored issue-for-issue", so that part is by design.

**(b) The 3 November work is not reflected [Medium].** **[verified]**
`TODO-Plan.md` §13 records that the deck "was synchronized with the outline on
2026-10-27" and the number sweep's three fixes, but not the 3 November slide-10
correction (PR #122/#123: the demo beat called for an MP3 upload the Streamlit
uploader cannot accept). `docs/presentation-outline.md` and
`docs/presentation.html` *were* corrected and agree with each other. So the
tracker is behind its own artifacts.

**(c) Self-disclosed status repairs are handled well [positive].**
`TODO-Plan.md` §5 explicitly records "The earlier version of this item was ticked
in error" for the SQLite `runs` table, and §3/§8 annotate open items with
"Checked 2026-10-27: …". I verified both negative claims (no `runs` table, no
PCA/K-means code) and both are accurate. This is the right pattern.

**(d) Presentation machine still unresolved [High].** **[verified]** The
conflict is documented rather than hidden — `environment.md` has a dedicated
"**Presentation machine: unresolved**" paragraph naming the three-way
disagreement (13 Oct decision names the DL120; all container/latency/parity
verification ran on `Linux VM`; `scripts/demo_setup.sh` describes the
presentation Mac as an M4 Pro ARM64, which would run the `linux/amd64` image under
emulation). The architecture plan's "Hosts used" section repeats the same
unresolved state. It is now past the date by which `roadmap.md` said to resolve it
(G4, 1 November). Open as #108.

**(e) Host naming is inconsistent across records [Low].** **[verified]** The same
machine is "Second host: `Linux VM`" in `environment.md`, "Linux VM Host" in
PR #121's title, and "Linux VM" in PR #124's title — both PRs changed the same
warning block. Nothing in the documents defines "Linux VM".

### D.7 Broken relative Markdown links — **none** [verified]

A script resolved every relative Markdown link in all 27 `.md` files (excluding
`http(s)`, `mailto`, bare anchors): **0 broken**. A second script validated every
cross-document heading anchor (e.g.
`research-backlog.md#pulling-backlog-items-into-phase-1-progress-gate`):
**0 broken**.

### D.8 References to files that no longer exist

**[verified]** A script extracted every backticked path-like reference from all
`.md` and `.html` files and resolved it. Results, by category:

| Category | Verdict |
| --- | --- |
| Bare module/script basenames (`manifest.py`, `docker_smoke.sh`, `cnn_study.yaml`, …) | All exist under `src/voice_detector/`, `scripts/` or `configs/`. Not broken — contextual shorthand. |
| Git-ignored generated artifacts (`artifacts/manifest/manifest.csv`, `manifest.sqlite`, `cnn_norm_stats.pt`, `reports/latency_benchmark.json`, `lr_run_record.json`, `freeze_g2_SCORED.json`) | Absent by policy (`AGENTS.md`, `artifacts/README.md`). `TODO-Plan.md` §13 already flags `reports/latency_benchmark.json` as uncommitted and its numbers as host-only. Not defects. |
| Dataset-internal files (`ASVspoof2019.LA.cm.*.txt`, `README.LA.txt`, `LICENSE_text.txt`) | Inside the un-redistributable dataset. Correct. |
| `docs/AI_Voice_Deepfake_Detector_Final_Project_Proposal_Revised.docx` | **Does not exist and was never tracked in Git** — confirmed with `git log --all`. See D.9. |
| `proposal-review-and-course-coverage.md` | Deliberately removed; `course-coverage.md` says "This file replaces" it and "The full review is in Git history". Correct. |
| 8 paths prescribed only by the untracked hardening plan (`DATASETS.md`, `EVALUATION_PROTOCOL.md`, `MODEL_CARD.md`, `REPRODUCIBILITY.md`, `reports/data_audit.md`, `reports/phase_inventory.md`, `reports/evaluation/*.csv`) | Do not exist and **should not be created** (section C.3). |

### D.9 Internal contradiction about the retired proposal candidate [Low]

**[verified]** Inside `docs/architecture-and-delivery-plan.md`, the sign-off
section (≈line 133) says the Revised candidate "is a non-selected historical
candidate with no governing authority. **Both `.docx` files remain unchanged**",
referring to it by the in-repo path `docs/AI_Voice_Deepfake_Detector_Final_Project_Proposal_Revised.docx`;
eight lines later, the "Revised-candidate retirement — 2026-10-11" section says it
"was safely dropped from the repository on 2026-10-11 after its unique details
were synced into the planning docs". Both statements are in the same document, and
the second is the true one — the file is absent and `git log --all` shows it was
never committed. `TODO-Plan.md`'s "How to use this plan" repeats the first framing
("`_Revised.docx` is a non-selected historical candidate") without saying it is
not in the repository. PR #113's own validation note acknowledges "one intentional
mention of the retired `_Revised.docx`", so this is known — but as written, two
sentences in one document disagree about whether a file is present.

### D.10 The documentation index no longer covers `docs/` [High]

**[verified]** `docs/README.md` asserts "Each active document below is the single
authority for the subject named next to it", but:

- it does not mention `docs/prompts/` or `docs/research/` (both untracked, both
  present in the working tree since 6 November);
- it does not mention `reports/` or `configs/`, even though
  `pre-tuning-protocol.md` cedes authority to `configs/pre_tuning_protocol.yaml`
  ("Where the two disagree, the YAML is authoritative, because the code reads the
  YAML") and `evaluation-protocol.md` cedes it to
  `configs/evaluation_protocol.yaml` and `reports/evaluation_freeze_g2.json`
  ("where the two differ, the YAML and the freeze record define what was
  computed").

So the index describes a strictly smaller authority graph than the one the
documents themselves establish.

### D.11 Expired mechanisms still written in the present tense [Medium]

**[verified]** Three mechanisms read as live but can no longer fire:

1. The Phase 1 **pull gate** (`research-backlog.md`), gated on the 22 and 26
   October freezes — referenced as live from both `TODO-Plan.md` and
   `roadmap.md`.
2. `roadmap.md`'s **"Not on the critical path"** cut list, which instructs "Done
   only if time remains after step 1" — step 1 (fresh-checkout reproduction,
   #107) is still open and the time is gone.
3. `roadmap.md`'s **"If the remaining time cannot cover step 1, cut from the end
   of the …"** rule, which presumes remaining time before 5 November.

### D.12 Claims I could not verify from the repository [disclosed, not defects]

The documents themselves flag these, which is the correct behaviour; I record them
so Step 2 does not mistake them for verified facts. All are **[unverified]** here
because the evidence lives on the hosts, outside Git:

- Latency figures (p50 19.4 ms, p95 107.8 ms, p99 113.2 ms, peak RSS 1016 MiB),
  cold start 39 s, container memory 612/49 MiB, image size 4.82 GB, 91 packages
  — `reports/latency_benchmark.json` and the Docker logs are not committed
  (`TODO-Plan.md` §13 says exactly this).
- The locked report JSON itself (sha256 `c0116ce2…`), the REPORTED copy of the
  freeze record (sha256 `88f27bda…`), the predictions store, the per-clip
  predictions, and `archive_g2_20261020` (80 files). The committed freeze record
  is the **FROZEN** copy (`state: FROZEN`, `scored_utc: null`,
  `reported_utc: null`) — **[verified]** — which is consistent with every
  document's description.
- The development EER tables in `deployed-model-decision.md` (run records on the
  host), the threshold set's 574 bona fide clips and 4.88% achieved FPR.
- The four CNN training run records.
- Container parity (198 clips, max |Δ| 0.0, 0 flips).

---

## E. Source of truth recommendations

The existing model is sound. The table below is what I would publish after Step 2,
with the changes from today's `docs/README.md` marked **(change)**.

| Topic | Authoritative document | Notes |
| --- | --- | --- |
| **Scope, commitments, acceptance** | `docs/AI_Voice_Deepfake_Detector_Final_Project_Proposal.docx` | Frozen, immutable; SHA-256 `b16f5124…a09aec` re-verified in this audit. Deviations are recorded in the architecture plan, never by editing it. |
| **Project status and TODOs** | `TODO-Plan.md` | Keep as the only tracker. Needs the #107–#112/#118 rows and a post-3-November reconciliation **(change)**. Do not create a second tracker in Phase 2 — this is also the hardening plan's and the Phase 2 prompt's own instruction. |
| **Milestone sequence / execution order** | `docs/roadmap.md` | Keep. Needs a current "where the project stands" section and closure of the three expired mechanisms (D.11) **(change)**. Milestone dates should live only here, with the architecture plan pointing at it **(change)**. |
| **Architecture, design contracts, deviations** | `docs/architecture-and-delivery-plan.md` | Keep. The deviation register is the single place deferred proposal commitments are accounted for. Fix the internal `_Revised.docx` contradiction (D.9) **(change)**. |
| **Implemented pipeline and command inventory** | `docs/pipeline.md` | Keep; accurate against `cli.py`. |
| **Data, provenance, preprocessing contract** | `docs/data.md` (prose) + `configs/*.yaml` (machine-readable, authoritative on conflict) | The YAML precedence rule should be stated in the index **(change)**. |
| **Development selection, thresholds, tuning budget** | `docs/pre-tuning-protocol.md` + `configs/pre_tuning_protocol.yaml` (wins on conflict) | Immutable dated declaration; annotate, never rewrite. |
| **Locked evaluation methodology** | `docs/evaluation-protocol.md` + `configs/evaluation_protocol.yaml` + `reports/evaluation_freeze_g2.json` (win on conflict) | Immutable; the 27 Oct amendment is correctly an annotation. |
| **Experimental results** | `docs/evaluation-results-g2.md` | Immutable (REPORTED). All other copies are declared one-way derivatives; keep the `tests/test_ui.py` pin on `configs/deployed_model_summary.yaml`. Consider noting in the index that the filename says G2 but the content is the final locked result **(change)**. |
| **Deployment / model-selection decisions** | `docs/deployed-model-decision.md`, with `configs/serving.yaml` + `configs/deployed_model_summary.yaml` as the enforced pins | Keep. The non-blind-selection disclosure must survive every future edit. |
| **Deployed model's intended use, performance, limits** | `docs/model-card.md` | Keep DRAFT until its five listed blockers close (or record a waiver); #110. |
| **Environment, hosts, which host produced which number** | `docs/environment.md` | Keep. Resolve the presentation machine (#108) **(change)**. It carries the host-safety warning for the live REPORTED record — highest-value page in the set for anyone touching a host. |
| **Reproduction** | `docs/reproduction.md` | Keep, with its verification banner, until #107 runs. |
| **Syllabus coverage** | `docs/course-coverage.md` | Keep. |
| **Research extensions (Phase 2)** | `docs/research-backlog.md` | Keep as the Phase 2 scope authority. Close the pull gate and absorb the Gates 0–3 structure from the Phase 2 prompt rather than leaving it only in an untracked file **(change)**. |
| **Presentation** | `docs/presentation.html` (wins) + `docs/presentation-outline.md` | Keep; the precedence rule is already stated in the outline. |
| **Contributor and agent conventions** | `AGENTS.md` | Binding; unchanged since 4 Oct and still correct. |
| **Historical records** | `docs/archive/` | Keep, with the existing "not edited to match the present" caveat. |
| **External inputs already consumed** | *(no owner today)* | `INCREMENTAL_UPGRADES_AND_ROBUSTNESS_HARDENING.md` and `PHASE2_READINESS_AND_FOUNDATION_PROMPT.md` need an explicit status: superseded input, or pending work order. Neither should enter the authority table **(change)**. |

---

## F. Proposed documentation structure

**Recommendation: do not restructure.** The current layout is flat, indexed, and
its authority model is explicit and working. A reorganization would break the
cross-references in 20+ documents, the pinned paths in `configs/serving.yaml`
comments and `configs/deployed_model_summary.yaml` (`source_doc:
docs/evaluation-results-g2.md`, test-checked), and the numerous
`docs/evaluation-results-g2.md` citations inside the immutable protocol and
results documents — for no navigational gain at 23 documents. The audit's
findings are about *content freshness and file status*, not location.

The two minimal placement changes worth making are about the **two untracked
files**, not the existing tree:

```text
.
├── README.md                      overview, status, quick start
├── AGENTS.md                      binding conventions (unchanged)
├── TODO-Plan.md                   THE status tracker
├── docs/
│   ├── README.md                  index — extend to name configs/ and reports/
│   │                              as machine-readable authorities, and to list
│   │                              any retained input documents as non-authorities
│   ├── AI_Voice_Deepfake_Detector_Final_Project_Proposal.docx   frozen
│   ├── roadmap.md                 sequence + current standing
│   ├── architecture-and-delivery-plan.md   contracts + deviation register
│   ├── pipeline.md
│   ├── data.md
│   ├── environment.md
│   ├── reproduction.md
│   ├── pre-tuning-protocol.md     immutable declaration
│   ├── evaluation-protocol.md     immutable declaration
│   ├── evaluation-results-g2.md   immutable results
│   ├── deployed-model-decision.md
│   ├── model-card.md
│   ├── course-coverage.md
│   ├── research-backlog.md        Phase 2 scope (absorb Gates 0–3)
│   ├── presentation.html
│   ├── presentation-outline.md
│   └── archive/                   historical, never edited to match the present
│       ├── architecture-review-2026-10-13.md
│       ├── dl120-verification-tracker.md
│       ├── lr-baseline-review-2026-10-14.md
│       └── (option) incremental-upgrades-hardening-plan-2026-10-26.md
│              ← only if the hardening plan is retained in-repo at all; needs a
│                header: superseded external input, reconciled by PR #113, its
│                file names and checklists are not to be created
├── configs/                       machine-readable authorities (wins on conflict)
├── reports/                       committed frozen records + this audit
└── (outside docs/) the Phase 2 readiness prompt
       ← a work order, not documentation. Keep it untracked, or place it where the
         index can mark it "pending task, not an authority"; migrate its durable
         design content into research-backlog.md and the architecture plan.
```

Rejected alternatives, with reasons:

- **`docs/evaluation/`, `docs/architecture/`, `docs/operations/` subdirectories** —
  23 files do not need three levels, and it would break every committed
  cross-reference and the test-pinned `source_doc` path.
- **The hardening plan's §3.3 recommended tree** — it was written without being
  able to read this repository (the plan says so), and the Phase 2 prompt itself
  says "Do not redesign the repository tree solely to resemble the source plan's
  example."
- **Renaming `evaluation-results-g2.md` to reflect that it is the final result** —
  tempting (the name misleads, D/B), but it is cited by immutable documents and
  by a test-checked config field. Clarify in the index instead.
- **A `docs/status.md` snapshot** — the 27 October consolidation deliberately
  deleted one of these; re-adding it would recreate the contradiction that
  cleanup removed.

---

## G. Prioritized action items

All items are documentation-only unless marked otherwise. None requires touching
locked results, thresholds, predictions, freeze records or model artifacts.

### Critical

| # | Action | Affected files | Rationale |
| --- | --- | --- | --- |
| C1 | Record the established Phase 1 outcome — **presented 5 November 2026, release not tagged** (owner, 2026-11-06): rewrite the `README.md` status paragraph, replace `roadmap.md`'s "Where the project stands (27 October 2026)" with a post-presentation standing, mark the presentation milestone done in `TODO-Plan.md`, and establish from the owner, commits, issues and host artifacts which G4/G5 items (#107–#111) actually completed before the talk. | `README.md` (status paragraph), `docs/roadmap.md` ("Where the project stands", remaining sequence), `TODO-Plan.md` (milestone table + next actions) | Every status document still describes the presentation as pending when it has happened, which understates the project; and still implies a release path that produced no tag. This blocks every other status fix. (D.5) |
| C1b | Decide the release: tag it and fill in the sign-off record, or record a dated waiver for each unmet acceptance line. `TODO-Plan.md` already requires "an unmet line with no waiver blocks sign-off" — today 7 of 13 sign-off fields and 4 acceptance lines are unresolved with neither a tag nor a waiver. | `TODO-Plan.md` (release sign-off record, final acceptance checklist), issue #111 | With the presentation delivered and no tag, the repository has no identified release commit, no recorded test/demo result, and no sign-off — and the project's own rule says that state blocks sign-off rather than silently closing it. (D.5, D.6a) |
| C2 | Add #107, #108, #109, #110, #111, #112 and #118 to `TODO-Plan.md`'s issue table with their sections, and re-state the table's "checked against `gh issue list`" date. | `TODO-Plan.md` | The declared single tracker omits the entire remaining critical path, all created the day before its last reconciliation and all still open. (D.6a) |

### High

| # | Action | Affected files | Rationale |
| --- | --- | --- | --- |
| H1 | Decide the status of `docs/research/INCREMENTAL_UPGRADES_AND_ROBUSTNESS_HARDENING.md`: leave untracked, or archive it with a header stating it is a superseded external input reconciled by PR #113 whose prescribed file names and checklists must not be created. Either way, keep it out of `docs/README.md`'s authority table. | `docs/research/…`, `docs/README.md` | PR #113 explicitly decided against adding it because its file names and "update checkmarks in this file" rule would reintroduce a competing tracker; it is now in `docs/` unlabeled, with a third expired timeline. (C.3, finding 3) |
| H2 | Decide the status of `docs/prompts/PHASE2_READINESS_AND_FOUNDATION_PROMPT.md` and migrate its durable content (Gates 0–3 with prerequisites/exit criteria, the multi-dataset manifest/adapter and freeze-identity decisions, the JSON-vs-SQLite run-record decision) into `docs/research-backlog.md` and `docs/architecture-and-delivery-plan.md`. | `docs/prompts/…`, `docs/research-backlog.md`, `docs/architecture-and-delivery-plan.md` | It is a work order, not documentation, yet it is currently the only place parts of the Phase 2 design intent exist. (C.4) |
| H3 | Extend `docs/README.md`: name `configs/` and `reports/` as machine-readable authorities that win over prose (quoting the two existing precedence clauses), and give every retained non-authority document an explicit label. | `docs/README.md` | The index claims completeness but describes a smaller authority graph than the documents themselves establish, and omits two files now present in `docs/`. (D.10) |
| H4 | Resolve and record the presentation machine, or convert it into a recorded deviation with its consequences (ARM64 emulation of the `linux/amd64` image; container config verified only on `Linux VM`). | `docs/environment.md`, `docs/architecture-and-delivery-plan.md` ("Hosts used"), `TODO-Plan.md` §10, issue #108 | Three records disagree, the resolution deadline (G4, 1 Nov) has passed, and the offline-fallback plan assumes an Intel Mac. (D.6d) |

### Medium

| # | Action | Affected files | Rationale |
| --- | --- | --- | --- |
| M1 | Close the three expired mechanisms: mark the Phase 1 pull gate closed (it can no longer admit anything), and retire or rewrite the "not on the critical path" and "if the remaining time cannot cover step 1" rules. | `docs/research-backlog.md`, `docs/roadmap.md`, `TODO-Plan.md` (references) | They read as live mechanisms gated on dates in the past. The Phase 2 prompt independently asks for the pull gate to be closed. (D.11, C.7) |
| M2 | Record the 3 November deck/outline/`demo_setup.sh` corrections in `TODO-Plan.md` §13, which currently stops at the 27 October sync. | `TODO-Plan.md` | The tracker is behind its own artifacts. (D.6b) |
| M3 | Decide `model-card.md`'s disposition explicitly: close its five listed blockers or record a dated waiver keeping it DRAFT. | `docs/model-card.md`, `TODO-Plan.md`, issue #110 | Four of the five blockers (#89 review, #107 reproduction, #109 demo provenance, redistribution) are unresolved and past their G4 date; the card is the instructor-facing artifact. |
| M4 | Keep milestone dates in `docs/roadmap.md` only; replace the architecture plan's timeline table with a pointer. | `docs/architecture-and-delivery-plan.md`, `docs/roadmap.md` | Three independent copies of the same dates agree today but will drift at the first revision. (C.1) |

### Low

| # | Action | Affected files | Rationale |
| --- | --- | --- | --- |
| L1 | Fix the internal contradiction about `_Revised.docx`: the sign-off section says "Both `.docx` files remain unchanged" and cites an in-repo path; the retirement section eight lines later says the file was dropped. Say once that it is not in the repository and identified only by SHA-256. Align `TODO-Plan.md`'s "How to use this plan" wording. | `docs/architecture-and-delivery-plan.md`, `TODO-Plan.md` | Two sentences in one document disagree about whether a file is present; verified absent and never tracked. (D.9) |
| L2 | State that the deployed CNN reuses the timing pilot's architecture unchanged, and name `src/voice_detector/cnn_pilot.py` as the definition site deliberately (or move the class, per open issue #72, which is a code change). | `docs/model-card.md` §1, `docs/course-coverage.md`, `docs/architecture-and-delivery-plan.md` | A reader auditing the production model is sent to a file whose docstring calls it a pilot that "may widen". (D.3) |
| L3 | Add the `configs/robustness_spec.yaml` → `training_augmentation` pointer to the model card's augmentation row (which currently cites `configs/cnn_study.yaml`, where the probabilities do not live). | `docs/model-card.md` | The values are correct; the pointer is not. (D.4) |
| L4 | Note in `docs/README.md` that `evaluation-results-g2.md` is named for G2 but holds the final locked (G3) results, and that the archived DL120 tracker covers runs mostly executed on `Linux VM`. | `docs/README.md` | Two filenames mislead; renaming is worse than annotating because immutable documents and a test-checked config field cite the paths. (section B note) |
| L5 | Define the host nicknames once and use the canonical name. | `docs/environment.md` | The same machine has three names across records; only one is defined anywhere. (D.6e) |
| L6 | Reconcile the stated window frame count (`401` in the deck and card prose vs `n_frames=399` default in code). | `docs/model-card.md`, `docs/presentation.html` | Immaterial to the model but the two numbers differ. (D.3) |

### Explicitly not recommended

- **Do not** restructure `docs/` into subdirectories (section F).
- **Do not** create `DATASETS.md`, `EVALUATION_PROTOCOL.md`, `MODEL_CARD.md`,
  `REPRODUCIBILITY.md`, `reports/data_audit.md` or `reports/phase_inventory.md` —
  duplicates of existing documents, prescribed only by a superseded external input
  that forbade them being created in its own successor prompt.
- **Do not** edit the archive to match the present; `docs/README.md`'s caveat
  covers it.
- **Do not** edit `pre-tuning-protocol.md`, `evaluation-protocol.md` or
  `evaluation-results-g2.md` other than by dated annotation, and do not touch the
  committed freeze or reservation records.
- **Do not** add a sixth copy of the locked numbers, or a status-snapshot
  document.

---

## Appendix — mechanical checks performed

| Check | Command/method | Result |
| --- | --- | --- |
| Relative Markdown links | Python script over all 27 `.md` files, resolving each non-URL target | 0 broken |
| Heading anchors | Python script slugifying every heading and resolving every `…#anchor` link | 0 broken |
| Backticked file references | Python script extracting `` `path.ext` `` and resolving against repo root and doc directory | categorized in D.8 |
| Frozen proposal integrity | `shasum -a 256 docs/AI_…Proposal.docx` | `b16f5124…a09aec` — matches the pin |
| Freeze record integrity | `shasum -a 256 reports/evaluation_freeze_g2.json` + JSON inspection | `cfd1406c…cb4d2`, `state: FROZEN`, `scored_utc`/`reported_utc` null — matches every document |
| Threshold / bundle-hash pins | JSON vs `configs/serving.yaml` vs `model-card.md` vs `evaluation-results-g2.md` | all 5 thresholds and the deployed bundle hash match |
| Reservation record | `reports/dev_reservation_frozen.json` vs `data.md` | 15 `select` / 5 `threshold` speakers — matches |
| UI summary | `configs/deployed_model_summary.yaml` vs `evaluation-results-g2.md` | all values match |
| Deck figures | HTML text extraction vs `evaluation-results-g2.md`, `data.md` | all match |
| CNN parameters | hand count from `CompactMelCNN` layer definitions | 23,650 ≈ "~23.7 K" |
| Negative implementation claims | `grep` for PCA/KMeans/silhouette in `src/`,`tests/`; `grep CREATE TABLE src/` | both confirmed absent, as documented |
| Issue status | `gh issue list --state all --limit 200` (read-only) | 76 issues, 54 open; #107–#112, #118 open and absent from the tracker |
| Release tag | `git tag -l` | empty — no release tag (owner-confirmed, 2026-11-06) |
| Repository history | `git log`, `git log --all -- 'docs/*Revised*'` | last commit 2026-11-03; `_Revised.docx` never tracked |
