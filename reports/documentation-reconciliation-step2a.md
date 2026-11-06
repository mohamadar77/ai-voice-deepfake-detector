# Documentation reconciliation — Step 2A

**Date:** 2026-11-06
**Repository state:** branch `main` at `d900088` (unchanged since the Step 1 audit), working tree modified, nothing staged, nothing committed.
**Input:** [`documentation-audit-step1.md`](documentation-audit-step1.md) (Step 1 audit, read in full; left unchanged by this task apart from the pre-task amendment noted in §2).
**Nature of this task:** documentation only. No code, test, config, dataset, model, prediction or evaluation artifact was touched.

---

## 1. Executive summary

**What was reconciled.** The status layer of the documentation now matches the
evidence. Before this task, every status document was dated 27 October 2026 and
described the 5 November presentation as upcoming and "tentative"; the single task
tracker omitted the entire G4/G5 issue set; and three dated mechanisms (the Phase 1
pull gate, the roadmap cut list, the remaining-time rule) still read as live. All of
that is now corrected, with each claim attributed to its evidence or marked as
resting on the owner's statement.

**Current Phase 1 status: presented, not released.**

| Question | Answer | Evidence |
| --- | --- | --- |
| Was the presentation delivered? | **Yes, 5 November 2026** | Owner statement, recorded 2026-11-06. **No repository artifact corroborates it** — no tag, and no commit after 2026-11-03. This is stated as owner-sourced everywhere it now appears. |
| Was it signed off? | **No** | `TODO-Plan.md` release sign-off record: 7 of 13 fields `pending`. Four acceptance-checklist lines unmet, none waived. The file's own rule — "an unmet line with no waiver blocks sign-off" — is therefore unsatisfied. |
| Was it released? | **No** | `git tag -l` is empty. No release commit identified, no test or demo result recorded. |
| Was G4 delivered? | **No** | All five issues (#107–#110, #112) OPEN, `updatedAt` still 2026-10-26. GitHub milestone *G4: release and slides*: 5 open / 0 closed. |
| Was G5 delivered? | **Not recorded** | #111 OPEN, untouched since 2026-10-26. Milestone *G5*: 1 open / 0 closed. No rehearsal, backup recording or offline-fallback verification exists anywhere in the repository. |

The honest reading: **the talk was given with the entire verification and rehearsal
tier still open.** That is now recorded rather than implied, and it is a stronger
statement about the project than the documents previously made in either direction —
they understated the delivery and overstated the readiness.

---

## 2. Audit delta

The repository had not changed since the Step 1 audit (same `HEAD`, same untracked
set, same frozen hashes), so every Step 1 finding was re-checked rather than assumed.

### Findings that remained valid and were resolved by this task

| Step 1 finding | Status after Step 2A |
| --- | --- |
| D.5 / C1 — status layer past its own horizon | **Resolved.** `README.md`, `docs/roadmap.md`, `TODO-Plan.md` and the architecture plan header now record the presentation as delivered and the release as open. |
| D.6a / C2 — tracker missing #107–#112, #118 | **Resolved.** Seven rows added to the issue table with milestone and section mapping; the table's "as of" date and provenance line updated. |
| D.6b — 3 November deck corrections unrecorded | **Resolved.** `TODO-Plan.md` §13 now records PR #122/#123 and the delivery date. |
| D.9 — `_Revised.docx` internal contradiction | **Resolved.** Three passages in the architecture plan and one in `TODO-Plan.md` now say once, consistently, that the candidate is absent, was never committed, and is identified only by SHA-256. |
| D.10 — index does not cover its own authority graph | **Resolved.** `docs/README.md` gained a "Machine-readable authorities" section (9 records, each with the prose that defers to it) and a "Not authoritative" section. |
| D.11 — three expired mechanisms in the present tense | **Resolved.** Pull gate marked CLOSED with its reasoning; the cut list became "Optional, and never required for sign-off"; the remaining-time stop/go rule was replaced. |
| D.6d / A4 — presentation machine unresolved | **Reframed, not resolved.** It is no longer a decision (the talk happened) but a missing record of a past event. `docs/environment.md` now says so and states exactly what only the owner can supply. Still open as #108. |
| D.4 / L3 — model card augmentation pointer | **Resolved.** The row now names `configs/robustness_spec.yaml` → `training_augmentation`. |
| C.1 / M4 — milestone dates duplicated in three documents | **Resolved.** The architecture plan's dated gate table was removed and now points at the roadmap; the roadmap keeps the dates. The plan retains the resource envelope, which is genuinely its own. |
| M3 — model card disposition | **Decided conservatively.** Kept DRAFT, with a dated line explaining that none of its five blockers closed. Promoting it would assert evidence that does not exist. |

### Findings deliberately left unresolved

| Step 1 finding | Why it was not resolved here |
| --- | --- |
| C.3 / H1 — hardening plan status | The work order's C2 settles it: do not delete, do not stage, add a factual header if the file remains. Done (see §5). Its disposition as a *tracked* file remains the owner's call. |
| C.4 / H2 — Phase 2 prompt status | Its durable content was migrated (see §5); the prompt itself stays untracked. Whether to retain it in-repo is the owner's call. |
| L2 — `CompactMelCNN` lives in `cnn_pilot.py` | Open issue #72 proposes moving the class. That is a code change, out of scope here, and a documentation-only note would duplicate #72. |
| L4 — `evaluation-results-g2.md` filename says G2, content is final | Renaming would break immutable documents and the test-pinned `source_doc` field. Annotating the index was considered and dropped as noise; the Step 1 audit records it. |
| L5 — host nicknames ("High-performance Linux VM") | The inconsistency is in PR titles, not in the documents; `docs/environment.md` already uses the canonical `Linux VM` throughout. Nothing to fix in-repo. |
| L6 — 401 vs 399 frames | Immaterial (adaptive pooling), and touching the model card's input row risks disturbing a frozen description for no reader benefit. |
| D.12 — host-only numbers unverifiable from the repository | Unchanged and still disclosed. `TODO-Plan.md` §13 now additionally records that the pre-presentation confirmation never happened. |

### Change to the audit report itself

`reports/documentation-audit-step1.md` was amended **before** this task began, in the
same session, to record the owner's "presented, not tagged" answer (header line,
finding 1, the D.5 table, action C1, a new C1b, and the appendix row). It has not
been touched since Step 2A started, per the work order's instruction to keep it
unchanged. No finding was weakened or removed by that amendment.

---

## 3. G4/G5 reconciliation

Every row was checked with `gh issue view` (state, `updatedAt`, comments, milestone)
and against the repository. **No item could be marked complete.** Nothing was closed
on GitHub; this task only records status.

| Item | Current status | Evidence inspected | Documentation updated | Remaining blocker / owner decision |
| --- | --- | --- | --- | --- |
| **#107** fresh-checkout reproduction (G4) | OPEN, untouched since 2026-10-26, 0 comments | Issue state; `docs/reproduction.md`'s own banner ("the pipeline itself was **not** re-run to write this guide"); no run record in the tree | `TODO-Plan.md` issue table + §10 (already annotated "Not done; no record of any such run exists"); roadmap item 3 | Needs an actual run on a clean host. Largest unmet acceptance claim; blocks "Baseline and CNN comparisons are reproducible". |
| **#108** presentation machine (G4) | OPEN, 1 comment (2026-10-27 static finding on ARM64 wheel divergence) | Issue + comment; `docs/environment.md`; `scripts/demo_setup.sh` ARM64 assumption; recorded container parity on `Linux VM` | `docs/environment.md` paragraph rewritten from "unresolved decision" to "unrecorded event", naming what only the owner can supply; roadmap item 2 | **Owner must state** which machine presented, whether the container or the offline fallback ran, and whether the venue access path worked. |
| **#109** demo set provenance (G4) | OPEN, untouched, 0 comments | Issue; `scripts/generate_demo_clips.py` (synthetic/too-short/MP3/silence inputs exist); no record of human recordings, TTS tool, version or terms | `TODO-Plan.md` §13 demo item annotated (explicitly: do not tick it because the talk happened); model card blocker 3; roadmap item 2 | **Owner must record** the clips actually demonstrated, with tool, version and terms. |
| **#110** model card disposition (G4) | OPEN, untouched, 0 comments | Issue; `docs/model-card.md` five listed blockers, each re-checked — none closed | Model card status line now dated 2026-11-06 and explains why DRAFT stands; roadmap item 4 | **Owner decides:** close the blockers, or accept DRAFT in the release record. |
| **#111** rehearsal, fallback, backup, release tag (G5) | OPEN, untouched, 0 comments | Issue; `git tag -l` empty; `scripts/demo_backup.py` exists but "needs host run"; no rehearsal record anywhere | `TODO-Plan.md` milestone table, §13 (three items annotated), acceptance line, sign-off preamble; roadmap item 1 | **The gating owner decision.** Tag + complete the sign-off record, or waive each unmet line with reason, impact, owner and date. |
| **#112** dataset mirror URL (G4) | OPEN, untouched, 0 comments | Issue; `docs/data.md` (the open item is already stated inline) | `TODO-Plan.md` issue table (§2 mapping); roadmap item 5 | **Owner supplies** the URL. Checksums remain the authoritative anchor meanwhile. |
| **#118** `reserve-development` `--force` audit trail | OPEN, Phase 2 milestone | Issue; milestone assignment | `TODO-Plan.md` issue table, Phase 2 row | Not a Phase 1 blocker. Correctly outside the release decision. |

**Sign-off fields and acceptance lines.** 7 of 13 sign-off fields remain `pending`
(release commit/tag, test/CI command and result, demo startup command and result,
unresolved acceptance items, sign-off date, plus the manifest-hash confirmation at
release time). Four acceptance-checklist lines are unmet, verified by count: real
data documented and auditable; baseline/CNN comparisons reproducible; the
application runs and handles valid and invalid inputs; presentation and backup
demonstration ready. **None is waived.** By the file's own rule, sign-off is blocked.
No waiver was written on the owner's behalf.

---

## 4. Documentation changes

Eight tracked Markdown files, +319 / −105 lines. No non-Markdown file was modified.

| File | Lines | Purpose of the change |
| --- | --- | --- |
| `README.md` | +2 / −2 | Status paragraph: "verification and rehearsal stage, tentative presentation" → presented 5 Nov, presented-but-not-released, with the G4/G5 issues named and the owner-sourced provenance stated. Deliverables: presentation added to delivered; demo-set provenance and sign-off added to still-open. All metrics untouched. |
| `TODO-Plan.md` | +52 / −26 | Header: reconciliation date, "Presented: 5 November 2026" replacing "Target presentation". Issue table: 7 new rows (#107–#112 as Phase 1, #118 as Phase 2), new "as of 6 November" provenance line disclosing that the G4/G5 set was missing. Milestone table: G4 "not delivered", G5 "not recorded", presentation "delivered", plus a new release row. Next actions rewritten around the release decision. §13: six items annotated (deck delivery + 3 Nov corrections, demo beat, release artifacts, presentation-machine test, rehearsal, host-only numbers). Acceptance line and sign-off preamble annotated. Two decision rows added (5 Nov delivery, 6 Nov reconciliation), in chronological position. Pull-gate sentence rewritten as closed. `_Revised.docx` wording aligned. |
| `docs/roadmap.md` | +64 / −43 | "Where the project stands" re-dated to 6 November and rewritten as presented-not-released; milestone table gained G4, G5 and presentation rows with milestone-counter evidence; the dated "Remaining sequence" became "Remaining work to close Phase 1", led by the owner's release decision; "Not on the critical path" became "Optional, and never required for sign-off"; the expired remaining-time stop/go rule replaced with "nothing still owed may be closed by assertion"; definition of done gained a dated "not met" statement. |
| `docs/architecture-and-delivery-plan.md` | +64 / −18 | Header reconciled. The duplicated gate-date table removed in favour of a pointer to the roadmap, keeping the resource envelope. The `_Revised.docx` contradiction fixed in three passages (sign-off, precedence, retirement) with the `git log --all` verification recorded. New section "Proposed Phase 2 foundation contracts (not adopted)" — five proposals, explicitly labelled as needing owner approval (§5). |
| `docs/README.md` | +36 / −0 | Intro now states that machine-readable records beat prose where a document says so. New "Machine-readable authorities" table: 9 records under `configs/` and `reports/`, each paired with the prose that defers to it. New "Not authoritative" table: the hardening plan, `prompts/`, the two dated report files, and the owner's untracked working notes. |
| `docs/research-backlog.md` | +74 / −8 | Intro records the delivered presentation and the open release without making Phase 2 wait on it. Pull gate marked CLOSED with a blockquote giving the reason, and its ranking preserved as the Phase 2 starting order. New "Phase 2 research gates (proposed)" section: Gates 0–3 with prerequisites, work mapped to existing backlog item numbers, and exit criteria (§5). |
| `docs/environment.md` | +21 / −7 | "Presentation machine: unresolved" rewritten as "Which machine presented: unrecorded", reframed from a pending decision to a missing record, with the ARM64 wheel divergence from #108's comment added and an explicit statement that no claim about the demonstrated configuration can be made until the owner records it. |
| `docs/model-card.md` | +4 / −3 | Status line re-dated and explains why DRAFT stands after the presentation. Augmentation row now names `configs/robustness_spec.yaml` → `training_augmentation`. Blocker 3 notes the demo provenance is still unrecorded post-presentation. **No metric, hash, threshold or architecture value changed** (verified by grepping the diff for every headline number: 0 hits). |

**Untracked file modified in place (not staged, not committed):**
`docs/research/INCREMENTAL_UPGRADES_AND_ROBUSTNESS_HARDENING.md` — a header block
prepended per the work order's C2. The plan text below it is unchanged. Flagged here
because it is the owner's own untracked file: if the header is unwanted, deleting the
block restores the file exactly.

**Anchor-stability note.** Closing the pull gate changed its heading, which would
have broken the inbound link from `TODO-Plan.md`. The heading uses a colon rather
than an em dash (`… (progress gate): CLOSED`) because GitHub and the local slugifier
disagree on how `" — "` collapses; the inbound link was updated to the new slug and
the anchor checker re-run clean.

---

## 5. Phase 2 preparation

**No Phase 2 implementation was started.** No code, no dataset, no download, no
training, no scoring, no new configuration file.

### Durable decisions migrated, and where

| Decision | Destination | Recorded as |
| --- | --- | --- |
| Gates 0–3 (Foundation, Cross-domain generalization, Model and robustness research, Reliability and publication readiness), each with prerequisites, work mapped to existing backlog items, and exit criteria | `docs/research-backlog.md`, new section before the backlog list | **Proposed, not approved.** Explicitly stated: the prioritized backlog remains operative until the owner approves the gates. |
| Canonical multi-dataset identity and per-dataset ingestion adapters, with the constraint that the LA adapter must reproduce manifest hash `b932b859…` or the change is wrong | `docs/architecture-and-delivery-plan.md`, new "Proposed Phase 2 foundation contracts" | **Proposal.** Includes the open question of how far to generalize before a second dataset exists. |
| Dataset-aware freeze and prediction identity (dataset, protocol version and source-map provenance in the frozen identity and every prediction row) | same | **Proposal.** Notes the specific hazard: LA's null `source_id` and `speaker_id` fallback must not be silently applied to a dataset that has real source maps. |
| Run records: per-run JSON authoritative and immutable; any SQLite `runs` table is a rebuildable index, never a second source of truth | same | **Proposal.** Resolves the half-implemented 13 October rule (#54) by deciding the ownership question instead of finishing both halves blindly; mirrors the existing canonical-CSV / rebuildable-SQLite manifest pattern. |
| Artifact integrity and publication for the archived Phase 1 evidence | same | **Proposal.** Linked to the open Phase 1 item on release-artifact retrieval, which waits on a redistribution decision. |
| Regression tests as the acceptance condition for each contract above | same | **Proposal.** Four concrete failing-test requirements named. |

Each carries the same boundary statement: the locked 2019 LA evaluation stays
REPORTED and immutable, and a Phase 2 dataset or protocol is always a **new** freeze
record, never a widening of the existing one.

### Decisions left pending

- Approval of Gates 0–3 as the governing Phase 2 structure.
- Approval of each of the five foundation contracts, individually — they are
  proposals in a document whose other contracts are adopted, and the section says so
  at the top.
- How far to generalize the manifest before the second dataset exists.
- Whether the Phase 2 prompt and the hardening plan are retained in-repo at all.
- Whether external datasets beyond ASVspoof 2021 DF (In-the-Wild, WaveFake) are
  approved, which the backlog already makes conditional on owner approval and a
  terms check.

### Hardening plan disposition

Per the work order's C2: **not deleted, not staged, not added to Git.** A header was
prepended identifying it as a historical/superseded input reconciled by PR #113,
quoting its own "could not retrieve the repository" boundary, stating that its
Phase 0–6 schedule is expired, and listing the eight filenames that must not be
created. `docs/README.md` lists it under "Not authoritative". **None of the eight
prescribed paths was created** (verified: `DATASETS.md`, `EVALUATION_PROTOCOL.md`,
`MODEL_CARD.md`, `REPRODUCIBILITY.md`, `reports/data_audit.md`,
`reports/phase_inventory.md`, `reports/evaluation/model_comparison.csv`,
`reports/evaluation/per_attack.csv` — all still absent).

---

## 6. Integrity and validation

| Check | Method | Result |
| --- | --- | --- |
| Frozen evaluation record | `shasum -a 256 reports/evaluation_freeze_g2.json` | `cfd1406c…cb4d2` — **unchanged**, matches every document's pin |
| Frozen reservation record | `shasum -a 256 reports/dev_reservation_frozen.json` | `4f518e87…25c1a` — **unchanged** |
| Frozen proposal | `shasum -a 256 docs/…Proposal.docx` | `b16f5124…a09aec` — **unchanged**, matches the immutability pin |
| Relative Markdown links | script over all `.md` files | **0 broken** |
| Heading anchors | script slugifying all headings, resolving all `…#anchor` links | **0 broken** (one break introduced by the pull-gate heading was found and fixed) |
| Removed/renamed path references | the two scripts above, plus a check that the eight forbidden paths were not created | none introduced |
| Cross-document contradictions | grep for residual stale phrasing (`tentative presentation`, `verification and rehearsal stage`, `Status (27 Oct`, `In progress` for G4) across all tracked docs outside `archive/` | **0 residual**; `data.md` and `environment.md` retain dated 27 October *record* headers, which remain accurate statements about past events |
| Acceptance-line count | `grep -c '^- \[ \]'` within the acceptance checklist | **4** — confirms the "four unmet lines" claim used in three documents |
| Whitespace / conflict markers | `git diff --check` | clean |
| Scope — code, tests, configs, data, models, evaluation artifacts | `git diff --name-only \| grep -v '\.md$'` | **empty — only `.md` files changed** |
| Full diff | reviewed hunk by hunk | no metric, hash, threshold, count or architecture value altered anywhere; model-card diff grepped for all nine headline numbers → 0 hits |
| Git state | `git status`, `git diff --cached` | nothing staged, nothing committed, no tag created, no branch created, no push |

Archive documents (`docs/archive/`) were not edited, per their standing "not edited
to match the present" rule.

---

## 7. Outstanding owner decisions

In the order that makes each cheapest to resolve.

1. **The release (gating).** Tag it and complete the sign-off record, or record a
   dated waiver for each of the four unmet acceptance lines with reason, impact and
   owner/date. Nothing else closes Phase 1 while this is open, and the project's own
   rule blocks sign-off until it is answered. *(#111)*
2. **What the presentation actually used.** Which machine presented; whether the live
   container or the offline fallback ran; whether the venue access path worked; which
   demo clips were shown, with tool, version and terms. This is pure recall and it
   decays fastest — it is the one item that gets harder every week. *(#108, #109)*
3. **Model card.** Accept DRAFT in the release record, or close its five blockers.
   *(#110)*
4. **Unverified G4/G5 completion.** The fresh-checkout reproduction (#107) was never
   run and remains the largest unmet acceptance claim; the dataset mirror URL (#112)
   needs supplying. Both are work, not just decisions.
5. **Host-only numbers.** The latency, cold-start, image-size and memory figures, the
   threshold set's 574 clips / 4.88% FPR, the development EER tables and the locked
   report hash were to be confirmed against host outputs before the presentation and
   were not. Confirm before any release or publication reuses them.
6. **Phase 2 design approvals.** Gates 0–3, and each of the five proposed foundation
   contracts; plus how far to generalize the manifest before a second dataset exists.
7. **Retention of the two untracked planning documents** in-repo, in any form.

---

## 8. Final recommendation

These are three different readiness levels and the project sits at exactly one of
them.

| Readiness | Verdict |
| --- | --- |
| **Owner review** | **Ready.** The documentation now states what is true, what is owner-sourced, and what is unverified. A reader can establish the project's position from `README.md` → `docs/roadmap.md` → `TODO-Plan.md` without contradiction, and the Phase 2 proposals are marked as proposals. |
| **Release / sign-off** | **Not ready, and not an agent's call.** Three acceptance lines are unmet with no waiver (four at the time of writing; see the addendum), 7 of 13 sign-off fields are pending, and the fresh-checkout reproduction was never run. The gap is the owner's decision in §7.1 plus the work in §7.4 — not more documentation. |
| **Phase 2 implementation** | **Not ready.** Gate 0 is not passed: the foundation contracts are proposals awaiting approval, none is implemented, and no regression test exists for any of them. The Phase 1 evidence is preserved and the boundary is documented, which is the prerequisite for Gate 0 — but not its exit. |

**Nothing in this task closed an item, created a tag, wrote a waiver, or asserted a
completion that evidence does not support.** Where the evidence ran out — the
presentation itself, what it demonstrated, whether it was rehearsed — the
documentation now says so explicitly and names who can answer.


---

## Addendum — demo record captured (2026-11-06, after the sections above)

§3 and §7.2 of this report listed "what the presentation actually used" as the item
that decays fastest and that only the owner could supply. The owner supplied it the
same day. Recorded here because it closes items this report had left open.

### What the owner stated

| Question | Answer |
| --- | --- |
| Which machine served the demo | **The presenting laptop, locally** — the `scripts/demo_setup.sh` path: bundle and clips copied from `Linux VM`, image run in Docker Desktop on the laptop. Not the DL120 (which the 13 October decision named), and not the SSH tunnel that slide 10 describes. |
| Live or pre-scored | **Live uploads, real responses.** Not `demo_backup.py` output, not screenshots. |
| Inputs shown | Own-voice recording; text-to-speech clips; the too-short and digital-silence WAVs through the Streamlit uploader; the MP3 rejection through `curl`. |
| Synthetic-clip provenance | **Ondoku3** (<https://ondoku3.com>), an online text-to-speech service — an external tool, not the repository's `generate_demo_clips.py` synthesis. |
| Observed behaviour | The Ondoku3 clips were **flagged spoof-like** — the correct direction. |

### What was written where

| File | Change |
| --- | --- |
| `docs/data.md` | New section "Demonstration clips (not dataset material)": a provenance table for the three input sources with their terms status, the anecdotal observation with its limits stated, and a note that the clips stay git-ignored because one is personal audio. |
| `docs/environment.md` | "Which machine presented" rewritten from *unrecorded* to the laptop-local answer, with the live-service detail, and a new caveat paragraph on the ARM64 build. Also records that the venue access path was never exercised, because no tunnel was used. |
| `TODO-Plan.md` | §9 demo-set item: status replaced with the recorded provenance, left open on Ondoku3's terms. §13: "Demonstrate a valid recording and an invalid input" → **[x]**, and "Test demonstration startup … on the presentation machine" → **[x]**, both with the evidence and its limits. Acceptance line "The application runs and handles valid and invalid inputs" → **[x]**. Unmet-line count corrected from four to three in three places. New decisions row. |
| `docs/roadmap.md` | Remaining-work item 2 marked mostly done, naming the two owner-only gaps and the parity gap; "Not met" count corrected to three. |
| `docs/model-card.md` | Blocker 3 rewritten (set recorded, terms still missing). New §5 paragraph on the live demo, explicitly labelled anecdotal and explicitly **not** evidence that the model detects contemporary synthesis. No metric, hash or threshold touched (diff grepped for 14 headline values: 0 hits). |
| `docs/README.md`, `README.md` | `data.md`'s authority line widened to cover demo-clip provenance; the README's still-open list updated. |

### What this closed, and what it did not

**Closed on evidence:** two §13 items and one acceptance line. Unmet acceptance lines
drop from four to three, verified by count. #108's central question — which machine
presented — is answered.

**Deliberately not closed:**

1. **Ondoku3's terms of use — checked 2026-11-06, see the second addendum below.** Use
   in a research presentation is expressly permitted; a free account obliges a visible
   credit notice. `#109` stays open on whether that notice was displayed, not on what the
   terms say.
2. **Clip counts and the own-voice clip's outcome** are unrecorded, so the proposal's
   "label the set illustrative with n reported" is still unmet. This matters more than
   it looks: #103 records two own-voice phone recordings being flagged spoof-like, so
   whether the demo's own-voice clip passed or failed is the single most informative
   unrecorded fact about the demonstration.
3. **The laptop's container was never parity-checked.** `demo_setup.sh` describes the
   presenting laptop as an M4 Pro and its step 1 warns that a native ARM64 image carries
   a different torch wheel from the `linux/amd64` wheel behind the locked evaluation and
   the recorded parity run. No `container_parity.py` run against that container exists.
   The demo therefore evidences **behaviour**, not score fidelity — which is why the
   acceptance line was ticked for input handling while the caveat was recorded beside it.
   This sub-item of [#108](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/108) stays open.
4. **The architecture was not separately confirmed.** The ARM64 inference rests on
   `demo_setup.sh`'s description of the laptop, not on a statement about the machine
   that ran on 5 November. Recorded as conditional wording, not as fact.

### Integrity

Frozen artifacts re-hashed after these edits and unchanged: `cfd1406c…cb4d2`,
`4f518e87…25c1a`, `b16f5124…a09aec`. Links and anchors clean. Only `.md` files changed.
Nothing staged, committed or tagged. No locked number, threshold, config, code or test
was touched.

One correction to §6 above: `git diff --check` is **not** silent any more. It reports
trailing whitespace at `TODO-Plan.md:5`, the line now reading "Presented: 5 November
2026 …". Those are two deliberate trailing spaces — Markdown's hard line break, which
keeps the Owner / Created / Presented header block on separate lines. The original line
carried them too (verified against `HEAD`), and three of the five header lines use the
same style, so this is pre-existing formatting surfacing because the line changed, not
a defect introduced here. Stripping them would reflow the header, so they stay.


---

## Second addendum — Ondoku3 terms checked (2026-11-06, at the owner's request)

The first addendum left Ondoku3's terms as the one gap an agent should not fill by
inference. The owner asked for them to be looked up, so they were read from the
service's own pages and summarised into
[`../docs/data.md`](../docs/data.md), "Ondoku3 terms".

**Sources**, both read 2026-11-06 and cited with that access date, because the licence
article states that the terms may change without notice:

- usage and licence article, <https://ondoku3.com/en/post/terms-licence/> (page dated
  22 July 2026);
- Terms of Use, <https://ondoku3.com/en/terms/> (operator: Mr. Co., Ltd.).

Summarised in the record, not reproduced.

### What the terms say about this use

| Point | Finding |
| --- | --- |
| Is a project presentation permitted? | **Yes, expressly.** The licence article lists use of Ondoku audio in lectures and research presentations, and use of Ondoku audio in a presentation one gives oneself, among permitted uses. |
| Does it need a Business Plan? | **No.** None of the Business-Plan-only purposes in Terms of Use Article 12 applies: no audio made on commission, nothing embedded in a product, no distribution of content whose main component is the audio, no broadcast use. |
| Is there a condition? | **Yes — a credit notice for free-account use**, visible to the audience wherever the audio plays. Terms of Use Article 13 §1 triggers the duty on disclosing the audio to a third party, expressly including playing it in a facility third parties enter, which describes a presentation room. It falls away only if a paid account produced the clips (Art. 13 §1(1)). Accepted wordings include `Audio: Ondoku` and `Voice by Ondoku3.com`. |
| Who owns the generated audio? | **The copyright holder of the input text** (Art. 14 §1); Art. 14 §4 requires prior permission where that text is someone else's. The prompt text was not recorded. |
| Redistribution duties? | **Not engaged.** No clip was committed or published (`demo_clips/` is git-ignored), so Art. 13 §2's third-party distribution obligations do not arise. |
| Machine-learning use? | **Not addressed by the terms.** In this project the clips were demonstration inputs only — never training, development or evaluation data — so the question does not arise. Flagged in the record as something to check before any future use of such audio *as data*. |

### What this changed, and the one thing it did not close

`#109` no longer waits on what the terms say. It now waits on three owner-only facts,
and the first is new information this lookup produced:

1. **Free or paid Ondoku3 account — answered: free. See the third addendum below.** The
   credit condition applied, the delivered deck carried no notice, and the deck has been
   corrected for reuse with the gap disclosed.
2. Clip counts and the own-voice clip's score (carried over).
3. Who authored the prompt text, because copyright in the audio follows it.

The record states that it is a factual summary for provenance and **not legal advice**,
and that the owner is the party who accepted the terms. No conclusion was drawn about
whether the owner complied; the condition and the missing fact are recorded separately
so the reader can see which is which.

### Propagation and integrity

`docs/data.md` (terms subsection and the provenance row), `TODO-Plan.md` (§9 status plus
a new decisions row), `docs/roadmap.md` (remaining-work item 2), `docs/model-card.md`
(blocker 3) and `README.md` (still-open list) were updated so none of them still says the
terms are unchecked. Frozen artifacts re-hashed unchanged; links and anchors clean; only
`.md` files touched; nothing staged, committed or tagged.


---

## Third addendum — free account confirmed, credit gap closed out (2026-11-06)

The owner confirmed the Ondoku3 clips came from a **free** account. That resolves the
question the second addendum left open and makes the credit condition applicable, so the
next step was to establish what the delivered deck actually carried rather than ask.

### Evidence gathered rather than requested

Searched [`../docs/presentation.html`](../docs/presentation.html) and
[`../docs/presentation-outline.md`](../docs/presentation-outline.md) for any Ondoku,
credit or attribution string. **The only attribution in either file was the ASVspoof
ODC-By notice.** So the delivered deck carried **no** Ondoku credit, and the free-tier
condition was not met on the slide at the 5 November delivery. This is a verified
negative, not an inference from silence: the deck is the artifact that was shown.

What is *not* established, and is not claimed in any direction: whether the owner
credited Ondoku aloud or on screen outside the deck.

### What was done about it

| File | Change |
| --- | --- |
| `docs/presentation.html` | Slide 10 — the slide on which the clips play — gained a visible line: `Synthetic demo clips — Voice by Ondoku3.com. Own-voice clip recorded and used with the speaker's consent.` `Voice by Ondoku3.com` is one of the wordings Ondoku3's licence article accepts. Visible body text, deliberately not a speaker note, because the condition is that the audience can see it. |
| `docs/presentation-outline.md` | Slide 10 notes record why the line must stay, that it was added on 2026-11-06 and **not** present at delivery, and what to do if the clips are replaced. |
| `docs/data.md` | "Ondoku3 terms" now states the free account, the verified absence of credit at delivery, the post-delivery fix, and that the obligation is continuing — it attaches to any recording, re-showing or reuse, not just the original talk. |
| `TODO-Plan.md` | §9 status updated; §13 deck item records the post-delivery change so the deck's divergence from the delivered version is on the record next to the 27 October and 3 November corrections; new decisions row. |
| `docs/roadmap.md`, `docs/model-card.md` | Remaining-work item and blocker 3 updated. |

### The judgement call, stated plainly

Editing a delivered artifact after the fact deserves to be explicit rather than quiet.
The deck file now differs from what the audience saw, in this one respect. That is
recorded in three places (`data.md`, `TODO-Plan.md` §13, the outline) with the date and
the reason, so nobody can later mistake the corrected deck for the delivered one. The
alternative — leaving a known, continuing licence condition unmet in the project's own
presentation artifact — would have been worse, and the obligation applies to every future
showing, not only the one that already happened. No claim was added anywhere that credit
*was* displayed on the day.

### Still open on #109

Two owner-only facts, both carried over: the clip count and what the own-voice clip
scored, and who authored the prompt text entered into Ondoku3 (copyright in the generated
audio follows the input text, Terms of Use Art. 14 §1). Neither blocks the release
decision ([#111](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/111)), which remains the gating item.

### Integrity

Frozen artifacts re-hashed unchanged. Links, anchors and placeholders clean. Changes are
confined to Markdown plus `docs/presentation.html`, which is a presentation artifact, not
code, config, test, dataset or evaluation output. Nothing staged, committed or tagged.

The deck was opened and advanced to slide 10 to confirm the line **renders as visible
body text** rather than merely existing in the markup - the condition is that the
audience can see it, so markup alone would not have settled it. Tag balance checked
against `HEAD`: paragraphs 17/17 before, 18/18 after; `div` count unchanged at 62/62.
