# AI Voice Deepfake Detector — Project Handoff

**Handoff date:** 2026-11-06  
**Purpose:** Resume the project and documentation/release-readiness work in a future ChatGPT conversation or Codex session.

---

## 1. Project identity and goals

- **Repository:** https://github.com/mohamadar77/ai-voice-deepfake-detector
- **Current branch/commit reported during audit:** `main` at `d900088`
- **Project:** AI Voice Deepfake Detector
- **Owner/team:** Individual project (the user is the sole builder).
- **Near-term goal:** Deliver an excellent AI Engineering course graduation project.
- **Longer-term goal:** Evolve toward a research-grade, potentially publishable deepfake detection system.
- **Current priorities:** on-time delivery; rigorous but manageable evaluation; reproducible models; a demonstrable end-to-end product; document research extensions without expanding the current delivery scope.
- The user is a Senior Software Engineer at Red Hat and is using this project to advance toward AI Engineering.

The project was selected over an AI Fake News/Misinformation Detector. Course syllabus coverage and principles-based engineering are important. A CNN was added to the proposal.

## 2. Collaboration and evidence rules

Work with a technically rigorous, direct style. The user values clear separation between:

1. **Verified repository evidence**
2. **Owner-provided recollection or statements**
3. **Inference**
4. **Unresolved questions**

Do not turn an owner statement into repository-verified evidence. Do not infer that an event, rehearsal, sign-off, release, or waiver occurred merely because it was discussed.

Preserve evaluation integrity:
- Do not modify frozen proposal/evaluation artifacts, thresholds, metrics, dataset partitions, model bundle, manifests, or hashes unless the user explicitly authorizes it.
- Do not silently change model results or evaluation protocol.
- Do not create release tags, commits, pushes, or acceptance waivers on the user's behalf.
- Keep Phase 1 delivery separate from Phase 2 research enhancements.
- Avoid documentation restructuring for aesthetics.

The user's `TODO-Plan.md` is the single status authority. Other documents have distinct responsibilities:
- `docs/roadmap.md`: milestone sequence and dates
- `docs/architecture-and-delivery-plan.md`: architecture, contracts, deviations
- `docs/research-backlog.md`: Phase 2 scope
- `docs/README.md`: documentation index
- Historical records belong in the archive.

## 3. Model/evaluation context

Previously recorded results (verify against the repository's frozen evidence before quoting or changing them):

### Logistic Regression baseline
- EER: 15.97%
- ROC AUC: 0.924
- Balanced accuracy: 83.0%
- Macro F1: 0.625
- Human FPR: 4.9% (target was below 5%)
- Synthetic detection: 70.9%
- C=0.01; threshold=1.297

### CNN clean evaluation
- EER: 0.63%
- ROC AUC: 0.9997
- Macro F1: 0.989
- Balanced accuracy: 99.1%
- Human FPR: 1.5%
- Synthetic detection: 99.7%
- Best checkpoint: epoch 12/17, early-stopping patience 5
- Approximately 62 minutes on CPU
- 23,650 parameters
- Compute environment previously documented: DL120 44-core Xeon; CUDA 13.0 available but not needed; peak RSS around 1.3 GiB. CPU host was chosen; Kaggle/Colab not required.

**Important:** These are historical context values, not permission to edit or re-baseline. Frozen records in the repository remain authoritative.

## 4. Step 1 — Documentation audit

A documentation audit report was created:
- **Report:** `documentation-audit-step1.md`
- **Audit date:** 2026-11-06
- **Repository state reported:** `main`, commit `d900088`

### Audit conclusion
The evidence layer was sound and independently verifiable, but the status/documentation layer was stale.

### Checks reported as passing
- Proposal SHA256 pinned.
- Evaluation freeze SHA256 pinned.
- Frozen thresholds agree across results, model card, and serving configuration.
- Deployed bundle hash and manifest hash match.
- Deployed summary matches results.
- Dataset counts and speaker partitions are consistent.
- CNN parameter count recomputed as 23,650.
- Architecture and Mel preprocessing are consistent.
- No broken Markdown links or anchors.
- No PCA/K-means.
- No SQLite `runs` table.

### Main audit findings
1. Status documents still described the presentation as upcoming/tentative although the owner said it was delivered on **2026-11-05**.
2. `TODO-Plan.md` omitted issues #107–#112 and #118.
3. An untracked hardening plan duplicated planning. PR #113 had already reconciled selected useful items. The hardening plan referenced eight nonexistent paths; those paths were not to be created.
4. Documentation index was incomplete regarding configs/reports.
5. Preserve the evidence layer; do not restructure it.

### Relevant issue references
- **#107:** fresh-checkout reproduction
- **#108:** presentation machine/environment
- **#109:** demo provenance
- **#110:** model card
- **#111:** rehearsal/fallback/release tag/sign-off
- **#112:** dataset mirror URL
- **#118:** reserve-development force-override audit trail (Phase 2; not a Phase 1 blocker)

## 5. Step 2A — Documentation reconciliation

A scoped Codex prompt was prepared. Its constraints:
- No code, tests, model, data, configuration, or evaluation changes.
- No frozen artifact changes.
- No tag, commit, or push.
- Do not guess completion.
- No Phase 2 implementation.
- Update relevant documentation only and produce `reports/documentation-reconciliation-step2a.md`.

### User-reported outcome
- Eight tracked Markdown files updated (+319/-105).
- Nothing staged or committed; no tags created.
- Phase 1 remains presented-but-not-released.
- No G4/G5 item was marked complete without evidence.
- Frozen artifacts were re-hashed and unchanged.
- Links and anchors were clean.

The Step 2A report was available in the earlier conversation as file reference `turn2file0` (245 lines). If the report itself is not available in the resumed conversation, use the repository copy rather than assuming its full contents.

### State described by the Step 2A report
- Repository still `main` at `d900088`, modified working tree.
- Owner says presentation was delivered 2026-11-05; no repository artifact corroborates delivery (no tag or commit after 3 Nov). This is an owner-sourced statement.
- No sign-off recorded: 7 of 13 fields pending.
- Four acceptance checklist lines unmet and none waived.
- No release tag exists; no release commit identified.
- G4 not delivered: #107–#110 and #112 all OPEN; G4 milestone 5 open / 0 closed.
- G5 not recorded: #111 OPEN; G5 milestone 1 open / 0 closed; no rehearsal, backup recording, or offline-fallback verification in repository.
- #107 fresh checkout not run; reproduction guide says pipeline itself was not re-run; clean-host run needed.
- #108 presentation machine unknown at the time of that report. Candidate ambiguity included DL120, `hp-xl420gen9-01`, or presentation Mac M4 Pro ARM64 running `linux/amd64` emulation.
- #109 provenance incomplete at the time: actual human recordings, TTS tool/version/terms, and related details were not all documented.
- #110 model card remained DRAFT with five blockers open.
- #111 requires owner decision: either complete sign-off and release, or explicitly waive each unmet checklist line with reason, impact, owner/date. No agent-created waiver.
- #112 requires owner to supply dataset mirror URL.
- #118 remains Phase 2, not a Phase 1 blocker.

### Four acceptance checklist lines reported unmet
- Real data documented/auditable.
- Baseline/CNN comparisons reproducible.
- App runs and handles valid/invalid inputs.
- Presentation and backup demo ready.

### Step 2A documentation changes
Reported as eight tracked Markdown files:
- `README.md`
- `TODO-Plan.md`
- `docs/roadmap.md`
- `docs/architecture-and-delivery-plan.md`
- `docs/README.md`
- `docs/research-backlog.md`
- `docs/environment.md`
- `docs/model-card.md`

The untracked hardening plan also received a superseded header but was not staged/added. The model-card augmentation pointer was corrected to `configs/robustness_spec.yaml`.

### Validation reported
- Frozen evaluation record, dev reservation record, and proposal hashes unchanged.
- No broken links or anchors.
- `git diff --check` clean.
- Only Markdown changed.
- No metrics, hashes, thresholds, counts, or architecture values altered.
- No tag, commit, or push.

### Readiness conclusion at that point
- Owner review: ready.
- Release/sign-off: not ready.
- Phase 2 implementation: not ready.

## 6. Later update — Issue #109 and presentation deck attribution

The user later shared a further report/update concerning demo provenance and the deck.

### Verified deck findings
- The delivered revision of `docs/presentation.html` and `docs/presentation-outline.md` had no Ondoku credit.
- The deck did contain the ASVspoof ODC-By notice.
- This is a verified negative about the delivered revision, not an inference from silence.
- It is **not known** whether the user verbally credited Ondoku or displayed credit elsewhere outside the deck. That remains unrecorded.

### Post-event correction
- Slide 10, where the audio clips play, was updated with visible body text crediting Ondoku3.com and stating that the own-voice clip was recorded and used with the speaker's consent.
- Credit was placed in visible slide body text, not only speaker notes, because the audience must see it.
- The rendered deck was opened and slide 10 checked; validation was not limited to HTML markup.
- HTML paragraph tag balance changed from 17/17 to 18/18; div balance unchanged.
- The outline explains why the credit must remain, that it was added on 2026-11-06 and was absent from the delivered version, and what to do if clips change.
- The post-event edit is documented in `docs/data.md`, `TODO-Plan.md` §13, and the presentation outline, alongside 27 Oct and 3 Nov corrections.
- The corrected deck must not be represented as the exact deck shown to the audience on 5 Nov.
- Reason for the correction: ongoing licensing obligations. The credit must remain wherever the audio is played, including recordings, re-showing, or reuse. If the owner wants the exact delivered deck, the credit paragraph can be removed, but the clips should not be shown again as-is without satisfying the attribution condition.
- The exact credit wording should be read from the current repository file; do not reconstruct it from this summary.

### Demo instruction discrepancy
- Slide 10 step 1 still says: “Open the Streamlit UI (SSH tunnel to host)”.
- User explicitly stated this is inaccurate: **the demo ran locally in Docker**, not through an SSH tunnel to a remote host.
- This was flagged rather than silently rewritten because the presentation script is the owner's responsibility.
- This establishes local Docker execution, but does not yet establish which local machine ran it or whether Docker used native ARM64 or `linux/amd64` emulation.

### Remaining #109 facts
The later update said #109 now rests on two remaining facts:
1. Clip count and the score obtained by the own-voice clip.
2. Who wrote the prompt text entered into Ondoku3.

Do not invent these values or authorship. Record them when the owner supplies them or when repository evidence establishes them.

### Scope and validation of this later update
- The user reported 11 files changed, 465 insertions.
- Frozen hashes unchanged; links/anchors clean; nothing staged/committed.
- This may have been a subsequent Codex run beyond the original Step 2A eight-file reconciliation. Verify current `git status` and diffs before assuming exact file state.
- Do not assume all 11 changed files are tracked or that this update was committed.

## 7. Current unresolved items and next actions

### First: resolve presentation machine (#108)
Known:
- Demo ran locally in Docker (owner-provided fact).
- The slide's SSH-tunnel instruction is inaccurate.

Still unknown:
- Which exact machine ran the presentation demo?
- Was it the MacBook M4 Pro, or another local machine?
- If MacBook M4 Pro: did Docker run the image natively as ARM64 or through `linux/amd64` emulation?
- What exact environment details should be recorded (OS, architecture, Docker version/image, invocation, and any relevant access path)?

**Next useful owner question:** Which exact local machine ran the demo? If it was the MacBook M4 Pro, confirm whether the container ran as native ARM64 or `linux/amd64` emulation.

### Second: close the remaining demo provenance facts (#109)
Obtain and record:
- Number of audio clips played.
- The own-voice clip's exact detector score/result and the interpretation shown during the demo.
- Who authored the prompt text submitted to Ondoku3.
- Any remaining source/tool/version/license metadata required by the repository's provenance checklist.
- Whether any verbal or off-deck Ondoku attribution was made is unknown; do not state that it happened or did not happen without owner confirmation.

### Third: decide release/sign-off (#111)
At the time of the report:
- Seven of thirteen sign-off fields were pending.
- Four acceptance checklist lines were unmet and none waived.
- No release tag existed.
- No rehearsal, backup recording, or offline-fallback verification was recorded.

Owner must choose and explicitly document one of:
1. Complete outstanding acceptance work, sign-off, and release; or
2. Explicitly waive individual unmet checklist items, each with reason, impact, owner, and date.

Never create waivers on behalf of the owner. Do not call the project released until repository evidence supports it.

### Other Phase 1 work
- #107: perform fresh-checkout reproduction on a clean host; distinguish documentation review from actually re-running the pipeline.
- #110: keep model card DRAFT until its blockers are closed or explicitly waived.
- #112: owner must provide the dataset mirror URL.
- #118: Phase 2; do not treat as a Phase 1 release blocker.

### Phase 2
The reconciliation report proposed (but did not approve or implement) future work:
- Gates 0–3.
- Multi-dataset identity/adapters.
- Dataset-aware freeze/prediction identity.
- Per-run JSON as immutable authoritative record, with SQLite as rebuildable index.
- Artifact integrity/publication.
- Regression tests.

Keep these as proposals until the owner approves Phase 2 scope.

## 8. Safe resume procedure for a new session

1. Read this handoff.
2. Inspect the current repository state; do not assume the working tree still matches the last reported state:
   - `git status --short --branch`
   - `git log -1 --oneline --decorate`
   - `git diff --stat`
   - `git diff --`
   - `git ls-files --others --exclude-standard`
3. Read current versions of:
   - `TODO-Plan.md`
   - `docs/roadmap.md`
   - `docs/architecture-and-delivery-plan.md`
   - `docs/README.md`
   - `docs/research-backlog.md`
   - `docs/environment.md`
   - `docs/model-card.md`
   - `docs/data.md`
   - `docs/presentation.html`
   - `docs/presentation-outline.md`
   - `reports/documentation-reconciliation-step2a.md`
   - `documentation-audit-step1.md` (or its actual repository location)
4. Compare the current files with the claims in this handoff. Treat the repository as current evidence and this handoff as historical context.
5. Re-hash frozen artifacts before/after any documentation-only work if the task requires integrity verification.
6. Make only the changes the owner explicitly requests. Preserve the frozen evaluation layer.
7. Update `TODO-Plan.md` as the status source of truth whenever project status legitimately changes.
8. Do not stage, commit, push, tag, or create release records unless the user explicitly asks.

## 9. Suggested Codex continuation prompt

Use this prompt after confirming the repository is at the intended state:

> Continue work on the AI Voice Deepfake Detector repository using the project handoff as context. First inspect the actual git status, current commit, tracked and untracked diffs, and current versions of the relevant documentation. Do not assume the handoff's historical working-tree state is still current.
>
> Preserve all frozen proposal/evaluation artifacts, hashes, thresholds, metrics, dataset partitions, model bundle, and manifests. Do not modify code, tests, models, data, configs, or evaluation results unless I explicitly authorize it. Do not stage, commit, push, tag, or create release/waiver records.
>
> The demo ran locally in Docker, not through the SSH tunnel currently described in slide 10. Identify the exact local presentation machine and architecture from evidence where possible; if not possible, report the missing owner input. Continue resolving issue #109 by identifying the exact clip count, own-voice clip score/result, and prompt-text author from repository evidence where possible. Clearly separate verified evidence from owner input and unknowns.
>
> Keep the corrected Ondoku attribution in the current deck unless I explicitly direct otherwise. Preserve the distinction between the deck delivered on 2026-11-05 and the post-event corrected deck. Do not imply that verbal attribution occurred unless I confirm it.
>
> Report findings, evidence paths, unresolved owner questions, and the smallest safe next step. Do not mark any milestone, acceptance criterion, sign-off field, issue, or release as complete without evidence.

---

## 10. Important caution

This handoff summarizes conversation and user-reported Codex outputs as of 2026-11-06. It is not a substitute for checking the live repository. The working tree may have changed since the last report. Repository files, Git history, frozen records, and explicit owner confirmation should be used to establish current truth.
