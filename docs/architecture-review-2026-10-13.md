# Architecture review — 13 October 2026

Role: senior architect review of the frozen proposal, the planning documents, the
milestones and the live GitHub tracker, at the start of G2 (day 9 of 31).

Inputs:
- the frozen proposal `docs/AI_Voice_Deepfake_Detector_Final_Project_Proposal.docx`, whose
  SHA-256 was re-verified as `b16f5124…a09aec` and is **unchanged**;
- `AGENTS.md`, `TODO-Plan.md`, and every file under `docs/`;
- the code under `src/voice_detector/`;
- GitHub milestones G0–G5 and issues #4–#30.

The proposal was not edited. Every amendment below lives in Markdown and GitHub, as
required by the sign-off.

## Verdict

The proposal is **strong enough to be an impressive project, but not as written.**
Its methodology is well above a typical course capstone. Its headline numbers will
not impress on their own, because MFCC + LR and a compact CNN are 2019-era
countermeasures. The project impresses if the presentation leads with *reliability
findings*: what the detector actually learned, where it breaks, and evidence that
the evaluation was honest. Leaderboard numbers should not be the lead. Three cheap
additions make that story concrete; they are listed below.

The plan is detailed and disciplined. It had one **protocol-critical gap**: nothing
scheduled the development reservation before the LR baseline tunes and sets its
threshold. It also had one **schedule gap**: G1 is overdue, the 25% contingency is
spent, and the owner's weekly hours were never recorded. Both are now addressed.

| Dimension | Assessment |
| --- | --- |
| Problem framing and honesty | **Strong.** A score is not proof of authenticity, there are no leaderboard claims, and calibration is used only if fitted. |
| Evaluation rigor | **Strong by design; not yet validated.** The design is sound: unseen speakers and attacks, thresholds on reserved development speakers, paired robustness, a grouped bootstrap, an all-candidate freeze, and kept negative results. A design is not evidence, though. The claim counts only once the artifacts in "Design is not evidence" below exist. |
| Engineering | **Strong in the parts built so far.** Train and serve share one versioned preprocessing path enforced in code, and the API contract is typed with transient uploads. Containerization and observability are still planned, not built. |
| Evidence of execution | **Good.** The manifest matches the published counts exactly. Option D caught a real zero-padding shortcut before any model trained. |
| Novelty / "wow" | **Moderate.** Classic models, and two of the three research questions are only partly answered (see below). |
| Feasibility by 5 Nov | **At risk.** G1 slipped about 4 days, the host is CPU-only, and owner hours are unknown. It is recoverable with the parallel lane and the cut list below. |

## What makes it strong (keep)

1. **The leakage discipline is the differentiator.** It includes:
   - `dev_select` / `dev_threshold` separation;
   - `DRAFT → FROZEN → SCORED → REPORTED` freeze records that refuse mismatched or overwritten results;
   - one score per clip;
   - a speaker- or source-grouped bootstrap.

   Few student projects can show that their test set was never touched. Make this
   one slide.
2. **The unseen-speaker and unseen-attack test is built into the data.** Evaluation
   speakers are disjoint from training. Training and development use A01–A06.
   Evaluation uses A07–A19, of which A16/A19 reuse known algorithms (see findings),
   so the data also contains a known-versus-unseen control. Per-attack results will
   show a wide spread, because some attacks are known to be hard for spectral
   countermeasures. That spread is the best failure-analysis material available.
3. **Train/serve parity is enforced, not hoped for.** The `PREPROCESSING_VERSION`
   code constant is asserted against the config, and the API sits behind a neutral
   `Scorer` protocol.
4. **The project already has an honest-shortcut story.** The option D decision (#4)
   removed zero-padding that would have leaked clip duration into pooled MFCC
   statistics. It is a real example of catching a shortcut before it shows up in a
   result.

## Design is not evidence (owner challenge, accepted)

The owner rightly challenged the original "top tier" wording. A rigorous design shows
neither that the implementation is correct nor that the models generalize. No
methodology claim goes into the report or slides until these artifacts exist:

1. **Leakage tests pass on the real manifest**, not only on fixtures:
   - speakers are disjoint across official splits (already asserted by the manifest builder);
   - no speaker holds two development roles (#31);
   - every fitted transform's row set is a subset of train (scaler, PCA, K-means, normalization);
   - evaluation rows are untouched until the freeze record says `FROZEN` (#35).
2. **Reproducible runs.** A re-run gives identical development predictions on CPU,
   or matches within a recorded tolerance for GPU runs. Run records carry the
   commit, manifest hash, config and seed.
3. **Per-attack results with counts**, separated into known and unseen algorithms
   (see the A16/A19 finding below), plus the pooled unseen result.
4. **Offline == API parity** on real clips (#9), and container parity on the
   presentation machine (#11).

**Model choice follows the same principle.** No advanced or pretrained models in
Phase 1. A reliable, thoroughly evaluated classical model is better research than an
impressive pretrained model with weak evaluation. SSL front-ends (wav2vec 2.0 and
others) go to the [post-graduation research backlog](research-backlog.md).

## Where it will under-impress, and the cheap fixes

| # | Gap | Why it matters | Fix (cost) | Tracked |
| --- | --- | --- | --- | --- |
| 1 | The silence/duration shortcut is only audited in EDA | ASVspoof 2019 LA's best-known pitfall is that leading/trailing silence differs by class, and models can lean on it (Müller et al., ASVspoof 2021 Workshop). The proposal's risk table already commits to "inspect silence and duration effects". | (a) A shortcut-only baseline: LR on duration + silence, evaluated on `dev_select`. (b) A paired, evaluation-only **silence-trimmed condition** declared before the freeze, scored for all three bundles. Likely the most memorable result slide. About 4–6 h. | #6, #34 |
| 2 | Headline EERs have no reference point | A 10% EER on unseen attacks sounds bad, but it is comparable to the published B01/B02 challenge baselines. Without context the audience cannot read the number. | Cite the published B01 (CQCC-GMM) and B02 (LFCC-GMM) evaluation results, labeled as published numbers that were not reproduced. Min t-DCF is deferred to Phase 2. About 1 h. | TODO §13 (#38 → Phase 2) |
| 3 | The demo shows only benchmark-style clips | The proposal itself says old benchmarks do not establish performance on current generators. Showing that live is the strongest possible demo moment. | Demo set: the owner's own consented recordings, plus synthetic clips the owner generates with a modern TTS/voice-clone tool whose terms permit it. Label the set *illustrative* and report n. A miss is a planned "honest failure". About 2–3 h. | TODO §9 |
| 4 | Two of three research questions are only partly answered | RQ2's "second evaluation domain" (2021 DF) and RQ3 (wav2vec 2.0) are deferred. The proposal requires unavailable transfer experiments to be clearly identified. | Final report and slides map each RQ to its evidence or to an explicit "not run". 2021 DF transfer leads the Phase 2 backlog. | TODO §7/§13, [research backlog](research-backlog.md) |
| 5 | The model comparison confounds features with architecture | MFCC+LR versus log-Mel CNN does not show which of the two caused a difference. The plan already caveats this. | Phase 1: state the caveat. The matched control (LR on pooled log-Mel statistics) is deferred to Phase 2. | #39 (Phase 2) |

## Plan and milestone findings

| Severity | Finding | Resolution |
| --- | --- | --- |
| **High** | The **development reservation was unscheduled**, yet #8 would tune LR and pick its threshold on unreserved development data. This violates AGENTS.md and cannot be undone once scores are seen. The selection metric, tie-break and threshold rule were also not yet predeclared. | New #31 blocks #8. It freezes `dev_select`/`dev_threshold` membership (about 5 of 20 speakers, with coverage checks) and the pre-tuning record. |
| **High** | **Schedule.** G1 was due 12 Oct with 6 of 7 issues open. The 25% contingency is consumed. The owner's focused hours per week, a G0 exit input, were **never recorded**, so feasibility is unverified. | Re-baselined forecast, critical path, parallel lane and cut list (below). G1 is narrowed to its critical path. The owner must record hours/week now. |
| **High** | The **evaluation harness** (freeze state machine, scoring every candidate × condition, persisted predictions, bootstrap) is the largest engineering item before the locked test, and it had no issue or time slot. | New #35, built in G2 while the CNN trains unattended. |
| **High** | **"A07–A19 = unseen attacks" is not quite true.** In ASVspoof 2019 LA evaluation, A16 and A19 reuse the A04 and A06 algorithms, which appear in training (Wang et al., 2020, *Computer Speech & Language*; check against the challenge PDFs bundled in `dataset.zip` on the host). Pooling them into "unseen" inflates the headline. They also give a free **known-algorithm control on unseen speakers**: A16/A19 against A07–A15/A17/A18 on the same 67 evaluation speakers, plus A01–A06 on development. That answers "known versus previously unseen attacks" directly. | #35 slices: development known (A01–A06); evaluation known-algorithm (A16, A19); evaluation unseen (A07–A15, A17, A18) pooled and per attack. `docs/data.md` corrected. |
| Medium | Clip-level metrics are needed by #8, the CNN and the locked evaluation, but nothing owned them. | New #32: one module with known-score tests. |
| Medium | The **CNN input padding** policy was deferred to G2 with no issue. Zero-padding short clips to 4 s would bring back the shortcut option D removed. | New #33: decide repeat-pad, masking or variable-length input before any training run. |
| Medium | The robustness spec lacked details: codec binaries (ffmpeg) sit outside `uv.lock`; the noise source was unspecified (a new noise dataset would be scope creep); augmentation/test overlap was undeclared. | New #34: pinned codec version, generated noise with per-clip seeds, overlap disclosure. |
| Medium | **Bootstrap unit.** The manifest records no 2019 LA source grouping, so the "group by source" commitment needs a predeclared fallback. `dev_threshold` has only about 5 speakers, so threshold uncertainty is large. | #35: resample by `speaker_id` (67 evaluation speakers) and disclose. Report threshold-set counts. |
| Medium | **Presentation-machine risk.** The locked x86_64 `torch` is the CUDA build, so the image may be several GB. The demo laptop cannot install the locked environment natively. A first test in G5 is too late. | #11 now records image size and cold start, and runs once on the presentation machine during G2/G3. |
| Medium | The **deviation register is incomplete**. The 11 Oct sign-off deferred 2021 DF, WaveFake and wav2vec 2.0 "with rationale", but did not record the fields the sign-off itself requires (commitment, reason, decision, date, acceptance impact). | Register added to the architecture plan, "Dated decisions — 13 October 2026". |
| Low | The **stretch choice inverts the proposal's priority order** (RAG is ranked above the proposal's own 2021 DF evaluation item). The S1 gate also cannot pass on the forecast. | **Resolved by the owner the same day:** S1 no-go; 2021 DF leads the Phase 2 backlog. #37 closed. |
| Low | Stale status text: roadmap "current position", README status and platform lines, pipeline "commands that exist". | Corrected. |
| Low | Duration slices were in the course review but not in the TODO. 1,277 clips are under 1 s (#14). | #35 predeclares bins <1, 1–<2, 2–<4, 4–10, >10 s, with per-class counts. |

## Re-baselined schedule

The proposal's dates remain the frozen baseline. These are forecasts, recorded on
GitHub milestone due dates and in the roadmap.

| Gate | Proposal baseline | Plan (6 Oct) | **Forecast (13 Oct)** | Exit content |
| --- | --- | --- | --- | --- |
| G1 | 5–11 Oct | 12 Oct | **16 Oct** | Critical path only: #4/PR #30 → #7 → #31 + #32 → #8 dev report → #10 CNN pilot |
| G2 | 12–18 Oct | 18 Oct | **22 Oct** | #33 CNN clean + augmented, thresholds on `dev_threshold`, all-candidate freeze (#35). Parallel lane: #6, #9, #11, #34, #36; decide #14 |
| Feature freeze | 26 Oct | 26 Oct | **26 Oct (held)** | UI evidence view and demo set done; no new models, datasets or extensions |
| G3 | 19–25 Oct | 25 Oct | **27 Oct** | Locked scoring of every bundle × condition, silence-shortcut diagnostic, grouped bootstrap, failure report, model-card draft |
| G4 | 26 Oct–1 Nov | 1 Nov | **1 Nov (held)** | Fresh-checkout reproduction, p50/p95/memory, container on the presentation machine, model card, slides |
| G5 | 2–4 Nov | 4 Nov | **4 Nov (held)** | Rehearsal ≤ 12 min, backup recording |

**Why it is recoverable.** CNN training is unattended compute on a 44-thread host.
G2 is split into two lanes: the critical path is #33 training plus selection and
thresholds; the *parallel lane* runs while the CNN trains and covers #35 evaluation
harness, #34 transform spec, #36 PCA/K-means, #6 EDA/shortcut audit, #9 scorer
parity and #11 container smoke.

The forecast assumes about 32–43 focused hours a week, the plan's own envelope.
**It is unverified until the owner records actual hours.**

**Cut list.** Items 1, 3 and 4 were applied by the owner on 13 Oct. Apply the rest in order if the forecast slips further; never cut below the line.

1. ~~S1 evidence assistant~~: no-go, decided.
2. The second CNN seed pair: disclose that training variability was not measured, as the proposal permits. With owner-run Kaggle/Colab GPU available, this cut is less likely to be needed.
3. ~~Nice-to-haves #38 and #39~~: moved to Phase 2.
4. ~~UI polish~~: cut to score, spectrogram, measured-summary table and limitations.
5. The K search shrinks to a single predeclared K (still training-fitted).

**Never cut:**
- the development reservation, the freeze record and the locked evaluation;
- the four robustness conditions and the silence diagnostic;
- the failure report, API/UI/container and model card;
- rehearsal and the backup recording.

## Changes made in this review

- **GitHub, new issues:**
  - #31 development reservation + pre-tuning record (G1, blocks #8);
  - #32 metrics module (G1, blocks #8);
  - #33 CNN study (G2);
  - #34 robustness + silence-shortcut spec (G2);
  - #35 evaluation protocol + freeze record (G2);
  - #36 PCA/K-means (G2);
  - #37 stretch-ranking decision (G3);
  - #38 and #39 nice-to-haves.

  All of them were added to the board.
- **GitHub, existing items:**
  - #6, #9 and #11 re-homed to the G2 parallel lane (retitled, with comments);
  - DoD items appended to #6 (shortcut audit + shortcut-only baseline), #8 (blocked by #31/#32; `dev_select` vs `dev_threshold` roles; SQLite `runs` table) and #11 (image size; presentation-machine run);
  - milestones G1/G2/G3 re-dated to 16/22/27 Oct, with baselines kept in their descriptions.
- **Docs:**
  - this file;
  - the architecture plan (dated decisions and deviation register);
  - `TODO-Plan.md` (slice table, forecast column, new checklist items, decision rows);
  - the roadmap (current position, forecast, lanes, cut list);
  - README and pipeline (stale status fixes).

## Open owner inputs

1. **Focused hours per week**, plus any fixed unavailable days before 5 Nov. This determines whether the forecast holds and whether item 2 of the cut list triggers.
2. **The presentation machine** (model, OS, CPU architecture, Docker available?) and whether the venue has network access.
3. ~~A decision on **#37**~~: decided by the owner the same day (see the addendum).
4. A decision on **#14** (sub-1 s duration gate) before the G2 freeze.
5. The demo TTS/voice-clone tool for the consented own-voice demo, and its terms.

## Addendum (same day): owner direction and the two-phase strategy

After reading this review, the owner set the direction. It does not change the
project's direction; it changes the **immediate focus from planning to experiments,
evidence and delivery**.

- **Phase 1 (graduation, 5 Nov)** optimizes for, in order:
  1. a working end-to-end system;
  2. correct methodology;
  3. reproducible experiments;
  4. measured results and failure analysis;
  5. the demo and presentation;
  6. delivery within time and compute.

  Experimental correctness is never traded for the deadline.
- **Phase 2 (post-graduation research)** lives in a separate
  [research backlog](research-backlog.md). Phase 1 keeps the interfaces extensible
  and preserves artifacts, but builds no research-only extensions.
- **Decided, ending #37:** the S1 evidence assistant (RAG) is a **no-go for this
  release**. RAG/MCP are outside the research track. In Phase 2, cross-domain
  transfer (2021 DF and others) leads the backlog.
- **Deferred to Phase 2:** #38 (min t-DCF context) and #39 (matched-input control),
  as optional comparisons. UI polish is cut to the minimum demo view.
- **Not deferred:** the silence/duration shortcut test (#6, #34) becomes the
  project's standout failure analysis. Known versus unseen attacks (A16/A19 control),
  codecs and noise stay core.
- **External GPU (owner-run Kaggle/Colab) is available** as a fallback. The #10 CPU
  pilot decides whether it is needed; the external-compute protocol is in the
  architecture plan.
- **Working mode:** issue by issue, with tested code and verifiable outputs; no
  further broad architecture reviews in Phase 1. The owner chooses the review gate
  for each slice.
- **The project's single question:** *Can we build a reliable AI-generated voice
  detector, and scientifically demonstrate where it succeeds, where it fails, and
  why?*

The Phase 1 critical path is in the [roadmap](roadmap.md).
