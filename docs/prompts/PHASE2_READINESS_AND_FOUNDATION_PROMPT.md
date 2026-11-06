# Agent Task: Prepare the AI Voice Deepfake Detector for Phase 2

**Repository:** https://github.com/mohamadar77/ai-voice-deepfake-detector  
**Primary source plan:** `INCREMENTAL_UPGRADES_AND_ROBUSTNESS_HARDENING.md` (689 lines; supplied separately / available in the project materials)  
**Current project documents:** `TODO-Plan.md`, `docs/roadmap.md`, `docs/architecture-and-delivery-plan.md`, `docs/research-backlog.md`, `docs/README.md`

## Objective

Prepare the repository, project planning, architecture contracts, and engineering foundations for the next research/development phase.

The next phase should build on the existing Phase 1 system and evidence. It should **not** redesign the project from scratch, rewrite working components, or immediately start model training. The central goal is to make the project genuinely ready for multi-dataset research, reliable experiment tracking, and rigorous cross-domain evaluation.

Use the hardening plan as a set of requirements to audit—not as a mandate to reproduce its proposed file structure or checklists. Reuse existing implementation and documentation where they already satisfy a requirement. Do not duplicate existing functionality or create another source of truth.

**Carry out the work in the repository.** Inspect current state, make justified planning/design/documentation/code-foundation changes, validate them, and report what remains before actual Phase 2 experiments can begin.

---

## 1. Mandatory operating rules

1. Start from the latest `main` branch and inspect the latest commits, issues, milestones, working-tree state, and actual source code before editing. Findings below are from a prior review and may have changed.
2. Use a dedicated branch or the repository's established PR workflow. Do not commit directly to `main` unless that is explicitly required by the repository workflow.
3. Preserve unrelated user changes. Never reset, overwrite, or discard uncommitted work.
4. Do not modify the frozen Phase 1 evaluation results, thresholds, predictions, checkpoint selection, or freeze records.
5. Do not use the Phase 1 locked evaluation set to tune, retrain, select models, select thresholds, choose transforms, or inspect new errors. A new experiment needs a separately declared protocol and an appropriate untouched evaluation set.
6. Do not start training, expensive scoring, dataset downloads, or a new research experiment as part of this foundation task.
7. Do not invent successful tests, data provenance, run records, research outcomes, or completion status. Verify claims from code, artifacts, tests, Git history, or authoritative issue records.
8. Do not create a new general-purpose project tracker. Retain the existing documentation ownership model.
9. Keep changes incremental. Prefer explicit interfaces and adapters over broad refactors.
10. If a proposed task is unnecessary, already implemented, unsafe for the frozen experiment, or too broad, explain why and leave it out.

## 2. Repository and source-of-truth audit

Read at least:

- `README.md`
- `AGENTS.md`
- `TODO-Plan.md`
- `docs/README.md`
- `docs/roadmap.md`
- `docs/architecture-and-delivery-plan.md`
- `docs/research-backlog.md`
- `docs/data.md`
- `docs/environment.md`
- `docs/evaluation-protocol.md`
- `docs/evaluation-results-g2.md`
- `docs/deployed-model-decision.md`
- `docs/model-card.md`
- `docs/reproduction.md`
- `docs/pipeline.md`
- `INCREMENTAL_UPGRADES_AND_ROBUSTNESS_HARDENING.md`, when supplied/available
- Relevant implementation files under `src/voice_detector/`, especially `manifest.py`, `prediction_store.py`, `evaluation.py`, `bootstrap.py`, `scoring.py`, `preprocessing.py`, `baseline.py`, `pretuning.py`, `augmentation.py`, and serving modules
- Relevant tests, configuration files, experiment artifacts, and GitHub issues/milestones

Produce an accurate mapping of requirements to current implementation:

- `implemented and verified` — include concrete code/test/artifact evidence;
- `partially implemented` — identify the missing guarantee;
- `documented but not implemented`;
- `not implemented`;
- `not applicable / explicitly deferred`.

Do not equate a design statement with implementation evidence. Do not mark an issue complete solely because it has a PR or a document reference.

### Prior review observations to re-check

The prior review found that:

- The project has a manifest builder, an SQLite manifest/catalog, a prediction store, a DRAFT → FROZEN → SCORED → REPORTED evaluation lifecycle, speaker-grouped bootstrap support, a shared scorer, FastAPI/Streamlit serving, Docker packaging, and a versioned model bundle.
- `manifest.py` contains ASVspoof 2019 LA-specific protocol paths, dataset identity, split assumptions, and audio resolution.
- The canonical manifest uses `source_id = None` for LA because the source-recording grouping is not published for that dataset; speaker grouping is the documented fallback.
- The prediction store is designed around the existing LA evaluation dimensions and should be assessed for multi-dataset identity and source/domain metadata.
- JSON run records exist, but a queryable SQLite `runs` registry is not implemented. The documentation should make a deliberate decision about the role of JSON versus SQLite.
- Some code review follow-ups raise potentially important future-training integrity issues.

Verify every observation against the latest branch. Do not assume it remains current.

---

## 3. Phase 1 closure and Phase 2 planning transition

The previous repository snapshot still showed Phase 1 release sign-off incomplete and old October/November dates. The target presentation date was 5 November 2026, which has now passed. Reconcile the actual outcome; do not assume the presentation or release did or did not happen.

### Required work

1. Inspect the latest release tags, commits, pull requests, issues, project milestones, and artifacts to establish the real Phase 1 outcome.
2. Reconcile old Phase 1 issues and the TODO plan. For each outstanding item, determine whether it was completed, waived, superseded, or remains unresolved.
3. Preserve `TODO-Plan.md` as the historical Phase 1 completion/sign-off record, or make only a clearly justified minimal transition update. Do not repurpose it into another Phase 2 tracker.
4. Update `docs/roadmap.md` to show the actual Phase 1 result and the new Phase 2 starting point. Remove expired planning dates and expired feature-pull rules where appropriate.
5. Update `README.md` and `docs/README.md` where the project status or links are stale.
6. Keep the frozen original proposal unchanged. Record deviations from it in the architecture/deviation record, not by editing the original proposal.
7. Preserve explicit caveats, including the fact that the deployed-model selection rule was documented after the locked report had already been read. Never describe this selection as blind or preregistered.
8. Preserve the Phase 1 locked evaluation as immutable historical evidence.

### Phase 1 issues to reconcile if still open

Check current status rather than blindly closing these:

- [#107 — fresh-checkout reproduction](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/107)
- [#108 — presentation host/container/fallback](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/108)
- [#109 — demo data provenance](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/109)
- [#110 — model card and redistribution decision](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/110)
- [#111 — rehearsal, backup, and release tag](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/111)
- [#112 — dataset mirror URL](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/112)

If an item cannot be completed or verified, retain it as an explicit limitation/deviation with an owner and next action. Do not fabricate a URL, release tag, test result, or rehearsal record.

### Phase 2 planning requirements

Create a clear Phase 2 milestone structure using the existing `docs/research-backlog.md` and GitHub milestone conventions. The research backlog is the Phase 2 scope authority; GitHub issues track actionable implementation tasks.

Close the Phase 1 pull gate. It must no longer suggest that Phase 1 extensions can still be pulled after the feature-freeze/presentation period.

Add a small set of research gates to `docs/research-backlog.md` (or an equally appropriate existing document), with prerequisites and exit criteria:

- **Gate 0 — Foundation:** Phase 1 evidence preserved; dataset-agnostic contracts, provenance, run identity, artifact integrity, and evaluation safeguards ready.
- **Gate 1 — Cross-domain generalization:** ASVspoof 2021 DF transfer evaluation, source-overlap analysis, and optionally a separately approved external dataset such as In-the-Wild or WaveFake.
- **Gate 2 — Model and robustness research:** shortcut-robust training, matched-input controls, stronger baselines, SSL/audio embeddings, and additional seeds under new protocols.
- **Gate 3 — Reliability and publication readiness:** calibration/abstention if justified, statistical analysis, modern generator evaluation, literature/novelty review, and a paper/no-paper decision.

These are gates, not fixed-date promises. Preserve the current backlog's priorities, caveats, and conditional scope. Do not add RAG/MCP as research work: the owner decision says they are not on the research track except potentially as a separate engineering demo.

---

## 4. Architecture preparation: multi-dataset ingestion

This is the most important structural preparation for Phase 2.

### Current issue to verify

The existing `src/voice_detector/manifest.py` contains ASVspoof 2019 LA-specific details such as protocol filenames, split names, directory mappings, dataset constants, and audio path resolution. This is reasonable for Phase 1 but will make additional datasets awkward if all logic is added to the same module.

### Required design

Introduce or document a clear boundary between:

1. **Dataset-specific adapters/builders** — understand each dataset's directory structure, protocol files, label conventions, metadata, source maps, official partitions, and license/terms.
2. **Canonical dataset/manifest contract** — provides stable project-level fields to shared processing and evaluation code.
3. **Shared pipeline** — preprocessing, feature extraction, training, scoring, evaluation, and reporting consume the canonical manifest/contracts rather than raw dataset layouts.

Support these fields where available:

- `dataset_id` and `dataset_version`;
- stable `clip_id`;
- `speaker_id`;
- `source_id` / `utterance_id` or another explicit source-recording grouping key;
- canonical label (`bonafide` / `spoof`) and label mapping provenance;
- `official_split` and project-specific split/role, kept distinct;
- attack, codec, vocoder, subset, domain, and other dataset-specific metadata;
- file checksum and, where useful, decoded-waveform checksum;
- source-map/protocol version and hashes;
- dataset terms/license reference and ingestion/build version;
- derived sample parent identity, window bounds, transform name/configuration/seed/version when relevant.

Rules:

- Never infer speaker, attack, codec, or source identity from a filename unless the mapping is documented, authoritative, and tested.
- Preserve the official dataset split labels. Do not overwrite them with project split names.
- Derived windows and transformations must remain linked to the original clip and cannot cross prohibited splits.
- Unknown source metadata must remain explicitly unknown; do not fabricate source grouping.
- Keep licensed audio out of Git unless redistribution is allowed.

### Implementation approach

- Retain reusable validation and manifest utilities in their existing module where practical.
- Add dataset-specific adapters/builders for additional datasets rather than rewriting the Phase 1 LA builder.
- Do not force the source plan's illustrative directory tree on the repository.
- Add a small dataset-adapter interface and tests before adding a large new dataset integration.
- First target should be ASVspoof 2021 DF because it is the highest-priority existing research-backlog item and is evaluation-only for the initial transfer experiment.

### Acceptance criteria

- The LA builder still produces the existing canonical LA manifest without changing the Phase 1 artifact identity.
- A new dataset can be ingested without editing the shared preprocessing/model/evaluation code for its raw folder structure.
- Dataset identity, official split, project role, source provenance, metadata availability, and terms are represented explicitly.
- Tests prove label mapping, protocol/split preservation, stable IDs, checksums, parent/derived identity, and refusal on invalid or conflicting rows.

---

## 5. Architecture preparation: evaluation and prediction identity

The current evaluation lifecycle and prediction store are valuable foundations. Extend their identity model carefully rather than creating a second evaluation framework.

### Required design decisions

Every evaluation/freeze should identify at least:

- dataset ID and dataset version;
- manifest hash and manifest schema version;
- official protocol identity/version/hash;
- source-map identity/version/hash where relevant;
- model candidate and frozen bundle hash;
- preprocessing and feature version/config hashes;
- scoring/aggregation rule and score direction;
- threshold and the data role used to select it;
- conditions/transforms and their versions/config hashes/seeds;
- evaluation split/subset and source grouping used for uncertainty;
- prediction-store identity/hash and report hash;
- Git commit, dirty-state disclosure, dependency/environment identity.

Each new dataset/experiment must have its own freeze record and a separately auditable lifecycle. Do not modify or reuse the Phase 1 G2 freeze record.

### Prediction-store requirements

Assess whether the existing SQLite `prediction_store.py` needs fields or related metadata for dataset/domain/source-map identity. Add only fields that are genuinely necessary for cross-dataset comparisons and traceability. Preserve one-prediction-per-original-clip/candidate/condition semantics where appropriate.

### Acceptance criteria

- A freeze cannot be scored using a different dataset, manifest, protocol, source map, config, or model artifact than the one it froze.
- A report identifies which dataset/domain and grouping unit each result represents.
- Overlapping sources are not described as independent evidence.
- Full official-set transfer results and any no-overlap subset are separate, clearly labeled analyses.
- Phase 1 report and prediction artifacts remain unchanged.

---

## 6. Experiment tracking and artifact publication

### Current design gap to resolve

The project has JSON run records, a SQLite manifest/catalog, and a SQLite prediction store. The architecture/backlog says a SQLite `runs` table is intended, but GitHub issue #54 identifies missing run persistence and incomplete metadata.

Make an explicit design decision before implementing a registry:

- JSON run records should remain durable, human-readable, exportable records for each experiment; and
- SQLite may act as a queryable index for datasets, runs, predictions, and artifacts.

Alternatively, propose a simpler JSON-only approach if it meets the research query/reporting needs. Do not add a database table merely because an older plan mentions it. Record the decision and rationale in the architecture plan and Phase 2 backlog.

### Required run metadata

Where applicable, record:

- unique run ID;
- dataset ID/version and manifest hash;
- config and preprocessing hashes/versions;
- seed and model family/version;
- Git commit and dirty working-tree state;
- dependency versions, Python/runtime, hardware and relevant library/codec versions;
- train/development/evaluation roles and protocol/freeze identity;
- checkpoint/bundle hash and threshold/selection-rule provenance;
- prediction and metric artifact paths/hashes;
- runtime, status, and failure information.

### Artifact integrity

- Publish a model plus run metadata as one coherent immutable bundle.
- Do not overwrite prior experiment evidence during retries or reruns.
- Stage, validate and checksum artifacts before publishing a reference to them.
- Ensure a failed write cannot leave a model paired with another run's threshold/record.
- Distinguish a fresh run from an idempotent restore.
- Keep large or licensed artifacts outside Git where appropriate, with retrieval instructions and checksums.

### Acceptance criteria

- A run can be traced from report → predictions → run record → checkpoint/config → dataset manifest/protocol.
- A run record cannot claim settings that the code did not actually execute.
- Failure injection proves that artifact publication is coherent and prior evidence survives failed writes.
- The chosen tracking model has tests and documentation; no duplicate tracking system is introduced.

---

## 7. Resolve critical integrity follow-ups before new training

Re-check each issue on the latest branch and confirm whether it is still open or has been fixed. The prior review identified these as especially important:

| Issue | Previously identified gap | Required action |
|---|---|---|
| [#52](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/52) | Stale/wrong-partition feature caches can be accepted by LR training | Validate partition, preprocessing/cache fingerprint, schema/dimensions, and exact clip/label alignment before fitting |
| [#53](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/53) | Current manifest not fully validated against frozen development reservation before LR tuning | Verify the full supplied catalog/manifest against the committed reservation; require exact selection/threshold membership and complete disjoint masks |
| [#55](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/55) | LR pipeline and threshold/run record can be published inconsistently | Stage and validate an immutable bundle; add write-failure tests proving prior bundle survives |
| [#57](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/57) | Pre-tuning config accepts declarations the implementation may not honor | Reject unsupported aggregation, selection, or regularization declarations, or implement them faithfully |
| [#54](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/54) | Missing/partial run registry and reproduction metadata | Decide JSON/SQLite responsibilities and implement the accepted design |
| [#63](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/63) | Robustness YAML version/parameters are not checked against code constants | Add strict loading/version/condition validation before new robustness experiments |
| [#66](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/66) | Duplicate `BootstrapSpec` definitions can drift | Consolidate to one canonical type before extending statistical analysis |

The prior issue text says some behaviors were demonstrated with synthetic reproductions. Do not reinterpret those reproductions as evidence that past Phase 1 evaluation actually used leaked data. Preserve the scope of each finding accurately.

Other issues to triage according to relevance:

- [#64](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/64): clarify `CodecResult.samples_delta` semantics or expose pre-truncation output length.
- [#58](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/58): use versioned workload inputs, data-preparation cost, and planning margin in capacity forecasting before expensive SSL/model matrices.
- [#59](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/59): keep Torch-dependent test collection compatible with the documented Intel Mac fallback while still running on Torch-equipped CI/host.
- [#67](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/67): low-priority dead evaluation metadata cleanup.
- [#89](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/89): finish serving security/reliability review before extending exposure or deployment.

Do not implement every historical issue automatically. Identify what blocks Phase 2 data ingestion, training, evaluation, reproducibility, or serving, then prioritize accordingly.

---

## 8. Phase 2 research design: first experiment

The first substantive research direction should be **cross-domain generalization using ASVspoof 2021 DF**, in line with the existing research backlog and the original proposal's second-domain research question.

This task is protocol/design preparation only. Do not download the dataset or score it yet unless a separately approved implementation task explicitly authorizes that action.

### Required protocol decisions

1. Dataset version, access source, terms, official protocol, and expected metadata.
2. Dataset-specific manifest adapter and canonical label mapping.
3. Source-overlap audit using available published source maps.
4. Confirm whether the initial experiment is strictly evaluation-only on frozen Phase 1 bundles.
5. Define full official-set reporting and separately labeled no-overlap subset analysis where feasible.
6. Define codec/vocoder/subset breakdowns based on metadata actually available.
7. Define source-grouped uncertainty and an explicit fallback when source identity is unavailable.
8. Define a new Phase 2 freeze record; do not touch Phase 1 G2.
9. Define all required counts, metrics, plots, error tables and limitation statements before scoring.
10. Define stop conditions if access, mappings, runtime, or label consistency is inadequate.

### Scientific safeguards

- Keep the complete official protocol intact for the primary transfer report.
- Do not silently drop samples to create a more favorable subset.
- Report overlapping and no-overlap results separately where possible.
- Do not describe transfer data with source overlap as independent confirmation.
- Do not change the Phase 1 model based on results from this test.
- If Phase 2 later trains on 2021 DF or other external data, establish a new, untouched evaluation set for that research question.

---

## 9. Phase 2 research gates and backlog structure

Retain the current backlog topics and priority order unless evidence justifies a change. Organize them into gates, with prerequisites and exit criteria, rather than turning the backlog into a date-based promise.

### Gate 0 — Foundation

- Phase 1 release and evidence status reconciled.
- Artifact archive verified and retrieval documented.
- Dataset adapter/canonical manifest contract agreed.
- Dataset-aware freeze/prediction identity agreed.
- Run-record and artifact-publication design agreed and tested.
- Critical cache/reservation/config-integrity issues resolved or explicitly excluded from the first experiment with a defensible reason.

**Exit:** a clean Phase 2 workspace can reproduce/inspect the Phase 1 evidence and safely ingest a new dataset without risking Phase 1 artifacts.

### Gate 1 — Cross-domain generalization

- ASVspoof 2021 DF adapter and protocol audit.
- Source-map overlap/no-overlap analysis.
- Evaluation-only transfer of frozen Phase 1 bundles.
- Separate reporting by the metadata actually available.
- Optionally, separately approved In-the-Wild or WaveFake external data after terms and provenance checks.

**Exit:** a protocol-compliant and traceable account of transfer performance and limitations.

### Gate 2 — Model and robustness research

Potential work from the existing backlog:

- Development-data-only investigation of silence/duration shortcut reliance.
- Shortcut-robust training with new predeclared candidate/freeze protocol.
- Matched-input controls (#39) to distinguish representation effects from model-architecture effects.
- Stronger baselines and controlled ablations.
- Frozen SSL/wav2vec 2.0/XLS-R/WavLM representations where resources permit.
- Additional seeds and bounded hyperparameter studies.

**Exit:** fair, repeatable comparisons under new declared protocols; results include uncertainty and compute/inference cost.

### Gate 3 — Reliability and publication readiness

Potential work from the existing backlog:

- Calibration and abstention, fitted and evaluated only with appropriate development/test separation.
- min t-DCF (#38) if official ASV scores and metric protocol are appropriate.
- Paired statistical analysis, grouped bootstrap/permutation tests, and multiple-comparison handling.
- Modern TTS/voice-cloning generator evaluation with tool/version/terms provenance.
- Contemporary related-work and novelty review.
- A paper/no-paper decision based on actual results and reproducibility.

**Exit:** a defensible research package and an evidence-based decision about publication. Publication is not guaranteed.

Do not add RAG/MCP/agents to this research track. They were explicitly removed from the research direction; only revisit them as a separate engineering demonstration if the owner requests it.

---

## 10. Testing and validation requirements for foundation changes

Add to the existing test suite. Do not create a new testing framework.

### Dataset and leakage tests

- Duplicate IDs/conflicting labels and malformed splits are rejected.
- Dataset adapter preserves the official partition and applies the intended label map.
- Source-group and derived-window/transform split leakage is detected where metadata permits.
- Unknown source metadata remains unknown rather than being fabricated.
- Manifest/catalog counts and joins reconcile.
- Dataset/version/protocol/source-map hashes participate in the appropriate experiment identity.

### Cache/training integrity

- Stale or mismatched preprocessing cache is rejected.
- Wrong-partition features cannot reach fitting.
- Cached clip IDs and labels exactly match the expected manifest/roles.
- Frozen reservation is checked against the actual current manifest before training.
- Unsupported config declarations are rejected.
- Failed bundle publication leaves the previous valid bundle intact.

### Evaluation and tracking

- Freeze refuses a changed dataset, manifest, protocol, source map, preprocessing config, model bundle, or condition spec.
- Predictions remain one per original clip/candidate/condition as intended.
- Grouped bootstrap samples groups rather than individual windows/clips when groups exist.
- Reported values trace to predictions and a run record.
- JSON records and any SQLite index reconcile.

### Serving (only if serving contracts are changed)

- Valid input returns the existing typed response with correct version/score semantics.
- Unsupported, corrupt, empty, silent, too-short/too-long, oversized or undecodable input gets a safe non-success response.
- Missing/corrupt model makes readiness fail.
- Temporary audio is cleaned on success and exceptions.
- Logs exclude raw audio/payloads and sensitive file paths.
- Offline scoring and serving golden tests remain in tolerance.

Run only fast/dataset-free tests for this task. Do not trigger large training or evaluation workloads.

---

## 11. Documentation and restructuring decisions

Keep the current documentation ownership model:

- `README.md`: project overview, supported usage and quick start.
- `TODO-Plan.md`: Phase 1 status/completion record; do not make it a second Phase 2 task tracker.
- `docs/README.md`: documentation index and authority per subject.
- `docs/roadmap.md`: current project phase and major gates, not another detailed checklist.
- `docs/architecture-and-delivery-plan.md`: architecture contracts and decisions/deviations.
- `docs/research-backlog.md`: Phase 2 research scope, priorities, gates and exit criteria.
- `docs/data.md`: LA dataset provenance and its specific contract; link to the general dataset adapter contract if one is added.
- `docs/evaluation-protocol.md` and `docs/evaluation-results-g2.md`: immutable Phase 1 protocol/results.
- `docs/reproduction.md`: verified Phase 1 reproduction instructions.
- `docs/model-card.md`: Phase 1 deployed model and its actual limitations.

Do not create generic duplicates such as `DATASETS.md`, `EVALUATION_PROTOCOL.md`, `REPRODUCIBILITY.md`, or `MODEL_CARD.md` if the existing lower-case documents already serve those roles.

The hardening plan itself need not be copied into `docs/` or kept as a second checklist. Its requirements should be mapped to the existing files and issues. If it is preserved, label it as historical input, not an active status authority.

---

## 12. Execution boundaries: what to do now and what to defer

### Do now — before substantial Phase 2 implementation

- Phase 1 status/commit/artifact reconciliation.
- Close the expired Phase 1 gate and define Phase 2 milestones/gates.
- Decide and document multi-dataset manifest/adapter contracts.
- Decide dataset-aware freeze and prediction identity.
- Decide authoritative JSON run record versus SQLite indexing responsibilities.
- Resolve or explicitly sequence critical cache, manifest/reservation, unsupported config, and artifact publication safeguards.
- Define the ASVspoof 2021 DF transfer protocol and source-overlap handling.
- Add regression tests for these foundations.

### Can be deferred until the relevant experiment

- Actual ASVspoof 2021 DF download, decoding, ingestion, and scoring.
- WaveFake/In-the-Wild ingestion until data terms and protocol are approved.
- SSL/wav2vec/XLS-R/WavLM implementation and GPU training.
- New architectures such as LCNN, RawNet-style, AASIST-class, or additional feature front-ends.
- Shortcut-robust model retraining.
- Matched-input controls and min t-DCF if not selected in the first research protocol.
- Calibration/abstention unless probability output is an explicit research objective.
- Cloud deployment, RAG, MCP, agents and multi-agent work.

### Do not do

- Do not retrain or retune against Phase 1 evaluation labels/predictions.
- Do not change locked Phase 1 numbers or artifacts.
- Do not redesign the repository tree solely to resemble the source plan's example.
- Do not build a generic multi-dataset abstraction beyond the requirements of the first planned dataset.
- Do not add optional technologies just to cover syllabus topics.
- Do not claim publication novelty before a contemporary literature review.

---

## 13. Required deliverables from this task

Make the repository changes and then provide a concise but evidence-based report with:

1. **Phase 1 closure:** actual release/presentation state, tag/commit if present, outstanding issues, waivers, unresolved evidence.
2. **Phase 2 planning:** updated milestone/gate sequence, ownership, prerequisites, exit criteria, and issue assignments.
3. **Architecture decisions:** canonical manifest/dataset adapter, evaluation identity/freeze model, prediction-store changes, run-record/SQLite decision, artifact publication model.
4. **Code changes:** files and interfaces changed, with a reason for each; no broad rewrite.
5. **Integrity fixes:** current state of #52, #53, #54, #55, #57, #63, #66 and other relevant issues; evidence for completion or reason for deferral.
6. **First experiment readiness:** proposed ASVspoof 2021 DF protocol, source-map requirements, overlap policy, metrics, uncertainty grouping, and stop conditions.
7. **Tests:** exact commands run and observed pass/fail/skip counts. Separate dataset-free checks from anything requiring actual data or the remote host.
8. **Remaining blockers:** explicitly identify everything that still prevents Gate 0 or Gate 1 from starting.
9. **Git status:** branch, changed files, commit/PR if created.

### Final acceptance condition for this task

The repository is ready to begin the first Phase 2 dataset-ingestion and transfer-evaluation implementation only when:

- Phase 1 evidence and artifacts are preserved and their status is unambiguous;
- the Phase 2 scope/gates and first experiment are explicit;
- the canonical multi-dataset identity and adapter contract is agreed;
- evaluation freeze/prediction identity covers dataset/protocol/source-map provenance;
- run records and immutable artifact publication have a documented, testable design;
- critical cache/reservation/config-integrity issues are fixed or explicitly dispositioned with a reason;
- tests demonstrate that the foundation cannot silently mix dataset versions, partitions, manifests, configurations, or artifacts.

If any condition is not met, do not claim Phase 2 implementation readiness. State exactly what remains and what should be done next.

**Start with the latest repository audit, then make the smallest set of changes that establishes these foundations.**
