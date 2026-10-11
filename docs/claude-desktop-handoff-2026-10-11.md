# Claude Desktop handoff — ai-voice-deepfake-detector

**Exported:** 11 October 2026 · **Purpose:** portable context for continuing this
project in Claude Desktop (or any fresh Claude session).

> How to use this file: create a Claude Desktop Project named something like
> "Voice Deepfake Detector", add this file as project knowledge (and ideally the
> repo's `AGENTS.md`, `TODO-Plan.md`, `docs/architecture-and-delivery-plan.md`,
> `docs/data.md`, `docs/environment.md`). Then start a chat with: *"Read the
> handoff; confirm the current state and the next action."* This file is a
> point-in-time snapshot — the repo docs remain canonical where they disagree.

---

## 1. What the project is

A reproducible graduation project that detects human vs. synthesized or
voice-converted speech on **ASVspoof 2019 Logical Access (LA)**. Core deliverable
path (in order, enhancements come last):

1. Canonical data contract + manifest (official train/dev/eval splits preserved).
2. Shared preprocessing (16 kHz mono, 4.0 s window).
3. **MFCC + Logistic Regression** baseline → versioned inference bundle.
4. Compact **log-Mel CNN** (one clean run + one augmented run).
5. **Locked final evaluation** with paired robustness + failure analysis.
6. **FastAPI** (`/predict`, `/health`, `/ready`) + **Streamlit** thin client,
   both calling the *same* versioned inference/preprocessing modules as eval.
7. Local **Docker** (CPU inference) + reproducibility/observability records.

Score direction: **synthetic = positive class; larger score = more synthetic.**
Optional stretch (evaluated evidence assistant / RAG) carries **zero committed
hours** and is likely deferred. Target presentation: **5 November 2026**.

The architecture, module contracts, leakage controls and freeze discipline are
**already settled**. Open issues are environmental/factual, not architectural.

## 2. Binding conventions (non-negotiable — from AGENTS.md)

- **Python 3.12 + `uv` only.** Install: `uv sync --frozen --all-groups`. Run via
  `uv run`. Do not add a second environment or lockfile. Don't change the
  lockfile without a recorded platform decision.
- Before completing a code change: `uv run ruff check .` + the smallest relevant
  `uv run pytest`. Tests stay fast and independent of datasets/network/GPU/checkpoints.
- Reusable code in `src/voice_detector/`; config in versioned `configs/*.yaml`
  (no hidden choices in notebooks/CLI defaults). Keep **training separate from
  serving**. Deterministic; record seeds explicitly.
- **Evaluation partition is locked.** Never use eval for fitting, feature
  selection, tuning, checkpoint selection, calibration, threshold selection, or
  post-result retraining. Dev partition does model/checkpoint selection + a
  predeclared **speaker-grouped** reservation for thresholds/calibration.
- Split **before** windowing/transforming. Keep every window/augmented copy/
  feature tied to its clip, speaker, source, official split. **Aggregate window
  scores to one score per clip** before clip-level metrics; bootstrap by source
  recording.
- Freeze protocol, configs, checkpoints, score direction, thresholds, test
  membership, and metrics **before** inspecting final predictions.
- Report EER, macro F1, balanced accuracy, ROC AUC, class counts, confusion
  matrices, human false-positive rate, synthetic detection rate (+ slice counts).
  Report unfamiliar attacks separately. Don't treat correlated 2019/2021 sources
  as independent. Label outputs as **scores**, not proof of authenticity/identity.
- **Privacy:** never commit ASVspoof/third-party/uploaded audio, feature caches,
  checkpoints, SQLite DBs, credentials, or `.env`. Uploaded audio is temporary —
  don't log or retain raw audio. Record dataset provenance/terms/checksums.
- Don't download large datasets, train models, inspect final eval predictions, or
  add cloud services unless the active task explicitly requires it.

## 3. Current state vs. plan (11 Oct 2026)

| Gate / phase | Planned | Actual status |
| --- | --- | --- |
| Scaffolding | repo, env, config, smoke CLI, tests, CI | **Done.** Clean, typed. |
| G0 — feasibility & data contract | revalidated env, data terms/access, capacity | **Blocked.** Data contract written; environment broken; dataset access unconfirmed. |
| G1 — data/preprocess/baseline/API | manifest audit, features, LR bundle, thin API/UI | **Not started.** Only a dataset-free manifest row validator exists. |
| G2–G5 | CNN, freeze, locked eval, robustness, release, demo | **Not started.** |

### What genuinely exists in code (~150 lines)

- `src/voice_detector/config.py` — YAML loader with `${VAR:-default}` env
  expansion + local path resolution.
- `src/voice_detector/cli.py` — `smoke` subcommand printing env/dependency versions.
- `src/voice_detector/manifest.py` — pure validation of manifest rows (required
  fields, label/split enums, checksum format, duplicate `clip_id`). **No manifest
  builder and no protocol parser yet.**
- `tests/` — 4 tests (config + manifest). CI runs on `ubuntu-latest`.

No data pipeline, preprocessing, features, model, training, evaluation, API,
Streamlit, Docker, dataset, or trained artifact exists. Docs are honest about
being plan-only.

## 4. Critical blockers

1. **Documented hardware ≠ this machine.** `docs/environment.md` claims an Apple
   M4 Pro, 48 GB, ARM64, MPS. The review machine is an Intel Core i5-1038NG7
   (4c/8t, 2020 Intel MacBook Pro 13"), 16 GB, x86_64, no Apple Silicon, ~470 GB
   free. The whole compute budget (110–150 h; ~4 CNN runs × up to 12 device-hours;
   MPS) was sized for the M4 Pro. On a 4-core CPU with no GPU, the committed CNN
   matrix likely isn't feasible without a data subset or a different compute target.
2. **Python 3.12 / PyTorch install is broken locally.** Shell Python is 3.14.6;
   project needs 3.12. `.venv` points at `/tmp/voice-detector-python/...python3.12`
   which no longer exists (macOS clears `/tmp`) — the venv is dead and `uv sync`
   never completed. Root cause: PyTorch has **no macOS x86_64 wheel** for
   `torch>=2.6` (last Intel-mac build was 2.2.2), so `uv sync --frozen` can't
   install on an Intel Mac. CI stays green only because it runs on Linux x86_64 —
   masking a locally unusable environment.
3. **Dataset absent, access unconfirmed.** `data/` has only a README. No ASVspoof
   audio/protocols; published-terms access not recorded.

## 5. Missing external resources (what the human must provide/decide)

1. **ASVspoof 2019 LA dataset + official protocols** — register/accept terms at
   the official source (Edinburgh DataShare / ASVspoof 2019 release). LA is
   roughly ~7–8 GB compressed / tens of GB extracted (verify at download); fits
   in ~470 GB free. Nothing progresses past the manifest schema without it.
2. **A decision on the real compute target** (biggest blocker):
   (a) M4 Pro exists and work runs there → fix `environment.md`, budget stands;
   (b) this Intel i5 is the real machine → torch can't install as pinned; needs a
   recorded platform decision (older torch / Linux or cloud training / declared
   data subset) and a renegotiated CNN budget;
   (c) training on another Linux machine.
3. **A persistent Python 3.12 interpreter** (not under `/tmp`).
4. **Confirmed hours/week.** At 15–20 h/week the full scope is not a credible
   5 Nov commitment without help or a scope cut.

## 6. Recommended path forward (G0 first; no implementation until decisions land)

1. **Resolve compute & environment (G0):** pick the compute target; get a
   persistent Python 3.12; make `uv sync --frozen --all-groups`,
   `uv run voice-detector smoke`, `ruff`, `pytest` green locally; update
   `environment.md` with real hardware; if Intel Mac, record the PyTorch platform
   decision explicitly.
2. **Confirm & record dataset access (G0):** accept ASVspoof terms; record source
   URLs, release IDs, checksums, terms in `docs/data.md`; then download LA.
3. **First code slice — data contract (G1):** official-protocol parser → canonical
   manifest builder (CSV + SQLite) → metadata-only audit (uniqueness,
   speaker/source overlap, class balance) → tiny fixtures. This is the correct
   first implementation target, **not** the CNN.
4. Then shared preprocessing → LR baseline + bundle → thin API/UI → CNN → freeze
   → locked eval → robustness → release/demo.

## 7. Canonical repo docs (source of truth; read these in the checkout)

- `AGENTS.md` — binding conventions + evaluation/privacy constraints.
- `TODO-Plan.md` — completion checklist (source of truth for "done").
- `docs/architecture-and-delivery-plan.md` — scope, module contracts, gates
  G0–G5, TODOs A0–A9, stretch S1, 110–150 h budget.
- `docs/data.md` — canonical manifest schema + data contract + G0 env check.
- `docs/environment.md` — hardware/env record (**contains the M4 Pro claim that
  contradicts the live Intel machine — reconcile before trusting**).
- `docs/pipeline.md`, `docs/project-management.md`, `docs/roadmap.md` — workflow,
  board/milestones, execution order.
- `docs/status-2026-10-11.md` — the in-repo version of this snapshot.

## 8. Immediate decision point for the human

Pick one so work can resume: **(a)** draft the corrected G0 environment/
feasibility update + dataset-access checklist; **(b)** lay out the detailed data-
contract (manifest builder) implementation design; or **(c)** decide the compute
target first. No code should be written until (a)/(c) is resolved.
