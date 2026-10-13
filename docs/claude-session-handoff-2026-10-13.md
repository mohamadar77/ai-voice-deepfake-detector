# Session handoff — 2026-10-13

How to use: paste this file as the first message of a new Claude Code session (or
point the session at `docs/claude-session-handoff-2026-10-13.md`). It is a
pointer map, not a substitute for the sources it names — read `AGENTS.md`,
`TODO-Plan.md`, `docs/data.md` and the memory files before acting.

## Project in one line

Reproducible detector for human vs. synthesized/voice-converted speech on
**ASVspoof 2019 Logical Access (LA)**. Core path: MFCC + Logistic Regression
baseline → compact log-Mel CNN → locked evaluation → robustness/failure analysis
→ FastAPI + Streamlit + Docker + reproducibility. Target presentation **5 Nov
2026**. Budget ~110–150 focused hours.

## Non-negotiables (see AGENTS.md — read it in full)

- Python 3.12 + `uv` only; install `uv sync --frozen --all-groups`; run via `uv run`.
- Preserve official LA train/development/evaluation boundaries. **Evaluation =
  locked final test**: no fitting, selection, tuning, calibration, or inspecting
  predictions on it.
- Split before windowing; aggregate window scores to **one score per clip**.
- Outputs are **scores**, not calibrated probabilities (unless calibration is
  separately fit on reserved dev).
- Never commit audio, features, checkpoints, DBs, `.env`. Record provenance,
  seeds, versions, hashes for every experiment.
- Before completing a code change: `uv run ruff check .` + the smallest relevant
  `uv run pytest`.

## Standing rules (from memory — `.../memory/MEMORY.md`)

1. **Consult an architect/tech-lead agent before non-trivial implementation**,
   and loop to the optimal design first; then implement. (`Agent` tool, `Plan`
   subagent; Codex not wired in.)
2. **Log out-of-scope findings as labeled GitHub issues** (don't re-ask): bug →
   `bug`, enhancement → `enhancement`, nice-to-have → `nice-to-have`, needs a
   frozen decision → `needs-decision`.
3. **CI hardening is parked** (GitHub milestone #7, issues #15–#29) until the
   core ML work is done — do not implement it yet.

## Where things stand (milestones)

- **G0 — DONE.** Environment installs green on the host; ASVspoof 2019 LA acquired,
  integrity-verified, provenance recorded (ODC-By 1.0, checksums, Todisco 2019
  citation) in `docs/data.md`; `docs/environment.md` reconciled to the host.
- **G1 — in progress.** Manifest builder shipped and verified on real data. Next:
  features → baseline. ~4-day slip carried from G0; recoverable.
- G2 CNN+freeze, G3 eval+demo, G4 release+slides, G5 rehearsal: not started.

## What is on `main` now

- `src/voice_detector/`: `protocol.py` (CM parser + `protocol-audit`),
  `manifest.py` (validators **+ the merged builder**: `build_la_manifest`,
  assertions, CSV/SQLite/`catalog_meta`, manifest hash), `preprocessing.py`
  (shared decode→window→MFCC; real-sample windowing via `window_spans`, version
  **mfcc-v2** after PR A), `api.py` (FastAPI
  /health,/ready,/predict; 1–30 s gate, 20 MB cap), `scoring.py` (neutral
  `Scorer` Protocol, no real model yet), `ui.py` (Streamlit client), `config.py`,
  `cli.py` (`smoke`, `protocol-audit`, `build-manifest`).
- Tests: 69 passing on the host (incl. 17 dataset-free manifest tests). CI
  (`.github/workflows/ci.yml`) present.
- Docs: `TODO-Plan.md` (master status), `docs/data.md`, `docs/environment.md`,
  `docs/roadmap.md`, `docs/architecture-and-delivery-plan.md`.

## Verified dataset facts (host build, 2026-10-13)

`build-manifest` over the real data matched published figures exactly:

| split | bonafide | spoof | total | speakers | attacks |
|---|---|---|---|---|---|
| train | 2,580 | 22,800 | 25,380 | 20 | A01–A06 |
| development | 2,548 | 22,296 | 24,844 | 20 | A01–A06 |
| evaluation | 7,355 | 63,882 | 71,237 | 67 | A07–A19 |

Total 121,461. Train/dev share attacks; eval is disjoint (A07–A19) → report
unfamiliar attacks separately. Class imbalance ≈ 1:8.8. Manifest hash
`b172e50c…`. **Finding:** 1,277 clips are shorter than the API's 1.0 s gate
(eval 1,059) — tracked as decision issue #14.

## The compute host

HPE ProLiant DL120 Gen10 (`hpe-dl120gen10-01`), Fedora 44, x86_64, glibc 2.43,
Xeon Gold 6152 (44 threads), 30 GB RAM, **CPU-only**, uv 0.13. Repo clone at
`~/ai-voice-deepfake-detector`; LA data at `data/raw/LA` (git-ignored); manifest
written to `artifacts/manifest/` (git-ignored).

**Workflow:** write code locally, the host runs it. This dev Mac **cannot install
the torch-locked env** (documented macOS x86_64 wheel gap), so verify locally
with `uvx ruff@0.9.10 check …` and an isolated test env:
`PYTHONPATH=src uv run --no-project --with numpy --with soundfile --with pytest python -m pytest tests/test_manifest.py -q`.
The full `uv run --all-groups` suite + real builds run on the host.

## GitHub (repo `mohamadar77/ai-voice-deepfake-detector`)

- Milestones G0(closed)…G5; board **Project #3** "AI Voice Deepfake Detector —
  Delivery" (Todo/In Progress/Done).
- Merged PRs: #1 preprocessing, #2 serving, #3 G0 docs, #12 G1 tracking, #13
  manifest builder (closed #5).
- Issues: **#4** windowing policy — **frozen as option D** (pool MFCCs over real
  samples only; fold sub-frame tails; `<4 s`→one window; bump mfcc-v1→**mfcc-v2**);
  implemented in #7, which closes it. #6 EDA, #7 MFCC features+cache, #8 LR
  baseline, #9 scorer+API parity, #10 CNN timing pilot, #11 CPU container smoke.
  #14 duration-gate decision (`needs-decision`, G2). CI issues #15–#29 parked.

## Current task (in flight)

The architect/tech-lead review for the next phase **converged**. Verdict: ship
as **two PRs, option-D first** (not bundled), because option-D changes feature
values on a tiny pure surface that should be reviewed on its own and #7 declares
"Blocked by #4".

- **PR A — option-D (closes #4), in progress on branch `claude/option-d-windowing`:**
  pure `window_spans` policy in `preprocessing.py`, `clip_feature_matrix` pools
  real samples only, `window_waveform` retired, `mfcc-v1`→`mfcc-v2` (code constant
  `PREPROCESSING_VERSION`, config asserted to match). Also carries the
  `TODO-Plan.md` board-line fix (board now exists as Project #3). Verified locally:
  ruff clean, 22 preprocessing tests pass.
- **PR B — #7 features + cache:** builds on the frozen v2 next (new `features.py`,
  `.npz`-per-split cache keyed on `(manifest_hash, preprocessing_fingerprint)`,
  `extract-features` CLI, no scaler). Awaiting PR A merge.

Local note: the preprocessing tests need librosa; on this x86_64 Mac pin
`llvmlite==0.44.0 numba==0.61.2` in the isolated `uv run --no-project` env
(llvmlite 0.50.0, the locked version, has no macOS x86_64 wheel). Full suite
still runs on the host.

## Next slices after this

#7 features → #8 LR baseline (first real metrics; EER/macro-F1/bal-acc/ROC-AUC on
dev; threshold ≤5% human FA). #6 EDA can run in parallel off the manifest. Freeze
#14 (duration gate) before #8 locks numbers.
