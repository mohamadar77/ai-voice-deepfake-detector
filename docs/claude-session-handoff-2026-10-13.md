# Phase 1 resume brief: for the implementing senior engineer (agent)

Updated: 13 October 2026, end of the architecture-review session. This replaces the
earlier same-day handoff.

**Use:** give this file to a fresh Claude Code session acting as **senior ML/software
engineer**. It is a pointer map and work order, not a substitute for its sources.
Read them in this order before acting:

1. `AGENTS.md`: binding conventions and evaluation non-negotiables.
2. [`docs/roadmap.md`](roadmap.md): the **Phase 1 critical path** (P0–P5), deferrals, stop/go rules.
3. [`TODO-Plan.md`](../TODO-Plan.md): master status, the critical-path table and the decisions log.
4. [`docs/architecture-and-delivery-plan.md`](architecture-and-delivery-plan.md), section "Dated decisions — 13 October 2026": deviation register, owner decisions, external-GPU protocol.
5. [`docs/architecture-review-2026-10-13.md`](architecture-review-2026-10-13.md): why the plan looks like this.
6. [`docs/data.md`](data.md): data contract, verified counts, the A16/A19 note.
7. The GitHub issue you are about to work on. Issues carry the definition of done.

## Mission

**Phase 1 (graduation, 5 Nov 2026):** a complete, scientifically credible,
reproducible voice-deepfake detector. It must answer one question: *can we build a
reliable AI-generated voice detector, and scientifically demonstrate where it
succeeds, where it fails, and why?*

Work **issue by issue, with tested code and verifiable outputs**. Do not commission
broad architecture reviews. Do not build Phase 2 items; they are in
[`research-backlog.md`](research-backlog.md) and can only be pulled in through its gate.

## Standing rules (owner, durable)

1. **Before each implementation slice, ask the owner which review gate to use:**
   - full `/consult` (two-model design review, then wait for a go);
   - implement, then `/code-review`;
   - no review.

   Never decide it yourself. Never batch several slices under one answer.
2. When `/code-review` is chosen, relay the ranked findings and apply fixes **only
   after the owner approves each one**.
3. Log out-of-scope findings as labeled GitHub issues: `bug`, `enhancement`,
   `nice-to-have`, `needs-decision`. Do not re-ask for permission.
4. CI/DevEx hardening (#15–#29) is **parked** until core ML is done.
5. Never edit the proposal `.docx` (SHA-256 `b16f5124…`). Plan changes go in Markdown,
   with dated decision rows.
6. Commit and push only when the owner asks. The established pattern is one PR per
   issue, with the attribution lines the harness provides.

## Non-negotiables you will touch soon

- **No tuning or threshold selection on development data before #31 freezes**
  `dev_select` / `dev_threshold` membership.
- **Evaluation partition is locked.** No fitting, selection, tuning or threshold
  choice on it, and no inspection of evaluation predictions before the #35 freeze
  record is `FROZEN`. Evaluation audio and features **never leave the DL120 host**;
  external GPUs see training and development data only.
- One score per clip (metrics reject duplicates). Synthetic = positive; higher
  score = more synthetic; `score >= t` means spoof. Outputs are *scores*.
- **Known versus unseen attacks:** evaluation A16/A19 reuse the A04/A06 algorithms.
  The headline unseen result is A07–A15, A17 and A18. Never pool A16/A19 into it.
- No advanced or pretrained models in Phase 1. Keep the declared tuning budget.
  Extensive hyperparameter optimization is Phase 2.

## Environment

- **Development loop (owner, 13 Oct):** develop **locally** on the laptop → push →
  CI → **test and execute on the DL120 host** (the target and demo environment).
  - A slice is done only when local checks, CI and the host's full `uv run pytest`
    are green on the **same commit**, and any real-data output has a run record
    citing that commit.
  - Never patch code on the host; fix locally and push.
  - Steps are in `docs/environment.md`, "Development loop".
- **Host access (owner, 13 Oct):** remote over SSH from the laptop, for testing,
  execution and the demo. Disk budget: robustness conditions on the fly, no cached
  augmentation, ≥30 GB free. Implementation rules:
  - long jobs run detached (`tmux` / `systemd-run --user`) with file logs, and are **resumable**: CNN checkpoints each epoch with optimizer and RNG state; cache writes are atomic;
  - API/UI bind to `localhost` and are reached via an SSH tunnel, with no public ports;
  - when giving the owner a host command, wrap long ones in `tmux new -s <name> '…'`.

  Details are in `docs/environment.md`, section "Access".
- **Local development (laptop, Intel Mac x86_64):** cannot install the locked env
  (no torch or llvmlite wheels). Make torch-dependent tests skip locally with
  `pytest.importorskip("torch")`; CI and the host run them. Lint with `uvx ruff@0.16.10 check .`. Test dataset-free modules in an
  isolated env pinned to the lock:

  ```bash
  PYTHONPATH=src uv run --no-project --python 3.12 --with numpy==2.5.3 --with scikit-learn==1.9.1 --with soundfile==0.14.0 --with pyyaml==6.0.3 --with pytest==9.1.1 python -m pytest <tests> -q -p no:cacheprovider
  ```

  Tests importing librosa need `--with llvmlite==0.44.0 --with numba==0.61.2 --with librosa`.
  The first run is slow (cold compile); warm runs take seconds.
- **DL120 host** (`hpe-dl120gen10-01`; Fedora 44, 44 threads, 30 GB, CPU-only):
  - target and demo environment: runs `uv sync --frozen --all-groups`, the full suite on every pushed slice, and every data, training and evaluation job;
  - repo at `~/ai-voice-deepfake-detector`, LA data at `data/raw/LA`, manifest at `artifacts/manifest/` (all git-ignored);
  - the owner runs host and Kaggle/Colab jobs; give exact commands.

## Open PRs and merge order

| PR | Content | Base | State at handoff |
| --- | --- | --- | --- |
| #30 | Option D windowing, `mfcc-v2` (closes #4) | `main` | Open, mergeable |
| #40 | Docs: review, Phase 1 critical path, Phase 2 backlog, this brief | #30's branch (stacked) | Open. **After #30 merges, retarget to `main`.** |
| #41 | Metrics module (closes #32) | `main` | Open. Review fixes applied (`8f1f6c8`; 42 metrics tests; all 9 formula mutants killed). **Before merge:** CI green, then the full host suite on `8f1f6c8` (expect 111 tests). |

## Work order (pick the top unblocked item)

| Pri | Issue | Ready? | Starting notes |
| --- | --- | --- | --- |
| P0 | #32 metrics | PR #41 | Host-verify `8f1f6c8`, then merge. **API for later slices:**<br>- `scoring.decide_spoof` is the only labeling rule (float64), and scorers (#9) must use it;<br>- `evaluate_clip_scores(ids, labels, scores, threshold=None \| float, score_direction=SCORE_DIRECTION)`;<br>- `threshold_at_target_fpr(ids, labels, scores, target_fpr=0.05, score_direction=SCORE_DIRECTION)`;<br>- `METRICS_VERSION` goes into every run record. |
| P0 | **#31** dev reservation + pre-tuning record | **Ready** | Code is local and dataset-free: deterministic seeded speaker assignment, coverage checks, `development_role` writer, YAML config, tests. The membership run happens **on the host** against the real manifest. **Do this before #7's host extraction** (see the hazard comment on #7/#31: roles change the manifest hash). The threshold rule is already implemented as `metrics.threshold_at_target_fpr`. |
| P1 | #7 MFCC features + cache | After #30 merges | Planned shape: `features.py`, `.npz` per split, `extract-features` CLI, no scaler. Choose the cache key (manifest hash after #31, or a role-independent content key) at the review gate. |
| P1 | #8 LR baseline dev report | After #31, #32, #7 | Tune three C values on `dev_select`, selecting with `evaluate_clip_scores(..., threshold=None)` (EER); set the threshold on `dev_threshold` clips via `threshold_at_target_fpr`; report with `evaluate_clip_scores(..., threshold=t)`; JSON run record plus SQLite `runs` table. **First real numbers.** |
| P1 | #10 CNN timing pilot | After log-Mel settings are declared (pilot-only) | Output is a compute-route decision: CPU host, or Kaggle/Colab under the external-GPU protocol. |
| P2 | #35 eval protocol + freeze record | Start once #8 exists | Rehearse `DRAFT → FROZEN → SCORED → REPORTED` end-to-end on **`dev_select`** with the LR bundle, using no evaluation data. Encode the three attack slices, duration bins and the speaker-grouped bootstrap. |
| P2 | #34 transforms + silence-trim diagnostic, #6 EDA + shortcut-only baseline | Parallel | The standout failure analysis. Parameters are predeclared, never tuned to results. |
| P3 | #33 CNN, #36 PCA/K-means, #14 decision | G2 | Decide the CNN input/padding policy before training. Freeze every candidate (#35). |
| P3‖ | #9 scorer parity, #11 container in the presentation configuration on the host | G2 lane | Run these while the CNN trains. Keep API/UI bound to localhost; reach them via SSH tunnel if presenting remotely. |

Milestone forecasts: G1 16 Oct; G2 freeze 22 Oct; feature freeze 26 Oct (hard); G3
27 Oct; G4 1 Nov; G5 4 Nov; presentation 5 Nov.

## Owner inputs still pending

Ask for these when relevant; do not guess:
1. focused hours per week (forecast validation);
2. ~~how the host is reached during the presentation~~: **remotely via SSH tunnel** (owner, 13 Oct). The offline fallback is the presentation image exported to the laptop's Docker (verify in G4), then the backup recording;
3. #14, the sub-1 s duration gate (before the G2 freeze);
4. #34 codec, SNR and trim parameters (before the G2 freeze);
5. the demo TTS/voice-clone tool and its terms.

## Verified facts (do not re-derive)

LA manifest (host, 2026-10-13), matching the published counts exactly:

| Split | Bona fide | Spoof | Total | Speakers | Attacks |
| --- | --- | --- | --- | --- | --- |
| Train | 2,580 | 22,800 | 25,380 | 20 | A01–A06 |
| Development | 2,548 | 22,296 | 24,844 | 20 | A01–A06 |
| Evaluation | 7,355 | 63,882 | 71,237 | 67 | A07–A19 |

- Speakers are disjoint across splits (asserted).
- Manifest hash `b172e50c…`, before #31 adds roles.
- 1,277 clips are shorter than 1.0 s (1,059 of them in evaluation; #14).
- No 2019 LA source grouping is available, so the bootstrap falls back to
  `speaker_id`.
