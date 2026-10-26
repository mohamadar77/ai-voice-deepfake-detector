# Reproduction guide

How to go from a fresh checkout to the Phase 1 results and the demo.

**Verification status (2026-10-27).** Subcommand names, flags and defaults below
were checked against `voice-detector <command> --help` and `src/voice_detector/cli.py`
at this commit. The pipeline itself was **not** re-run to write this guide, and a
full fresh-checkout reproduction (data preparation, baseline, CNN, report) has not
been performed or recorded; that item is still open in
[`../TODO-Plan.md`](../TODO-Plan.md) §10. Treat the procedure as the intended
path, and expect to correct small details when it is first executed end to end.

## Prerequisites

- Linux x86_64 host (verified hosts: [`environment.md`](environment.md)). macOS
  cannot install the locked CPU `torch` wheel (`uv sync` fails on macOS x86_64), so
  nothing below runs on a Mac except `ruff`, which the CI also runs.
- Python 3.12 and [`uv`](https://docs.astral.sh/uv/). Git. Docker with Compose v2
  for section 11.
- ASVspoof 2019 LA, extracted, with the official CM protocols
  (provenance, licence, archive checksums: [`data.md`](data.md)). The repository
  does not download it. Point `VOICE_DETECTOR_LA_ROOT` at the extracted `LA`
  directory; it must also be set for every later command that reads audio.
- `ffmpeg` with `libopus` on `PATH`. The codec robustness condition and augmented
  CNN training call it (`augmentation.check_ffmpeg`); it sits outside `uv.lock`.
  The original run used ffmpeg 8.1.3 with libopus
  ([`archive/dl120-verification-tracker.md`](archive/dl120-verification-tracker.md)).
- Disk: the Phase 1 budget in [`environment.md`](environment.md) is about 50-60 GB.
- CPU only, no GPU. The long steps are multi-hour. Recorded durations: locked
  scoring 4,926 s and the locked report 2 h 16 min, both with 30 workers on
  hp-xl420gen9-01 ([`evaluation-results-g2.md`](evaluation-results-g2.md)). Run
  long steps detached (`tmux`).

What is **not** in Git and so cannot come from a fresh checkout: audio, the manifest,
feature caches, model bundles (weights), prediction stores, and the REPORTED copy of
the freeze record. Section 12 covers the retained archive.

## 1. Environment

```bash
git clone https://github.com/mohamadar77/ai-voice-deepfake-detector.git
cd ai-voice-deepfake-detector
uv sync --frozen --all-groups
uv run voice-detector smoke --config configs/default.yaml
uv run ruff check .
uv run pytest
```

`smoke` and `pytest` need no dataset, network or trained checkpoint.

## 2. Protocol audit and manifest

Optional metadata-only audit of each official protocol (reads no audio); the
counts must match [`data.md`](data.md):

```bash
uv run voice-detector protocol-audit \
    --protocol "$VOICE_DETECTOR_LA_ROOT/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt" \
    --split train
```

Build the per-clip manifest (reads audio headers and hashes the files; takes
hours):

```bash
export VOICE_DETECTOR_LA_ROOT=/path/to/LA
uv run voice-detector build-manifest --la-root "$VOICE_DETECTOR_LA_ROOT"
```

Output: `artifacts/manifest/manifest.csv` and `manifest.sqlite`. The command refuses
a manifest that disagrees with the published ASVspoof 2019 LA counts (121,461
clips) unless `--no-expected-counts` is passed. The role-free manifest hash
recorded in the Phase 1 evidence is `b172e50c…`.

## 3. Development reservation

The speaker-grouped `select` / `threshold` split is committed as
`reports/dev_reservation_frozen.json` (commit `81c3cf7`). This step re-derives it
from `configs/dev_reservation.yaml` and stamps the roles into the manifest. It
refuses if the derived membership differs from the committed record.

```bash
uv run voice-detector reserve-development --dry-run
uv run voice-detector reserve-development
```

This changes the manifest hash by design (role-bearing hash `b932b859…`), so it
must run before feature extraction. Defaults: `--reservation-config
configs/dev_reservation.yaml`, `--record-out reports/dev_reservation_frozen.json`.
Do not pass `--force` unless deliberately replacing a frozen reservation.

## 4. MFCC features

```bash
uv run voice-detector extract-features
```

The default splits are `train development`; `evaluation` is refused. Output:
`artifacts/features/features_train.npz` and `features_development.npz`.

## 5. Logistic regression baseline

```bash
uv run voice-detector train-lr
```

Defaults: `--protocol configs/pre_tuning_protocol.yaml`, `--frozen-record
reports/dev_reservation_frozen.json`, `--out-dir artifacts/lr_baseline`. It selects
`C` on `dev_select`, sets the threshold on `dev_threshold`, and writes the bundle
and a versioned development report. The bundle directory name (`lr_baseline`) is
the candidate id used by section 7.

## 6. CNN training (four runs)

Optional timing pilot on synthetic data (no dataset needed): `uv run voice-detector
cnn-pilot`.

Four runs: clean and augmented, each with the two seeds declared in
`configs/cnn_study.yaml`. `eval-freeze` and `eval-score` resolve a candidate as
`<artifacts_root>/<candidate_id>`, where the candidate id is the bundle directory
name (`cli.py`, `eval-freeze`), so write each bundle to its candidate directory with
`--out-dir`; the default `artifacts/cnn_study/<run_label>` would be refused.

```bash
uv run voice-detector train-cnn --run-label clean_seed1 --seed 20261004 \
    --out-dir artifacts/cnn_clean_seed1
uv run voice-detector train-cnn --run-label clean_seed2 --seed 20261005 \
    --out-dir artifacts/cnn_clean_seed2
uv run voice-detector train-cnn --run-label augmented_seed1 --seed 20261004 --augmented \
    --out-dir artifacts/cnn_augmented_seed1
uv run voice-detector train-cnn --run-label augmented_seed2 --seed 20261005 --augmented \
    --out-dir artifacts/cnn_augmented_seed2
```

Defaults: `--study-config configs/cnn_study.yaml`, `--robustness-config
configs/robustness_spec.yaml` (used with `--augmented`), `--frozen-record
reports/dev_reservation_frozen.json`. Each run is at most 20 epochs with early
stopping on development clip EER.

Retraining produces new bundles. Bit-identical weights, and therefore the recorded
bundle hashes (e.g. `999bd10a…` for `cnn_augmented_seed1`), are **not** claimed or
verified; to reproduce the reported numbers exactly, use the archived bundles
(section 12).

## 7. Evaluation freeze, scoring and report

These stages are one-way (`DRAFT -> FROZEN -> SCORED -> REPORTED`; see
[`evaluation-protocol.md`](evaluation-protocol.md)). Write a **new** freeze record for
a reproduction; never run `eval-score` or `eval-report` against the committed
`reports/evaluation_freeze_g2.json`, whose published REPORTED state belongs to the
original locked run.

```bash
uv run voice-detector eval-freeze \
    --bundle artifacts/lr_baseline \
    --bundle artifacts/cnn_clean_seed1 \
    --bundle artifacts/cnn_clean_seed2 \
    --bundle artifacts/cnn_augmented_seed1 \
    --bundle artifacts/cnn_augmented_seed2 \
    --out reports/evaluation_freeze_repro.json
```

Development rehearsal first (it also supplies the parity gate), then the locked
evaluation partition:

```bash
uv run voice-detector eval-score \
    --freeze-record reports/evaluation_freeze_repro.json \
    --partition development \
    --store artifacts/evaluation/predictions_dev.db \
    --workers 5

uv run voice-detector eval-score \
    --freeze-record reports/evaluation_freeze_repro.json \
    --partition evaluation \
    --store artifacts/evaluation/predictions.db \
    --dev-store artifacts/evaluation/predictions_dev.db \
    --workers 5
```

`--resume` continues a partial store; `--max-clips N` is a development-only smoke
option. Then the report (the bootstrap is slow; issue #78 measured about 5.5 h on
the development store):

```bash
uv run voice-detector eval-report \
    --freeze-record reports/evaluation_freeze_repro.json \
    --store artifacts/evaluation/predictions.db \
    --dev-store artifacts/evaluation/predictions_dev.db \
    --bootstrap --workers 5
```

Output: a report JSON under `artifacts/evaluation/`. The original run's hashes are
in [`evaluation-results-g2.md`](evaluation-results-g2.md): report file sha256
`c0116ce2…`, predictions hash `1ef12a29…`. A reproduction on newly trained bundles is
not expected to match them; a match is evidence only when the archived bundles and
the original commit `c4445f50…` are used.

`eval-supplement` (post-report analysis of a scored store) exists but is outside
Phase 1 and requires explicit owner authorization before use on an evaluation
store (deviation register, [`architecture-and-delivery-plan.md`](architecture-and-delivery-plan.md)).

## 8. Serving

The service reads the pinned bundle from `<artifacts_root>/cnn_augmented_seed1`
(`configs/serving.yaml`) and refuses to start if its hash or the exact threshold
differs from the pins.

```bash
uv run voice-detector serve --check \
    --freeze-record reports/evaluation_freeze_g2.json
uv run voice-detector serve                      # API on 127.0.0.1:8000
uv run streamlit run src/voice_detector/ui.py    # UI, second terminal
```

The UI calls the API at `VOICE_DETECTOR_API_URL` (default `http://127.0.0.1:8000`).
Both bind to localhost; reach a remote host through an SSH tunnel
([`environment.md`](environment.md)). `--check` with the committed FROZEN record
only passes with the original pinned bundle; with a retrained bundle, update the
two pins in `configs/serving.yaml` in a reviewed change (a recorded decision).

## 9. Docker

```bash
VOICE_DETECTOR_BUNDLE_DIR=/path/to/cnn_augmented_seed1 docker compose up --build --wait
```

One image, two services (`api`, `ui`); the bundle is mounted read-only. The full
smoke test, which also prints image size and cold start, is
`VOICE_DETECTOR_BUNDLE_DIR=... scripts/docker_smoke.sh`. Measured results and the
host they came from: [`model-card.md`](model-card.md) section 6 and
[`environment.md`](environment.md).

## 10. Benchmarks and parity checks

Both scripts run on the host with the bundle available, on development or
synthetic clips only (never the evaluation partition):

```bash
uv run python scripts/latency_benchmark.py \
    --config configs/latency_benchmark.yaml --output reports/latency_benchmark.json

VOICE_DETECTOR_ARTIFACTS_DIR=/path/to/artifacts \
  uv run python scripts/container_parity.py --clip-dir /path/to/LA/ASVspoof2019_LA_dev/flac \
  --count 200 --seed 20261021 --output reports/container_parity.json
```

## 11. Demo materials

`scripts/generate_demo_clips.py --output-dir demo_clips` writes the synthetic,
too-short, MP3 and digital-silence test inputs. The human-speech clip must be
recorded by the presenter. `scripts/demo_backup.py` writes pre-scored fallback
JSON. `scripts/demo_setup.sh` is a step-by-step presentation-day checklist, to be
read and run section by section, not executed blindly. Demo clips are git-ignored
and must not be committed.

## 12. Archive and retention

```bash
bash scripts/archive_artifacts.sh \
    --artifacts artifacts \
    --reported-record /path/to/evaluation_freeze_g2.REPORTED.json \
    --frozen-record reports/evaluation_freeze_g2.json \
    --dest /path/outside/the/repo/archive_g2 \
    --check-published-hashes
```

The original archive (`archive_g2_20261020`: 80 files, SHA-256 checksums, verified
off-host) is held outside Git. Redistribution of the model bundle has not been
decided, so there are no public artifact-retrieval instructions yet (open item in
the TODO plan).

## Temporary audio handling

Uploaded audio is decoded in memory for the request. The API does not log, write or
retain audio after the response, on success or error; logs carry allowlisted
fields only. The Streamlit UI keeps the upload in memory for display and drops it
on rerun. See [`pipeline.md`](pipeline.md) and `tests/test_serving_observability.py`.

## Reproducibility records

Training and evaluation steps write JSON run records (configuration, seed,
dependency versions, Git commit, manifest hash, preprocessing settings, timing,
hardware, metrics). The freeze record carries the lifecycle hashes. Records for the
original runs are in the retained archive, not in Git, except the committed FROZEN
freeze record and the development-reservation record.
