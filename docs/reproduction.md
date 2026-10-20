# Reproduction guide

How to reproduce the Phase 1 results from a fresh checkout. All commands run
on a Linux x86_64 host with the ASVspoof 2019 LA dataset available locally.
Nothing runs on macOS (the torch wheels do not resolve there).

## Prerequisites

- Python 3.12, [uv](https://docs.astral.sh/uv/) (tested with 0.12.x).
- ASVspoof 2019 LA dataset under `$VOICE_DETECTOR_LA_ROOT` (see `docs/data.md`
  for provenance and terms; ODC-By 1.0 license requires citation).
- About 30 GB free disk (dataset, artifacts, predictions store).
- CPU only; no GPU required. Expect multi-hour runs for CNN training and locked
  scoring.

## 1. Environment

```bash
git clone https://github.com/mohamadar77/ai-voice-deepfake-detector.git
cd ai-voice-deepfake-detector
uv sync --frozen --all-groups
uv run voice-detector smoke          # validates imports and runtime
uv run ruff check .
uv run pytest                        # fast dataset-free tests
```

## 2. Manifest

Build the per-clip manifest from the official protocol files and audio headers.
The manifest hash `b172e50c…` is checked on write.

```bash
export VOICE_DETECTOR_LA_ROOT=/path/to/ASVspoof2019_LA

uv run voice-detector build-manifest \
    --la-root "$VOICE_DETECTOR_LA_ROOT"
```

Output: `artifacts/manifest/manifest.sqlite`.

## 3. Development reservation

Reserve 5 speakers from the development partition for threshold setting. The
frozen record is already committed (`reports/dev_reservation_frozen.json`); this
step reproduces it.

```bash
uv run voice-detector reserve-dev \
    --spec configs/dev_reservation_spec.yaml \
    --frozen-record reports/dev_reservation_frozen.json
```

## 4. MFCC feature extraction

Extract MFCC features for the training and development partitions.

```bash
uv run voice-detector extract \
    --splits train development
```

Output: `artifacts/features/features_train.npz`,
`artifacts/features/features_development.npz`.

## 5. Logistic regression baseline

Train the LR baseline using the pre-tuning protocol.

```bash
uv run voice-detector train-lr \
    --pre-tuning-record configs/pre_tuning_protocol.yaml \
    --frozen-record reports/dev_reservation_frozen.json
```

Output: `artifacts/lr_baseline/` (bundle with scaler, model, threshold,
run record).

## 6. CNN training (4 runs)

Train the compact log-Mel CNN. Four runs: clean and augmented, each with two
seeds (20261004, 20261005). Run labels: `clean_seed1`, `clean_seed2`,
`augmented_seed1`, `augmented_seed2`.

```bash
# Clean, seed 1
uv run voice-detector train-cnn \
    --run-label clean_seed1 --seed 20261004

# Clean, seed 2
uv run voice-detector train-cnn \
    --run-label clean_seed2 --seed 20261005

# Augmented, seed 1
uv run voice-detector train-cnn \
    --run-label augmented_seed1 --seed 20261004 --augmented

# Augmented, seed 2
uv run voice-detector train-cnn \
    --run-label augmented_seed2 --seed 20261005 --augmented
```

Output: `artifacts/cnn_study/<run_label>/` (bundle with checkpoint, scaler,
mel config, threshold, run record). Each run is at most 20 epochs with early
stopping (patience 5) on development clip EER.

## 7. Evaluation freeze

Declare all candidates and conditions, then freeze the evaluation protocol.
The freeze record transitions: DRAFT → FROZEN.

```bash
uv run voice-detector eval-freeze \
    --bundle artifacts/lr_baseline \
    --bundle artifacts/cnn_study/clean_seed1 \
    --bundle artifacts/cnn_study/clean_seed2 \
    --bundle artifacts/cnn_study/augmented_seed1 \
    --bundle artifacts/cnn_study/augmented_seed2 \
    --out reports/evaluation_freeze_g2.json
```

## 8. Locked scoring

Score all clips under all frozen conditions. Run development first as a
rehearsal, then evaluation.

```bash
# Development rehearsal
uv run voice-detector eval-score \
    --freeze-record reports/evaluation_freeze_g2.json \
    --partition development \
    --store artifacts/evaluation/predictions_dev.db \
    --workers 5

# Evaluation (locked)
uv run voice-detector eval-score \
    --freeze-record reports/evaluation_freeze_g2.json \
    --partition evaluation \
    --store artifacts/evaluation/predictions.db \
    --dev-store artifacts/evaluation/predictions_dev.db \
    --workers 5
```

## 9. Report

Generate the evaluation report with bootstrap intervals.

```bash
uv run voice-detector eval-report \
    --freeze-record reports/evaluation_freeze_g2.json \
    --store artifacts/evaluation/predictions.db \
    --dev-store artifacts/evaluation/predictions_dev.db \
    --bootstrap \
    --workers 5
```

Output: `artifacts/evaluation/evaluation_report.json`. The report hash should
match `c0116ce295b3c7f50a667bff79b77e3c476494b9e4cd895f36aae4421427ed2b`
(recorded in `docs/evaluation-results-g2.md`). The predictions hash should
match `1ef12a29a78ec5fe59d9bb6c068a218fd69f4f9430ad70f7a51f0b06b3f3d23a`.

## 10. Serving

Verify the pinned model bundle and start the API.

```bash
uv run voice-detector serve --check \
    --freeze-record reports/evaluation_freeze_g2.json

# Start the API (requires the bundle at the configured artifacts path)
uv run voice-detector serve
```

The Streamlit UI (in a second terminal):

```bash
uv run streamlit run src/voice_detector/ui.py
```

## 11. Docker

```bash
VOICE_DETECTOR_BUNDLE_DIR=artifacts/cnn_study/augmented_seed1 \
    docker compose up --build --wait
```

See `compose.yaml` and `scripts/docker_smoke.sh` for the full smoke test.

## 12. Archive

Archive frozen artifacts with checksums for off-host retention.

```bash
bash scripts/archive_artifacts.sh \
    artifacts /path/to/archive_g2 \
    --check-published-hashes
```

## Temporary audio handling

Uploaded audio is decoded in memory for the scoring request. The API does not
log, write, or retain audio data after the response is sent (success or error).
The Streamlit UI holds the upload in memory for the session's display and drops
it on rerun; `del data` in the render function releases the reference
explicitly. No audio is written to disk by the serving path.

## Reproducibility records

Each training run and evaluation step writes a structured run record (JSON)
with: configuration, seed, dependency versions, Git commit, preprocessing
settings, model version, timing, hardware, and metric outputs. The evaluation
freeze record tracks the DRAFT → FROZEN → SCORED → REPORTED lifecycle with
hashes at each transition. See `docs/evaluation-protocol.md`.
