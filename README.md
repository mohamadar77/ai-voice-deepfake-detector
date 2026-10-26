# AI Voice Deepfake Detector

A reproducible AI Engineering graduation project for distinguishing bona fide human speech from synthesized or voice-converted speech.

## Status

Phase 1 (graduation release, tentative presentation 5 November 2026) is in its verification and rehearsal stage; the feature freeze passed on 26 October. The deployed demo model is `cnn_augmented_seed1`, a compact log-Mel CNN (~23.7 K parameters) served through FastAPI and a Streamlit demo, packaged in Docker. The locked evaluation on ASVspoof 2019 LA is REPORTED. On unseen attacks the clean EER is 19.93% for the MFCC + Logistic Regression baseline and 6.68-11.77% for the four CNN runs; the deployed model is the highest of the four (11.77%) because it was chosen by a development-only rule written after the locked report had been read. The frozen decision threshold does not transfer to compressed or noisy audio. See the [model card](docs/model-card.md) (still a draft) and the [locked results](docs/evaluation-results-g2.md).

This is a project evaluation, **not an official ASVspoof leaderboard result**. The output is an uncalibrated score and a decision at a fixed threshold. It is not proof of authenticity, not a speaker verifier and not for consequential decisions.

All documentation is indexed in [`docs/README.md`](docs/README.md), which names the authoritative document for each topic. Task status is in [`TODO-Plan.md`](TODO-Plan.md); the milestone sequence is in the [roadmap](docs/roadmap.md). Post-graduation research is kept separately in the [research backlog](docs/research-backlog.md).

## Core scope

- ASVspoof 2019 Logical Access (LA), preserving the official train, development, and evaluation partitions.
- An MFCC feature baseline with Logistic Regression.
- A compact CNN trained on log-Mel spectrograms.
- A locked final evaluation, robustness experiments, and failure analysis.
- A shared inference path exposed through FastAPI and a Streamlit demo.
- Docker packaging, reproducibility records, focused tests, and privacy-aware observability.

The study includes, all implemented and reported:

- clean and augmented CNN training (two seeds each);
- known- versus unseen-attack evaluation;
- paired noise/compression evaluation;
- a silence-trim shortcut diagnostic.

Committed in the proposal but **not yet implemented**: training-fitted PCA/K-means diagnostics, the EDA / exclusion report and the shortcut-only baseline (open in [`TODO-Plan.md`](TODO-Plan.md)).

The evidence-assistant stretch is a no-go for this release. Audio transformers, extra datasets/models, cloud deployment, calibration/abstention, agents and MCP belong to the post-graduation research backlog.

See the [architecture and delivery plan](docs/architecture-and-delivery-plan.md) for design contracts, decisions and the deviation register. It is the addendum to the [final frozen proposal](docs/AI_Voice_Deepfake_Detector_Final_Project_Proposal.docx) (signed off 2026-10-11, SHA-256 `b16f5124…`), which is the immutable source of truth; the root TODO tracks completion.

## Repository layout

```text
app/                  Placeholder (README only); the service is src/voice_detector/serving/, the UI is src/voice_detector/ui.py
artifacts/            Local generated models, predictions, and reports (ignored)
configs/              Versioned experiment and application configuration
data/                 Local datasets and feature caches (ignored)
docs/                 Proposal and project documentation (start at docs/README.md)
notebooks/            Exploratory notebooks that call reusable package code
reports/              Versioned small summaries and report sources (reservation and freeze records)
scripts/              Host, container, benchmark and presentation helper scripts
src/voice_detector/   Reusable data, modeling, evaluation, and inference code
tests/                Fast, dataset-free checks
```

## Prerequisites

- macOS or Linux
- Python 3.12
- [`uv`](https://docs.astral.sh/uv/)
- Git

The verified install target for training and evaluation is Linux x86_64 (see [`docs/environment.md`](docs/environment.md)). macOS x86_64 cannot install the locked PyTorch build; use the Linux host or the container there. Accelerator-specific behavior must be verified on the target machine.

## Setup

Create the isolated environment and install the exact locked dependency set:

```bash
uv sync --frozen --all-groups
```

Copy the optional environment example only when local path overrides are needed:

```bash
cp .env.example .env
```

Run the dataset-free smoke command:

```bash
uv run voice-detector smoke --config configs/default.yaml
```

Run the repository checks:

```bash
uv run ruff check .
uv run pytest
```

## Configuration

[`configs/default.yaml`](configs/default.yaml) is the committed, credential-free default. Local storage can be moved outside the repository with:

```bash
export VOICE_DETECTOR_DATA_DIR=/path/to/asvspoof-data
export VOICE_DETECTOR_ARTIFACTS_DIR=/path/to/generated-artifacts
```

The extracted ASVspoof 2019 LA root (the directory containing
`ASVspoof2019_LA_cm_protocols/`, `ASVspoof2019_LA_train/`, etc.) can likewise be
pointed to wherever it actually lives, instead of symlinking it into the
working tree:

```bash
export VOICE_DETECTOR_LA_ROOT=/path/to/LA
uv run voice-detector build-manifest --config configs/default.yaml
```

`--la-root` still overrides the environment variable when passed explicitly.
Manifest rows store each clip's audio path relative to that root, so the same
`VOICE_DETECTOR_LA_ROOT` variable also has to be set (not just for
`build-manifest`) whenever a later command reads the audio itself —
`extract-features`, `train-lr`, `train-cnn`, `eval-score` — unless it is run
with its working directory already at the LA root.

The local `data/` and `artifacts/` directories are intentionally excluded from version control except for their documentation files. Do not commit audio, extracted features, checkpoints, databases, credentials, or uploaded recordings.

## Evaluation safeguards

- Fit models and preprocessing statistics on the official training partition only.
- Use development data for model selection and for a speaker-grouped threshold/calibration reservation.
- Keep the ASVspoof 2019 LA evaluation partition locked until the evaluation protocol, models, score direction, and thresholds are frozen.
- Aggregate all windows to one score per original clip before reporting clip-level metrics.
- Never tune on final evaluation labels or report an unofficial subset as an official benchmark result.
- Treat outputs as model scores, not probabilities, unless calibration has been implemented and evaluated.

See [`TODO-Plan.md`](TODO-Plan.md) for the implementation plan and [`AGENTS.md`](AGENTS.md) for contributor conventions.

## Reproduce and run the demo

[`docs/reproduction.md`](docs/reproduction.md) is the verified command sequence from a
fresh checkout (manifest, features, LR baseline, four CNN runs, evaluation freeze,
locked scoring, report, serving, Docker). The pinned demo model bundle and the dataset
are not in Git, so the demo needs the trained `cnn_augmented_seed1` bundle:

```bash
uv run voice-detector serve --check        # verify the pinned bundle
uv run voice-detector serve                # API on 127.0.0.1:8000
uv run streamlit run src/voice_detector/ui.py
```

or, with Docker, `VOICE_DETECTOR_BUNDLE_DIR=/path/to/cnn_augmented_seed1 docker compose up --build --wait`. Fresh-checkout reproduction of the full pipeline has not yet been performed and recorded (open in the TODO plan).

## Deliverables

Delivered: audited manifest, shared preprocessing, reproducible baseline and CNN experiments, locked evaluation and robustness reports with failure analysis, a versioned inference service and UI, container packaging, a draft model card and a 12-slide deck ([`docs/presentation.html`](docs/presentation.html)). Still open: the EDA report, PCA/K-means diagnostics, fresh-checkout reproduction, container verification on the presentation machine, rehearsal and release tag; see [`TODO-Plan.md`](TODO-Plan.md).

## Data

Dataset acquisition is deliberately not part of repository setup. ASVspoof 2019 LA audio must be obtained under its published terms (ODC-By 1.0; provenance, checksums and attribution notice in [`docs/data.md`](docs/data.md)). Large datasets and generated artifacts remain local or are distributed separately where their terms allow.

> Contains information from the ASVspoof 2019 database (<https://datashare.ed.ac.uk/handle/10283/3336>), made available under the ODC Attribution License (<https://opendatacommons.org/licenses/by/1-0/>).
