# AI Voice Deepfake Detector

A reproducible AI Engineering graduation project for distinguishing bona fide human speech from synthesized or voice-converted speech.

## Status

Phase 1 (graduation release, 5 November 2026) is in progress. These exist:

- the data contract and the official-protocol parser/audit;
- the ASVspoof 2019 LA manifest builder, verified against the published counts on the training host;
- shared MFCC preprocessing with frozen windowing;
- a FastAPI/Streamlit serving skeleton.

**No model has been trained, and no measured results exist yet.** The prioritized critical path is in the [roadmap](docs/roadmap.md). Post-graduation research is kept separately in the [research backlog](docs/research-backlog.md).

## Core scope

- ASVspoof 2019 Logical Access (LA), preserving the official train, development, and evaluation partitions.
- An MFCC feature baseline with Logistic Regression.
- A compact CNN trained on log-Mel spectrograms.
- A locked final evaluation, robustness experiments, and failure analysis.
- A shared inference path exposed through FastAPI and a Streamlit demo.
- Docker packaging, reproducibility records, focused tests, and privacy-aware observability.

The committed study includes:

- clean and augmented CNN training;
- known- versus unseen-attack evaluation;
- paired noise/compression evaluation;
- a silence/duration shortcut diagnostic;
- bounded training-fitted PCA/K-means diagnostics.

The evidence-assistant stretch is a no-go for this release. Audio transformers, extra datasets/models, cloud deployment, calibration/abstention, agents and MCP belong to the post-graduation research backlog.

See the [architecture and delivery plan](docs/architecture-and-delivery-plan.md) for agreed contracts, TODOs, resource estimates and the tentative November 5 schedule. It is the current scope/schedule addendum to the [final frozen proposal](docs/AI_Voice_Deepfake_Detector_Final_Project_Proposal.docx) (signed off 2026-10-11, SHA-256 `b16f5124…`), which is the immutable source of truth; the root TODO tracks completion.

## Repository layout

```text
app/                  FastAPI and Streamlit entry points
artifacts/            Local generated models, predictions, and reports (ignored)
configs/              Versioned experiment and application configuration
data/                 Local datasets and feature caches (ignored)
docs/                 Proposal and project documentation
notebooks/            Exploratory notebooks that call reusable package code
reports/              Versioned small summaries and report sources
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

The local `data/` and `artifacts/` directories are intentionally excluded from version control except for their documentation files. Do not commit audio, extracted features, checkpoints, databases, credentials, or uploaded recordings.

## Evaluation safeguards

- Fit models and preprocessing statistics on the official training partition only.
- Use development data for model selection and for a speaker-grouped threshold/calibration reservation.
- Keep the ASVspoof 2019 LA evaluation partition locked until the evaluation protocol, models, score direction, and thresholds are frozen.
- Aggregate all windows to one score per original clip before reporting clip-level metrics.
- Never tune on final evaluation labels or report an unofficial subset as an official benchmark result.
- Treat outputs as model scores, not probabilities, unless calibration has been implemented and evaluated.

See [`TODO-Plan.md`](TODO-Plan.md) for the implementation plan and [`AGENTS.md`](AGENTS.md) for contributor conventions.

## Planned deliverables

The planned release includes the audited data manifest and EDA, reproducible baseline and CNN experiments, locked evaluation and robustness reports, failure analysis, a versioned inference service and UI, container startup instructions, a model card, and presentation materials.

## Data

Dataset acquisition is deliberately not part of repository setup. ASVspoof audio must be obtained under its published terms during the data-access milestone. Large datasets and generated artifacts remain local or are distributed separately where their terms allow.
