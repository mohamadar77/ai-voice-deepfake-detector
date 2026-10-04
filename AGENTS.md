# Project conventions

These instructions apply to the entire repository.

## Scope and priorities

- Complete the core path before enhancements: ASVspoof 2019 LA, MFCC plus Logistic Regression, log-Mel CNN, locked evaluation, robustness and failure analysis, FastAPI, Streamlit, Docker, reproducibility, and observability.
- Treat frozen transformer embeddings as a later enhancement. Treat RAG and MCP as optional.
- Do not download large datasets, train models, inspect final evaluation predictions, or add cloud services unless the active task explicitly requires it.

## Environment and commands

- Use Python 3.12 and `uv`; do not introduce a second environment or lockfile system.
- Install with `uv sync --frozen --all-groups`.
- Run commands through `uv run`.
- Before completing a code change, run `uv run ruff check .` and the smallest relevant `uv run pytest` selection. Run the full test suite when practical.
- Keep tests fast and independent of full datasets, network access, GPUs, and trained checkpoints unless explicitly marked otherwise.

## Code organization

- Put reusable implementation in `src/voice_detector/`; keep notebooks thin and import package code.
- Keep configuration in versioned YAML files under `configs/`. Do not hide experiment choices in notebooks or CLI defaults.
- Keep training separate from serving. FastAPI and Streamlit must call the same versioned inference and preprocessing modules used by evaluation.
- Use type hints for public functions and concise docstrings where behavior or constraints are not obvious.
- Prefer deterministic behavior. Pass and record random seeds explicitly.
- Write small, structured outputs. Generated datasets, feature caches, checkpoints, prediction files, databases, and large plots belong in ignored paths.

## Data and privacy

- Never commit ASVspoof audio, third-party audio, uploaded recordings, feature caches, model checkpoints, SQLite databases, credentials, or `.env` files.
- Keep committed schemas, configuration, checksums, manifests without restricted paths, and small approved summaries.
- Uploaded audio is temporary: do not log raw audio or retain it after successful or failed requests.
- Record dataset provenance, release identifiers, terms, checksums, exclusions, and transformations.

## Non-negotiable evaluation constraints

- Preserve the official ASVspoof 2019 LA train, development, and evaluation boundaries.
- Training partition: fit supervised models, scalers, normalization statistics, PCA, and clustering models.
- Development partition: model selection, checkpoint selection, and a predeclared speaker-grouped reservation for thresholds or calibration.
- Evaluation partition: locked final test only. Do not use it for fitting, feature selection, hyperparameter tuning, checkpoint selection, calibration, threshold selection, or post-result retraining.
- Keep every window, augmented copy, and derived feature assigned to its original clip, speaker, source, and official partition. Split before windowing or transformation.
- Aggregate window scores to one score per original clip before clip-level metrics. Bootstrap repeated observations by original source recording when provenance permits.
- Freeze the evaluation protocol, candidate configurations, checkpoints, score direction, operating thresholds, test membership, and metrics before inspecting final predictions.
- Report EER, macro F1, balanced accuracy, ROC AUC, class counts, confusion matrices, human false-positive rate, and synthetic detection rate. Include sample counts for slices.
- Report unfamiliar attacks separately. Do not treat correlated 2019/2021 source recordings as independent evidence.
- Apply robustness transformations consistently to both classes and the same underlying clips with frozen parameters and seeds.
- Label outputs as scores unless calibration is separately fit on reserved development data and evaluated. Do not imply that a score proves authenticity, identifies a speaker, or verifies spoken claims.
- Never describe subset evaluation, externally pretrained models, or modified protocols as official leaderboard results.

## Reproducibility records

For every experiment, record the configuration, seed, dependency versions, Git commit, manifest hash, preprocessing settings, model version, runtime, hardware, predictions, and metric outputs. Record negative results and protocol deviations rather than silently replacing them.
