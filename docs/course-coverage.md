# Course coverage

Maps the AI Engineering syllabus to what this project actually contains. It is an
educational appendix, not a menu of promised features: the frozen proposal and the
[architecture plan](architecture-and-delivery-plan.md) define scope, and task
status lives in [`../TODO-Plan.md`](../TODO-Plan.md). This file replaces the
6 October proposal review (`proposal-review-and-course-coverage.md`); its project
critique was folded into the architecture plan and its remaining planning prose was
removed. The full review is in Git history.

The graduation brief requires real data, a clearly defined practical problem, an
end-to-end AI/ML solution, evaluation and a working demonstration, and does not
require every course algorithm to be implemented. A design explanation is never
presented here as an implementation.

**Status values.** *Implemented*: code and tests exist and the evidence is named.
*Partial*: some of the topic exists. *Design only*: explained, not built. *Not
implemented*: committed or considered, not built. *Out of scope*: decided against
or deferred.

## Syllabus map

| Syllabus topic (sessions) | Project evidence | Status |
| --- | --- | --- |
| Python environment, structures, functional design, engineering (1-6) | Python 3.12, uv lockfile, typed modules under `src/voice_detector/`, YAML configuration, focused tests (`tests/`) | Implemented |
| Data lifecycle, data as a resource (7) | Provenance, licence, checksums and exclusions in [`data.md`](data.md); manifest builder | Implemented |
| NumPy, Pandas, EDA, SQL (9) | NumPy features; CSV and SQLite manifest catalog (`clips`, `catalog_meta`) and a SQLite prediction store. The EDA report, figures and SQL query set are not done (#6) | Partial |
| Regression systems (10) | Logistic Regression is used as a classifier only. No continuous-target regression | Design only |
| Classification and evaluation (12) | MFCC + Logistic Regression; EER, ROC AUC, macro F1, balanced accuracy, confusion matrices, per-class precision/recall (`metrics.py`, [`evaluation-results-g2.md`](evaluation-results-g2.md)) | Implemented |
| Generalization (13) | Official partitions, speaker-grouped development reservation, locked evaluation, unseen-attack slices, noise/codec conditions | Implemented |
| Advanced ML models (15) | No tree-ensemble or other classical comparator was built | Not implemented |
| Clustering (16), PCA (18) | Contract written ([architecture plan](architecture-and-delivery-plan.md)); no PCA or K-means code exists (#36) | Not implemented |
| NLP: TF-IDF, BM25, search (20); retrieval engineering (21); embeddings and semantic similarity (27, 32, 39) | Evidence assistant is a no-go for this release (#37); no retrieval code | Out of scope |
| Neural networks / MLP (23) | Compact CNN with a small MLP head: loss, Adam, early stopping, balanced class weights, learning curves (`cnn_train.py`). No standalone MLP comparator | Implemented (head only) |
| CNN (24) | Log-Mel spectrogram CNN, 3 conv blocks, ~23.7 K parameters; four runs (clean/augmented x 2 seeds). YOLO and box detection have no target in clip classification | Implemented |
| Sequence models, LSTM (26) | Explained as an alternative; no LSTM was trained | Design only |
| Attention and Transformers (29-30); transformers to LLM systems (33) | Frozen wav2vec 2.0 embeddings were not run; deferred, no transfer-learning claim | Out of scope (Phase 2 backlog) |
| LLM ecosystem, LLM as engineering instrument, RAG, agents, multi-agent systems, MCP (35, 37, 40, 42, 43, 45) | No LLM, RAG, agent or MCP component exists; RAG/MCP are not on the research track either | Out of scope |
| Governance, control, enterprise AI (47) | Model card ([`model-card.md`](model-card.md), draft), provenance, stated limitations, temporary uploads, bundle hash pinning, input limits | Implemented |
| Observability and reliability (49) | Request IDs, allowlisted structured logs without audio, health/readiness endpoints, structured error responses (`serving/`, `tests/test_serving_*.py`) | Implemented |
| Cloud deployment, infrastructure (51) | Local Docker/Compose only (`Dockerfile`, `compose.yaml`); no cloud deployment | Partial (local only) |
| Performance testing (53) | p50/p95 latency, memory, cold start, failure rate on named hardware ([`model-card.md`](model-card.md) section 6) | Implemented |
| End-to-end AI workshop (55) | The same preprocessing and scorer serve evaluation, API and UI; container-versus-library parity checked | Implemented |
| Final project (61) | Working demonstration, measured conclusions, failure analysis ([`presentation.html`](presentation.html)). Fresh-start demonstration and rehearsal are still open in the TODO plan | Partial |

Practice and office-hour sessions introduce no separately named topics. The generic
advanced-ML session has no supplied algorithm list, so what it taught is unverified.

## Boundaries of the claims

- An MLP head demonstrates neural-network principles, not an independently
  benchmarked MLP model.
- Using a frozen pretrained encoder (not done here) would not demonstrate
  transformer pretraining or sequence-to-sequence training.
- A fixed API chain is not an agent, and a REST API does not demonstrate MCP.
- Docker does not demonstrate cloud deployment.
- Visual separation in PCA or clustering views would be a diagnostic, never evidence
  of detection performance.

## Deferred coverage

Items marked *Out of scope* or *Not implemented* appear in
[`research-backlog.md`](research-backlog.md) only where they are research
directions (transformer front-ends, stronger baselines, calibration). The
evidence assistant and MCP are not part of that backlog. If the instructor
separately requires an implementation of every syllabus topic, the project scope
would need renegotiation.
