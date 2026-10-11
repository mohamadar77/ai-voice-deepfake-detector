# Final project proposal review and complete course coverage

Reviewed 6 October 2026. Owner: Mohamad Abo-Ras.

**Scope update:** the subsequent [technical architecture and delivery agreement](architecture-and-delivery-plan.md) is the canonical dated record of implementation decisions and deviations, subordinate to the frozen proposal. The optional examples below are educational possibilities, not authorized release tasks. Extra model families, external datasets, calibration/abstention, cloud, agents and MCP are deferred. The evidence assistant is the only selected stretch, with zero committed hours and a full-core-completion gate before 26 October. Final planning targets are baseline/API by 12 October, all-candidate freeze by 18 October, report/demo by 25 October, then release and rehearsal.

**Proposal decision (2026-10-11):** the original Word proposal (`AI_Voice_Deepfake_Detector_Final_Project_Proposal.docx`) is adopted as the final, frozen, immutable source of truth — see the **Final proposal sign-off** in the architecture plan. The "Review judgment" below (keep the original as the foundation; do not let `_Revised.docx` replace it) is the accepted outcome; `_Revised.docx` is a non-selected historical candidate with no governing authority. Dated schedule/scope deviations are recorded in the architecture plan and do not alter the frozen proposal. This withdraws the earlier "the original Word proposal remains background / superseded by that addendum" language.

## Recommended project definition

**AI Voice Deepfake Detector: Detection, Robustness, and Evidence-Based Reporting**

I will design, build, evaluate, and demonstrate an audio classification system that estimates whether a speech recording is bona fide or generated through synthesis or voice conversion. I will compare MFCC features with Logistic Regression against a compact CNN operating on log-Mel spectrograms, using the official ASVspoof 2019 LA partitions. The study will measure performance on unfamiliar attacks, controlled noise and compression, and investigate false alarms and missed synthetic recordings. A shared, versioned inference pipeline will power evaluation, FastAPI, and a Streamlit demonstration.

The main contribution is a reproducible account of detection reliability: which model works better, under which conditions, at what inference cost, and where it fails. An optional evidence assistant can explain approved results with citations. Audio classification remains the responsibility of the evaluated detector. Outputs are model scores unless calibration is separately implemented and evaluated.

## Review judgment

Keep the original proposal as the foundation. It already contains a strong research question, baseline comparison, official partition discipline, reserved development data for thresholds, source-overlap auditing, bootstrap intervals, deployment design, a model card, dated milestones, and a 12-minute presentation plan.

The earlier `_Revised.docx` should not replace it: that version loses the explicit 5% development false-alarm target, dated milestones, detailed RAG evaluation, external hyperlinks, and some benchmarking qualifications. This review supersedes the earlier assessment that simply narrowing the original would necessarily improve it.

The meaningful weaknesses are:

1. **Incomplete course mapping.** Regression, advanced ML alternatives, RNN/LSTM, TF-IDF, encoder–decoder concepts, autoencoders, and several agent principles need explicit treatment.
2. **Uneven experimental detail.** PCA/clustering need a named method and selection rule; optional RAG needs a separate development/test question split; model comparisons need a fixed tuning budget.
3. **Too many possible extensions.** Transformer embeddings, multiple external datasets, cloud hosting, retrieval, agents, and MCP compete for the same time. Choose extensions at an explicit gate.
4. **Unsupported completion impression.** The repository currently contains setup/configuration scaffolding, not a trained detector or completed API. All proposal experiments must remain future-tense.
5. **Coverage versus implementation ambiguity.** An MLP classification head demonstrates neural-network principles but not an independently benchmarked MLP model. Audio embeddings do not demonstrate text retrieval. Calling a pretrained model does not demonstrate training a transformer.

## What the supplied brief actually requires

`Graduation Project.pdf`, pages 1–2, requires real data, a clearly defined practical problem, an end-to-end AI/ML solution, evaluation, and a working demonstration. It recommends early data collection/cleaning/EDA and milestones. It allows groups of up to three and focused projects in modeling, systems, research, or evaluation tooling. Presentation: up to 12 minutes, no code, emphasize choices, results, what worked, and what failed.

The brief does **not** say every course algorithm must be implemented. The user's broader request is addressed below by accounting for every substantive syllabus topic and the main additional principles in the supplied summaries. A design explanation is explicitly distinguished from an implemented experiment. If the instructor separately requires implementation of every topic, the proposed scope needs renegotiation; it would otherwise expand far beyond this focused project.

## Complete syllabus map

Status means planned scope, not completed work. **Core** = committed implementation; **Extension** = gated implementation after core; **Design** = explicit explanation/comparison, not an implementation claim. Practice and office-hour sessions introduce no separately named syllabus topics.

| Syllabus topic / sessions | Project use and evidence | Status |
| --- | --- | --- |
| Python environment, core structures, control flow, functional design, engineering/lab (1–6) | Python 3.12, uv lockfile, small reusable modules, typed interfaces, exceptions, deterministic configuration, focused tests. | Core |
| Data lifecycle and web as data source (7) | Dataset provenance, release/terms/checksums, documented acquisition, exclusions and transformations. Arbitrary scraping is unnecessary. | Core |
| NumPy, Pandas, EDA, SQL (9) | Arrays/features, manifest and SQLite joins, imbalance/duration/silence/duplicate audit with plots. | Core |
| Regression systems (10) | Explain continuous-target regression versus this binary task. Logistic Regression is a classifier and does not by itself cover continuous regression. Optional small latency-versus-duration regression using development benchmark timings; compare against a mean predictor on held-out clips, report MAE/RMSE and hardware. | Design; optional extension |
| Classification and evaluation (12) | MFCC + Logistic Regression, class weighting, fixed score direction, confusion matrix and clip-level metrics. | Core |
| Generalization (13) | Official boundaries, grouped provenance, development selection, locked unfamiliar-attack and robustness tests. | Core |
| Advanced ML models (15) | Compare nonlinear model options and tuning/compute tradeoffs. Optional small tree-ensemble comparator on the same MFCC features; select and freeze using development only. The supplied syllabus does not identify the exact algorithms taught in this session. | Design; optional extension |
| Clustering (16) | Train-fitted K-means; select K using a small predeclared training-only inertia/silhouette study; assign development/evaluation clips to frozen centroids and report errors/counts. | Core |
| PCA (18) | Train-fitted scaler and PCA; report explained variance and a fixed two-dimensional diagnostic projection. | Core |
| NLP: TF-IDF, BM25, search (20) | Retrieve approved documentation/report passages. Compare TF-IDF cosine search and BM25 under a fixed corpus and preprocessing. | Extension |
| Retrieval system engineering (21) | Document IDs, chunk IDs, versions, index rebuild procedure, relevance labels, recall@k and query latency. | Extension |
| Neural networks / MLP (23) | CNN classifier head plus tensors, loss, backpropagation/autograd, optimizer, batch size, learning rate, regularization, train/eval modes, training curves. A standalone MFCC + MLP comparator is optional. | Core; optional comparator |
| CNN including YOLO (24) | Spectrogram convolution, receptive fields, weight sharing, activation and pooling. Explain why bounding-box object detection/YOLO has no direct target in clip classification. | Core CNN; Design YOLO |
| Sequence models / LSTM (26) | Explain temporal order and many-to-one prediction. Optional compact LSTM on frame-level MFCCs, with padding masks/lengths and the same clip groups; compare pooled MFCC baseline, cost and errors. Do not replace official partitions with a forecasting split. | Design; extension |
| Word/document embeddings (27) | Distinguish word vectors, pooled document vectors, contextual representations and audio embeddings. Optional text embedding retriever compared with TF-IDF/BM25. | Design; extension |
| Attention / Transformer architecture (29–30) | Explain Q/K/V, self-attention, positional information, encoder versus decoder. Optional frozen audio encoder plus linear head; record upstream pretraining and resource cost. | Design; extension |
| Embeddings / semantic similarity (32) | Similarity over document embeddings; use labeled retrieval questions to test whether semantic search improves lexical search. | Extension |
| From transformers to LLM systems (33) | Explain tokenization, next-token generation and context limits; optional cited report generation with schema validation. | Design; extension |
| LLM ecosystem / model infrastructure (35) | Document model choice by task quality, privacy, latency, cost and deployment constraints. Record model/version, prompts, decoding settings and token usage if implemented. | Design; extension |
| LLM as engineering instrument (37) | Ingest approved documents, cleanse, chunk, enrich metadata, validate structured responses; numerical claims come from saved metrics. | Extension |
| Embeddings / retrieval infrastructure (39) | Version corpus, embedding model, chunks and index together; cache embeddings and test stale-version behavior. A small local index is sufficient. | Extension |
| RAG architecture, variants and evaluation (40) | Compare retrieval-only and retrieval-plus-generation; separate question development and held-out testing, citation/number correctness, unanswerable and injection cases. | Extension |
| AI agents (42) | A bounded assistant may select validated report/audio tools; record actions, tool arguments, results, task success and failures. A fixed API chain is not automatically an agent. | Extension |
| Multi-agent systems / orchestration (43) | Explain sequential versus hierarchical workflows and state handoff. A single orchestrator suffices here; multiple agents need measured benefit before becoming scope. | Design |
| MCP / FastMCP (45) | Optional typed wrappers around inspect_audio, predict_audio and get_model_report; test schema, permissions, timeouts and parity with API results. An ordinary REST API does not demonstrate MCP. | Extension |
| Governance, control, enterprise AI (47) | Model card, provenance, explicit limitations, temporary uploads, versioning, input/resource controls and documented access boundary. | Core |
| Observability / AI system reliability (49) | Request IDs, model version, status/latency, cleanup on errors, health endpoint and failure tests. | Core |
| Cloud deployment / AI infrastructure (51) | Docker and reproducible local startup are core infrastructure. Cloud-specific deployment remains optional and cannot be claimed from Docker alone. | Core local; extension cloud |
| Observability, evaluation, performance testing (53) | p50/p95 latency, memory, failures, named hardware, concurrency and cold/warm conditions. | Core |
| End-to-end AI workshop (55) | Demonstrate the same frozen preprocessing/model from evaluation through API/UI. | Core |
| Final project / graduation (61) | Working demonstration, measured conclusions, failure analysis and a rehearsed presentation. Tentative date: 5 November 2026. | Core |

## Additional principles from the supplied summaries

- **Deep Learning summary:** loss/optimization, gradient descent, learning rate, autograd, initialization, overfitting/underfitting, L1/L2/weight decay, dropout, normalization and representation learning belong in the model-design explanation and training evidence. Explain autoencoder/denoising reconstruction and self-supervised learning as alternatives; do not claim an autoencoder was built. Denoising could remove discriminative artifacts, so it cannot be silently added to preprocessing.
- **Clustering summary:** explain DBSCAN density/noise versus K-means and t-SNE visualization versus PCA. Use training-only demonstrations if added. Do not fit DBSCAN or t-SNE on final evaluation data under this project's rules or treat visual separation as performance evidence.
- **NLP summary:** cover structured/semi-structured/unstructured data, extraction, normalization, vocabulary, tokenization, stop-word/stemming/lemmatization choices, Boolean/Jaccard/BoW/TF-IDF/BM25. In an optional retrieval notebook, simple Boolean/Jaccard baselines can illustrate the progression without becoming extra product services. Preserve attack IDs, model names and numbers during normalization.
- **Sequence/CNN/RNN/LSTM summaries:** explain hidden/cell state, gates, BPTT, vanishing/exploding gradients, gradient clipping, shape/length handling and temporal pooling. Forecasting, recursive rollout and chronological splitting are contrasted with this clip-classification task, not claimed as implemented.
- **Seq2Seq/Transformers summary:** explain encoder–decoder bottlenecks, attention, Q/K/V, positional information, causal/cross-attention, teacher forcing and autoregressive decoding. Classification needs no sequence-generating decoder. Using a frozen encoder does not demonstrate teacher-forcing training or transformer pretraining.
- **LLM methodology:** distinguish pretraining, fine-tuning, instruction tuning and RLHF; compare zero/few-shot prompts and context selection if the assistant is built. LLM training/RLHF are conceptual scope. Audio remains local; only approved report text may enter a configured language-model service.
- **LLM as an instrument:** document parsing, OCR only when required, cleansing, metadata-enriched chunks, typed output, validation and function calling belong to the optional evidence workflow.
- **Agents/Crews:** account for tools, planning, short/long-term memory, guardrails, sequential/hierarchical orchestration, routing/events, roles/tasks/processes and framework choices. Prefer bounded session state and a single orchestrator; long-term storage of recordings is outside scope. Evaluate tool correctness, task success, latency and token use if implemented. Do not label deterministic schema checks as an independent reviewer agent.

## Changes to make the research stronger

1. **Freeze a small experiment matrix.** Required: MFCC + Logistic Regression; clean-training CNN; CNN with one fixed augmentation recipe. Use identical test clips and clip-level aggregation. Freeze tuning budgets/configurations and any optional model candidates before inspecting final predictions.
2. **Separate feature and architecture claims.** MFCC + Logistic Regression versus log-Mel CNN compares complete systems. It does not isolate the architecture effect. If needed, add an MFCC + MLP control or another matched-input comparator within the development budget.
3. **Make shortcut testing visible.** Examine duration, silence ratio and level distributions before final testing. A small metadata-only classifier using training/development can flag suspiciously easy dataset cues; never feed speaker/source/attack IDs to the detector. Do not change the model after final-test diagnostics.
4. **Preserve the operating point.** Define synthetic as the positive class and larger scores as more synthetic. Fix the threshold on reserved development speakers to target at most 5% human false alarms. Record sample counts and uncertainty; this is a development target, not a production guarantee. EER is a descriptive test metric, not the deployed threshold.
5. **Declare selection and calibration rules.** Keep model-selection development separate from threshold/calibration reservation. If adding both calibration and threshold optimization, predeclare the internal split or cross-fitting procedure; avoid choosing calibration approaches using the same final reservation.
6. **Measure paired changes.** Freeze codec, bitrate, noise construction/SNR, random seeds and clipping handling. Compare clean/transformed versions of the same clips, for both classes. Compute uncertainty for paired metric differences by resampling original source groups. Never count transformed copies as independent evidence.
7. **Specify cluster diagnostics.** Use training-fitted PCA and K-means. Freeze K and transformations before evaluation; report cluster counts and error rates after the final run. These are descriptive diagnoses, not new attack labels.
8. **Show quality and cost together.** A compact results table should include EER, macro F1, balanced accuracy, ROC AUC, false-positive rate, detection rate, class counts, inference latency and model size. Retain confusion matrices, slice counts and confidence intervals in the report.
9. **Demonstrate a failure.** Show a consented or licensed example, a recorded robustness comparison, and an honest false alarm/miss from the frozen report. A saved demo set is illustrative and does not establish general accuracy.
10. **Close the short-audio gap.** Predeclare duration slices of 1–<2, 2–<4, 4–10 and >10–30 seconds where enough clips exist, with counts for both classes. Freeze padding, test-window stride and clip aggregation. If making causal claims about duration, use paired controlled crops with a fixed crop rule and original-source grouping; natural duration slices alone show association. Qualify any accepted input range that lacks evaluation evidence.
11. **Bound feasibility and repeatability.** At the first-week gate, record available disk/RAM/hardware, measured pilot runtime and a maximum tuning-run budget per model. Retain the original goal of at least two CNN seeds if compute allows; otherwise disclose unmeasured training variability. Build a thin upload-to-result path early. Do not promise runtime or hardware capacity before measurement.

## Extension decision

Finish the required 2019 LA experiment and deployable demo first. The only selected stretch is a small evidence assistant over documentation and approved results: TF-IDF/BM25, optional semantic retrieval, cited generation if implemented, and explicit evaluation. It has zero committed hours and is likely deferred. LSTM and frozen audio transformers are future research, not alternative release commitments; agent/MCP integrations are also deferred.

If choosing the assistant, author 20 questions before prompt/index tuning: 10 for development and 10 held out for final testing, including numerical, unanswerable and malicious-instruction cases. Freeze expected passages and answers against a corpus version. Report recall@3, citation correctness, numerical exactness, unsupported-claim rate, correct abstention, task/tool success, latency and tokens. This is a small pilot evaluation with explicit counts, not a broad reliability claim. Retrieved documents are evidence, never permission to execute instructions.

If choosing LSTM, reuse frame-level MFCC extraction, original clip groups and the frozen evaluation interface. Predeclare hidden size, length masking, gradient clipping and development tuning budget. Report temporal-order benefits only if a suitable matched-input/order-control experiment supports them.

## Milestones and acceptance

Use the architecture agreement's aligned schedule: 6–7 October feasibility; 8–12 October data/baseline and early API/container pilot; 13–18 October CNN and all-candidate protocol freeze; 19–25 October locked report/demo; 26 October feature freeze; 27 October–1 November release; 2–4 November rehearsal; tentative 5 November presentation. Core effort is estimated at 110–150 focused hours before contingency; actual availability, access and runtime must be validated at G0/G1.

Acceptance requires reproducible configurations/manifests, the baseline and CNN, locked evaluation with all required metrics and counts, robustness ablation, failure analysis, shared inference, tested API/UI, local Docker startup, upload cleanup, model card and a rehearsed 12-minute demo. Improvement over the baseline is a hypothesis. Any incomplete extension is marked as future work.

Suggested presentation: 1 minute problem; 2 data/protocol; 2 model decisions; 3 results/failures; 2 demo; 1 limits/conclusions; 1 buffer. Put the complete course matrix in an appendix rather than attempting to explain every algorithm on stage.

## Sources and consultation status

Primary local sources: `Graduation Project.pdf` (2 pages); `לו_ז של הקורס.pdf` (7 pages); the nine DOCX summaries in `AI course summaries`; and the original proposal. The extracted archive duplicates the summary documents. The schedule explicitly says it is tentative. The generic advanced-ML session lacks a supplied algorithm-specific summary, so its exact taught algorithm list remains unverified.

The requested session `01a11238-7c72-72c2-96f2-ee77ba12524e` could not be opened through the local queue API: it returned “no rollout found.” A fresh independent read-only CLI consultation was started as session `01a11240-cb6b-7d82-b249-00f0c6a31894`, with `gpt-6-astra` and medium reasoning effort, matching the current local CLI configuration. This is a replacement consultation, not a claimed reply from the requested session.

The replacement consultation completed. It independently confirmed the missing coverage, complete-pipeline comparison caveat, robustness ambiguity, threshold reservation, short-audio gap, feasibility budgeting and lost references. Those findings are incorporated above. The reviewer preferred the revised document's organization with restored commitments; this review recommends preserving the original as the source because its safeguards, references and schedule are already intact. Both recommendations agree on the scientific scope and the need for a full coverage appendix.

External source verification was attempted but the web tool returned an authentication error. This review therefore relies on supplied materials and does not claim fresh verification of dataset release terms or pretrained-model licenses. Preserve the original proposal's links and recheck terms during the data-access milestone.
