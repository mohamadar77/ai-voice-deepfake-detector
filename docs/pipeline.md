# Project Workflow & Pipeline

This document describes the end-to-end workflow for the AI Voice Deepfake Detector project, from data acquisition through model serving.

## High-Level Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   AI VOICE DEEPFAKE DETECTOR PROJECT                        │
└─────────────────────────────────────────────────────────────────────────────┘

    1. DATA & SETUP                2. FEATURE ENGINEERING            3. MODEL TRAINING
    ──────────────────────         ─────────────────────────         ──────────────────
    
    ASVspoof 2019 LA               Audio Preprocessing                MFCC + Logistic
    ├─ Train (6K+)                ├─ Decode (mono, 16kHz)           Regression Baseline
    ├─ Dev (3K+)                  ├─ Slice (4-sec windows)          │
    └─ Eval (2.6K)                ├─ Extract Features                └─ Log-Mel CNN
                                   │  ├─ MFCC                       
    Official protocol              │  └─ Log-Mel spectrograms      4. MODEL SELECTION
    └─ Manifest (ID, speaker,     │                                 ──────────────────
      attack, label, split)        └─ Normalize & Scaler Fit        Dev metrics only
                                                                     ├─ Baseline F1
    ↓                              ↓                                 ├─ CNN F1
    ────────────────────────────────────────────────────────────────├─ Select best
                                                                     └─ Freeze checkpoint


    5. LOCKED EVALUATION            6. ROBUSTNESS ANALYSIS            7. DEMO & SERVE
    ─────────────────────────        ──────────────────────────        ────────────────
    
    Frozen checkpoint               Test distortions on Eval:        FastAPI service
    Eval data (locked)              ├─ Noise                         ├─ Audio upload
    ├─ Aggregate windows            ├─ Compression                   ├─ Feature extract
    ├─ Clip-level scores            ├─ Speed change                  ├─ Score inference
    ├─ Metrics (EER, F1, AUC)       ├─ Pitch shift                   └─ Return score
    └─ Frozen protocol              └─ Measure robustness
                                                                      Streamlit demo
    Failure analysis                Failure analysis                 ├─ Upload UI
    └─ Inspect misclassifications   └─ Understand failure modes      ├─ Visualize
                                                                      └─ Display results
    ↓                              ↓                                 ↓
    Report: eval_results.json      Report: robustness.json          Docker container
    + model card                   + failure_analysis.md            + documentation

```

## Detailed Data Flow

### Phase 1: Data Acquisition & Validation
```
ASVspoof 2019 LA (Local or Cloud)
            │
            ├─ Verify checksums
            ├─ Parse manifest
            └─ Audit metadata → docs/data.md
                     │
                     ├─ Label mapping
                     ├─ Train/Dev/Eval splits (LOCKED)
                     ├─ Speaker grouping
                     └─ Duration statistics
```

### Phase 2: Preprocessing & Feature Engineering
```
Audio files (MP4)
    │
    ├─ Decode audio
    │  └─ Sample rate: 16 kHz
    │  └─ Channels: mono
    │
    ├─ Window into segments
    │  └─ Train: 4-second non-overlapping windows
    │  └─ Eval: aggregate windows back to clips
    │
    ├─ Feature extraction
    │  ├─ Path A: MFCC (13-40 coefficients)
    │  └─ Path B: Log-Mel spectrogram (64-128 bins)
    │
    └─ Normalization & preprocessing statistics
       └─ Fit ONLY on training data
       └─ Apply to dev & eval

    ↓ TRAINING PARTITION ONLY
    Feature cache → artifacts/features_train.pkl
```

### Phase 3: Model Training

#### Baseline (MFCC + Logistic Regression)
```
Training features (MFCC)
    │
    ├─ Fit Logistic Regression
    │  ├─ Loss: binary cross-entropy
    │  ├─ Regularization: L2
    │  └─ Random seed: 20261004
    │
    └─ Checkpoint: artifacts/baseline_model.pkl
       (Contains scaler, encoder, model)

    ↓ DEVELOPMENT PARTITION
    Development metrics:
    ├─ EER (Equal Error Rate)
    ├─ Macro F1
    ├─ Balanced Accuracy
    └─ ROC AUC
    
    → baseline_dev_metrics.json
```

#### CNN Pilot (Log-Mel + Neural Network)
```
Training features (Log-Mel spectrograms)
    │
    ├─ Build compact CNN
    │  ├─ Conv blocks
    │  ├─ Batch normalization
    │  ├─ Dropout (0.2-0.5)
    │  └─ Binary classification head
    │
    ├─ Training loop
    │  ├─ Optimizer: Adam
    │  ├─ Loss: BCE
    │  ├─ Epochs: TBD
    │  └─ Early stopping on dev loss
    │
    └─ Checkpoint: artifacts/cnn_checkpoint.pt
       (PyTorch model state)

    ↓ DEVELOPMENT PARTITION
    Development metrics:
    ├─ EER
    ├─ Macro F1
    ├─ Balanced Accuracy
    └─ ROC AUC
    
    → cnn_dev_metrics.json
```

### Phase 4: Model Selection
```
Baseline metrics           CNN metrics
    │                            │
    └──────────────┬─────────────┘
                   │
                   ├─ Compare on dev data
                   ├─ Account for variance
                   ├─ Select better model
                   │
                   ↓
          Selected checkpoint
          (e.g., CNN v7)
                   │
                   └─ FREEZE for evaluation
```

### Phase 5: Locked Evaluation
```
LOCKED EVALUATION DATA (never touched during development)
            │
            ├─ Load selected checkpoint
            ├─ Extract features (same pipeline)
            ├─ Generate predictions
            │
            ├─ Aggregate window scores → clip scores
            │  └─ Strategy: mean (or configurable)
            │
            ├─ Compute clip-level metrics
            │  ├─ EER
            │  ├─ Macro F1
            │  ├─ Balanced Accuracy
            │  ├─ ROC AUC
            │  └─ Confusion matrix
            │
            ├─ Slice analysis
            │  ├─ By attack type
            │  ├─ By speaker
            │  └─ False positive/negative rates
            │
            └─ → eval_results.json (immutable)
               → predictions.csv (locked)
               → confusion_matrix.png
```

### Phase 6: Robustness Experiments
```
Original evaluation clips
    │
    ├─ Apply distortions (frozen parameters)
    │  ├─ Additive noise (SNR: TBD)
    │  ├─ Compression (codec, bitrate)
    │  ├─ Speed perturbation (0.95x, 1.05x)
    │  └─ Pitch shift (semitones: TBD)
    │
    ├─ Re-extract features
    ├─ Run inference with locked checkpoint
    ├─ Compare scores & metrics
    │
    └─ → robustness_metrics.json
       → failure_analysis.md
       → per_distortion_results/
```

### Phase 7: Serving & Demo

#### FastAPI Backend
```
User uploads audio
    │
    ├─ Store temporarily
    ├─ Validate (duration, format)
    │
    ├─ Preprocessing pipeline
    │  ├─ Decode
    │  ├─ Window (if needed)
    │  └─ Extract features (apply training normalization)
    │
    ├─ Load locked inference model
    ├─ Generate predictions
    ├─ Aggregate (if windowed)
    │
    ├─ Return JSON response
    │  └─ { score: float, label: "bonafide"|"spoof", confidence: float }
    │
    └─ DELETE temporary audio
       (Privacy-preserving)

    Endpoint: POST /detect
    Logs: Request metadata only (no audio)
```

#### Streamlit UI
```
User visits http://localhost:8501
    │
    ├─ Upload audio file
    ├─ Display metadata (duration, sample rate)
    │
    ├─ Call FastAPI backend
    ├─ Display results
    │  ├─ Score visualization
    │  ├─ Confidence indicator
    │  └─ Interpretation
    │
    └─ Show model card & disclaimers
```

#### Docker Container
```
Build image
    │
    ├─ Python 3.12 base
    ├─ Install dependencies (uv sync)
    ├─ Copy model checkpoint & config
    │
    └─ docker build -t voice-detector .

Run container
    │
    ├─ docker run -p 8000:8000 -p 8501:8501 voice-detector
    │
    ├─ FastAPI on port 8000
    ├─ Streamlit on port 8501
    │
    └─ Persistent logs (configured)
```

## Configuration & Reproducibility

```
configs/default.yaml
├─ Project name & seed
├─ Data paths (env overridable)
├─ Audio settings (16 kHz, mono, 4 sec windows)
├─ Evaluation protocol (locked partitions)
└─ Feature extraction parameters

Version control:
├─ ✓ Configs (YAML)
├─ ✓ Source code
├─ ✓ Test fixtures
├─ ✓ Data manifest & checksums
├─ ✗ Audio files (ignored)
├─ ✗ Feature cache (ignored)
├─ ✗ Model checkpoints (ignored)
└─ ✗ Database (ignored)

Reproducibility record:
└─ artifacts/experiment_log.json
   ├─ Git commit
   ├─ Configuration hash
   ├─ Manifest hash
   ├─ Dependency versions
   ├─ Random seed
   ├─ Runtime & hardware
   └─ Metric outputs
```

## Key Constraints & Safeguards

| Constraint | Enforcement | Purpose |
|-----------|------------|---------|
| Train/Dev/Eval split | YAML config + validation | No data leakage |
| Train-only fitting | Preprocessing module | Fair evaluation |
| Development-only tuning | Checkpoints frozen before eval | No overfitting to test set |
| Window aggregation | Clip ID mapping | Per-clip metrics are honest |
| Deterministic preprocessing | Explicit random seeds | Reproducibility |
| Locked evaluation protocol | Immutable prediction file | No p-hacking |
| Frozen thresholds | Recorded before prediction inspection | Fair reporting |
| Feature parity | Shared module for train & serve | Train-serve consistency |
| Temporary audio | Auto-delete after inference | Privacy |
| Versioned outputs | Small JSON + metadata | Audit trail |

## Command Reference

```bash
# Setup
uv sync --frozen --all-groups

# Smoke test
uv run voice-detector smoke --config configs/default.yaml

# Preprocessing & EDA
uv run python -m voice_detector.data audit_manifest
uv run python -m voice_detector.preprocessing extract_features

# Training
uv run python -m voice_detector.models train_baseline --config configs/default.yaml
uv run python -m voice_detector.models train_cnn --config configs/default.yaml

# Evaluation
uv run python -m voice_detector.evaluation locked_eval --checkpoint artifacts/cnn_checkpoint.pt

# Robustness
uv run python -m voice_detector.robustness distort_and_test --distortion noise

# Serving
uv run fastapi run app/main.py --port 8000
uv run streamlit run app/streamlit_app.py --server.port 8501

# Docker
docker build -t voice-detector .
docker run -p 8000:8000 -p 8501:8501 voice-detector

# Testing & linting
uv run pytest
uv run ruff check .
```

## Timeline & Milestones

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| 1. Data & Setup | Oct 5-11 | Manifest, audit, preprocessing module, tests |
| 2. Baselines | Oct 12-18 | MFCC baseline, CNN timing run, comparison |
| 3. Evaluation & Robustness | Oct 19-25 | Locked eval, robustness report, failure analysis |
| 4. Demo & Release | Oct 26-Nov 4 | FastAPI, Streamlit, Docker, model card, slides |
| 5. Presentation | Nov 5 | Live demo & slides |

