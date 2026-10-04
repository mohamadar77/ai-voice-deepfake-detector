# Implementation Roadmap

This roadmap turns the proposal into a sequence of small, verifiable vertical slices. It complements `TODO-Plan.md`: the checklist records all work, while this file identifies what to do next and when to stop.

## Current position

Repository and environment setup is complete. No ASVspoof audio has been downloaded, no evaluation predictions have been inspected, and no model has been trained.

## Core path

| Slice | Output | Gate before continuing |
| --- | --- | --- |
| 1. Data contract | `docs/data.md`, manifest schema, protocol parser and metadata audit | Official train/development/evaluation boundaries and label mapping are verified |
| 2. Shared preprocessing | Deterministic decoder, four-second training windows, evaluation aggregation and feature-shape tests | Train-only fitting and train/serve parity are demonstrated |
| 3. MFCC baseline | Logistic Regression bundle, development metrics and reproduction command | Development report exists; evaluation remains locked |
| 4. CNN pilot | Compact log-Mel CNN timing run and declared training budget | Compute and storage are sufficient for the main run |
| 5. Model comparison | CNN and baseline development comparison, selected checkpoint | Selection uses development data only |
| 6. Locked evaluation | Frozen protocol, clip-level predictions and metrics | Protocol, thresholds and checkpoints are frozen before prediction inspection |
| 7. Robustness and failures | Paired distortion results, slices and failure analysis | Transformations use frozen parameters and original-clip provenance |
| 8. Demo and release | FastAPI, Streamlit, Docker, observability and model card | Serving uses the evaluated bundle and temporary audio is deleted |

## First session

1. Verify that the official ASVspoof 2019 LA protocol files can be obtained under their terms.
2. Write the source, release identifier, checksums and permitted uses in `docs/data.md`.
3. Define the manifest columns: clip ID, path, source ID, speaker ID, attack ID, label, official split, duration, sample rate, checksum and provenance.
4. Implement a metadata-only audit command and test it on a tiny fixture.
5. Record the storage and runtime decision in the experiment log before downloading audio or starting training.

## Stop/go rules

- If data access is blocked, finish the parser, fixtures, schemas and synthetic smoke tests; do not invent benchmark results.
- If full processing is too expensive, freeze a documented stratified subset before inspecting model scores and report it as a subset study.
- If ASVspoof 2021 DF access is unavailable, keep it deferred and complete the 2019 LA core path.
- If a model improves only on development data, prefer the simpler or more reliable model and record the negative result.

## Definition of done

The core project is complete when a fresh checkout can reproduce the manifest, preprocessing, baseline, CNN comparison, locked evaluation report and demo using versioned configuration and recorded provenance. Extensions are separate and never change the core acceptance criteria.
