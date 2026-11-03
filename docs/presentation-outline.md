# Presentation outline — AI Voice Deepfake Detector

Speaker notes and timing for [`presentation.html`](presentation.html), which is the
delivered deck. Slide order, titles, metrics and claims were synchronized with the
HTML on 2026-10-27; if the two ever differ, the HTML wins and this file is corrected.
The "Visual" lines are the original design suggestions; the delivered visuals are
the ones in the HTML.

Target: 12 minutes. Planned talk time 11:00 (slide timings below) leaving a 1-minute
buffer. Allocation: 1 motivation, 2 data, 2 methods, 3 results, 2 demo, 1 conclusions. Show ideas, plots and behavior, not code.

---

## Slide 1 — Title (~15 s)

**AI Voice Deepfake Detector: Detecting Synthesized Speech on ASVspoof 2019 LA**

Mohamad Abo-Ras · Graduation project · 5 November 2026

---

## Slide 2 — Motivation (~45 s)

**Why this matters**

- Voice cloning is cheap and fast: a few seconds of audio, a public model, a
  convincing fake. Real incidents: phone scams, deepfake political audio.
- Automatic detection is a measurable signal, not a verdict — it helps human
  reviewers focus, not replace them.
- This project: build a detector, measure it honestly, show its limits.

Speaker note: emphasize that the detector is a score, not ground truth.

---

## Slide 3 — Dataset (1 min)

**ASVspoof 2019 Logical Access**

- Standard benchmark for speech anti-spoofing research (Todisco et al., 2019).
- 121,461 clips: 12,483 bona fide (human), 108,978 spoofed (19 attacks: A01-A06
  seen in training; A07-A15, A17, A18 unseen; A16/A19 reuse the A04/A06 algorithms).
  107 speakers.
- Official partitions preserved: train (20 speakers, attacks A01-A06),
  development (20 speakers, A01-A06), evaluation (67 speakers, A07-A19).
- Evaluation attacks are unseen at training time — the real test.
- Licence: ODC-By 1.0. Footer on every data slide: "Contains information from the
  ASVspoof 2019 database (datashare.ed.ac.uk/handle/10283/3336), made available under
  the ODC Attribution License."
- Context only (never as our result): published ASVspoof 2019 baselines B01 (CQCC-GMM)
  and B02 (LFCC-GMM).

Visual: partition/attack table or stacked bar chart showing class balance and
the known/unseen split.

---

## Slide 4 — Evaluation integrity (45 s)

**Keeping the test honest**

- Strict partition walls: no evaluation data in training, tuning, threshold
  setting, or checkpoint selection.
- Windows and augmentations assigned to their original clip and partition —
  split before transformation.
- Development speakers reserved for the operating threshold (5 of 20,
  SHA-256 ranked selection).
- Protocol, thresholds, candidates and metrics frozen before any evaluation
  prediction was inspected.
- One score per original clip (mean of 4-second window logits).

Speaker note: this is the boring-but-crucial part. Honest evaluation is
the contribution, not the model.

---

## Slide 5 — Methods: baseline (45 s)

**MFCC + Logistic Regression baseline**

- 20 MFCCs plus their first-order deltas per frame; per 4 s window the mean and
  standard deviation of each of the 40 channels give an 80-dimensional vector
  (`configs/default.yaml`, `preprocessing.py`). Pooled over real audio only — no
  zero-pad contamination, "option D".
- Logistic Regression, 3 hyperparameter settings, class-balanced.
- Selected on development EER; threshold at 5% false-positive rate on
  5 reserved speakers.
- Purpose: a reference that the CNN must beat on the same protocol.

Visual: MFCC feature pipeline diagram (audio → windows → MFCC → pool → LR).

---

## Slide 6 — Methods: CNN (1 min 15 s)

**Compact log-Mel CNN (~23.7K parameters)**

- 64-bin log-Mel spectrogram, 4-second windows at 16 kHz.
- 3 conv blocks (16 → 32 → 64 channels, 3×3, batch norm, ReLU; max-pool after
  blocks 1 and 2 only), adaptive average pool, linear head.
- Trained with balanced cross-entropy, Adam, early stopping on dev clip EER.
- Two training conditions × two seeds:
  - **Clean:** trained on unmodified audio.
  - **Augmented:** per-clip Opus 24 kbps codec and white noise 10-30 dB SNR
    (each 50%, drawn only for the 50% of clips that are augmented, so about 25% of
    clips per epoch get each), applied before the random crop.
- Same inference pipeline for evaluation and serving.
- CPU-only: Xeon E5-2650L, no GPU. The architecture is designed to be
  small enough.

Visual: CNN architecture diagram. Keep it simple — 3 blocks + pool + head.

---

## Slide 7 — Results: headline (1 min)

**Clean EER on unseen attacks (the primary endpoint)**

| Model | EER % [95% CI] |
| --- | --- |
| LR baseline | 19.93 [17.18, 22.80] |
| CNN clean seed 1 | 8.30 [6.98, 9.50] |
| CNN clean seed 2 | 6.68 [5.52, 8.06] |
| CNN augmented seed 1 (deployed) | 11.77 [10.28, 13.04] |
| CNN augmented seed 2 | 7.35 [6.18, 8.38] |

- CNNs reduce LR's EER by 41–66% (6.7–11.8% vs 19.9%).
- Seed variance is large: augmented seeds differ by 4.4 points with
  non-overlapping intervals. Clean-vs-augmented is not decided by these
  point estimates alone.
- The deployed model (augmented seed 1) has the highest clean CNN EER. It was chosen
  by a development-only rule written after the locked report was read: disclosed,
  not blind.

Visual: grouped bar chart or dot plot with CI whiskers.

---

## Slide 8 — Results: robustness (1 min)

**What happens under noise and compression?**

EER % on unseen attacks by condition. The augmented models were trained with codec
and noise, so their gains under those conditions are matched-condition gains. The
condition parameters are frozen; the clean-audio threshold is not recalibrated.

| Model | clean | codec 24k | noise 20 dB | noise 10 dB |
| --- | --- | --- | --- | --- |
| LR | 19.9 | 17.6 | 25.7 | 32.9 |
| CNN clean s1 | 8.3 | 13.4 | 27.9 | **61.4** |
| CNN aug s1 (deployed) | 11.8 | 13.1 | 19.7 | 20.6 |
| CNN aug s2 | 7.4 | 7.7 | 17.8 | 22.1 |

- **Augmentation helps under matched noise**: augmented CNNs hold up under
  10 dB noise (EER ~20-22%) while clean CNN seed 1 drops to 61.4% (below
  chance; clean seed 2 is 30.1%).
- **Thresholds don't transfer**: the clean-dev threshold flags 25-33% of
  human clips under noise/codec for the deployed model.
- Negative result published, not hidden.

Visual: line chart, EER by condition, one line per model family.

---

## Slide 9 — Results: per-attack and failures (45 s)

**Not all attacks are equal**

- A09 is nearly solved by every CNN (EER ≈ 0%).
- A17 is hardest for LR and three of the four CNNs (CNN EER 17-22%); for the
  deployed model A10 is (27.1% vs 19.5% on A17).
- A10, A13, A15 remain challenging.
- Silence trimming raises CNN EER from 7-12% → 20-23% (partial reliance
  on silence cues — shortcut diagnostic, not proof).
- Known-algorithm attacks (A16/A19, reusing A04/A06) are much easier
  (CNN EER 0.4-6.5%).

Visual: heatmap of per-attack EER (attacks × models), color-scaled.

Speaker note: honest reporting — some attacks are hard, one model seed
has a below-chance negative result, and there's evidence of a silence
shortcut.

---

## Slide 10 — Live demo (2 min)

**Show the detector working**

1. Open the Streamlit UI (via SSH tunnel to the host).
2. Upload a clean human speech clip → show score, decision, spectrogram.
3. Upload a synthetic clip → show the score. Clips come from the predeclared demo
   set (N, sentences, tool, version and terms to be recorded; still open in the TODO plan); each is scored once and all
   results are shown, including misses. Report as "n=..., illustrative".
4. Upload an invalid input via the UI — `too_short.wav` (under 1 s) and
   `digital_silence.wav` (silence) — to show the structured rejection errors.
   MP3 cannot be uploaded (the UI only accepts WAV/FLAC); to show the format
   rejection, use `curl -s -F file=@demo_clips/real.mp3 localhost:8000/predict`
   in a terminal.
5. Discuss what the output means: "flagged_spoof_like" is not "fake",
   "not_flagged" is not "real".
6. Show the model info endpoint — bundle hash, threshold, limitations notice.

Fallback: saved screenshots and pre-recorded results if the live demo fails.

If a real-voice phone recording is flagged spoof-like, say so (issue #103: both of
the owner's two test recordings were; anecdotal, n = 2). It is the measured
threshold-transfer limitation (#83), not a bug and not a finding about the speaker.

Speaker note: keep the demo quick. The point is that it works, shows
measured limitations, and doesn't over-claim.

---

## Slide 11 — Architecture and reliability (~30 s)

**Serving stack**

- FastAPI + Streamlit, one Docker image (~4.8 GB, CPU-only torch).
- Latency: p50 19.4 ms, p95 107.8 ms (benchmark on declared input mix).
- Bundle mounted read-only, verified at startup by hash.
- No audio logged or retained. Every response carries a limitations notice.
- Same inference code path for evaluation and serving (verified: 198 dev clips, max score difference 0.0, 0 decision flips).

Visual: simple serving diagram (upload → API → scorer → response).

---

## Slide 12 — Conclusions and future work (1 min)

**What we learned**

1. A 23.7K-parameter CNN on log-Mel spectrograms cuts the LR baseline's
   EER by 41-66% on unseen attacks (6.7-11.8% vs 19.9%).
2. Training augmentation (noise + codec) improves robustness under the matched
   conditions; seed 1 of the augmented family also has the highest clean EER,
   but seed 2 does not, so the clean cost is not established.
3. Seed variance is large — single-seed results can be misleading.
4. Thresholds set on clean audio do not transfer under distortion.
5. There is evidence of partial reliance on silence cues.
6. Honest evaluation matters more than the number: frozen protocol,
   published negatives, disclosed limitations. The deployed model was chosen by a
   development-only rule written after the locked report was read, and it has the
   highest clean unseen EER of the four CNNs; both are disclosed.

**Future work (Phase 2)**

- ASVspoof 2021 DF transfer evaluation.
- Frozen wav2vec 2.0 embeddings (if compute allows).
- Calibration and abstention on uncertain scores.

**Not claimed:** this is not an official leaderboard result, not a speaker
verifier, and not proof of authenticity.

---

## Backup materials

- Pre-saved demo screenshots and scoring results.
- Model card (`docs/model-card.md`).
- Full evaluation report (`docs/evaluation-results-g2.md`).
- Reproduction guide (`docs/reproduction.md`).
