# Model card: `cnn_augmented_seed1` (deployed demo model)

Status: draft of 2026-10-20 for the Phase 1 release. Every number below is copied from
a committed source (`docs/evaluation-results-g2.md`, `docs/deployed-model-decision.md`,
`configs/*.yaml`) or from the host verification logs named in the text. Nothing here is
a new measurement. This is a project evaluation, **not an official ASVspoof leaderboard
result**.

## 1. What it is

A compact log-Mel CNN that returns a **score** for how spoof-like a short speech clip is,
and a decision at one fixed threshold. It is a score, not a probability: it is
uncalibrated, and no calibration was fitted or evaluated.

| Item | Value |
| --- | --- |
| Model id / version | `cnn_augmented_seed1` / `cnn:augmented_seed1` |
| Bundle hash (pinned, checked at startup) | `999bd10a831fcdff43829fbc197c3d4991491153fcead7cf594c1a971e3fe8a0` |
| Architecture | 3 conv blocks (16, 32, 64 channels; 3x3; batch norm; ReLU; max pool), adaptive average pool, linear head to 2 classes. About 23.7 thousand trainable parameters (computed from the layer definitions in `cnn_pilot.py`) |
| Input | 4-second windows at 16 kHz mono; 64-bin log-Mel (`mel-v1`: n_fft 400, hop 160, 20-8000 Hz); per-bin standardization fitted on training data only |
| Clip score | Mean of the per-window logits (`mean_window_logit`); higher means more spoof-like |
| Operating threshold | -1.2519543 (float64 `-1.251954287290573`), fixed on clean development audio from 5 reserved development speakers (574 bona fide clips; achieved false-positive rate 4.88% against a 5% target) |
| Decision | `flagged_spoof_like` if score >= threshold, else `not_flagged`. The API never returns "real", "fake", a probability or a confidence |
| Training | Cross-entropy, balanced class weights, Adam (lr 0.001, weight decay 0.0001), batch 32, at most 20 epochs, early stopping (patience 5) on development clip EER. Declared seeds 20261004 and 20261005; see `configs/cnn_study.yaml` |
| Augmentation (this model) | Per clip with probability 0.5: Opus 24 kbps codec (probability 0.5) and white noise at 10-30 dB SNR (probability 0.5), applied before the random crop. The clean model family was trained without it |

## 2. Intended use

- **Intended:** an educational and research demonstration of a spoof-versus-bona-fide
  score on short read-speech clips of the kind found in ASVspoof 2019 LA,
  shown next to its measured error rates and limitations.
- **Not intended:** deciding whether a recording is authentic or synthetic in any
  consequential setting (legal, journalistic, security, hiring, fraud handling);
  identifying or verifying a speaker; checking what was said; screening phone calls or
  compressed or noisy audio; any use that treats `not_flagged` as proof of human speech.
- Uploaded audio is processed in memory for the request and is not logged or retained.
  The service accepts WAV or FLAC, 1 to 30 seconds, up to 20 MiB.

## 3. Training and evaluation data

ASVspoof 2019 Logical Access, official partitions preserved (`docs/data.md`). Licence
ODC-By 1.0; required attribution and citation are recorded there.

| Partition | Bona fide | Spoof | Attacks | Speakers | Used for |
| --- | --- | --- | --- | --- | --- |
| Train | 2,580 | 22,800 | A01-A06 | 20 | fitting, scaler statistics |
| Development | 2,548 | 22,296 | A01-A06 | 20 | checkpoint and model selection; 5 speakers reserved for the threshold |
| Evaluation | 7,355 | 63,882 | A07-A19 | 67 | locked final test only |

The evaluation partition was not used for fitting, tuning, checkpoint selection or the
threshold. Attacks A16 and A19 reuse the A04 and A06 algorithms, so they are reported as
a separate "known algorithm" group and never pooled with the unseen attacks.

Language and speaker scope: the project did not characterize the language or speaker
population beyond the dataset's own documentation. Do not assume the model works for
other languages, accents, microphones, channels or newer voice-cloning systems; none
were tested.

## 4. Measured performance (locked evaluation, REPORTED)

95% intervals are speaker-grouped percentile bootstrap intervals (10,000 resamples,
seed 20261015, 67 evaluation speakers). They cover test-sampling variance only. No
result is described as significant: about 675 unadjusted intervals exist in the report.

Primary endpoint, clean EER on unseen attacks (A07-A15, A17, A18; 7,355 bona fide,
54,054 spoof):

| Model | EER % [95% CI] |
| --- | --- |
| **cnn_augmented_seed1 (deployed)** | **11.77 [10.28, 13.04]** |
| cnn_augmented_seed2 | 7.35 [6.18, 8.38] |
| cnn_clean_seed1 | 8.30 [6.98, 9.50] |
| cnn_clean_seed2 | 6.68 [5.52, 8.06] |
| lr_baseline (MFCC + logistic regression) | 19.93 [17.18, 22.80] |

The deployed model has the highest clean EER of the four CNNs on this endpoint. It was
chosen by a rule fixed in advance from **development** results only (lowest mean
development EER over clean, codec, noise 20 dB and noise 10 dB: 13.44 against 15.25 for
the runner-up). The locked report had already been read when that rule was written down,
so the choice should be read as made with knowledge of the evaluation (disclosed in
`docs/deployed-model-decision.md`).

Known-algorithm group (A16, A19; 7,355 bona fide, 9,828 spoof): EER 3.70 [2.76, 4.82].

Pooled clean evaluation, all attacks (mixed seen and unseen; not a headline), at the
frozen threshold:

| Measure | Value |
| --- | --- |
| Clips | 7,355 bona fide, 63,882 spoof |
| EER / ROC AUC | 10.74% / 95.76% |
| Macro F1 / balanced accuracy | 76.42% / 89.78% |
| Human false-positive rate | 7.21% (target was 5%; development gave 4.91%) |
| Synthetic detection rate | 86.78% |
| Confusion (TN / FP / FN / TP) | 6,825 / 530 / 8,448 / 55,434 |

Per-attack clean EER for this model (%, no intervals): A07 6.80, A08 1.12, A09 0.04,
A10 27.09, A11 7.90, A12 2.71, A13 16.26, A14 6.06, A15 14.47, A16 (known) 2.98,
A17 19.47, A18 2.71, A19 (known) 4.26. The hardest unseen attacks are A10, A13, A15 and
A17.

Seed variance is large: the two augmented seeds differ by 4.4 points on the primary
endpoint with non-overlapping intervals. The deployed model is a single-seed pick.

## 5. Known failures

**Degraded audio (issue #83).** Unseen-attack group, this model:

| Condition | EER % | Human flagged % | Synthetic flagged % |
| --- | --- | --- | --- |
| clean | 11.77 | 7.21 | 84.72 |
| Opus 24 kbps codec | 13.06 | 32.56 | 94.01 |
| noise 20 dB SNR | 19.73 | 26.15 | 84.36 |
| noise 10 dB SNR | 20.64 | 25.33 | 81.61 |

The ranking (EER) degrades gradually, but the **frozen threshold does not transfer**:
about a quarter to a third of human speech is flagged under codec or noise. On
development audio the same model flags 21-27% of bona fide clips under these conditions.
The training augmentation overlaps the codec and noise test conditions, so these are
matched-condition numbers, not evidence of general robustness.

**Silence shortcut (issue #84).** Removing leading and trailing silence raises this
model's EER from 11.77% to 23.29% (all CNNs: from 6.7-11.8% to 20.1-23.3%). This is
consistent with partial reliance on silence cues. It is a diagnostic, not proof, and not
a robustness claim.

**Other limits.**
- Only ASVspoof 2019 LA was evaluated; no other domain, dataset or recent cloning system.
- Scores average 4-second windows, so partial synthesis can be diluted.
- Clips under 1 second are rejected: the benchmark scored them, but they were not
  validated separately (issue #14).
- Evaluation-only duration results are not available: the locked report's `duration_*`
  slices pool development and evaluation clips (issue #85).
- 67 evaluation speakers are the resampling unit because source recordings cannot be
  mapped; candidates and conditions share the same clips.
- The threshold rests on 5 development speakers and one clean-audio condition.

## 6. Serving and resources

Measured on hp-xl420gen9-01, CPU only, torch 2.14.1 with one thread:

- Container cold start to healthy: 39 s; warmup of a 4-second clip 29 s (cold start), then
  70 ms for the second warmup clip (12 seconds, multi-window). A single live request returned in about 22 ms during
  host verification. These are single observations, **not** the p50/p95 benchmark with a
  declared input mix, which is still to be run.
- Memory: API container 612 MiB, UI container 49 MiB.
- Image size 19.97 GB, because `uv.lock` resolves CUDA-capable torch wheels.

## 7. Provenance and reproducibility

- Freeze record: `reports/evaluation_freeze_g2.json` (FROZEN copy sha256
  `cfd1406ca977362207228aa2af8a494c45c0f3cbe1216ef9509aa3ea1d8cb4d2`; REPORTED copy
  `88f27bda2c317136826a5337b13cc4ac3224e948c865efa7fb120665d321fb18`, not committed).
- Locked report sha256 `c0116ce295b3c7f50a667bff79b77e3c476494b9e4cd895f36aae4421427ed2b`;
  predictions hash `1ef12a29a78ec5fe59d9bb6c068a218fd69f4f9430ad70f7a51f0b06b3f3d23a`.
- Archive with SHA-256 checksums (outside Git): `archive_g2_20261020`, 80 files, written
  2026-10-20 by `scripts/archive_artifacts.sh`.
- Code commit of the evaluation: `c4445f5083d80a91b2014fa57937eb328eb8ae5d`.
- Model weights are not committed. Redistribution of the bundle has not been decided.

## 8. Ethical and misuse considerations

A spoof-like score can be wrong in both directions, and a false alarm labels a real
person's speech as suspicious. At the frozen threshold roughly 7% of clean human clips
and 25-33% of degraded human clips are flagged. Do not use the output to accuse anyone.
A determined attacker can adapt to a public detector, and this model was not tested
against adaptive attacks.
