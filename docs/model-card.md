# Model card: `cnn_augmented_seed1` (deployed demo model)

**Status: DRAFT** (written 2026-10-20; re-checked against the repository 2026-10-27 and again 2026-11-06). It stays DRAFT: the presentation was delivered on 5 November 2026 but none of the five blockers below was closed, and promoting the card would assert evidence that does not exist ([#110](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/110)).
Every number below is copied from a committed source (`docs/evaluation-results-g2.md`,
`docs/deployed-model-decision.md`, `configs/*.yaml`) or from the host verification logs
named in the text. Nothing here is a new measurement. This is a project evaluation,
**not an official ASVspoof leaderboard result**.

The card stays a draft until these are resolved (none is hidden elsewhere):

1. The serving code (PR #88) has not had its `/code-review` (#89), so section 6 describes
   host-verified behavior, not reviewed behavior.
2. A fresh-checkout reproduction of data preparation, training and the report has not been
   run ([`reproduction.md`](reproduction.md); open in the TODO plan).
3. The demo set is now recorded (`docs/data.md`, "Demonstration clips"): the owner's
   own consented recording, text-to-speech clips from Ondoku3, and the locally generated
   invalid inputs, with Ondoku3's terms checked and cited there; the free-tier credit
   notice the delivered deck lacked has been added for reuse. **Still missing: the clip
   counts and the own-voice clip's outcome** ([#109](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/109)). Nothing in this card characterizes
   modern voice-cloning systems; see the demo note in section 5, which is an anecdote,
   not a measurement.
4. Real-device audio (phone and laptop microphones) was never evaluated; the only
   observation is the anecdote in section 5 (#103).
5. Evaluation-only duration results do not exist (#85), and redistribution of the model
   bundle is undecided.

## 1. What it is

A compact log-Mel CNN that returns a **score** for how spoof-like a short speech clip is,
and a decision at one fixed threshold. It is a score, not a probability: it is
uncalibrated, and no calibration was fitted or evaluated.

| Item | Value |
| --- | --- |
| Model id / version | `cnn_augmented_seed1` / `cnn:augmented_seed1` |
| Bundle hash (pinned, checked at startup) | `999bd10a831fcdff43829fbc197c3d4991491153fcead7cf594c1a971e3fe8a0` |
| Architecture | 3 conv blocks (16, 32, 64 channels; 3x3; batch norm; ReLU), 2x2 max pool after blocks 1 and 2 only, adaptive average pool, linear head to 2 classes. About 23.7 thousand trainable parameters (computed from the layer definitions in `cnn_pilot.py`) |
| Input | 4-second windows at 16 kHz mono; 64-bin log-Mel (`mel-v1`: n_fft 400, hop 160, 20-8000 Hz); per-bin standardization fitted on training data only |
| Clip score | Mean of the per-window logits (`mean_window_logit`); higher means more spoof-like |
| Operating threshold | -1.2519543 (float64 `-1.251954287290573`), fixed on clean development audio from 5 reserved development speakers (574 bona fide clips; achieved false-positive rate 4.88% against a 5% target) |
| Decision | `flagged_spoof_like` if score >= threshold, else `not_flagged`. The API never returns "real", "fake", a probability or a confidence |
| Training | Cross-entropy, balanced class weights, Adam (lr 0.001, weight decay 0.0001), batch 32, at most 20 epochs, early stopping (patience 5) on development clip EER. Declared seeds 20261004 and 20261005; see `configs/cnn_study.yaml` |
| Augmentation (this model) | Specified in `configs/robustness_spec.yaml` under `training_augmentation`. Per clip with probability 0.5: Opus 24 kbps codec (probability 0.5) and white noise at 10-30 dB SNR (probability 0.5), applied before the random crop. The probabilities are nested and redrawn each epoch, so each transform reaches about 25% of clips per epoch and about 62.5% of clips are left clean. The clean model family was trained without it |

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

> Contains information from the ASVspoof 2019 database
> (<https://datashare.ed.ac.uk/handle/10283/3336>), made available under the ODC
> Attribution License (<https://opendatacommons.org/licenses/by/1-0/>).

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
chosen by a rule applied to **development** results only (lowest mean
development EER over clean, codec, noise 20 dB and noise 10 dB: 13.44 against 15.25 for
the runner-up). The locked report had already been read when that rule was written down,
so the selection was not blind and the rule was not pre-registered: read the choice as
made with knowledge of the evaluation (disclosed in `docs/deployed-model-decision.md`).

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

**Own-voice recordings (issue #103, anecdotal).** Two phone recordings of the owner's own
voice (one speaker, one device, 8.43 s and 7.87 s, converted to 16 kHz mono WAV) were both
flagged `flagged_spoof_like` by the deployed service (scores +2.65 and -0.52 against the
threshold -1.2520), while five development bona fide clips scored -4.42 to -2.75 and were
not flagged. n = 2 is not a measured rate and real-device audio was not evaluated; it is
consistent with the threshold non-transfer above. The model must not be used to judge a
person's own recording.

**Live demo, 5 November 2026 (anecdotal, uncounted).** In the presentation the deployed
service flagged text-to-speech clips from Ondoku3 (an online TTS service) as spoof-like
- the correct direction for synthetic speech. The clip count, the scores and the
outcome for the own-voice clip were not recorded, and the clips were never scored under
a declared protocol. **This is not evidence that the model detects contemporary
speech synthesis.** It is one impression from one unrecorded demonstration, it does not
change any number above, and it does not soften the statement in section 3 that no
other dataset or recent cloning system was evaluated. The opposite anecdote (#103,
above) came from the same kind of informal check. Provenance: `docs/data.md`,
"Demonstration clips". The container that produced those scores was an ARM64 build on
the presenting laptop that was never parity-checked against the `linux/amd64`
evaluation path ([#108](https://github.com/mohamadar77/ai-voice-deepfake-detector/issues/108)).

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

Measured on hp-xl420gen9-01 (Xeon E5-2650L v3 @ 1.80 GHz), CPU only, torch
2.14.1+cpu with one thread:

- **Latency benchmark** (`configs/latency_benchmark.yaml`, `scripts/latency_benchmark.py`):
  250 rounds over five clip types (1 s, 4 s, 12 s, 30 s mono 16 kHz + 4 s stereo 44.1 kHz),
  seed 20261020, 3 warmup rounds excluded.
  - Overall: p50 19.4 ms, p95 107.8 ms, p99 113.2 ms, 0 failures.
  - Per-bin p50/p95: 1 s = 10.4/10.7 ms, 4 s = 11.5/11.8 ms, 12 s = 31.3/31.6 ms,
    30 s = 102.3/113.2 ms, 4 s stereo = 19.4/19.7 ms.
  - Peak RSS 1016 MiB.
- Container cold start to healthy: 39 s; warmup 28.6 s (cold) then 71.8 ms (warm).
- Memory: API container 612 MiB, UI container 49 MiB.
- Image size 4.82 GB (CPU-only torch wheels via `[tool.uv.sources]`; was 19.97 GB with
  CUDA wheels, a 76% reduction). 91 packages (was 110).
- **Container-vs-library score parity** (`scripts/container_parity.py`, seed 20261021,
  198 development clips on hp-xl420gen9-01):
  max |score_container − score_library| = 0.0, 0 decision flips, 67 multi-window clips
  included. 2 clips rejected by the service (duration < 1 s). Same inference code path
  confirmed between the Docker container (via HTTP) and the in-process library scorer.

## 7. Provenance and reproducibility

- Freeze record: `reports/evaluation_freeze_g2.json` (FROZEN copy sha256
  `cfd1406ca977362207228aa2af8a494c45c0f3cbe1216ef9509aa3ea1d8cb4d2`; REPORTED copy
  `88f27bda2c317136826a5337b13cc4ac3224e948c865efa7fb120665d321fb18`, not committed).
- Locked report sha256 `c0116ce295b3c7f50a667bff79b77e3c476494b9e4cd895f36aae4421427ed2b`;
  predictions hash `1ef12a29a78ec5fe59d9bb6c068a218fd69f4f9430ad70f7a51f0b06b3f3d23a`.
- Archive with SHA-256 checksums (outside Git): `archive_g2_20261020`, 80 files, written
  2026-10-20 by `scripts/archive_artifacts.sh`.
- Code commit of the evaluation: `c4445f5083d80a91b2014fa57937eb328eb8ae5d`.
- Selection-order deviation: the development-only rule that chose this model was written
  after the locked report had been read (`docs/deployed-model-decision.md`; deviation
  register in `docs/architecture-and-delivery-plan.md`). It is not a blind selection.
- Model weights are not committed. Redistribution of the bundle has not been decided.

## 8. Ethical and misuse considerations

A spoof-like score can be wrong in both directions, and a false alarm labels a real
person's speech as suspicious. At the frozen threshold roughly 7% of clean human clips
and 25-33% of degraded human clips are flagged. Do not use the output to accuse anyone.
A determined attacker can adapt to a public detector, and this model was not tested
against adaptive attacks.
