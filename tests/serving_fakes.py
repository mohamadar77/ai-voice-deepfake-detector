"""Shared helpers for the serving tests: a fake scorer, settings and audio bytes.

No torch, no trained bundle and no dataset: the fake satisfies ``ServableScorer`` so
the API, limits and privacy behaviour are tested in isolation.
"""

from __future__ import annotations

import copy
import io
import threading
from typing import Any

import numpy as np
import soundfile as sf

from voice_detector.preprocessing import window_spans
from voice_detector.scoring import ScoredClip, WaveformScore, clip_score_from
from voice_detector.serving.settings import ServingSettings

WINDOW_SAMPLES = 64000  # 4 s at 16 kHz, the deployed model's window
N_FFT = 400
BUNDLE_HASH = "a" * 64


class FakeScorer:
    """Deterministic stand-in for the CNN scorer; records what it was given."""

    model_family = "cnn"
    model_version = "cnn:fake"
    preprocessing_version = "mel-v1"
    aggregation_rule = "mean_window_logit"
    sample_rate_hz = 16000
    bundle_hash: str | None = BUNDLE_HASH

    def __init__(
        self,
        *,
        score: float = 1.0,
        threshold: float = 0.0,
        raises: Exception | None = None,
        gate: threading.Event | None = None,
    ) -> None:
        self.score = score
        self.threshold = threshold
        self.raises = raises
        self.gate = gate
        self.calls = 0
        self.last_waveform: np.ndarray | None = None

    def score_decoded(self, waveform: np.ndarray) -> ScoredClip:
        self.calls += 1
        self.last_waveform = waveform.copy()
        if self.gate is not None:
            self.gate.wait(timeout=5.0)
        if self.raises is not None:
            raise self.raises
        spans = window_spans(waveform.size, window_samples=WINDOW_SAMPLES, n_fft=N_FFT)
        clip = clip_score_from(WaveformScore(self.score, len(spans)), self.threshold)
        return ScoredClip(
            clip=clip,
            window_scores=tuple(self.score for _ in spans),
            window_real_samples=tuple(stop - start for start, stop in spans),
        )

    def describe(self) -> dict[str, Any]:
        return {
            "candidate_id": "cnn_fake",
            "threshold_basis": {"value": self.threshold, "target_fpr": 0.05},
        }


_BASE_SETTINGS: dict[str, Any] = {
    "version": "serving-v1",
    "model": {
        "candidate_id": "cnn_fake",
        "expected_bundle_hash": BUNDLE_HASH,
        "expected_threshold": 0.0,
    },
    "runtime": {
        "torch_threads": 1,
        "interop_threads": 1,
        "deterministic_algorithms": True,
        "max_concurrent_scoring": 1,
        "queue_timeout_seconds": 0.3,
        "warmup": {"enabled": False, "seed": 1, "seconds": [1.0], "resample_from_hz": 16000},
    },
    "server": {
        "host": "127.0.0.1",
        "port": 8000,
        "allowed_hosts": ["testserver"],
        "cors_allow_origins": [],
        "limit_concurrency": 4,
        "timeout_keep_alive_seconds": 5,
        "timeout_graceful_shutdown_seconds": 5,
        "upload_timeout_seconds": 5.0,
    },
    "input": {
        "max_upload_bytes": 2_000_000,
        "multipart_overhead_bytes": 65536,
        "accepted_formats": ["WAV", "FLAC"],
        "max_channels": 2,
        "min_source_sample_rate_hz": 8000,
        "max_source_sample_rate_hz": 96000,
        "min_duration_seconds": 1.0,
        "max_duration_seconds": 30.0,
        "silence_peak_dbfs": -60.0,
    },
    "hints": {
        "version": "hints-test",
        "provisional": True,
        "warn_below_seconds": 1.0,
        "warn_above_seconds": 10.0,
        "clipping_level": 0.999,
        "clipping_fraction": 0.001,
        "low_rms_dbfs": -45.0,
        "short_window_seconds": 1.0,
    },
    "observability": {"log_level": "INFO", "debug_tracebacks": False},
    "framing": {
        "notice": "NOTICE: a score, not a verdict.",
        "limitations": ["limitation one"],
        "evidence_refs": ["docs/deployed-model-decision.md"],
    },
}


def make_settings(**sections: dict[str, Any]) -> ServingSettings:
    """Valid settings with per-section overrides, e.g. ``make_settings(input={...})``."""

    data = copy.deepcopy(_BASE_SETTINGS)
    for name, override in sections.items():
        data[name].update(override)
    return ServingSettings.model_validate(data)


def audio_bytes(
    seconds: float = 2.0,
    *,
    sample_rate: int = 16000,
    channels: int = 1,
    amplitude: float = 0.3,
    fmt: str = "WAV",
    subtype: str | None = "PCM_16",
    seed: int = 0,
) -> bytes:
    """A seeded tone-plus-noise clip as encoded bytes."""

    rng = np.random.default_rng(seed)
    frames = int(round(seconds * sample_rate))
    t = np.arange(frames) / sample_rate
    tone = np.sin(2 * np.pi * 220.0 * t) + 0.1 * rng.standard_normal(frames)
    data = (amplitude * tone / 1.2).astype(np.float32)
    if channels > 1:
        data = np.stack([data] * channels, axis=1)
    buffer = io.BytesIO()
    sf.write(buffer, data, sample_rate, format=fmt, subtype=subtype)
    return buffer.getvalue()
