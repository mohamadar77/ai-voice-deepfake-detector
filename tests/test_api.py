"""Tests for the serving layer, using a stub scorer and synthetic audio.

No trained bundle, feature code or dataset is involved: the stub satisfies the
Scorer contract, so the API's validation and wiring are exercised in isolation.
"""

from __future__ import annotations

import io

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from voice_detector.api import create_app
from voice_detector.scoring import AudioSource, ClipScore


class StubScorer:
    """Minimal in-memory scorer implementing the Scorer protocol."""

    model_version = "stub-model-1"
    preprocessing_version = "mfcc-v1"

    def __init__(self, *, score: float = 0.9, threshold: float = 0.5) -> None:
        self.score = score
        self.threshold = threshold
        self.calls = 0

    def score_source(self, source: AudioSource) -> ClipScore:
        self.calls += 1
        return ClipScore(
            score=self.score,
            label="spoof" if self.score >= self.threshold else "bonafide",
            score_direction="higher_more_spoof",
            threshold=self.threshold,
            n_windows=3,
        )


def _wav_bytes(seconds: float, sample_rate: int = 16000, fmt: str = "WAV") -> bytes:
    t = np.linspace(0.0, seconds, int(seconds * sample_rate), endpoint=False)
    tone = (0.5 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, tone, sample_rate, format=fmt)
    return buffer.getvalue()


# --- health / ready ---------------------------------------------------------


def test_health_is_always_alive() -> None:
    client = TestClient(create_app(scorer=None))
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_ready_is_503_without_a_scorer() -> None:
    client = TestClient(create_app(scorer=None))
    assert client.get("/ready").status_code == 503


def test_ready_reports_versions_with_a_scorer() -> None:
    client = TestClient(create_app(scorer=StubScorer()))
    response = client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["model_version"] == "stub-model-1"
    assert body["preprocessing_version"] == "mfcc-v1"


# --- predict happy path -----------------------------------------------------


def test_predict_returns_the_full_contract() -> None:
    scorer = StubScorer(score=0.9, threshold=0.5)
    client = TestClient(create_app(scorer=scorer))
    response = client.post(
        "/predict",
        files={"file": ("clip.wav", _wav_bytes(2.0), "audio/wav")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "spoof"
    assert body["score"] == 0.9
    assert body["score_direction"] == "higher_more_spoof"
    assert body["threshold"] == 0.5
    assert body["model_version"] == "stub-model-1"
    assert body["preprocessing_version"] == "mfcc-v1"
    assert 1.9 < body["duration_seconds"] < 2.1
    assert body["processing_ms"] >= 0.0
    assert body["request_id"]
    assert "confidence" not in body  # scores are not probabilities of authenticity
    assert scorer.calls == 1


def test_predict_accepts_flac() -> None:
    client = TestClient(create_app(scorer=StubScorer()))
    response = client.post(
        "/predict",
        files={"file": ("clip.flac", _wav_bytes(2.0, fmt="FLAC"), "audio/flac")},
    )
    assert response.status_code == 200


# --- predict guards ---------------------------------------------------------


def test_predict_is_503_without_a_scorer() -> None:
    client = TestClient(create_app(scorer=None))
    response = client.post(
        "/predict",
        files={"file": ("clip.wav", _wav_bytes(2.0), "audio/wav")},
    )
    assert response.status_code == 503


def test_predict_rejects_unsupported_extension() -> None:
    client = TestClient(create_app(scorer=StubScorer()))
    response = client.post(
        "/predict",
        files={"file": ("notes.txt", b"not audio", "text/plain")},
    )
    assert response.status_code == 415


def test_predict_rejects_oversize_upload() -> None:
    client = TestClient(create_app(scorer=StubScorer(), max_upload_bytes=1024))
    response = client.post(
        "/predict",
        files={"file": ("clip.wav", _wav_bytes(2.0), "audio/wav")},  # well over 1 KiB
    )
    assert response.status_code == 413


def test_predict_rejects_undecodable_bytes() -> None:
    client = TestClient(create_app(scorer=StubScorer()))
    response = client.post(
        "/predict",
        files={"file": ("clip.wav", b"RIFFgarbage-not-a-real-wav", "audio/wav")},
    )
    assert response.status_code == 400


def test_predict_rejects_too_short_clip() -> None:
    client = TestClient(create_app(scorer=StubScorer(), min_duration_seconds=1.0))
    response = client.post(
        "/predict",
        files={"file": ("clip.wav", _wav_bytes(0.5), "audio/wav")},
    )
    assert response.status_code == 422


def test_predict_rejects_too_long_clip() -> None:
    client = TestClient(create_app(scorer=StubScorer(), max_duration_seconds=2.0))
    response = client.post(
        "/predict",
        files={"file": ("clip.wav", _wav_bytes(3.0), "audio/wav")},
    )
    assert response.status_code == 422
