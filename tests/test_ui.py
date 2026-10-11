"""Tests for the UI's HTTP client logic, run in-process against the ASGI app.

The FastAPI ``TestClient`` is an httpx-style client, so the UI functions can be
exercised against a real app with a stub scorer — no server, no model, no
dataset, and Streamlit is never imported.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from voice_detector.api import create_app
from voice_detector.scoring import AudioSource, ClipScore
from voice_detector.ui import ApiError, check_ready, content_type_for, score_clip


class StubScorer:
    """Minimal scorer implementing the Scorer protocol for UI tests."""

    model_version = "stub-model-1"
    preprocessing_version = "mfcc-v1"

    def __init__(self, *, score: float = 0.9, threshold: float = 0.5) -> None:
        self.score = score
        self.threshold = threshold

    def score_source(self, source: AudioSource) -> ClipScore:
        return ClipScore(
            score=self.score,
            label="spoof" if self.score >= self.threshold else "bonafide",
            score_direction="higher_more_spoof",
            threshold=self.threshold,
            n_windows=3,
        )


def _wav_bytes(seconds: float = 2.0, sample_rate: int = 16000) -> bytes:
    t = np.linspace(0.0, seconds, int(seconds * sample_rate), endpoint=False)
    tone = (0.5 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, tone, sample_rate, format="WAV")
    return buffer.getvalue()


def test_content_type_mapping() -> None:
    assert content_type_for("clip.wav") == "audio/wav"
    assert content_type_for("CLIP.FLAC") == "audio/flac"
    assert content_type_for("mystery.bin") == "application/octet-stream"


def test_check_ready_true_with_scorer() -> None:
    client = TestClient(create_app(scorer=StubScorer()))
    readiness = check_ready(client)
    assert readiness.ready is True
    assert readiness.model_version == "stub-model-1"
    assert readiness.preprocessing_version == "mfcc-v1"


def test_check_ready_false_without_scorer() -> None:
    client = TestClient(create_app(scorer=None))
    readiness = check_ready(client)
    assert readiness.ready is False
    assert "not ready" in readiness.detail.lower() or readiness.detail


def test_score_clip_returns_parsed_contract() -> None:
    client = TestClient(create_app(scorer=StubScorer(score=0.8, threshold=0.5)))
    result = score_clip(client, "clip.wav", _wav_bytes())
    assert result["label"] == "spoof"
    assert result["score"] == 0.8
    assert result["model_version"] == "stub-model-1"
    assert result["request_id"]


def test_score_clip_raises_apierror_on_bad_type() -> None:
    client = TestClient(create_app(scorer=StubScorer()))
    with pytest.raises(ApiError) as excinfo:
        score_clip(client, "notes.txt", b"not audio")
    assert excinfo.value.status_code == 415


def test_score_clip_raises_apierror_without_scorer() -> None:
    client = TestClient(create_app(scorer=None))
    with pytest.raises(ApiError) as excinfo:
        score_clip(client, "clip.wav", _wav_bytes())
    assert excinfo.value.status_code == 503
