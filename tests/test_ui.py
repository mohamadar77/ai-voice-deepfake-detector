"""Tests for the UI's HTTP client logic, run in-process against the ASGI app.

The FastAPI ``TestClient`` is an httpx-style client, so the UI functions can be
exercised against the real serving app with a fake scorer: no server, no model, no
dataset, and Streamlit is never imported.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from serving_fakes import FakeScorer, audio_bytes, make_settings

from voice_detector.serving.app import create_app
from voice_detector.ui import (
    ApiError,
    check_ready,
    content_type_for,
    decision_text,
    fetch_model_info,
    score_clip,
)


def _client(scorer: FakeScorer | None) -> TestClient:
    return TestClient(create_app(make_settings(), scorer))


def test_content_type_mapping() -> None:
    assert content_type_for("clip.wav") == "audio/wav"
    assert content_type_for("CLIP.FLAC") == "audio/flac"
    assert content_type_for("mystery.bin") == "application/octet-stream"


def test_check_ready_true_with_scorer() -> None:
    readiness = check_ready(_client(FakeScorer()))
    assert readiness.ready is True
    assert readiness.model_version == "cnn:fake"
    assert readiness.preprocessing_version == "mel-v1"
    assert readiness.model_id == "cnn_fake"


def test_check_ready_false_without_scorer() -> None:
    readiness = check_ready(_client(None))
    assert readiness.ready is False
    assert readiness.detail == "The model is not ready."


def test_score_clip_returns_parsed_contract() -> None:
    result = score_clip(_client(FakeScorer(score=0.8, threshold=0.5)), "clip.wav", audio_bytes())
    assert result["decision"] == "flagged_spoof_like"
    assert result["score"] == 0.8
    assert result["model_version"] == "cnn:fake"
    assert result["request_id"]
    assert result["notice"]


def test_score_clip_raises_apierror_on_undecodable_audio() -> None:
    with pytest.raises(ApiError) as excinfo:
        score_clip(_client(FakeScorer()), "notes.txt", b"not audio")
    assert excinfo.value.status_code == 400


def test_score_clip_raises_apierror_without_scorer() -> None:
    with pytest.raises(ApiError) as excinfo:
        score_clip(_client(None), "clip.wav", audio_bytes())
    assert excinfo.value.status_code == 503


def test_fetch_model_info_returns_the_limitations() -> None:
    info = fetch_model_info(_client(FakeScorer()))
    assert info is not None and info["limitations"] == ["limitation one"]
    assert fetch_model_info(_client(None)) is None


def test_decision_text_never_says_authentic_or_fake() -> None:
    for decision in ("flagged_spoof_like", "not_flagged"):
        text = decision_text(decision).lower()
        assert "authentic" not in text and "fake" not in text and "real" not in text
    assert decision_text("anything else") == "Unknown decision"
