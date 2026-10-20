"""Tests for the UI's HTTP client logic, run in-process against the ASGI app.

The FastAPI ``TestClient`` is an httpx-style client, so the UI functions can be
exercised against the real serving app with a fake scorer: no server, no model, no
dataset, and Streamlit is never imported.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient
from serving_fakes import FakeScorer, audio_bytes, make_settings

from voice_detector.serving.app import create_app
from voice_detector.ui import (
    ApiError,
    check_ready,
    content_type_for,
    decision_text,
    display_spectrogram,
    fetch_model_info,
    headline_rows,
    load_evaluation_summary,
    robustness_rows,
    score_clip,
    spectrogram_figure,
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


# --- evaluation summary and spectrogram view -------------------------------------------

REPO = Path(__file__).resolve().parents[1]


def test_summary_loads_and_matches_the_serving_candidate() -> None:
    summary = load_evaluation_summary()
    serving = yaml.safe_load((REPO / "configs" / "serving.yaml").read_text())
    assert summary["candidate_id"] == serving["model"]["candidate_id"]
    assert summary["threshold"] == pytest.approx(serving["model"]["expected_threshold"], abs=1e-6)


def test_summary_rejects_a_wrong_version(tmp_path: Path) -> None:
    bad = tmp_path / "summary.yaml"
    bad.write_text("version: other\n")
    with pytest.raises(ValueError):
        load_evaluation_summary(bad)


def test_summary_numbers_appear_in_the_source_document() -> None:
    """Every summary number must be copied from the REPORTED results document."""

    summary = load_evaluation_summary()
    doc = (REPO / summary["source_doc"]).read_text(encoding="utf-8")
    candidate_lines = [
        line for line in doc.splitlines() if line.startswith(f"| {summary['candidate_id']} |")
    ]
    assert candidate_lines
    for item in summary["headline"]:
        needle = f"{item['value']:.2f} [{item['ci'][0]:.2f}, {item['ci'][1]:.2f}]"
        assert any(needle in line for line in candidate_lines), needle
    robustness = summary["robustness"]
    for key in ("eer", "human_false_positive_rate", "synthetic_detection_rate"):
        formatted = [f"{value:.2f}" for value in robustness[key]]
        assert any(
            all(value in line for value in formatted) for line in candidate_lines
        ), (key, formatted)
    pooled = summary["pooled_clean"]
    confusion = pooled["confusion"]
    cells = (confusion["tn"], confusion["fp"], confusion["fn"], confusion["tp"])
    confusion_text = " / ".join(str(cell) for cell in cells)
    assert any(confusion_text in line for line in candidate_lines)


def test_summary_tables_carry_counts_and_stay_descriptive() -> None:
    summary = load_evaluation_summary()
    rows = headline_rows(summary)
    assert all(row["Human clips"] > 0 and row["Synthetic clips"] > 0 for row in rows)
    assert len(robustness_rows(summary)) == len(summary["robustness"]["conditions"])
    assert "significant" not in yaml.safe_dump(summary).lower()
    assert "not an official leaderboard result" in summary["framing"]


def _wav(seconds: float = 2.0, rate: int = 16000) -> bytes:
    import io

    import soundfile as sf

    rng = np.random.default_rng(0)
    buffer = io.BytesIO()
    noise = (0.1 * rng.standard_normal(int(seconds * rate))).astype("float32")
    sf.write(buffer, noise, rate, format="WAV")
    return buffer.getvalue()


def test_display_spectrogram_uses_the_shared_log_mel() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torchaudio")
    spectrogram = display_spectrogram(_wav(2.0))
    assert spectrogram.shape[0] == 64
    assert spectrogram.shape[1] == 1 + 64000 // 160  # one repeat-padded 4 s window
    assert spectrogram.dtype.kind == "f" and bool(np.isfinite(spectrogram).all())


def test_display_spectrogram_rejects_non_audio() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torchaudio")
    with pytest.raises(Exception):  # noqa: B017 - soundfile raises its own error type
        display_spectrogram(b"not audio")


def test_upload_to_display_flow() -> None:
    """Upload -> API score -> decision text -> spectrogram figure, on one clip."""

    pytest.importorskip("torch")
    pytest.importorskip("torchaudio")
    data = audio_bytes()
    client = _client(FakeScorer(score=0.8, threshold=0.5))
    assert check_ready(client).ready
    result = score_clip(client, "clip.wav", data)
    assert decision_text(result["decision"]).startswith("Flagged")
    figure = spectrogram_figure(display_spectrogram(data))
    assert len(figure.axes) >= 1
    figure.clear()
