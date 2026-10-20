"""Serving with a real (tiny, random-weight) CNN bundle. Needs torch; no real weights."""

from __future__ import annotations

import io
import threading
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from conftest import requires_librosa
from fastapi.testclient import TestClient
from serving_fakes import audio_bytes, make_settings
from tiny_bundle import RUN_LABEL, THRESHOLD, make_bundle

pytest.importorskip("torch")

from voice_detector.cnn_scoring import load_cnn_scorer  # noqa: E402
from voice_detector.hashing import hash_directory  # noqa: E402
from voice_detector.preprocessing import AudioConfig, load_waveform  # noqa: E402
from voice_detector.serving.app import create_app  # noqa: E402
from voice_detector.serving.loader import LoaderError, build_scorer  # noqa: E402


@pytest.fixture
def artifacts(tmp_path: Path) -> Path:
    make_bundle(tmp_path / "cnn_unit")
    return tmp_path


def _settings(artifacts: Path, **model_overrides):
    pins = {
        "candidate_id": "cnn_unit",
        "expected_bundle_hash": hash_directory(artifacts / "cnn_unit"),
        "expected_threshold": THRESHOLD,
    }
    pins.update(model_overrides)
    return make_settings(model=pins)


def test_build_scorer_verifies_and_retains_the_bundle_hash(artifacts: Path):
    settings = _settings(artifacts)
    scorer = build_scorer(settings, artifacts)
    assert scorer.bundle_hash == settings.model.expected_bundle_hash
    assert scorer.threshold == THRESHOLD
    assert scorer.model_version == f"cnn:{RUN_LABEL}"
    assert scorer.describe()["candidate_id"] == "cnn_unit"
    assert scorer.describe()["threshold_basis"]["target_fpr"] == 0.05


def test_a_wrong_hash_pin_refuses_to_load(artifacts: Path):
    settings = _settings(artifacts, expected_bundle_hash="0" * 64)
    with pytest.raises(LoaderError) as caught:
        build_scorer(settings, artifacts)
    assert caught.value.reason == "bundle_hash_mismatch"


def test_a_wrong_threshold_pin_refuses_to_load(artifacts: Path):
    settings = _settings(artifacts, expected_threshold=THRESHOLD + 1.0)
    with pytest.raises(LoaderError) as caught:
        build_scorer(settings, artifacts)
    assert caught.value.reason == "threshold_mismatch"


def test_a_stray_file_in_the_bundle_changes_the_hash_and_is_refused(artifacts: Path):
    settings = _settings(artifacts)
    (artifacts / "cnn_unit" / "stray.txt").write_text("x", encoding="utf-8")
    with pytest.raises(LoaderError) as caught:
        build_scorer(settings, artifacts)
    assert caught.value.reason == "bundle_hash_mismatch"


def test_a_missing_bundle_directory_is_reported(tmp_path: Path):
    settings = make_settings()
    with pytest.raises(LoaderError) as caught:
        build_scorer(settings, tmp_path)
    assert caught.value.reason == "bundle_missing"


def test_score_decoded_equals_the_evaluation_entry_points(artifacts: Path):
    scorer = build_scorer(_settings(artifacts), artifacts)
    data = audio_bytes(2.5, seed=2)
    waveform = load_waveform(io.BytesIO(data), AudioConfig(sample_rate_hz=16000))
    decoded = scorer.score_decoded(waveform)
    assert decoded.clip.score == scorer.score_waveform(waveform).clip_score
    assert decoded.clip.score == scorer.score_source(io.BytesIO(data)).score
    assert decoded.clip.n_windows == len(decoded.window_scores) == len(decoded.window_real_samples)
    assert sum(decoded.window_real_samples) == waveform.size


def test_the_api_returns_the_same_score_as_the_library(artifacts: Path):
    settings = _settings(artifacts)
    scorer = build_scorer(settings, artifacts)
    data = audio_bytes(2.5, seed=5)
    want = scorer.score_source(io.BytesIO(data))
    with TestClient(create_app(settings, scorer)) as client:
        body = client.post("/predict", files={"file": ("c.wav", data, "audio/wav")}).json()
    assert body["score"] == want.score
    assert body["threshold"] == THRESHOLD
    assert body["n_windows"] == want.n_windows
    assert body["bundle_hash"] == settings.model.expected_bundle_hash
    assert body["decision"] == ("flagged_spoof_like" if want.label == "spoof" else "not_flagged")


def test_concurrent_requests_give_bitwise_equal_scores(artifacts: Path):
    settings = _settings(artifacts)
    scorer = build_scorer(settings, artifacts)
    waveform = load_waveform(io.BytesIO(audio_bytes(2.5)), AudioConfig(sample_rate_hz=16000))
    expected = scorer.score_decoded(waveform).clip.score
    results: list[float] = []

    def run() -> None:
        for _ in range(5):
            results.append(scorer.score_decoded(waveform).clip.score)

    threads = [threading.Thread(target=run) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(results) == 40 and set(results) == {expected}


def test_huge_but_finite_samples_are_unscorable_not_a_server_error(artifacts: Path):
    settings = _settings(artifacts)
    scorer = build_scorer(settings, artifacts)
    samples = np.full(32000, 1e30, dtype=np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, samples, 16000, format="WAV", subtype="FLOAT")
    with TestClient(create_app(settings, scorer)) as client:
        response = client.post(
            "/predict", files={"file": ("c.wav", buffer.getvalue(), "audio/wav")}
        )
    assert response.status_code == 422
    assert response.json()["code"] == "unscorable_audio"


@requires_librosa
def test_warmup_runs_the_resample_path(artifacts: Path):
    settings = make_settings(
        model={
            "candidate_id": "cnn_unit",
            "expected_bundle_hash": hash_directory(artifacts / "cnn_unit"),
            "expected_threshold": THRESHOLD,
        },
        runtime={
            "warmup": {"enabled": True, "seed": 7, "seconds": [1.5, 3.0], "resample_from_hz": 44100}
        },
    )
    scorer = build_scorer(settings, artifacts)
    with TestClient(create_app(settings, scorer)) as client:
        assert client.get("/ready").status_code == 200


def test_the_library_loader_still_works_without_a_pin(artifacts: Path):
    scorer = load_cnn_scorer(artifacts / "cnn_unit")
    assert scorer.bundle_hash is None  # a hash is retained only when it was verified
