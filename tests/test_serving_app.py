"""The serving API, exercised in-process with a fake scorer and synthetic audio."""

from __future__ import annotations

import io
import json
import logging
import tempfile
import threading
import time

import numpy as np
import pytest
import soundfile as sf
from conftest import requires_librosa
from fastapi.testclient import TestClient
from serving_fakes import BUNDLE_HASH, FakeScorer, audio_bytes, make_settings
from starlette import formparsers

from voice_detector.preprocessing import AudioConfig, load_waveform
from voice_detector.scoring import DECISION_RULE, SCORE_DIRECTION
from voice_detector.serving.app import create_app
from voice_detector.serving.service import decision_from_label

SCORE_KEYS = {
    "schema_version", "request_id", "score", "score_direction", "threshold", "decision",
    "decision_rule", "n_windows", "duration_seconds", "source_sample_rate_hz", "model_id",
    "model_version", "preprocessing_version", "aggregation_rule", "bundle_hash",
    "processing_ms", "warnings", "notice",
}
FORBIDDEN_KEYS = {"label", "probability", "confidence", "prob", "verdict", "authentic"}


def _client(scorer: FakeScorer | None = None, **sections) -> TestClient:
    return TestClient(create_app(make_settings(**sections), scorer))


def _post(client: TestClient, data: bytes, name: str = "clip.wav"):
    return client.post("/predict", files={"file": (name, data, "audio/wav")})


def test_health_is_alive_without_a_model():
    with _client(None) as client:
        assert client.get("/health").json() == {"status": "alive"}


def test_not_ready_without_a_model_and_predict_is_503():
    with _client(None) as client:
        response = client.get("/ready")
        assert response.status_code == 503
        assert response.json()["code"] == "model_not_ready"
        assert client.get("/model").status_code == 503
        assert _post(client, audio_bytes(2.0)).status_code == 503


def test_ready_and_model_info():
    with _client(FakeScorer(threshold=0.25)) as client:
        ready = client.get("/ready")
        assert ready.status_code == 200
        assert ready.json() == {
            "status": "ready",
            "model_id": "cnn_fake",
            "model_version": "cnn:fake",
            "preprocessing_version": "mel-v1",
        }
        info = client.get("/model").json()
        assert info["threshold"] == 0.25
        assert info["bundle_hash"] == BUNDLE_HASH
        assert info["score_direction"] == SCORE_DIRECTION
        assert info["decision_rule"] == DECISION_RULE
        assert info["notice"].startswith("NOTICE")
        assert info["limitations"] == ["limitation one"]
        assert info["hints_provisional"] is True
        assert info["threshold_basis"]["target_fpr"] == 0.05
        assert "/" not in info["model_id"]  # no filesystem paths


def test_predict_returns_exactly_the_documented_fields():
    with _client(FakeScorer(score=1.5, threshold=0.0)) as client:
        response = _post(client, audio_bytes(2.0))
    assert response.status_code == 200
    body = response.json()
    assert set(body) == SCORE_KEYS
    assert not (FORBIDDEN_KEYS & set(body))
    assert body["decision"] == "flagged_spoof_like"
    assert body["score"] == 1.5 and body["threshold"] == 0.0
    assert body["score_direction"] == SCORE_DIRECTION
    assert body["decision_rule"] == DECISION_RULE
    assert body["duration_seconds"] == 2.0 and body["source_sample_rate_hz"] == 16000
    assert body["bundle_hash"] == BUNDLE_HASH and body["model_id"] == "cnn_fake"
    assert body["notice"].startswith("NOTICE")
    assert response.headers["x-request-id"] == body["request_id"]
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"


def test_decision_boundary_is_float64_and_inclusive():
    score = 0.1234567890123
    above = float(np.nextafter(score, np.inf))
    with _client(FakeScorer(score=score, threshold=score)) as client:
        assert _post(client, audio_bytes(2.0)).json()["decision"] == "flagged_spoof_like"
    with _client(FakeScorer(score=score, threshold=above)) as client:
        assert _post(client, audio_bytes(2.0)).json()["decision"] == "not_flagged"


def test_label_mapping_is_total_and_neutral():
    assert decision_from_label("spoof") == "flagged_spoof_like"
    assert decision_from_label("bonafide") == "not_flagged"
    with pytest.raises(ValueError):
        decision_from_label("authentic")


def test_same_upload_gives_the_same_response_apart_from_request_data():
    data = audio_bytes(3.0, seed=4)
    with _client(FakeScorer()) as client:
        first = _post(client, data).json()
        second = _post(client, data).json()
    for key in ("request_id", "processing_ms"):
        first.pop(key)
        second.pop(key)
    assert first == second


@requires_librosa
def test_the_scorer_receives_the_shared_decoders_waveform():
    data = audio_bytes(2.0, sample_rate=48000, channels=2)
    scorer = FakeScorer()
    with _client(scorer) as client:
        assert _post(client, data).status_code == 200
    expected = load_waveform(io.BytesIO(data), AudioConfig(sample_rate_hz=16000))
    assert np.array_equal(scorer.last_waveform, expected)


def test_short_tail_window_and_clip_warnings():
    # 4.03 s: a 4 s window plus a 0.03 s tail that is its own window (>= n_fft samples).
    with _client(FakeScorer()) as client:
        body = _post(client, audio_bytes(4.03)).json()
    assert body["n_windows"] == 2
    assert "short_tail_window" in [w["code"] for w in body["warnings"]]
    with _client(FakeScorer()) as client:
        long_body = _post(client, audio_bytes(11.0)).json()
    assert "long_clip" in [w["code"] for w in long_body["warnings"]]


def _clipped_wav() -> bytes:
    t = np.arange(32000) / 16000
    data = np.clip(2.0 * np.sin(2 * np.pi * 220.0 * t), -1.0, 1.0).astype(np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, data, 16000, format="WAV", subtype="FLOAT")
    return buffer.getvalue()


def test_clipping_and_low_level_warnings():
    with _client(FakeScorer()) as client:
        clipped = _post(client, _clipped_wav()).json()
        quiet = _post(client, audio_bytes(2.0, amplitude=0.002)).json()
    assert "clipping" in [w["code"] for w in clipped["warnings"]]
    assert "low_level" in [w["code"] for w in quiet["warnings"]]


def test_silent_audio_is_rejected():
    silent = io.BytesIO()
    sf.write(silent, np.zeros(32000, dtype=np.float32), 16000, format="WAV", subtype="PCM_16")
    scorer = FakeScorer()
    with _client(scorer) as client:
        response = _post(client, silent.getvalue())
    assert response.status_code == 422 and response.json()["code"] == "silent_audio"
    assert scorer.calls == 0


def test_short_clip_is_rejected_with_a_fixed_message():
    with _client(FakeScorer()) as client:
        response = _post(client, audio_bytes(0.5))
    assert response.status_code == 422
    assert response.json()["code"] == "duration_out_of_range"


def test_unsupported_and_undecodable_uploads():
    with _client(FakeScorer()) as client:
        aiff = _post(client, audio_bytes(2.0, fmt="AIFF"), "x.aiff")
        junk = _post(client, b"not audio", "x.wav")
    assert aiff.status_code == 415 and aiff.json()["code"] == "unsupported_format"
    assert junk.status_code == 400 and junk.json()["code"] == "undecodable_audio"


def test_request_shape_is_enforced():
    with _client(FakeScorer()) as client:
        two = client.post(
            "/predict",
            files=[("file", ("a.wav", audio_bytes(2.0))), ("file", ("b.wav", audio_bytes(2.0)))],
        )
        extra = client.post(
            "/predict", files={"file": ("a.wav", audio_bytes(2.0))}, data={"note": "hi"}
        )
        none = client.post("/predict", data={"x": "y"})
        wrong_name = client.post("/predict", files={"audio": ("a.wav", audio_bytes(2.0))})
    for response in (two, extra, none, wrong_name):
        assert response.status_code == 400
        assert response.json()["code"] == "invalid_request"


def test_oversize_upload_is_rejected_before_scoring():
    scorer = FakeScorer()
    with _client(scorer, input={"max_upload_bytes": 5000, "multipart_overhead_bytes": 1000}) as c:
        response = _post(c, audio_bytes(2.0))  # ~64 kB
    assert response.status_code == 413
    assert response.json()["code"] == "payload_too_large"
    assert scorer.calls == 0


def test_a_non_finite_score_is_unscorable_not_a_500():
    with _client(FakeScorer(score=float("nan"))) as client:
        response = _post(client, audio_bytes(2.0))
    assert response.status_code == 422 and response.json()["code"] == "unscorable_audio"


def test_scorer_errors_become_a_generic_500_without_the_message(
    caplog: pytest.LogCaptureFixture,
):
    caplog.set_level(logging.DEBUG)
    scorer = FakeScorer(raises=RuntimeError("SECRETMARK-from-exception"))
    with _client(scorer) as client:
        response = _post(client, audio_bytes(2.0))
    assert response.status_code == 500
    assert response.json()["code"] == "internal_error"
    assert "SECRETMARK" not in response.text
    assert "SECRETMARK" not in caplog.text
    assert "RuntimeError" in caplog.text  # the class name is allowed


def test_a_busy_service_answers_503_and_recovers():
    gate = threading.Event()
    scorer = FakeScorer(gate=gate)
    results: list[int] = []
    with _client(scorer, runtime={"queue_timeout_seconds": 0.2}) as client:

        def slow() -> None:
            results.append(_post(client, audio_bytes(2.0)).status_code)

        worker = threading.Thread(target=slow)
        worker.start()
        deadline = time.monotonic() + 3.0
        while scorer.calls == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        busy = _post(client, audio_bytes(2.0))
        gate.set()
        worker.join(timeout=5.0)
        after = _post(client, audio_bytes(2.0))
    assert busy.status_code == 503 and busy.json()["code"] == "server_busy"
    assert results == [200] and after.status_code == 200


def test_unlisted_host_header_is_refused():
    with _client(FakeScorer()) as client:
        response = client.get("/health", headers={"Host": "evil.example"})
    assert response.status_code == 400


def test_no_cors_headers_are_sent():
    with _client(FakeScorer()) as client:
        response = client.options(
            "/predict",
            headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"},
        )
    assert "access-control-allow-origin" not in response.headers


def test_failed_warmup_keeps_the_service_not_ready():
    scorer = FakeScorer(raises=RuntimeError("boom"))
    settings = make_settings(runtime={"warmup": {
        "enabled": True, "seed": 1, "seconds": [1.0], "resample_from_hz": 16000}})
    with TestClient(create_app(settings, scorer)) as client:
        assert client.get("/ready").status_code == 503


def test_successful_warmup_marks_ready():
    settings = make_settings(runtime={"warmup": {
        "enabled": True, "seed": 1, "seconds": [1.0, 2.0], "resample_from_hz": 16000}})
    scorer = FakeScorer()
    with TestClient(create_app(settings, scorer)) as client:
        assert client.get("/ready").status_code == 200
    assert scorer.calls == 2


def test_validation_errors_do_not_echo_input():
    with _client(FakeScorer()) as client:
        response = client.get("/health", params={"x": "SECRETMARK"})
        bad = client.post("/predict", content=b"SECRETMARK", headers={"content-type": "text/plain"})
    assert response.status_code == 200
    assert "SECRETMARK" not in bad.text


def test_upload_content_and_filename_never_reach_the_logs(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.DEBUG)
    marker_name = "SECRETMARK-speaker-name.wav"
    with _client(FakeScorer()) as client:
        _post(client, audio_bytes(2.0), marker_name)
        _post(client, b"SECRETMARK" * 100, marker_name)
    assert "SECRETMARK" not in caplog.text
    assert "speaker-name" not in caplog.text


class _RecordingSpool(tempfile.SpooledTemporaryFile):
    instances: list[_RecordingSpool] = []

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        type(self).instances.append(self)


def test_every_upload_spool_is_closed_on_success_and_failure(monkeypatch: pytest.MonkeyPatch):
    _RecordingSpool.instances = []
    monkeypatch.setattr(formparsers, "SpooledTemporaryFile", _RecordingSpool)
    with _client(FakeScorer()) as client:
        assert _post(client, audio_bytes(2.0)).status_code == 200
        assert _post(client, b"garbage").status_code == 400
        assert _post(client, audio_bytes(0.5)).status_code == 422
    assert _RecordingSpool.instances, "the multipart parser did not use the patched spool"
    assert all(spool.closed for spool in _RecordingSpool.instances)


def test_responses_are_valid_json_without_nan():
    with _client(FakeScorer()) as client:
        text = _post(client, audio_bytes(2.0)).text
    json.loads(text, parse_constant=lambda c: pytest.fail(f"non-finite constant {c}"))
