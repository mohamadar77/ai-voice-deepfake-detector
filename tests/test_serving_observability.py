"""Logging allowlist: nothing outside it, and nothing unsafe inside it."""

from __future__ import annotations

import json
import logging

import pytest

from voice_detector.serving.observability import log_event


def _last(caplog: pytest.LogCaptureFixture) -> dict:
    return json.loads(caplog.records[-1].getMessage())


def test_unknown_fields_are_refused(caplog: pytest.LogCaptureFixture):
    with pytest.raises(ValueError):
        log_event("x", filename="clip.wav")
    with pytest.raises(ValueError):
        log_event("x", score=0.5)


def test_unsafe_strings_are_redacted(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO, logger="voice_detector.serving")
    log_event("score_failed", error_type="Bad Type /etc/passwd", request_id="abc-123")
    record = _last(caplog)
    assert record["error_type"] == "redacted"
    assert record["request_id"] == "abc-123"


def test_non_finite_numbers_become_null(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO, logger="voice_detector.serving")
    log_event("request", total_ms=float("nan"), status=200)
    record = _last(caplog)
    assert record["total_ms"] is None and record["status"] == 200


def test_event_names_must_be_identifiers():
    with pytest.raises(ValueError):
        log_event("has space")
