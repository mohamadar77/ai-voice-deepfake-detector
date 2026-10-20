"""Response and error models for the serving API.

No field is named ``label``, ``probability`` or ``confidence``: the output is a
model score and a decision at one frozen threshold, never a verdict.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCORE_RESPONSE_VERSION = "score-response-v1"

Decision = Literal["flagged_spoof_like", "not_flagged"]
WarningCode = Literal[
    "short_clip", "long_clip", "clipping", "low_level", "short_tail_window"
]
ErrorCode = Literal[
    "invalid_request",
    "undecodable_audio",
    "payload_too_large",
    "unsupported_format",
    "unsupported_audio_parameters",
    "duration_out_of_range",
    "empty_audio",
    "silent_audio",
    "unscorable_audio",
    "upload_timeout",
    "server_busy",
    "model_not_ready",
    "internal_error",
]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ScoreWarning(_Model):
    code: WarningCode
    message: str


class ScoreResponse(_Model):
    schema_version: Literal["score-response-v1"] = SCORE_RESPONSE_VERSION
    request_id: str
    score: float = Field(allow_inf_nan=False, description="float64 clip score, never rounded")
    score_direction: str
    threshold: float = Field(allow_inf_nan=False)
    decision: Decision
    decision_rule: str
    n_windows: int
    duration_seconds: float
    source_sample_rate_hz: int
    model_id: str
    model_version: str
    preprocessing_version: str
    aggregation_rule: str
    bundle_hash: str
    processing_ms: float
    warnings: list[ScoreWarning]
    notice: str


class ErrorResponse(_Model):
    detail: str
    code: ErrorCode
    request_id: str


class ModelInfo(_Model):
    schema_version: Literal["model-info-v1"] = "model-info-v1"
    model_id: str
    model_version: str
    model_family: str
    preprocessing_version: str
    aggregation_rule: str
    score_direction: str
    decision_rule: str
    threshold: float = Field(allow_inf_nan=False)
    threshold_basis: dict[str, Any]
    bundle_hash: str
    config_sha256: str
    hints_version: str
    hints_provisional: bool
    limits: dict[str, Any]
    runtime: dict[str, Any]
    notice: str
    limitations: list[str]
    evidence_refs: list[str]


class ReadyResponse(_Model):
    status: Literal["ready"] = "ready"
    model_id: str
    model_version: str
    preprocessing_version: str
