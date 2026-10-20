"""Serving configuration, loaded from ``configs/serving.yaml``.

Every limit and policy lives in versioned YAML, not in code. Unknown keys are
rejected so a typo cannot silently disable a guard.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from voice_detector.config import expand_environment_values

SERVING_CONFIG_VERSION = "serving-v1"

Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelSettings(_Strict):
    candidate_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    expected_bundle_hash: Sha256Hex
    # Exact float64 from the freeze record; the loader refuses a bundle whose own
    # threshold differs. There is deliberately no key that overrides the threshold.
    expected_threshold: float = Field(allow_inf_nan=False)


class WarmupSettings(_Strict):
    enabled: bool = True
    seed: int = 0
    seconds: list[float] = Field(min_length=1)
    resample_from_hz: int = Field(gt=0)


class RuntimeSettings(_Strict):
    torch_threads: int = Field(ge=1)
    interop_threads: int = Field(ge=1)
    deterministic_algorithms: bool
    max_concurrent_scoring: int = Field(ge=1)
    queue_timeout_seconds: float = Field(gt=0)
    warmup: WarmupSettings


class ServerSettings(_Strict):
    host: str
    port: int = Field(ge=1, le=65535)
    allowed_hosts: list[str] = Field(min_length=1)
    cors_allow_origins: list[str] = Field(default_factory=list)
    limit_concurrency: int = Field(ge=1)
    timeout_keep_alive_seconds: int = Field(ge=1)
    timeout_graceful_shutdown_seconds: int = Field(ge=1)
    upload_timeout_seconds: float = Field(gt=0)


class InputSettings(_Strict):
    max_upload_bytes: int = Field(gt=0)
    multipart_overhead_bytes: int = Field(ge=0)
    accepted_formats: list[Literal["WAV", "FLAC"]] = Field(min_length=1)
    max_channels: int = Field(ge=1)
    min_source_sample_rate_hz: int = Field(gt=0)
    max_source_sample_rate_hz: int = Field(gt=0)
    min_duration_seconds: float = Field(gt=0)
    max_duration_seconds: float = Field(gt=0)
    silence_peak_dbfs: float = Field(allow_inf_nan=False)

    @model_validator(mode="after")
    def _ordered(self) -> InputSettings:
        if self.min_duration_seconds >= self.max_duration_seconds:
            raise ValueError("min_duration_seconds must be below max_duration_seconds")
        if self.min_source_sample_rate_hz >= self.max_source_sample_rate_hz:
            raise ValueError("min_source_sample_rate_hz must be below max_source_sample_rate_hz")
        return self


class HintSettings(_Strict):
    """Advisory input checks. They never change a score or a decision."""

    version: str
    provisional: bool
    warn_below_seconds: float = Field(gt=0)
    warn_above_seconds: float = Field(gt=0)
    clipping_level: float = Field(gt=0, le=1)
    clipping_fraction: float = Field(gt=0, le=1)
    low_rms_dbfs: float = Field(allow_inf_nan=False)
    short_window_seconds: float = Field(gt=0)


class ObservabilitySettings(_Strict):
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]
    debug_tracebacks: bool


class FramingSettings(_Strict):
    notice: str = Field(min_length=1)
    limitations: list[str] = Field(min_length=1)
    evidence_refs: list[str]


class ServingSettings(_Strict):
    version: Literal["serving-v1"]
    model: ModelSettings
    runtime: RuntimeSettings
    server: ServerSettings
    input: InputSettings
    hints: HintSettings
    observability: ObservabilitySettings
    framing: FramingSettings
    # SHA-256 of the YAML bytes, filled in by ``load_serving_settings``.
    config_sha256: str = ""


def load_serving_settings(path: Path) -> ServingSettings:
    """Load and validate the serving configuration (environment variables expanded)."""

    raw_bytes = Path(path).read_bytes()
    data = yaml.safe_load(raw_bytes)
    if not isinstance(data, dict):
        raise ValueError(f"serving configuration must be a mapping: {path}")
    settings = ServingSettings.model_validate(expand_environment_values(data))
    return settings.model_copy(update={"config_sha256": hashlib.sha256(raw_bytes).hexdigest()})
