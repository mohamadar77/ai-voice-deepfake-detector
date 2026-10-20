"""Transport-independent scoring service.

``ScoringService.score_bytes`` is the one code path from uploaded bytes to a
response: probe the header, decode with the shared evaluation decoder, apply the
advisory checks, score, and guard the output. The FastAPI app, the warmup and the
host parity test all call it, so they cannot drift apart.
"""

from __future__ import annotations

import io
import math
import time
from dataclasses import dataclass

import numpy as np
import soundfile as sf

from voice_detector.runtime import effective_torch_settings, runtime_versions
from voice_detector.scoring import DECISION_RULE, SCORE_DIRECTION, ServableScorer
from voice_detector.serving.decode import InputRejected, decode_upload, probe_upload
from voice_detector.serving.quality import collect_warnings, reject_if_silent, summarize
from voice_detector.serving.schema import (
    Decision,
    ModelInfo,
    ScoreResponse,
)
from voice_detector.serving.settings import ServingSettings

_DECISIONS: dict[str, Decision] = {"spoof": "flagged_spoof_like", "bonafide": "not_flagged"}


def decision_from_label(label: str) -> Decision:
    """Map the scorer's internal label to the neutral public decision."""

    try:
        return _DECISIONS[label]
    except KeyError as error:  # a new label must be mapped deliberately
        raise ValueError(f"unmapped scorer label {label!r}") from error


@dataclass(frozen=True)
class ScoreOutcome:
    response: ScoreResponse
    decode_ms: float
    inference_ms: float


class ScoringService:
    def __init__(self, scorer: ServableScorer, settings: ServingSettings) -> None:
        if not scorer.bundle_hash:
            raise ValueError("serving requires a hash-verified model bundle")
        self._scorer = scorer
        self._settings = settings
        self._model_id = str(scorer.describe().get("candidate_id") or settings.model.candidate_id)

    @property
    def scorer(self) -> ServableScorer:
        return self._scorer

    def score_bytes(self, data: bytes, *, request_id: str) -> ScoreOutcome:
        """Blocking: run in a worker thread. Raises ``InputRejected`` for bad input."""

        limits = self._settings.input
        started = time.perf_counter()
        probe = probe_upload(data, limits)
        waveform = decode_upload(data, probe, self._scorer.sample_rate_hz, limits)
        summary = summarize(waveform, self._settings.hints.clipping_level)
        reject_if_silent(summary, limits)
        decoded = time.perf_counter()

        scored = self._scorer.score_decoded(waveform)
        finished = time.perf_counter()

        clip = scored.clip
        if not math.isfinite(clip.score):
            raise InputRejected("unscorable_audio", 422, "The audio could not be scored.")

        warnings = collect_warnings(
            summary,
            probe.duration_seconds,
            scored.window_real_samples,
            self._scorer.sample_rate_hz,
            self._settings.hints,
        )
        response = ScoreResponse(
            request_id=request_id,
            score=clip.score,
            score_direction=clip.score_direction,
            threshold=clip.threshold,
            decision=decision_from_label(clip.label),
            decision_rule=DECISION_RULE,
            n_windows=clip.n_windows,
            duration_seconds=round(probe.duration_seconds, 3),
            source_sample_rate_hz=probe.sample_rate_hz,
            model_id=self._model_id,
            model_version=self._scorer.model_version,
            preprocessing_version=self._scorer.preprocessing_version,
            aggregation_rule=self._scorer.aggregation_rule,
            bundle_hash=str(self._scorer.bundle_hash),
            processing_ms=round((finished - started) * 1000.0, 3),
            warnings=warnings,
            notice=self._settings.framing.notice,
        )
        return ScoreOutcome(
            response=response,
            decode_ms=(decoded - started) * 1000.0,
            inference_ms=(finished - decoded) * 1000.0,
        )

    def warmup(self) -> list[float]:
        """Score seeded synthetic clips through the whole path; returns milliseconds each.

        The clips are stereo at ``resample_from_hz``, so the resampler (and its lazy
        librosa import) runs before the service reports ready.
        """

        cfg = self._settings.runtime.warmup
        timings: list[float] = []
        for index, seconds in enumerate(cfg.seconds):
            clip = _synthetic_wav(seconds, cfg.resample_from_hz, cfg.seed + index)
            begun = time.perf_counter()
            self.score_bytes(clip, request_id="warmup")
            timings.append((time.perf_counter() - begun) * 1000.0)
        return timings

    def model_info(self) -> ModelInfo:
        s = self._settings
        described = self._scorer.describe()
        return ModelInfo(
            model_id=self._model_id,
            model_version=self._scorer.model_version,
            model_family=self._scorer.model_family,
            preprocessing_version=self._scorer.preprocessing_version,
            aggregation_rule=self._scorer.aggregation_rule,
            score_direction=SCORE_DIRECTION,
            decision_rule=DECISION_RULE,
            threshold=self._scorer.threshold,
            threshold_basis=dict(described.get("threshold_basis", {})),
            bundle_hash=str(self._scorer.bundle_hash),
            config_sha256=s.config_sha256,
            hints_version=s.hints.version,
            hints_provisional=s.hints.provisional,
            limits={
                "max_upload_bytes": s.input.max_upload_bytes,
                "accepted_formats": list(s.input.accepted_formats),
                "min_duration_seconds": s.input.min_duration_seconds,
                "max_duration_seconds": s.input.max_duration_seconds,
                "max_channels": s.input.max_channels,
                "source_sample_rate_hz": [
                    s.input.min_source_sample_rate_hz,
                    s.input.max_source_sample_rate_hz,
                ],
            },
            runtime={
                "configured_torch_threads": s.runtime.torch_threads,
                "max_concurrent_scoring": s.runtime.max_concurrent_scoring,
                "versions": runtime_versions(),
                **effective_torch_settings(),
            },
            notice=s.framing.notice,
            limitations=list(s.framing.limitations),
            evidence_refs=list(s.framing.evidence_refs),
        )


def _synthetic_wav(seconds: float, sample_rate_hz: int, seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    frames = int(round(seconds * sample_rate_hz))
    samples = (0.1 * rng.standard_normal((frames, 2))).astype(np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, samples, sample_rate_hz, format="WAV", subtype="PCM_16")
    return buffer.getvalue()
