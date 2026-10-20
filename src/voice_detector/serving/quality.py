"""Cheap, advisory input checks (numpy only).

They never change a score or a decision. Digital silence is the only hard reject.
Compression and background noise cannot be detected reliably from 16 kHz audio
without a validated detector, so no hint claims either; the static notice carries
that caveat. Thresholds are provisional until derived from development statistics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from voice_detector.serving.decode import InputRejected
from voice_detector.serving.schema import ScoreWarning
from voice_detector.serving.settings import HintSettings, InputSettings

_EPS = 1e-12


@dataclass(frozen=True)
class SignalSummary:
    peak_dbfs: float
    rms_dbfs: float
    clipped_fraction: float


def summarize(waveform: np.ndarray, clipping_level: float) -> SignalSummary:
    samples = np.asarray(waveform, dtype=np.float64)
    peak = float(np.max(np.abs(samples)))
    rms = float(np.sqrt(np.mean(samples**2)))
    return SignalSummary(
        peak_dbfs=20.0 * math.log10(max(peak, _EPS)),
        rms_dbfs=20.0 * math.log10(max(rms, _EPS)),
        clipped_fraction=float(np.mean(np.abs(samples) >= clipping_level)),
    )


def reject_if_silent(summary: SignalSummary, limits: InputSettings) -> None:
    if summary.peak_dbfs < limits.silence_peak_dbfs:
        raise InputRejected("silent_audio", 422, "The audio is silent or nearly silent.")


def collect_warnings(
    summary: SignalSummary,
    duration_seconds: float,
    window_real_samples: tuple[int, ...],
    sample_rate_hz: int,
    hints: HintSettings,
) -> list[ScoreWarning]:
    warnings: list[ScoreWarning] = []
    if duration_seconds < hints.warn_below_seconds:
        warnings.append(
            ScoreWarning(code="short_clip", message="Very short clips give less reliable scores.")
        )
    if duration_seconds > hints.warn_above_seconds:
        warnings.append(
            ScoreWarning(
                code="long_clip",
                message="Scores average 4-second windows; partial synthesis can be diluted.",
            )
        )
    if summary.clipped_fraction >= hints.clipping_fraction:
        warnings.append(
            ScoreWarning(code="clipping", message="The recording is clipped (distorted peaks).")
        )
    if summary.rms_dbfs < hints.low_rms_dbfs:
        warnings.append(
            ScoreWarning(code="low_level", message="The recording level is very low.")
        )
    short_window = int(round(hints.short_window_seconds * sample_rate_hz))
    if len(window_real_samples) > 1 and min(window_real_samples) < short_window:
        warnings.append(
            ScoreWarning(
                code="short_tail_window",
                message="One analysis window holds under 1 s of real audio and is repeat-padded.",
            )
        )
    return warnings
