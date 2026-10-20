"""Upload probing and decoding, strictly in memory.

The service receives ``bytes`` and never a path: nothing a client sends is used as
a filename. Header facts (format, channels, rate, frame count) are checked with
``soundfile.info`` before any sample is decoded, so a small file that declares a
huge signal is rejected before it can expand in memory. Decoding then goes through
the same ``load_waveform`` the evaluation uses.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import soundfile as sf

from voice_detector.preprocessing import AudioConfig, load_waveform
from voice_detector.serving.schema import ErrorCode
from voice_detector.serving.settings import InputSettings


class InputRejected(Exception):
    """A request the service refuses, with a fixed, content-free message."""

    def __init__(self, code: ErrorCode, status: int, message: str) -> None:
        super().__init__(code)
        self.code = code
        self.status = status
        self.message = message


@dataclass(frozen=True)
class AudioProbe:
    format: str
    channels: int
    sample_rate_hz: int
    frames: int
    duration_seconds: float


def probe_upload(data: bytes, limits: InputSettings) -> AudioProbe:
    """Validate the container header against the configured limits."""

    try:
        info = sf.info(io.BytesIO(data))
    except Exception as error:  # noqa: BLE001 - any parse failure is an undecodable upload
        raise InputRejected(
            "undecodable_audio", 400, "The upload could not be read as audio."
        ) from error

    if info.format not in limits.accepted_formats:
        raise InputRejected(
            "unsupported_format",
            415,
            f"Unsupported audio format. Accepted: {', '.join(limits.accepted_formats)}.",
        )
    if info.channels < 1 or info.channels > limits.max_channels:
        raise InputRejected(
            "unsupported_audio_parameters", 422, "Unsupported number of audio channels."
        )
    if not limits.min_source_sample_rate_hz <= info.samplerate <= limits.max_source_sample_rate_hz:
        raise InputRejected(
            "unsupported_audio_parameters", 422, "Unsupported audio sample rate."
        )
    if info.frames <= 0:
        raise InputRejected("empty_audio", 422, "The audio contains no samples.")

    duration = info.frames / info.samplerate
    if not limits.min_duration_seconds <= duration <= limits.max_duration_seconds:
        raise InputRejected(
            "duration_out_of_range",
            422,
            "Audio duration is outside the accepted range "
            f"[{limits.min_duration_seconds:g}, {limits.max_duration_seconds:g}] seconds.",
        )
    return AudioProbe(
        format=info.format,
        channels=int(info.channels),
        sample_rate_hz=int(info.samplerate),
        frames=int(info.frames),
        duration_seconds=float(duration),
    )


def decode_upload(
    data: bytes, probe: AudioProbe, target_rate_hz: int, limits: InputSettings
) -> np.ndarray:
    """Decode with the shared evaluation decoder and re-check the decoded signal."""

    try:
        waveform = load_waveform(io.BytesIO(data), AudioConfig(sample_rate_hz=target_rate_hz))
    except Exception as error:  # noqa: BLE001 - decoder errors may echo content; never forward
        raise InputRejected(
            "undecodable_audio", 400, "The upload could not be read as audio."
        ) from error

    if waveform.size == 0:
        raise InputRejected("empty_audio", 422, "The audio contains no samples.")
    # Header and decoded lengths must agree up to resampler rounding.
    if waveform.size > int(limits.max_duration_seconds * target_rate_hz) + target_rate_hz:
        raise InputRejected(
            "duration_out_of_range", 422, "Decoded audio is longer than the accepted range."
        )
    if not np.all(np.isfinite(waveform)):
        raise InputRejected(
            "unscorable_audio", 422, "The audio contains non-finite samples."
        )
    return waveform
