"""Shared, deterministic audio preprocessing and MFCC feature extraction.

This is the single decode -> window -> feature path used by training, locked
evaluation and serving. Keeping one implementation guarantees that a bundle
scored in the API sees exactly the features it was fitted on. Everything here
is deterministic: there is no randomness, so the same waveform always yields
the same features. Random cropping and augmentation are separate, explicitly
seeded training operations and deliberately live elsewhere.

The feature vector for one window is the per-coefficient mean and standard
deviation of its MFCCs (and, when enabled, their first-order deltas) pooled
over time. That gives a fixed-length vector regardless of window content, which
is what the Logistic Regression baseline consumes. A clip becomes a matrix of
one such vector per fixed-length window; window scores are aggregated to one
score per clip downstream, never treated as independent clips.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import librosa
import numpy as np
import soundfile as sf

AudioSource = str | Path | BinaryIO


@dataclass(frozen=True)
class AudioConfig:
    """Decoding and windowing settings shared across the project."""

    sample_rate_hz: int = 16000
    mono: bool = True
    window_seconds: float = 4.0

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be a positive integer")
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be positive")

    @property
    def window_samples(self) -> int:
        """Number of samples in one fixed-length analysis window."""

        return int(round(self.sample_rate_hz * self.window_seconds))


@dataclass(frozen=True)
class MfccConfig:
    """MFCC feature settings. Defaults are tuned for 16 kHz speech."""

    n_mfcc: int = 20
    n_fft: int = 400
    hop_length: int = 160
    fmin: float = 20.0
    fmax: float = 8000.0
    include_delta: bool = True

    def __post_init__(self) -> None:
        for name in ("n_mfcc", "n_fft", "hop_length"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.fmin < 0:
            raise ValueError("fmin must be non-negative")
        if self.fmax <= self.fmin:
            raise ValueError("fmax must be greater than fmin")

    @property
    def feature_dim(self) -> int:
        """Length of the pooled feature vector produced per window."""

        stacks = 2 if self.include_delta else 1
        return 2 * stacks * self.n_mfcc  # mean and std of (MFCC [+ delta])


@dataclass(frozen=True)
class PreprocessingConfig:
    """The full shared preprocessing specification plus its version tag."""

    audio: AudioConfig = AudioConfig()
    mfcc: MfccConfig = MfccConfig()
    version: str = "mfcc-v1"

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> PreprocessingConfig:
        """Build from a project configuration mapping (``config.raw``).

        Reads the top-level ``audio`` section and the ``preprocessing`` section;
        both are optional and fall back to the documented defaults.
        """

        audio_raw = dict(raw.get("audio") or {})
        audio = AudioConfig(
            sample_rate_hz=int(audio_raw.get("sample_rate_hz", AudioConfig.sample_rate_hz)),
            mono=bool(audio_raw.get("mono", AudioConfig.mono)),
            window_seconds=float(audio_raw.get("window_seconds", AudioConfig.window_seconds)),
        )

        pre_raw = dict(raw.get("preprocessing") or {})
        mfcc_raw = dict(pre_raw.get("mfcc") or {})
        mfcc = MfccConfig(
            n_mfcc=int(mfcc_raw.get("n_mfcc", MfccConfig.n_mfcc)),
            n_fft=int(mfcc_raw.get("n_fft", MfccConfig.n_fft)),
            hop_length=int(mfcc_raw.get("hop_length", MfccConfig.hop_length)),
            fmin=float(mfcc_raw.get("fmin", MfccConfig.fmin)),
            fmax=float(mfcc_raw.get("fmax", MfccConfig.fmax)),
            include_delta=bool(mfcc_raw.get("include_delta", MfccConfig.include_delta)),
        )
        version = str(pre_raw.get("version", cls.version))
        return cls(audio=audio, mfcc=mfcc, version=version)


def load_waveform(source: AudioSource, audio: AudioConfig) -> np.ndarray:
    """Decode audio to a 1-D ``float32`` waveform at the configured rate.

    Accepts a path or an open binary file (so the API can decode an upload
    without writing it to disk). Multi-channel audio is averaged to mono when
    ``audio.mono`` is set, and the signal is resampled to
    ``audio.sample_rate_hz`` when the source rate differs.
    """

    data, source_rate = sf.read(source, dtype="float32", always_2d=False)
    waveform = np.asarray(data, dtype=np.float32)

    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1) if audio.mono else waveform[:, 0]
    elif waveform.ndim != 1:
        raise ValueError(f"Unexpected decoded audio shape: {waveform.shape}")

    if source_rate != audio.sample_rate_hz:
        waveform = librosa.resample(waveform, orig_sr=source_rate, target_sr=audio.sample_rate_hz)

    return np.ascontiguousarray(waveform, dtype=np.float32)


def window_waveform(waveform: np.ndarray, audio: AudioConfig) -> np.ndarray:
    """Split a waveform into non-overlapping fixed-length windows.

    Returns an array of shape ``(n_windows, window_samples)``. The trailing
    partial window is zero-padded, and a clip shorter than one window becomes a
    single zero-padded window, so every clip yields at least one window.
    """

    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if waveform.size == 0:
        raise ValueError("Cannot window an empty waveform")

    window_samples = audio.window_samples
    n_windows = max(1, math.ceil(waveform.size / window_samples))
    padded_length = n_windows * window_samples
    if waveform.size < padded_length:
        waveform = np.pad(waveform, (0, padded_length - waveform.size))
    return waveform.reshape(n_windows, window_samples)


def mfcc_feature_vector(window: np.ndarray, audio: AudioConfig, mfcc: MfccConfig) -> np.ndarray:
    """Pool one window's MFCCs into a fixed-length ``float32`` feature vector.

    The vector is ``[mean(coeffs), std(coeffs)]`` pooled over time, where
    ``coeffs`` is the MFCCs stacked with their first-order deltas when
    ``mfcc.include_delta`` is set. Length is always ``mfcc.feature_dim``.
    """

    window = np.asarray(window, dtype=np.float32).reshape(-1)
    coeffs = librosa.feature.mfcc(
        y=window,
        sr=audio.sample_rate_hz,
        n_mfcc=mfcc.n_mfcc,
        n_fft=mfcc.n_fft,
        hop_length=mfcc.hop_length,
        fmin=mfcc.fmin,
        fmax=mfcc.fmax,
    )
    if mfcc.include_delta:
        coeffs = np.vstack([coeffs, librosa.feature.delta(coeffs)])

    pooled = np.concatenate([coeffs.mean(axis=1), coeffs.std(axis=1)])
    return pooled.astype(np.float32, copy=False)


def clip_feature_matrix(waveform: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    """Produce one pooled feature vector per window for a whole clip.

    Returns an array of shape ``(n_windows, feature_dim)``. Downstream code
    aggregates these window scores into a single clip-level score; the windows
    are never treated as independent clips.
    """

    windows = window_waveform(waveform, config.audio)
    features = [mfcc_feature_vector(window, config.audio, config.mfcc) for window in windows]
    return np.stack(features).astype(np.float32, copy=False)


def clip_features_from_source(source: AudioSource, config: PreprocessingConfig) -> np.ndarray:
    """Decode an audio source and return its per-window feature matrix."""

    waveform = load_waveform(source, config.audio)
    return clip_feature_matrix(waveform, config)
