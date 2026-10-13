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
is what the Logistic Regression baseline consumes. A clip is split into windows
that span real (non-padded) samples only (see ``window_spans`` for the frozen
policy); each window yields one such vector, and window scores are aggregated to
one score per clip downstream, never treated as independent clips.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import librosa
import numpy as np
import soundfile as sf

AudioSource = str | Path | BinaryIO

# Preprocessing epoch. Bump this whenever the decode -> window -> feature path
# changes feature values in a way a parameter hash would not capture. The code
# constant is authoritative; configs/default.yaml declares the same string and
# is asserted to match in PreprocessingConfig.from_mapping, so the two can never
# silently diverge. mfcc-v1 -> mfcc-v2: pooled MFCC statistics are now computed
# over real samples only (option D), so feature values change under this bump.
PREPROCESSING_VERSION = "mfcc-v2"


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
    version: str = PREPROCESSING_VERSION

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
        version = str(pre_raw.get("version", PREPROCESSING_VERSION))
        if version != PREPROCESSING_VERSION:
            raise ValueError(
                f"configs pin preprocessing.version {version!r} but this code produces "
                f"{PREPROCESSING_VERSION!r}; feature values would not match. Align the "
                f"config or check out the matching code revision."
            )
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


def window_spans(
    n_samples: int, *, window_samples: int, n_fft: int
) -> tuple[tuple[int, int], ...]:
    """Half-open ``(start, stop)`` spans of real samples for one clip.

    This is the single source of truth for the frozen windowing policy
    (option D): windows cover real, non-padded samples only, and any trailing
    remainder shorter than one analysis frame (``n_fft`` samples) is folded into
    the preceding window instead of being zero-padded or dropped. A clip no
    longer than one window becomes a single window over its real audio. Every
    branch yields at least one span, so every clip produces at least one feature
    vector. The spans are pure integer arithmetic with no audio dependency,
    which makes the policy exhaustively unit-testable.

    Rules for ``n_samples`` greater than ``window_samples`` (remainder
    ``r = n_samples % window_samples``):

    - ``r == 0``: exact full windows.
    - ``0 < r < n_fft``: the final window is extended to absorb the remainder
      (one fewer window than a naive split).
    - ``r >= n_fft``: the remainder forms its own final window.
    """

    if n_samples <= 0:
        raise ValueError("Cannot window an empty waveform")
    if window_samples <= 0:
        raise ValueError("window_samples must be a positive integer")
    if n_fft <= 0:
        raise ValueError("n_fft must be a positive integer")

    if n_samples <= window_samples:
        return ((0, n_samples),)

    full = n_samples // window_samples
    remainder = n_samples % window_samples
    spans = [(i * window_samples, (i + 1) * window_samples) for i in range(full)]

    if remainder == 0:
        return tuple(spans)
    if remainder < n_fft:
        start, _ = spans[-1]
        spans[-1] = (start, n_samples)  # fold the sub-frame tail into the last window
    else:
        spans.append((full * window_samples, n_samples))
    return tuple(spans)


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

    Returns an array of shape ``(n_windows, feature_dim)``. Windows span real
    samples only, per ``window_spans``; no window is zero-padded, so pooled
    statistics never see padding. Downstream code aggregates these window scores
    into a single clip-level score; the windows are never treated as independent
    clips.
    """

    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    spans = window_spans(
        waveform.size,
        window_samples=config.audio.window_samples,
        n_fft=config.mfcc.n_fft,
    )
    features = [
        mfcc_feature_vector(waveform[start:stop], config.audio, config.mfcc)
        for start, stop in spans
    ]
    return np.stack(features).astype(np.float32, copy=False)


def clip_features_from_source(source: AudioSource, config: PreprocessingConfig) -> np.ndarray:
    """Decode an audio source and return its per-window feature matrix."""

    waveform = load_waveform(source, config.audio)
    return clip_feature_matrix(waveform, config)
