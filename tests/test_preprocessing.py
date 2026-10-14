"""Fast, dataset-free tests for the shared preprocessing path.

All audio here is synthesised in-memory or written to ``tmp_path``; none of it
touches the ASVspoof dataset, so these tests stay quick and runnable before any
download exists.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voice_detector.preprocessing import (
    PREPROCESSING_VERSION,
    AudioConfig,
    MfccConfig,
    PreprocessingConfig,
    clip_feature_matrix,
    clip_features_from_source,
    load_waveform,
    mfcc_feature_vector,
    window_spans,
)


def _sine(seconds: float, sample_rate: int, freq: float = 220.0) -> np.ndarray:
    t = np.linspace(0.0, seconds, int(seconds * sample_rate), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# --- config -----------------------------------------------------------------


def test_window_samples_follows_rate_and_duration() -> None:
    assert AudioConfig(sample_rate_hz=16000, window_seconds=4.0).window_samples == 64000


def test_feature_dim_tracks_delta_flag() -> None:
    assert MfccConfig(n_mfcc=20, include_delta=True).feature_dim == 80
    assert MfccConfig(n_mfcc=20, include_delta=False).feature_dim == 40


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"sample_rate_hz": 0}, "sample_rate_hz"),
        ({"window_seconds": 0}, "window_seconds"),
    ],
)
def test_audio_config_rejects_bad_values(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        AudioConfig(**kwargs)


def test_mfcc_config_rejects_fmax_below_fmin() -> None:
    with pytest.raises(ValueError, match="fmax must be greater than fmin"):
        MfccConfig(fmin=8000.0, fmax=20.0)


def test_from_mapping_reads_values_and_falls_back() -> None:
    config = PreprocessingConfig.from_mapping(
        {
            "audio": {"sample_rate_hz": 8000, "window_seconds": 2.0},
            "preprocessing": {"version": PREPROCESSING_VERSION, "mfcc": {"n_mfcc": 13}},
        }
    )
    assert config.audio.sample_rate_hz == 8000
    assert config.audio.window_seconds == 2.0
    assert config.mfcc.n_mfcc == 13
    assert config.mfcc.include_delta is True  # fell back to default
    assert config.version == PREPROCESSING_VERSION


def test_from_mapping_uses_defaults_when_sections_absent() -> None:
    config = PreprocessingConfig.from_mapping({})
    assert config.audio == AudioConfig()
    assert config.mfcc == MfccConfig()
    assert config.version == PREPROCESSING_VERSION


def test_from_mapping_rejects_version_mismatch() -> None:
    # A config that pins a different epoch than this code produces must fail
    # loudly: its features would not match the code's.
    with pytest.raises(ValueError, match="preprocessing.version"):
        PreprocessingConfig.from_mapping({"preprocessing": {"version": "mfcc-v1"}})


# --- windowing (option D: real-sample spans, sub-frame tails folded) --------

# window=1000 samples, n_fft=400: large enough to exercise every fold branch
# with small integers and no audio.
_W = 1000
_NFFT = 400


def _spans(n: int) -> tuple[tuple[int, int], ...]:
    return window_spans(n, window_samples=_W, n_fft=_NFFT)


def test_clip_no_longer_than_one_window_is_a_single_span() -> None:
    assert _spans(_W) == ((0, _W),)  # exactly one window
    assert _spans(250) == ((0, 250),)  # shorter than one window: real audio only
    assert _spans(_NFFT) == ((0, _NFFT),)  # exactly one analysis frame


def test_exact_multiple_produces_whole_windows() -> None:
    assert _spans(2 * _W) == ((0, _W), (_W, 2 * _W))


def test_remainder_at_least_a_frame_becomes_its_own_window() -> None:
    # 2W + r with r >= n_fft -> three spans, the last one the remainder.
    r = _NFFT + 50
    assert _spans(2 * _W + r) == ((0, _W), (_W, 2 * _W), (2 * _W, 2 * _W + r))


def test_subframe_tail_is_folded_into_the_previous_window() -> None:
    # 2W + r with 0 < r < n_fft -> two spans; the last absorbs the remainder.
    r = _NFFT - 1
    assert _spans(2 * _W + r) == ((0, _W), (_W, 2 * _W + r))
    # W + r folded -> a single span over the whole clip.
    assert _spans(_W + r) == ((0, _W + r),)


def test_pathological_sub_frame_clip_is_one_span() -> None:
    # Shorter than a single analysis frame: still exactly one span, no crash.
    assert _spans(_NFFT - 1) == ((0, _NFFT - 1),)


def test_empty_waveform_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty waveform"):
        window_spans(0, window_samples=_W, n_fft=_NFFT)


# --- features ---------------------------------------------------------------


_has_librosa = __import__("importlib").util.find_spec("librosa") is not None
requires_librosa = pytest.mark.skipif(not _has_librosa, reason="librosa not installed")


@requires_librosa
def test_feature_vector_has_fixed_length_and_is_finite() -> None:
    audio = AudioConfig(sample_rate_hz=16000, window_seconds=1.0)
    mfcc = MfccConfig(n_mfcc=20, include_delta=True)
    vector = mfcc_feature_vector(_sine(1.0, 16000), audio, mfcc)
    assert vector.shape == (mfcc.feature_dim,)
    assert vector.dtype == np.float32
    assert np.all(np.isfinite(vector))


@requires_librosa
def test_feature_vector_length_shrinks_without_deltas() -> None:
    audio = AudioConfig(sample_rate_hz=16000, window_seconds=1.0)
    mfcc = MfccConfig(n_mfcc=13, include_delta=False)
    vector = mfcc_feature_vector(_sine(1.0, 16000), audio, mfcc)
    assert vector.shape == (26,)


@requires_librosa
def test_clip_feature_matrix_is_one_row_per_window() -> None:
    config = PreprocessingConfig(
        audio=AudioConfig(sample_rate_hz=16000, window_seconds=1.0),
        mfcc=MfccConfig(n_mfcc=20, include_delta=True),
    )
    # 40000 samples over 16000-sample windows: two full windows plus an 8000
    # sample remainder (>= n_fft), which becomes its own third window.
    matrix = clip_feature_matrix(_sine(2.5, 16000), config)
    assert matrix.shape == (3, config.mfcc.feature_dim)


@requires_librosa
def test_short_clip_pools_over_real_samples_only() -> None:
    # A clip shorter than one window yields a single window whose features are
    # exactly those of the raw real audio: no zero-padding enters the pooled
    # statistics (this is the point of option D). Also guards the sub-1s clips
    # the manifest found (issue #14): extraction must process them, not crash.
    config = PreprocessingConfig(
        audio=AudioConfig(sample_rate_hz=16000, window_seconds=4.0),
        mfcc=MfccConfig(n_mfcc=20, include_delta=True),
    )
    waveform = _sine(0.1, 16000)  # 1600 samples, far below one 64000-sample window
    matrix = clip_feature_matrix(waveform, config)
    assert matrix.shape == (1, config.mfcc.feature_dim)
    assert np.all(np.isfinite(matrix))
    expected = mfcc_feature_vector(waveform, config.audio, config.mfcc)
    assert np.array_equal(matrix[0], expected)


@requires_librosa
def test_features_are_deterministic() -> None:
    config = PreprocessingConfig(
        audio=AudioConfig(sample_rate_hz=16000, window_seconds=1.0),
        mfcc=MfccConfig(n_mfcc=20),
    )
    waveform = _sine(2.0, 16000)
    first = clip_feature_matrix(waveform, config)
    second = clip_feature_matrix(waveform, config)
    assert np.array_equal(first, second)


# --- decoding ---------------------------------------------------------------


def test_load_waveform_averages_channels_to_mono(tmp_path: Path) -> None:
    path = tmp_path / "stereo.wav"
    left = _sine(0.5, 16000, freq=200.0)
    right = _sine(0.5, 16000, freq=400.0)
    sf.write(path, np.stack([left, right], axis=1), 16000, subtype="FLOAT")

    waveform = load_waveform(path, AudioConfig(sample_rate_hz=16000, mono=True))

    assert waveform.ndim == 1
    assert waveform.dtype == np.float32
    np.testing.assert_allclose(waveform, (left + right) / 2.0, atol=1e-6)


@requires_librosa
def test_load_waveform_resamples_to_target_rate(tmp_path: Path) -> None:
    path = tmp_path / "eight_k.wav"
    sf.write(path, _sine(1.0, 8000), 8000, subtype="FLOAT")

    waveform = load_waveform(path, AudioConfig(sample_rate_hz=16000))

    assert abs(waveform.size - 16000) <= 8  # ~doubled length, modulo resampler edges


@requires_librosa
def test_clip_features_from_flac_source(tmp_path: Path) -> None:
    # FLAC mirrors the real ASVspoof distribution format.
    path = tmp_path / "clip.flac"
    sf.write(path, _sine(2.5, 16000), 16000)

    config = PreprocessingConfig(
        audio=AudioConfig(sample_rate_hz=16000, window_seconds=1.0),
        mfcc=MfccConfig(n_mfcc=20, include_delta=True),
    )
    matrix = clip_features_from_source(path, config)

    assert matrix.shape == (3, config.mfcc.feature_dim)
    assert np.all(np.isfinite(matrix))
