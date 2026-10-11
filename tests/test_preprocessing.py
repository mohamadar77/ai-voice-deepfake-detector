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
    AudioConfig,
    MfccConfig,
    PreprocessingConfig,
    clip_feature_matrix,
    clip_features_from_source,
    load_waveform,
    mfcc_feature_vector,
    window_waveform,
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
            "preprocessing": {"version": "mfcc-test", "mfcc": {"n_mfcc": 13}},
        }
    )
    assert config.audio.sample_rate_hz == 8000
    assert config.audio.window_seconds == 2.0
    assert config.mfcc.n_mfcc == 13
    assert config.mfcc.include_delta is True  # fell back to default
    assert config.version == "mfcc-test"


def test_from_mapping_uses_defaults_when_sections_absent() -> None:
    config = PreprocessingConfig.from_mapping({})
    assert config.audio == AudioConfig()
    assert config.mfcc == MfccConfig()
    assert config.version == "mfcc-v1"


# --- windowing --------------------------------------------------------------


def test_short_clip_becomes_one_padded_window() -> None:
    audio = AudioConfig(sample_rate_hz=16000, window_seconds=1.0)  # 16000 samples
    windows = window_waveform(_sine(0.25, 16000), audio)  # 4000 samples
    assert windows.shape == (1, 16000)
    assert np.all(windows[0, 4000:] == 0.0)  # tail is zero-padded


def test_exact_multiple_produces_whole_windows_without_padding() -> None:
    audio = AudioConfig(sample_rate_hz=16000, window_seconds=1.0)
    windows = window_waveform(_sine(3.0, 16000), audio)
    assert windows.shape == (3, 16000)
    assert np.any(windows[-1] != 0.0)  # last window is real audio, not padding


def test_partial_final_window_is_padded() -> None:
    audio = AudioConfig(sample_rate_hz=16000, window_seconds=1.0)
    windows = window_waveform(_sine(2.5, 16000), audio)  # 2.5 windows
    assert windows.shape == (3, 16000)
    assert np.all(windows[2, 8000:] == 0.0)


def test_empty_waveform_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty waveform"):
        window_waveform(np.array([], dtype=np.float32), AudioConfig())


# --- features ---------------------------------------------------------------


def test_feature_vector_has_fixed_length_and_is_finite() -> None:
    audio = AudioConfig(sample_rate_hz=16000, window_seconds=1.0)
    mfcc = MfccConfig(n_mfcc=20, include_delta=True)
    vector = mfcc_feature_vector(_sine(1.0, 16000), audio, mfcc)
    assert vector.shape == (mfcc.feature_dim,)
    assert vector.dtype == np.float32
    assert np.all(np.isfinite(vector))


def test_feature_vector_length_shrinks_without_deltas() -> None:
    audio = AudioConfig(sample_rate_hz=16000, window_seconds=1.0)
    mfcc = MfccConfig(n_mfcc=13, include_delta=False)
    vector = mfcc_feature_vector(_sine(1.0, 16000), audio, mfcc)
    assert vector.shape == (26,)


def test_clip_feature_matrix_is_one_row_per_window() -> None:
    config = PreprocessingConfig(
        audio=AudioConfig(sample_rate_hz=16000, window_seconds=1.0),
        mfcc=MfccConfig(n_mfcc=20, include_delta=True),
    )
    matrix = clip_feature_matrix(_sine(2.5, 16000), config)
    assert matrix.shape == (3, config.mfcc.feature_dim)


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


def test_load_waveform_resamples_to_target_rate(tmp_path: Path) -> None:
    path = tmp_path / "eight_k.wav"
    sf.write(path, _sine(1.0, 8000), 8000, subtype="FLOAT")

    waveform = load_waveform(path, AudioConfig(sample_rate_hz=16000))

    assert abs(waveform.size - 16000) <= 8  # ~doubled length, modulo resampler edges


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
