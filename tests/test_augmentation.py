"""Tests for robustness transformations (#34).

All tests are dataset-free: they use synthetic waveforms and don't require
ffmpeg, ASVspoof audio, or GPU. Codec tests are marked with ``requires_ffmpeg``
and skipped when ffmpeg is unavailable.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from voice_detector.augmentation import (
    AUGMENTATION_VERSION,
    CodecResult,
    NoiseResult,
    TrainingAugmentor,
    TransformationError,
    TrimResult,
    check_ffmpeg,
    codec_transform,
    derive_clip_seed,
    noise_transform,
    silence_trim,
)

requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="ffmpeg not available",
)


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

def test_augmentation_version() -> None:
    assert AUGMENTATION_VERSION == "robustness-v1"


# ---------------------------------------------------------------------------
# Seed derivation
# ---------------------------------------------------------------------------

def test_derive_clip_seed_deterministic() -> None:
    s1 = derive_clip_seed("LA_E_1234567", 20261004)
    s2 = derive_clip_seed("LA_E_1234567", 20261004)
    assert s1 == s2


def test_derive_clip_seed_differs_by_clip() -> None:
    s1 = derive_clip_seed("clip_a", 20261004)
    s2 = derive_clip_seed("clip_b", 20261004)
    assert s1 != s2


def test_derive_clip_seed_differs_by_global_seed() -> None:
    s1 = derive_clip_seed("clip_a", 20261004)
    s2 = derive_clip_seed("clip_a", 99999)
    assert s1 != s2


def test_derive_clip_seed_is_positive_int() -> None:
    s = derive_clip_seed("test", 42)
    assert isinstance(s, int)
    assert s >= 0


# ---------------------------------------------------------------------------
# Noise transform
# ---------------------------------------------------------------------------

def _sine_wave(duration_s: float = 1.0, freq: float = 440.0, sr: int = 16000) -> np.ndarray:
    t = np.arange(int(sr * duration_s)) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_noise_determinism() -> None:
    wav = _sine_wave()
    r1 = noise_transform(wav, snr_db=20, clip_id="c1", global_seed=42)
    r2 = noise_transform(wav, snr_db=20, clip_id="c1", global_seed=42)
    np.testing.assert_array_equal(r1.waveform, r2.waveform)


def test_noise_different_clips_differ() -> None:
    wav = _sine_wave()
    r1 = noise_transform(wav, snr_db=20, clip_id="c1", global_seed=42)
    r2 = noise_transform(wav, snr_db=20, clip_id="c2", global_seed=42)
    assert not np.array_equal(r1.waveform, r2.waveform)


def test_noise_epoch_changes_output() -> None:
    wav = _sine_wave()
    r1 = noise_transform(wav, snr_db=20, clip_id="c1", global_seed=42, epoch=0)
    r2 = noise_transform(wav, snr_db=20, clip_id="c1", global_seed=42, epoch=1)
    assert not np.array_equal(r1.waveform, r2.waveform)


def test_noise_snr_approximately_correct() -> None:
    wav = _sine_wave(duration_s=5.0)
    result = noise_transform(wav, snr_db=20, clip_id="test", global_seed=42)
    noise = result.waveform - wav
    signal_power = np.mean(wav.astype(np.float64) ** 2)
    noise_power = np.mean(noise.astype(np.float64) ** 2)
    measured_snr = 10 * np.log10(signal_power / noise_power)
    assert abs(measured_snr - 20) < 1.0, f"Expected ~20 dB, got {measured_snr:.1f}"


def test_noise_result_type() -> None:
    wav = _sine_wave()
    result = noise_transform(wav, snr_db=10, clip_id="c1", global_seed=42)
    assert isinstance(result, NoiseResult)
    assert result.waveform.shape == wav.shape
    assert result.snr_db == 10
    assert result.reference_rms > 0
    assert result.noise_rms > 0


def test_noise_clips_to_range() -> None:
    wav = np.ones(1000, dtype=np.float32) * 0.9
    result = noise_transform(wav, snr_db=0, clip_id="loud", global_seed=42)
    assert np.all(result.waveform >= -1.0)
    assert np.all(result.waveform <= 1.0)


def test_noise_reports_clipping() -> None:
    wav = np.ones(1000, dtype=np.float32) * 0.99
    result = noise_transform(wav, snr_db=0, clip_id="test", global_seed=42)
    assert result.n_clipped >= 0
    assert result.max_abs_before_clip >= 0


def test_noise_zero_rms_returns_unchanged() -> None:
    wav = np.zeros(1000, dtype=np.float32)
    result = noise_transform(wav, snr_db=20, clip_id="silent", global_seed=42)
    np.testing.assert_array_equal(result.waveform, wav)
    assert result.noise_rms == 0.0
    assert result.n_clipped == 0


def test_noise_class_symmetry() -> None:
    """Transform signature does not accept a label — class is never an input."""
    import inspect

    sig = inspect.signature(noise_transform)
    param_names = set(sig.parameters)
    assert "label" not in param_names
    assert "class_label" not in param_names


# ---------------------------------------------------------------------------
# Silence trim
# ---------------------------------------------------------------------------

def test_silence_trim_removes_silence() -> None:
    sr = 16000
    silence = np.zeros(sr, dtype=np.float32)
    tone = _sine_wave(duration_s=1.0, sr=sr)
    wav = np.concatenate([silence, tone, silence])
    result = silence_trim(wav, sample_rate=sr, top_db=30)
    assert isinstance(result, TrimResult)
    assert result.trimmed_length < result.original_length
    assert result.samples_removed > 0
    assert not result.kept_original


def test_silence_trim_short_keeps_original() -> None:
    sr = 16000
    silence = np.zeros(int(sr * 0.9), dtype=np.float32)
    tiny_tone = _sine_wave(duration_s=0.1, sr=sr)
    wav = np.concatenate([silence, tiny_tone, silence])
    result = silence_trim(wav, sample_rate=sr, top_db=20, min_duration_seconds=0.5)
    assert result.kept_original
    assert result.trimmed_length == result.original_length
    assert result.samples_removed == 0


def test_silence_trim_all_silence() -> None:
    wav = np.zeros(16000, dtype=np.float32)
    result = silence_trim(wav, sample_rate=16000, top_db=30, min_duration_seconds=0.3)
    assert result.kept_original
    assert result.waveform.size == wav.size


def test_silence_trim_no_silence() -> None:
    wav = _sine_wave(duration_s=1.0)
    result = silence_trim(wav, sample_rate=16000, top_db=30)
    assert result.trimmed_length > 0
    assert not result.kept_original or result.trimmed_length == result.original_length


def test_silence_trim_result_fields() -> None:
    wav = _sine_wave(duration_s=1.0)
    result = silence_trim(wav, sample_rate=16000, top_db=30, min_duration_seconds=0.3)
    assert result.top_db == 30
    assert result.min_duration_seconds == 0.3


# ---------------------------------------------------------------------------
# Codec transform (requires ffmpeg)
# ---------------------------------------------------------------------------

@requires_ffmpeg
def test_codec_determinism() -> None:
    wav = _sine_wave(duration_s=1.0)
    r1 = codec_transform(wav)
    r2 = codec_transform(wav)
    np.testing.assert_array_equal(r1.waveform, r2.waveform)


@requires_ffmpeg
def test_codec_result_type() -> None:
    wav = _sine_wave(duration_s=1.0)
    result = codec_transform(wav)
    assert isinstance(result, CodecResult)
    assert result.codec == "libopus"
    assert result.bitrate_kbps == 24
    assert result.container == "ogg"
    assert len(result.ffmpeg_version) > 0


@requires_ffmpeg
def test_codec_truncates_to_original_length() -> None:
    wav = _sine_wave(duration_s=1.0)
    result = codec_transform(wav)
    assert result.waveform.size == wav.size
    assert result.samples_delta == 0


@requires_ffmpeg
def test_codec_modifies_waveform() -> None:
    wav = _sine_wave(duration_s=1.0)
    result = codec_transform(wav)
    assert not np.array_equal(result.waveform, wav)


@requires_ffmpeg
def test_codec_short_clip() -> None:
    wav = _sine_wave(duration_s=0.1)
    result = codec_transform(wav)
    assert result.waveform.size == wav.size


# ---------------------------------------------------------------------------
# check_ffmpeg
# ---------------------------------------------------------------------------

def test_check_ffmpeg_or_error() -> None:
    if shutil.which("ffmpeg") is not None:
        version = check_ffmpeg()
        assert "ffmpeg" in version.lower()
    else:
        with pytest.raises(TransformationError, match="ffmpeg not found"):
            check_ffmpeg()


# ---------------------------------------------------------------------------
# TrainingAugmentor
# ---------------------------------------------------------------------------

def test_augmentor_no_ffmpeg_noise_only() -> None:
    """Without codec, augmentor still applies noise."""
    aug = TrainingAugmentor(
        global_seed=42,
        probability=1.0,
        codec_probability=0.0,
        noise_probability=1.0,
        noise_snr_range=(15.0, 25.0),
    )
    wav = _sine_wave()
    result = aug(wav, "clip_1")
    assert result.shape == wav.shape


def test_augmentor_epoch_changes_output() -> None:
    aug = TrainingAugmentor(
        global_seed=42,
        probability=1.0,
        codec_probability=0.0,
        noise_probability=1.0,
    )
    wav = _sine_wave()
    aug.set_epoch(0)
    r0 = aug(wav, "clip_1")
    aug.set_epoch(1)
    r1 = aug(wav, "clip_1")
    assert not np.array_equal(r0, r1)


def test_augmentor_deterministic() -> None:
    aug = TrainingAugmentor(
        global_seed=42,
        probability=1.0,
        codec_probability=0.0,
        noise_probability=1.0,
    )
    wav = _sine_wave()
    r1 = aug(wav, "clip_1")
    r2 = aug(wav, "clip_1")
    np.testing.assert_array_equal(r1, r2)


def test_augmentor_probability_zero_returns_original() -> None:
    aug = TrainingAugmentor(probability=0.0)
    wav = _sine_wave()
    result = aug(wav, "clip_1")
    np.testing.assert_array_equal(result, wav)


def test_augmentor_set_epoch() -> None:
    aug = TrainingAugmentor()
    assert aug.epoch == 0
    aug.set_epoch(5)
    assert aug.epoch == 5
