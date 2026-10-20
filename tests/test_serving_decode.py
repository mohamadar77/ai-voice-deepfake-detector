"""Upload probing and decoding: header limits are enforced before any decode."""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf
from conftest import requires_librosa
from serving_fakes import audio_bytes, make_settings

from voice_detector.preprocessing import AudioConfig, load_waveform
from voice_detector.serving import decode
from voice_detector.serving.decode import InputRejected, decode_upload, probe_upload

LIMITS = make_settings().input


def _reject(data: bytes) -> InputRejected:
    with pytest.raises(InputRejected) as caught:
        probe_upload(data, LIMITS)
    return caught.value


def test_wav_and_flac_are_accepted():
    for fmt, subtype in (("WAV", "PCM_16"), ("FLAC", "PCM_16")):
        probe = probe_upload(audio_bytes(2.0, fmt=fmt, subtype=subtype), LIMITS)
        assert probe.format == fmt
        assert probe.duration_seconds == pytest.approx(2.0)
        assert probe.channels == 1 and probe.sample_rate_hz == 16000


def test_garbage_is_undecodable_and_the_message_is_fixed():
    rejected = _reject(b"SECRETMARK not audio at all")
    assert (rejected.code, rejected.status) == ("undecodable_audio", 400)
    assert "SECRETMARK" not in rejected.message


def test_other_containers_are_unsupported():
    rejected = _reject(audio_bytes(2.0, fmt="AIFF", subtype="PCM_16"))
    assert (rejected.code, rejected.status) == ("unsupported_format", 415)


def test_too_many_channels_and_extreme_rates_are_rejected():
    assert _reject(audio_bytes(2.0, channels=3)).code == "unsupported_audio_parameters"
    assert _reject(audio_bytes(2.0, sample_rate=7000)).code == "unsupported_audio_parameters"
    assert _reject(audio_bytes(2.0, sample_rate=100000)).code == "unsupported_audio_parameters"


@pytest.mark.parametrize("seconds", [0.5, 0.999])
def test_below_the_floor_is_rejected(seconds: float):
    assert _reject(audio_bytes(seconds)).code == "duration_out_of_range"


def test_the_floor_and_ceiling_are_inclusive():
    assert probe_upload(audio_bytes(1.0), LIMITS).duration_seconds == pytest.approx(1.0)
    assert probe_upload(audio_bytes(30.0), LIMITS).duration_seconds == pytest.approx(30.0)


def test_over_the_ceiling_is_rejected_from_the_header_before_decoding(
    monkeypatch: pytest.MonkeyPatch,
):
    # 31 s of digital silence compresses to a tiny FLAC; it must be refused from the
    # header, and the decoder must never run.
    silent = np.zeros(31 * 16000, dtype=np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, silent, 16000, format="FLAC", subtype="PCM_16")
    data = buffer.getvalue()
    assert len(data) < 50_000

    def boom(*args, **kwargs):
        raise AssertionError("decode ran before the header limits")

    monkeypatch.setattr(decode, "load_waveform", boom)
    assert _reject(data).code == "duration_out_of_range"


def test_zero_frame_audio_is_empty():
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(0, dtype=np.float32), 16000, format="WAV", subtype="PCM_16")
    assert _reject(buffer.getvalue()).code == "empty_audio"


def test_truncated_audio_is_not_scored_silently():
    data = audio_bytes(2.0)
    try:
        probe = probe_upload(data[: len(data) // 2], LIMITS)
    except InputRejected as rejected:
        assert rejected.code in {"undecodable_audio", "duration_out_of_range"}
    else:
        # Decoding a truncated file must not crash with a raw traceback.
        decode_upload(data[: len(data) // 2], probe, 16000, LIMITS)


def test_decode_matches_the_shared_evaluation_decoder_at_16k():
    data = audio_bytes(2.0)
    probe = probe_upload(data, LIMITS)
    got = decode_upload(data, probe, 16000, LIMITS)
    want = load_waveform(io.BytesIO(data), AudioConfig(sample_rate_hz=16000))
    assert got.dtype == np.float32
    assert np.array_equal(got, want)


@requires_librosa
def test_stereo_48k_is_mixed_to_mono_and_resampled_like_evaluation():
    data = audio_bytes(2.0, sample_rate=48000, channels=2)
    probe = probe_upload(data, LIMITS)
    got = decode_upload(data, probe, 16000, LIMITS)
    want = load_waveform(io.BytesIO(data), AudioConfig(sample_rate_hz=16000))
    assert np.array_equal(got, want)
    assert got.ndim == 1 and abs(got.size - 32000) <= 2


def test_non_finite_samples_are_unscorable():
    samples = np.full(32000, 0.1, dtype=np.float32)
    samples[100] = np.nan
    buffer = io.BytesIO()
    sf.write(buffer, samples, 16000, format="WAV", subtype="FLOAT")
    data = buffer.getvalue()
    probe = probe_upload(data, LIMITS)
    with pytest.raises(InputRejected) as caught:
        decode_upload(data, probe, 16000, LIMITS)
    assert caught.value.code == "unscorable_audio"
