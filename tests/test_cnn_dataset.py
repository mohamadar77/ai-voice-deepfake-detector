"""Dataset-free tests for log-Mel CNN dataset utilities (#33)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from voice_detector.cnn_dataset import (
    LABEL_MAP,
    EvalWindowDataset,
    MelDataset,
    MelSpec,
    _random_crop,
    _repeat_pad,
    compute_norm_stats,
)


def _write_wav(path: Path, duration_s: float, sr: int = 16000) -> Path:
    samples = np.random.default_rng(42).standard_normal(int(sr * duration_s)).astype(np.float32)
    sf.write(str(path), samples, sr)
    return path


def _make_rows(tmpdir: Path, n: int = 6, duration_s: float = 5.0) -> list[dict]:
    rows = []
    for i in range(n):
        label = "bonafide" if i % 2 == 0 else "spoof"
        wav = _write_wav(tmpdir / f"clip_{i}.flac", duration_s)
        rows.append({
            "clip_id": f"clip_{i}",
            "label": label,
            "speaker_id": f"LA_00{i:02d}",
            "official_split": "train",
            "local_path": str(wav),
        })
    return rows


def test_mel_spec_defaults() -> None:
    spec = MelSpec()
    assert spec.sample_rate_hz == 16000
    assert spec.n_mels == 64
    assert spec.window_samples == 64000
    assert spec.n_frames == (64000 - 400) // 160 + 1


def test_mel_spec_from_mapping() -> None:
    raw = {"mel": {"n_mels": 40, "window_seconds": 2.0}}
    spec = MelSpec.from_mapping(raw)
    assert spec.n_mels == 40
    assert spec.window_seconds == 2.0
    assert spec.window_samples == 32000


def test_repeat_pad_short() -> None:
    wav = np.array([1.0, 2.0, 3.0])
    out = _repeat_pad(wav, 8)
    assert len(out) == 8
    np.testing.assert_array_equal(out[:3], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(out[3:6], [1.0, 2.0, 3.0])


def test_repeat_pad_exact() -> None:
    wav = np.array([1.0, 2.0, 3.0])
    out = _repeat_pad(wav, 3)
    assert len(out) == 3
    np.testing.assert_array_equal(out, wav)


def test_repeat_pad_long() -> None:
    wav = np.arange(10, dtype=np.float32)
    out = _repeat_pad(wav, 5)
    assert len(out) == 5
    np.testing.assert_array_equal(out, wav[:5])


def test_random_crop_long() -> None:
    rng = np.random.default_rng(0)
    wav = np.arange(100, dtype=np.float32)
    out = _random_crop(wav, 20, rng)
    assert len(out) == 20


def test_random_crop_short_falls_back_to_repeat_pad() -> None:
    rng = np.random.default_rng(0)
    wav = np.array([1.0, 2.0])
    out = _random_crop(wav, 6, rng)
    assert len(out) == 6
    np.testing.assert_array_equal(out[:2], [1.0, 2.0])


def test_label_map() -> None:
    assert LABEL_MAP["bonafide"] == 0
    assert LABEL_MAP["spoof"] == 1


def test_mel_dataset_training_shapes() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        rows = _make_rows(Path(tmpdir), n=4, duration_s=5.0)
        spec = MelSpec(window_seconds=2.0)
        ds = MelDataset(rows, spec, training=True, seed=42)
        assert len(ds) == 4
        item = ds[0]
        assert item["spectrogram"].shape[0] == 1
        assert item["spectrogram"].shape[1] == spec.n_mels
        assert item["spectrogram"].shape[2] == spec.n_frames
        assert item["label"] in (0, 1)
        assert isinstance(item["clip_id"], str)


def test_mel_dataset_eval_shapes() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        rows = _make_rows(Path(tmpdir), n=2, duration_s=2.0)
        spec = MelSpec(window_seconds=2.0)
        ds = MelDataset(rows, spec, training=False, seed=42)
        assert len(ds) == 2
        item = ds[0]
        assert item["spectrogram"].shape == (1, spec.n_mels, spec.n_frames)


def test_mel_dataset_short_clip_repeat_pads() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        rows = _make_rows(Path(tmpdir), n=1, duration_s=0.5)
        spec = MelSpec(window_seconds=4.0)
        ds = MelDataset(rows, spec, training=False, seed=42)
        item = ds[0]
        assert item["spectrogram"].shape == (1, spec.n_mels, spec.n_frames)


def test_mel_dataset_normalization() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        rows = _make_rows(Path(tmpdir), n=4, duration_s=2.0)
        spec = MelSpec(window_seconds=2.0)
        mean, std = compute_norm_stats(rows, spec, max_clips=4, seed=42)
        ds = MelDataset(rows, spec, training=False, seed=42, mean=mean, std=std)
        item = ds[0]
        assert item["spectrogram"].shape == (1, spec.n_mels, spec.n_frames)


def test_eval_window_dataset() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        rows = _make_rows(Path(tmpdir), n=2, duration_s=5.0)
        spec = MelSpec(window_seconds=2.0)
        ds = EvalWindowDataset(rows, spec)
        assert len(ds) > 0
        item = ds[0]
        assert item["spectrogram"].shape == (1, spec.n_mels, spec.n_frames)
        assert isinstance(item["clip_id"], str)
        assert item["label"] in (0, 1)


def test_compute_norm_stats_shapes() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        rows = _make_rows(Path(tmpdir), n=4, duration_s=2.0)
        spec = MelSpec(window_seconds=2.0)
        mean, std = compute_norm_stats(rows, spec, max_clips=4, seed=42)
        assert mean.shape == (spec.n_mels,)
        assert std.shape == (spec.n_mels,)
        assert mean.dtype == torch.float32
        assert std.dtype == torch.float32
        assert (std >= 0).all()
