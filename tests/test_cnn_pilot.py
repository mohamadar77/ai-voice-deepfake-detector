"""Dataset-free tests for the CNN timing pilot (#10)."""

from __future__ import annotations

import torch

from voice_detector.cnn_pilot import (
    PILOT_VERSION,
    CompactMelCNN,
    MelConfig,
    _synthetic_mel_batch,
    run_timing_pilot,
)


def test_mel_config_defaults() -> None:
    cfg = MelConfig()
    assert cfg.sample_rate_hz == 16000
    assert cfg.n_mels == 64
    assert cfg.n_frames == 398


def test_compact_cnn_forward_shape() -> None:
    cfg = MelConfig()
    model = CompactMelCNN(n_mels=cfg.n_mels, n_frames=cfg.n_frames)
    x = torch.randn(4, 1, cfg.n_mels, cfg.n_frames)
    out = model(x)
    assert out.shape == (4, 2)


def test_compact_cnn_backward_runs() -> None:
    cfg = MelConfig()
    model = CompactMelCNN(n_mels=cfg.n_mels, n_frames=cfg.n_frames)
    x = torch.randn(2, 1, cfg.n_mels, cfg.n_frames)
    out = model(x)
    loss = out.sum()
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None


def test_synthetic_batch_shapes() -> None:
    cfg = MelConfig()
    data, labels = _synthetic_mel_batch(8, cfg, seed=42)
    assert data.shape == (8, 1, cfg.n_mels, cfg.n_frames)
    assert labels.shape == (8,)
    assert data.dtype == torch.float32
    assert labels.dtype == torch.int64


def test_synthetic_batch_deterministic() -> None:
    cfg = MelConfig()
    d1, l1 = _synthetic_mel_batch(4, cfg, seed=99)
    d2, l2 = _synthetic_mel_batch(4, cfg, seed=99)
    assert torch.equal(d1, d2)
    assert torch.equal(l1, l2)


def test_run_timing_pilot_produces_valid_record() -> None:
    record = run_timing_pilot(
        batch_sizes=[4],
        n_warmup=1,
        n_timed=2,
        train_n_clips=100,
        train_n_windows=120,
        target_epochs=2,
    )
    assert record["schema_version"] == PILOT_VERSION
    assert record["seed"] == 20261004
    assert len(record["results"]) == 1

    r = record["results"][0]
    assert r["batch_size"] == 4
    assert r["mean_step_seconds"] > 0
    assert r["steps_per_epoch"] == 30
    assert r["extrapolated"]["target_epochs"] == 2
    assert isinstance(r["extrapolated"]["fits_single_run_cap"], bool)
    assert isinstance(r["extrapolated"]["fits_matrix_cap"], bool)

    assert "model" in record
    assert record["model"]["n_params"] > 0
    assert "hardware" in record
    assert record["hardware"]["cpu_count"] is not None


def test_run_timing_pilot_multiple_batch_sizes() -> None:
    record = run_timing_pilot(
        batch_sizes=[4, 8],
        n_warmup=1,
        n_timed=1,
        train_n_clips=50,
        train_n_windows=60,
        target_epochs=1,
    )
    assert len(record["results"]) == 2
    assert record["results"][0]["batch_size"] == 4
    assert record["results"][1]["batch_size"] == 8
