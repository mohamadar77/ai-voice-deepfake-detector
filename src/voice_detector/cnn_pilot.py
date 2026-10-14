"""CNN timing pilot: measure forward/backward cost on CPU before full training (#10).

This module defines a minimal compact log-Mel CNN, runs a short
forward/backward loop on a tiny declared subset of training data, and
records wall-clock per step, peak memory, and an extrapolated full-run
estimate. The output is a JSON-serializable timing record that informs
the compute-route decision (CPU host vs external GPU).

Nothing here touches the evaluation partition or produces model-selection
metrics. The CNN architecture is deliberately small and fixed for the pilot;
architecture search belongs in the full CNN study (#33).
"""

from __future__ import annotations

import gc
import os
import platform
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from voice_detector.manifest import git_commit, package_version

PILOT_VERSION = "cnn-pilot-v1"


@dataclass(frozen=True)
class MelConfig:
    """Log-Mel spectrogram settings for the CNN input."""

    sample_rate_hz: int = 16000
    n_fft: int = 400
    hop_length: int = 160
    n_mels: int = 64
    fmin: float = 20.0
    fmax: float = 8000.0
    window_seconds: float = 4.0

    @property
    def window_samples(self) -> int:
        return int(round(self.sample_rate_hz * self.window_seconds))

    @property
    def n_frames(self) -> int:
        return (self.window_samples - self.n_fft) // self.hop_length + 1


class CompactMelCNN(nn.Module):
    """Minimal log-Mel CNN for the timing pilot.

    Architecture: 3 conv blocks (conv2d + batchnorm + relu + maxpool),
    adaptive average pool to (1, 1), linear head. Deliberately compact
    to get a lower bound on per-step cost; the full study may widen.
    """

    def __init__(self, n_mels: int = 64, n_frames: int = 399) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def _synthetic_mel_batch(
    batch_size: int,
    mel_config: MelConfig,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a synthetic batch of log-Mel spectrograms for timing."""

    rng = np.random.default_rng(seed)
    spectrograms = rng.standard_normal(
        (batch_size, 1, mel_config.n_mels, mel_config.n_frames)
    ).astype(np.float32)
    labels = rng.integers(0, 2, size=batch_size).astype(np.int64)
    return torch.from_numpy(spectrograms), torch.from_numpy(labels)


def _peak_memory_mb() -> float:
    """Peak RSS in MiB (Unix only, 0.0 elsewhere)."""

    try:
        import resource
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        if platform.system() == "Darwin":
            return rusage.ru_maxrss / (1024 * 1024)
        return rusage.ru_maxrss / 1024
    except (ImportError, AttributeError):
        return 0.0


def run_timing_pilot(
    *,
    mel_config: MelConfig | None = None,
    batch_sizes: list[int] | None = None,
    n_warmup: int = 2,
    n_timed: int = 10,
    seed: int = 20261004,
    train_n_clips: int = 25380,
    train_n_windows: int = 32459,
    target_epochs: int = 30,
    max_hours_per_run: float = 12.0,
    max_hours_total: float = 48.0,
    n_runs_matrix: int = 4,
) -> dict[str, Any]:
    """Run forward/backward timing on CPU with synthetic data.

    Returns a JSON-serializable record with per-batch-size timings and
    extrapolated full-run estimates.
    """

    if mel_config is None:
        mel_config = MelConfig()
    if batch_sizes is None:
        batch_sizes = [16, 32, 64]

    torch.manual_seed(seed)
    results_per_bs: list[dict[str, Any]] = []
    mem_before = _peak_memory_mb()

    for bs in batch_sizes:
        model = CompactMelCNN(n_mels=mel_config.n_mels, n_frames=mel_config.n_frames)
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        data, labels = _synthetic_mel_batch(bs, mel_config, seed=seed)
        dataset = TensorDataset(data, labels)
        loader = DataLoader(dataset, batch_size=bs, shuffle=False)

        for _ in range(n_warmup):
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                out = model(x_batch)
                loss = criterion(out, y_batch)
                loss.backward()
                optimizer.step()

        step_times: list[float] = []
        for _ in range(n_timed):
            for x_batch, y_batch in loader:
                t0 = time.perf_counter()
                optimizer.zero_grad()
                out = model(x_batch)
                loss = criterion(out, y_batch)
                loss.backward()
                optimizer.step()
                step_times.append(time.perf_counter() - t0)

        mean_step_s = float(np.mean(step_times))
        std_step_s = float(np.std(step_times))
        steps_per_epoch = (train_n_windows + bs - 1) // bs
        epoch_s = mean_step_s * steps_per_epoch
        full_run_s = epoch_s * target_epochs
        full_run_h = full_run_s / 3600
        matrix_h = full_run_h * n_runs_matrix
        fits_single = full_run_h <= max_hours_per_run
        fits_matrix = matrix_h <= max_hours_total

        n_params = sum(p.numel() for p in model.parameters())

        results_per_bs.append({
            "batch_size": bs,
            "n_warmup_steps": n_warmup,
            "n_timed_steps": len(step_times),
            "mean_step_seconds": round(mean_step_s, 6),
            "std_step_seconds": round(std_step_s, 6),
            "steps_per_epoch": steps_per_epoch,
            "epoch_seconds": round(epoch_s, 2),
            "epoch_minutes": round(epoch_s / 60, 2),
            "extrapolated": {
                "target_epochs": target_epochs,
                "full_run_hours": round(full_run_h, 2),
                "n_runs_matrix": n_runs_matrix,
                "matrix_hours": round(matrix_h, 2),
                "fits_single_run_cap": fits_single,
                "fits_matrix_cap": fits_matrix,
                "single_run_cap_hours": max_hours_per_run,
                "matrix_cap_hours": max_hours_total,
            },
        })

        del model, optimizer, criterion, data, labels, dataset, loader
        gc.collect()

    mem_after = _peak_memory_mb()

    return {
        "schema_version": PILOT_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_commit(),
        "package_version": package_version(),
        "seed": seed,
        "mel_config": {
            "sample_rate_hz": mel_config.sample_rate_hz,
            "n_fft": mel_config.n_fft,
            "hop_length": mel_config.hop_length,
            "n_mels": mel_config.n_mels,
            "fmin": mel_config.fmin,
            "fmax": mel_config.fmax,
            "window_seconds": mel_config.window_seconds,
            "window_samples": mel_config.window_samples,
            "n_frames": mel_config.n_frames,
        },
        "model": {
            "class": "CompactMelCNN",
            "n_params": n_params,
        },
        "dataset_context": {
            "train_n_clips": train_n_clips,
            "train_n_windows": train_n_windows,
        },
        "hardware": {
            "cpu": platform.processor() or platform.machine(),
            "cpu_count": os.cpu_count(),
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "peak_rss_mib_before": round(mem_before, 1),
            "peak_rss_mib_after": round(mem_after, 1),
        },
        "results": results_per_bs,
    }
