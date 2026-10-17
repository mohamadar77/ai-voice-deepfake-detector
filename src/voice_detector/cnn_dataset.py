"""PyTorch Dataset for log-Mel spectrogram CNN training (#33).

Loads waveforms from the manifest, computes log-Mel spectrograms on the fly,
and applies the declared input policy: repeat-pad short clips, random crop
long clips during training, deterministic option-D windows during evaluation.

Normalization statistics (per-channel mean/std) are fitted on training data
only and applied identically to all splits.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

from voice_detector.manifest import resolve_local_path
from voice_detector.preprocessing import AudioConfig, load_waveform, window_spans


@dataclass(frozen=True)
class MelSpec:
    """Log-Mel spectrogram transform, matching cnn_study.yaml mel section."""

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

    def transform(self) -> torchaudio.transforms.MelSpectrogram:
        return torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate_hz,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=self.fmin,
            f_max=self.fmax,
        )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> MelSpec:
        mel_raw = dict(raw.get("mel") or {})
        return cls(
            sample_rate_hz=int(mel_raw.get("sample_rate_hz", cls.sample_rate_hz)),
            n_fft=int(mel_raw.get("n_fft", cls.n_fft)),
            hop_length=int(mel_raw.get("hop_length", cls.hop_length)),
            n_mels=int(mel_raw.get("n_mels", cls.n_mels)),
            fmin=float(mel_raw.get("fmin", cls.fmin)),
            fmax=float(mel_raw.get("fmax", cls.fmax)),
            window_seconds=float(mel_raw.get("window_seconds", cls.window_seconds)),
        )


LABEL_MAP = {"bonafide": 0, "spoof": 1}


def repeat_pad(waveform: np.ndarray, target_len: int) -> np.ndarray:
    """Repeat-pad a waveform to target_len samples (no zero-padding).

    Longer input is truncated to ``target_len``; this drops the folded tail
    that ``window_spans`` may append to the last window.
    """

    if waveform.size >= target_len:
        return waveform[:target_len]
    repeats = (target_len // waveform.size) + 1
    return np.tile(waveform, repeats)[:target_len]


_repeat_pad = repeat_pad


def segment_to_spectrogram(
    segment: np.ndarray,
    mel_transform: torch.nn.Module,
    mean: torch.Tensor | None,
    std: torch.Tensor | None,
) -> torch.Tensor:
    """One fixed-length segment to a normalized log-Mel tensor ``(1, n_mels, n_frames)``.

    The single definition of the numerics shared by training evaluation and
    ``eval-score``. Keep the operation order: mel, log of the clamped mel, then
    per-channel standardization with ``std`` clamped at 1e-6.
    """

    mel = mel_transform(torch.from_numpy(segment))
    log_mel = torch.log(mel.clamp(min=1e-9))
    if mean is not None and std is not None:
        log_mel = (log_mel - mean.unsqueeze(-1)) / std.unsqueeze(-1).clamp(min=1e-6)
    return log_mel.unsqueeze(0)


def _random_crop(waveform: np.ndarray, target_len: int, rng: np.random.Generator) -> np.ndarray:
    """Random crop a waveform to target_len samples."""

    if waveform.size <= target_len:
        return _repeat_pad(waveform, target_len)
    start = rng.integers(0, waveform.size - target_len + 1)
    return waveform[start : start + target_len]


class MelDataset(Dataset):
    """Per-clip dataset: one log-Mel spectrogram per clip.

    In training mode, long clips are randomly cropped and short clips are
    repeat-padded. With ``training=False`` only the first ``window_seconds`` of
    each clip is used; it is not a scoring path. Use ``EvalWindowDataset`` for
    option-D deterministic windows.
    """

    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        mel_spec: MelSpec,
        *,
        training: bool = True,
        seed: int = 20261004,
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
        augment_fn: Callable[[np.ndarray, str], np.ndarray] | None = None,
    ) -> None:
        self.rows = list(rows)
        self.mel_spec = mel_spec
        self.training = training
        self.seed = seed
        self._mel_transform = mel_spec.transform()
        self._audio_config = AudioConfig(
            sample_rate_hz=mel_spec.sample_rate_hz,
            window_seconds=mel_spec.window_seconds,
        )
        self.mean = mean
        self.std = std
        self.augment_fn = augment_fn

    def set_epoch(self, epoch: int) -> None:
        """Forward epoch to the augmentor so per-clip seeds vary each epoch."""
        if self.augment_fn is not None and hasattr(self.augment_fn, "set_epoch"):
            self.augment_fn.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        clip_id = str(row["clip_id"])
        label = LABEL_MAP[str(row["label"])]
        local_path = row.get("local_path")
        if not local_path:
            raise ValueError(f"clip {clip_id!r} has no local_path")

        waveform = load_waveform(resolve_local_path(local_path), self._audio_config)

        if self.training and self.augment_fn is not None:
            waveform = self.augment_fn(waveform, clip_id)

        target_len = self.mel_spec.window_samples

        if self.training:
            rng = np.random.default_rng(self.seed + idx)
            segment = _random_crop(waveform, target_len, rng)
        else:
            segment = _repeat_pad(waveform, target_len)

        tensor = torch.from_numpy(segment)
        mel = self._mel_transform(tensor)
        log_mel = torch.log(mel.clamp(min=1e-9))

        if self.mean is not None and self.std is not None:
            log_mel = (log_mel - self.mean.unsqueeze(-1)) / self.std.unsqueeze(-1).clamp(min=1e-6)

        return {
            "spectrogram": log_mel.unsqueeze(0),  # (1, n_mels, n_frames)
            "label": label,
            "clip_id": clip_id,
        }


class EvalWindowDataset(Dataset):
    """Per-window dataset for deterministic evaluation.

    Each item is one option-D window of one clip, repeat-padded to
    window_samples if needed. Window scores are aggregated to clip scores
    downstream.
    """

    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        mel_spec: MelSpec,
        *,
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
    ) -> None:
        self.mel_spec = mel_spec
        self._mel_transform = mel_spec.transform()
        self._audio_config = AudioConfig(
            sample_rate_hz=mel_spec.sample_rate_hz,
            window_seconds=mel_spec.window_seconds,
        )
        self.mean = mean
        self.std = std

        self._items: list[tuple[str, int, int, int, str]] = []
        for row in rows:
            clip_id = str(row["clip_id"])
            label = LABEL_MAP[str(row["label"])]
            local_path = str(row.get("local_path", ""))
            waveform = load_waveform(resolve_local_path(local_path), self._audio_config)
            spans = window_spans(
                waveform.size,
                window_samples=mel_spec.window_samples,
                n_fft=mel_spec.n_fft,
            )
            for start, stop in spans:
                self._items.append((clip_id, label, start, stop, local_path))

        self._waveform_cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        clip_id, label, start, stop, local_path = self._items[idx]

        if local_path not in self._waveform_cache:
            self._waveform_cache[local_path] = load_waveform(
                resolve_local_path(local_path), self._audio_config
            )
        waveform = self._waveform_cache[local_path]
        segment = waveform[start:stop]
        segment = repeat_pad(segment, self.mel_spec.window_samples)

        return {
            "spectrogram": segment_to_spectrogram(
                segment, self._mel_transform, self.mean, self.std
            ),
            "label": label,
            "clip_id": clip_id,
        }


def compute_norm_stats(
    rows: Sequence[Mapping[str, Any]],
    mel_spec: MelSpec,
    *,
    max_clips: int = 2000,
    seed: int = 20261004,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-channel mean and std from a subset of training clips.

    Returns (mean, std) tensors of shape (n_mels,).
    """

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(rows), size=min(max_clips, len(rows)), replace=False)
    audio_config = AudioConfig(
        sample_rate_hz=mel_spec.sample_rate_hz,
        window_seconds=mel_spec.window_seconds,
    )
    mel_transform = mel_spec.transform()

    channel_sum = torch.zeros(mel_spec.n_mels, dtype=torch.float64)
    channel_sq_sum = torch.zeros(mel_spec.n_mels, dtype=torch.float64)
    total_frames = 0

    for i in indices:
        row = rows[int(i)]
        local_path = row.get("local_path")
        if not local_path:
            continue
        waveform = load_waveform(resolve_local_path(local_path), audio_config)
        segment = _repeat_pad(waveform, mel_spec.window_samples)
        tensor = torch.from_numpy(segment)
        mel = mel_transform(tensor)
        log_mel = torch.log(mel.clamp(min=1e-9)).to(torch.float64)

        channel_sum += log_mel.sum(dim=-1)
        channel_sq_sum += (log_mel ** 2).sum(dim=-1)
        total_frames += log_mel.shape[-1]

    mean = (channel_sum / total_frames).float()
    variance = (channel_sq_sum / total_frames - (channel_sum / total_frames) ** 2).clamp(min=0)
    std = variance.sqrt().float()
    return mean, std
