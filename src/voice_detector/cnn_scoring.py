"""CNN clip scorer shared by ``eval-score`` and, later, the serving API (#9).

The scorer reproduces the window-to-clip quantity that ``cnn_train`` scored when
it fitted the operating threshold: option-D windows from ``window_spans``,
repeat-padding, train-fitted per-channel normalization, and one score per
window equal to ``logit[spoof] - logit[bonafide]``. The clip score is the
arithmetic mean of the window scores in float64 (``mean_window_logit``).

This module imports torch. It must not import ``cnn_train``, ``metrics`` or
``reservation``: serving does not depend on training code.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from voice_detector.cnn_dataset import MelSpec, repeat_pad, segment_to_spectrogram
from voice_detector.cnn_pilot import CompactMelCNN
from voice_detector.hashing import hash_directory
from voice_detector.preprocessing import AudioConfig, load_waveform, window_spans
from voice_detector.scoring import (
    CNN_AGGREGATION_RULE,
    SCORE_DIRECTION,
    AudioSource,
    ClipScore,
    ScoredClip,
    WaveformScore,
    clip_score_from,
)

CNN_SCHEMA_VERSION = "cnn-study-v1"
CNN_RUN_RECORD = "cnn_run_record.json"
CNN_CHECKPOINT = "cnn_checkpoint.pt"
CNN_NORM_STATS = "cnn_norm_stats.pt"

# A clip's windows are scored together; only very long clips are chunked, so a
# clip score never depends on which other clips are being scored.
_WINDOW_CHUNK = 32


class CnnBundleError(ValueError):
    """The CNN bundle failed an integrity or consistency check."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CnnBundleError(message)


def _mel_spec_from_record(record: dict[str, Any]) -> MelSpec:
    mel = record.get("mel_config")
    _require(isinstance(mel, dict), "run record has no mel_config")
    try:
        return MelSpec(
            sample_rate_hz=int(mel["sample_rate_hz"]),
            n_fft=int(mel["n_fft"]),
            hop_length=int(mel["hop_length"]),
            n_mels=int(mel["n_mels"]),
            fmin=float(mel["fmin"]),
            fmax=float(mel["fmax"]),
            window_seconds=float(mel["window_seconds"]),
        )
    except KeyError as exc:
        raise CnnBundleError(f"run record mel_config is missing {exc}") from exc


def _check_run_record(record: dict[str, Any]) -> None:
    _require(
        record.get("schema_version") == CNN_SCHEMA_VERSION,
        f"unexpected run record schema_version {record.get('schema_version')!r}",
    )
    _require(
        record.get("model", {}).get("class") == "CompactMelCNN",
        "run record model.class is not CompactMelCNN",
    )
    _require(
        record.get("score_direction") == SCORE_DIRECTION,
        f"run record score_direction {record.get('score_direction')!r} "
        f"is not {SCORE_DIRECTION!r}",
    )
    policy = record.get("input_policy", {})
    _require(
        policy.get("eval_aggregation") == CNN_AGGREGATION_RULE,
        f"input_policy.eval_aggregation {policy.get('eval_aggregation')!r} "
        f"is not {CNN_AGGREGATION_RULE!r}",
    )
    _require(
        policy.get("eval_windowing") == "option_d_deterministic",
        "input_policy.eval_windowing is not option_d_deterministic",
    )
    _require(
        policy.get("short_clip_handling") == "repeat_pad",
        "input_policy.short_clip_handling is not repeat_pad",
    )
    _require(
        record.get("normalization", {}).get("method") == "per_channel_standardize",
        "normalization.method is not per_channel_standardize",
    )


_THRESHOLD_BASIS_KEYS = ("value", "decision_rule", "target_fpr", "achieved_fpr", "n_bonafide")


def _describe_record(record: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    """Allowlisted run-record summary for ``CnnClipScorer.describe`` (no paths)."""

    mel = record.get("mel_config", {})
    threshold = record.get("threshold", {})
    return {
        "candidate_id": candidate_id,
        "run_label": record.get("run_label"),
        "model_family": "cnn",
        "mel": {
            key: mel.get(key)
            for key in ("sample_rate_hz", "n_fft", "hop_length", "n_mels", "window_seconds")
        },
        "threshold_basis": {k: threshold[k] for k in _THRESHOLD_BASIS_KEYS if k in threshold},
    }


class CnnClipScorer:
    """Scores decoded waveforms with a loaded CNN bundle."""

    model_family = "cnn"
    aggregation_rule = CNN_AGGREGATION_RULE

    def __init__(
        self,
        model: CompactMelCNN,
        mel_spec: MelSpec,
        mean: torch.Tensor,
        std: torch.Tensor,
        threshold: float,
        *,
        model_version: str,
        preprocessing_version: str,
        bundle_hash: str | None = None,
        description: Mapping[str, Any] | None = None,
    ) -> None:
        self._model = model.eval()
        self._mel_spec = mel_spec
        self._mel_transform = mel_spec.transform()
        self._mean = mean
        self._std = std
        self.threshold = float(threshold)
        self.model_version = model_version
        self.preprocessing_version = preprocessing_version
        # Set only when the bundle was hash-verified at load time.
        self.bundle_hash = bundle_hash
        self._description = dict(description or {})

    @property
    def sample_rate_hz(self) -> int:
        return self._mel_spec.sample_rate_hz

    def _windows(self, waveform: np.ndarray) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
        """Per-window ``logit[spoof] - logit[bonafide]`` (float64) and the window spans."""

        if waveform.ndim != 1 or waveform.size == 0:
            raise ValueError("waveform must be a non-empty 1-D array")
        if not np.all(np.isfinite(waveform)):
            raise ValueError("waveform contains non-finite samples")
        samples = np.ascontiguousarray(waveform, dtype=np.float32)
        spec = self._mel_spec
        spans = window_spans(
            samples.size, window_samples=spec.window_samples, n_fft=spec.n_fft
        )
        scores: list[np.ndarray] = []
        with torch.inference_mode():
            spectrograms = [
                segment_to_spectrogram(
                    repeat_pad(samples[start:stop], spec.window_samples),
                    self._mel_transform,
                    self._mean,
                    self._std,
                )
                for start, stop in spans
            ]
            for offset in range(0, len(spectrograms), _WINDOW_CHUNK):
                batch = torch.stack(spectrograms[offset : offset + _WINDOW_CHUNK])
                logits = self._model(batch)
                scores.append((logits[:, 1] - logits[:, 0]).cpu().numpy())
        return np.concatenate(scores).astype(np.float64), spans

    def window_scores(self, waveform: np.ndarray) -> np.ndarray:
        """Per-window ``logit[spoof] - logit[bonafide]`` as float64."""

        return self._windows(waveform)[0]

    def score_waveform(self, waveform: np.ndarray) -> WaveformScore:
        window_scores = self.window_scores(waveform)
        return WaveformScore(
            clip_score=float(np.mean(window_scores)), n_windows=int(window_scores.size)
        )

    def score_decoded(self, waveform: np.ndarray) -> ScoredClip:
        """Score a decoded waveform and keep the per-window evidence (serving entry point)."""

        window_scores, spans = self._windows(waveform)
        result = WaveformScore(
            clip_score=float(np.mean(window_scores)), n_windows=int(window_scores.size)
        )
        return ScoredClip(
            clip=clip_score_from(result, self.threshold),
            window_scores=tuple(float(v) for v in window_scores),
            window_real_samples=tuple(stop - start for start, stop in spans),
        )

    def describe(self) -> Mapping[str, Any]:
        """Allowlisted run-record summary; never contains filesystem paths."""

        return dict(self._description)

    def score_source(self, source: AudioSource) -> ClipScore:
        """Decode, score and label one source (the entry point for #9)."""

        waveform = load_waveform(source, AudioConfig(sample_rate_hz=self.sample_rate_hz))
        return clip_score_from(self.score_waveform(waveform), self.threshold)


def load_cnn_scorer(
    bundle_dir: Path,
    *,
    expected_bundle_hash: str | None = None,
    expected_threshold: float | None = None,
) -> CnnClipScorer:
    """Load and verify a CNN bundle.

    The bundle hash is checked before any file is deserialized. Checkpoints are
    loaded with ``weights_only=True``. ``cnn_norm_stats.pt`` is authoritative
    and must equal the normalization recorded in the run record.
    """

    bundle_dir = Path(bundle_dir)
    verified_hash: str | None = None
    if expected_bundle_hash is not None:
        actual = hash_directory(bundle_dir)
        _require(
            actual == expected_bundle_hash,
            f"bundle hash mismatch for {bundle_dir.name!r}: "
            f"declared={expected_bundle_hash[:16]}…, current={actual[:16]}…",
        )
        verified_hash = actual

    record_path = bundle_dir / CNN_RUN_RECORD
    for name in (CNN_RUN_RECORD, CNN_CHECKPOINT, CNN_NORM_STATS):
        _require((bundle_dir / name).is_file(), f"bundle is missing {name}")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    _check_run_record(record)

    threshold = record.get("threshold", {}).get("value")
    _require(
        isinstance(threshold, int | float) and math.isfinite(threshold),
        "run record threshold.value is missing or not finite",
    )
    if expected_threshold is not None:
        _require(
            float(threshold) == float(expected_threshold),
            f"run record threshold {threshold!r} differs from the declared "
            f"threshold {expected_threshold!r}",
        )

    mel_spec = _mel_spec_from_record(record)

    norm = torch.load(bundle_dir / CNN_NORM_STATS, map_location="cpu", weights_only=True)
    _require(
        isinstance(norm, dict) and {"mean", "std"} <= set(norm),
        "normalization file must hold mean and std",
    )
    mean, std = norm["mean"], norm["std"]
    expected_shape = (mel_spec.n_mels,)
    _require(
        tuple(mean.shape) == expected_shape and tuple(std.shape) == expected_shape,
        f"normalization statistics must have shape {expected_shape}",
    )
    recorded = record["normalization"]
    _require(
        torch.equal(mean, torch.tensor(recorded["mean"], dtype=mean.dtype))
        and torch.equal(std, torch.tensor(recorded["std"], dtype=std.dtype)),
        "cnn_norm_stats.pt differs from the normalization in the run record",
    )

    model = CompactMelCNN(n_mels=mel_spec.n_mels, n_frames=mel_spec.n_frames)
    n_params = sum(p.numel() for p in model.parameters())
    _require(
        record["model"].get("n_params") == n_params,
        f"run record n_params {record['model'].get('n_params')!r} != {n_params}",
    )
    state = torch.load(bundle_dir / CNN_CHECKPOINT, map_location="cpu", weights_only=True)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise CnnBundleError(f"checkpoint does not match CompactMelCNN: {exc}") from exc

    return CnnClipScorer(
        model,
        mel_spec,
        mean,
        std,
        float(threshold),
        model_version=f"cnn:{record.get('run_label', bundle_dir.name)}",
        preprocessing_version=str(record["mel_config"].get("mel_version", "mel-v1")),
        bundle_hash=verified_hash,
        description=_describe_record(record, bundle_dir.name),
    )
