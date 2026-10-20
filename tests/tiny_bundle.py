"""A tiny random-weight CNN bundle for torch-backed serving tests (no real weights).

Mirrors the fixture in ``test_cnn_scoring.py``: 1 s windows and 16 mel bands, so the
bundle loads and scores in milliseconds. Tests may import training code; serving
code may not.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

THRESHOLD = 0.125
RUN_LABEL = "unit"


def make_bundle(directory: Path) -> Path:
    """Write a hash-verifiable bundle to ``directory`` and return it."""

    import torch

    from voice_detector.cnn_dataset import MelSpec
    from voice_detector.cnn_pilot import CompactMelCNN
    from voice_detector.cnn_train import CNN_STUDY_VERSION, save_cnn_bundle
    from voice_detector.metrics import METRICS_VERSION
    from voice_detector.scoring import CNN_AGGREGATION_RULE, DECISION_RULE, SCORE_DIRECTION

    mel = MelSpec(n_mels=16, window_seconds=1.0)
    torch.manual_seed(0)
    model = CompactMelCNN(n_mels=mel.n_mels, n_frames=mel.n_frames)
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.running_mean.normal_()
            module.running_var.uniform_(0.5, 2.0)
    model.eval()
    generator = torch.Generator().manual_seed(3)
    mean = torch.randn(mel.n_mels, generator=generator)
    std = torch.rand(mel.n_mels, generator=generator) + 0.5

    record: dict[str, Any] = {
        "schema_version": CNN_STUDY_VERSION,
        "run_label": RUN_LABEL,
        "seed": 1,
        "metrics_version": METRICS_VERSION,
        "score_direction": SCORE_DIRECTION,
        "mel_config": {
            "mel_version": "mel-v1",
            "sample_rate_hz": mel.sample_rate_hz,
            "n_fft": mel.n_fft,
            "hop_length": mel.hop_length,
            "n_mels": mel.n_mels,
            "fmin": mel.fmin,
            "fmax": mel.fmax,
            "window_seconds": mel.window_seconds,
        },
        "model": {
            "class": "CompactMelCNN",
            "n_params": sum(p.numel() for p in model.parameters()),
        },
        "input_policy": {
            "short_clip_handling": "repeat_pad",
            "eval_windowing": "option_d_deterministic",
            "eval_aggregation": CNN_AGGREGATION_RULE,
        },
        "normalization": {
            "method": "per_channel_standardize",
            "mean": mean.tolist(),
            "std": std.tolist(),
        },
        "threshold": {
            "value": THRESHOLD,
            "decision_rule": DECISION_RULE,
            "target_fpr": 0.05,
        },
    }
    save_cnn_bundle(model, record, (mean, std), directory)
    return directory
