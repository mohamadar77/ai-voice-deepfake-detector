"""Tests for the CNN clip scorer used by eval-score (#35).

Uses tiny random-weight bundles built with the production ``save_cnn_bundle``;
no datasets and no trained checkpoints. Requires torch and torchaudio.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchaudio")
sf = pytest.importorskip("soundfile")

from voice_detector.cnn_dataset import (  # noqa: E402
    EvalWindowDataset,
    MelSpec,
    repeat_pad,
    segment_to_spectrogram,
)
from voice_detector.cnn_pilot import CompactMelCNN  # noqa: E402
from voice_detector.cnn_scoring import (  # noqa: E402
    CNN_SCHEMA_VERSION,
    CnnBundleError,
    load_cnn_scorer,
)
from voice_detector.cnn_train import (  # noqa: E402
    CNN_STUDY_VERSION,
    _score_eval_dataset,
    save_cnn_bundle,
)
from voice_detector.evaluation import hash_directory  # noqa: E402
from voice_detector.metrics import METRICS_VERSION  # noqa: E402
from voice_detector.preprocessing import AudioConfig, load_waveform, window_spans  # noqa: E402
from voice_detector.scoring import (  # noqa: E402
    CNN_AGGREGATION_RULE,
    DECISION_RULE,
    SCORE_DIRECTION,
)

_MEL = MelSpec(n_mels=16, window_seconds=1.0)
_WINDOW = _MEL.window_samples
_THRESHOLD = 0.125


def _make_model(seed: int = 0) -> CompactMelCNN:
    torch.manual_seed(seed)
    model = CompactMelCNN(n_mels=_MEL.n_mels, n_frames=_MEL.n_frames)
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.running_mean.normal_()
            module.running_var.uniform_(0.5, 2.0)
    return model.eval()


def _run_record(model: CompactMelCNN, mean: torch.Tensor, std: torch.Tensor) -> dict[str, Any]:
    return {
        "schema_version": CNN_STUDY_VERSION,
        "run_label": "unit",
        "seed": 1,
        "metrics_version": METRICS_VERSION,
        "score_direction": SCORE_DIRECTION,
        "mel_config": {
            "mel_version": "mel-v1",
            "sample_rate_hz": _MEL.sample_rate_hz,
            "n_fft": _MEL.n_fft,
            "hop_length": _MEL.hop_length,
            "n_mels": _MEL.n_mels,
            "fmin": _MEL.fmin,
            "fmax": _MEL.fmax,
            "window_seconds": _MEL.window_seconds,
        },
        "model": {"class": "CompactMelCNN", "n_params": sum(p.numel() for p in model.parameters())},
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
        "threshold": {"value": _THRESHOLD, "decision_rule": DECISION_RULE},
    }


@pytest.fixture
def bundle(tmp_path: Path) -> tuple[Path, CompactMelCNN, tuple[torch.Tensor, torch.Tensor]]:
    model = _make_model()
    generator = torch.Generator().manual_seed(3)
    mean = torch.randn(_MEL.n_mels, generator=generator)
    std = torch.rand(_MEL.n_mels, generator=generator) + 0.5
    directory = tmp_path / "cnn_unit"
    save_cnn_bundle(model, _run_record(model, mean, std), (mean, std), directory)
    return directory, model, (mean, std)


def _wave(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(n).astype(np.float32) * 0.1


_LENGTHS = [100, 4800, _WINDOW, _WINDOW + 399, _WINDOW + 400, int(2.5 * _WINDOW)]


def test_schema_constant_matches_training():
    assert CNN_SCHEMA_VERSION == CNN_STUDY_VERSION


def test_clip_scores_match_training_scoring_path(bundle, tmp_path: Path):
    directory, model, (mean, std) = bundle
    scorer = load_cnn_scorer(directory)
    rows = []
    for i, n in enumerate(_LENGTHS):
        path = tmp_path / f"clip_{i}.wav"
        sf.write(str(path), _wave(n, seed=i), 16000)
        rows.append({"clip_id": f"c{i}", "label": "bonafide", "local_path": str(path)})

    dataset = EvalWindowDataset(rows, _MEL, mean=mean, std=std)
    ids, _, expected = _score_eval_dataset(model, dataset, batch_size=4)

    audio = AudioConfig(sample_rate_hz=16000)
    for clip_id, want, row in zip(ids, expected, rows, strict=True):
        got = scorer.score_waveform(load_waveform(Path(row["local_path"]), audio))
        assert got.clip_score == pytest.approx(want, abs=1e-5), clip_id
        spans = window_spans(
            load_waveform(Path(row["local_path"]), audio).size,
            window_samples=_WINDOW,
            n_fft=_MEL.n_fft,
        )
        assert got.n_windows == len(spans)


def test_clip_score_is_float64_mean_of_window_scores(bundle):
    scorer = load_cnn_scorer(bundle[0])
    wave = _wave(int(2.5 * _WINDOW))
    windows = scorer.window_scores(wave)
    assert windows.dtype == np.float64
    assert scorer.score_waveform(wave).clip_score == float(np.mean(windows))


def test_folded_tail_is_dropped_like_training(bundle):
    scorer = load_cnn_scorer(bundle[0])
    wave = _wave(_WINDOW + 100)
    assert scorer.score_waveform(wave) == scorer.score_waveform(wave[:_WINDOW])


def test_score_direction_is_spoof_minus_bonafide(bundle):
    directory, model, (mean, std) = bundle
    with torch.no_grad():
        for p in model.classifier.parameters():
            p.zero_()
        model.classifier.bias.copy_(torch.tensor([0.0, 2.5]))
    save_cnn_bundle(model, _run_record(model, mean, std), (mean, std), directory)
    scorer = load_cnn_scorer(directory)
    assert scorer.score_waveform(_wave(_WINDOW)).clip_score == pytest.approx(2.5)


def test_scoring_is_repeatable_and_does_not_mutate_input(bundle):
    scorer = load_cnn_scorer(bundle[0])
    wave = _wave(_WINDOW + 5000)
    wave.flags.writeable = False
    first = scorer.score_waveform(wave)
    assert scorer.score_waveform(wave) == first


def test_clip_score_does_not_depend_on_other_clips(bundle):
    scorer = load_cnn_scorer(bundle[0])
    a, b = _wave(3 * _WINDOW, 1), _wave(_WINDOW, 2)
    alone = scorer.score_waveform(a)
    scorer.score_waveform(b)
    assert scorer.score_waveform(a) == alone


@pytest.mark.parametrize(
    "bad",
    [np.zeros(0, dtype=np.float32), np.zeros((2, 10), dtype=np.float32),
     np.array([0.0, np.nan, 0.1], dtype=np.float32)],
)
def test_invalid_waveform_is_rejected(bundle, bad):
    with pytest.raises(ValueError):
        load_cnn_scorer(bundle[0]).score_waveform(bad)


def test_float64_input_is_coerced(bundle):
    scorer = load_cnn_scorer(bundle[0])
    wave = _wave(_WINDOW)
    assert scorer.score_waveform(wave.astype(np.float64)) == scorer.score_waveform(wave)


def test_segment_to_spectrogram_matches_manual_ops():
    mel = _MEL.transform()
    mean, std = torch.randn(_MEL.n_mels), torch.rand(_MEL.n_mels) + 0.5
    seg = repeat_pad(_wave(5000), _WINDOW)
    expected = torch.log(mel(torch.from_numpy(seg)).clamp(min=1e-9))
    expected = (expected - mean.unsqueeze(-1)) / std.unsqueeze(-1).clamp(min=1e-6)
    out = segment_to_spectrogram(seg, mel, mean, std)
    assert out.shape == (1, _MEL.n_mels, _MEL.n_frames)
    assert torch.equal(out[0], expected)


# --- loader integrity -------------------------------------------------------


def test_loader_accepts_matching_hash_and_threshold(bundle):
    directory = bundle[0]
    scorer = load_cnn_scorer(
        directory, expected_bundle_hash=hash_directory(directory), expected_threshold=_THRESHOLD
    )
    assert scorer.threshold == _THRESHOLD
    assert scorer.model_family == "cnn"
    assert scorer.aggregation_rule == CNN_AGGREGATION_RULE


def test_wrong_bundle_hash_fails_before_deserializing(bundle, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("torch.load must not run")

    monkeypatch.setattr(torch, "load", boom)
    with pytest.raises(CnnBundleError, match="hash"):
        load_cnn_scorer(bundle[0], expected_bundle_hash="0" * 64)


def test_threshold_mismatch_is_refused(bundle):
    with pytest.raises(CnnBundleError, match="threshold"):
        load_cnn_scorer(bundle[0], expected_threshold=_THRESHOLD + 1e-9)


def test_loads_use_weights_only(bundle, monkeypatch):
    seen: list[bool] = []
    real = torch.load

    def spy(*args, **kwargs):
        seen.append(kwargs.get("weights_only"))
        return real(*args, **kwargs)

    monkeypatch.setattr(torch, "load", spy)
    load_cnn_scorer(bundle[0])
    assert seen == [True, True]


def test_pickled_objects_are_rejected(bundle):
    directory = bundle[0]

    class Payload:
        pass

    torch.save(Payload(), directory / "cnn_checkpoint.pt")
    with pytest.raises(Exception, match="(?i)weights|unpickl|Payload"):
        load_cnn_scorer(directory)


def test_norm_stats_must_match_run_record(bundle):
    directory = bundle[0]
    mean, std = bundle[2]
    torch.save({"mean": mean + 1.0, "std": std}, directory / "cnn_norm_stats.pt")
    with pytest.raises(CnnBundleError, match="normalization"):
        load_cnn_scorer(directory)


def test_norm_stats_shape_is_checked(bundle):
    directory = bundle[0]
    torch.save(
        {"mean": torch.zeros(3), "std": torch.ones(3)}, directory / "cnn_norm_stats.pt"
    )
    with pytest.raises(CnnBundleError, match="shape"):
        load_cnn_scorer(directory)


def test_strict_state_dict_mismatch_is_refused(bundle):
    directory = bundle[0]
    state = torch.load(directory / "cnn_checkpoint.pt", weights_only=True)
    state.pop("classifier.bias")
    torch.save(state, directory / "cnn_checkpoint.pt")
    with pytest.raises(CnnBundleError, match="checkpoint"):
        load_cnn_scorer(directory)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.update(schema_version="cnn-study-v0"),
        lambda r: r["input_policy"].update(eval_aggregation="max_window"),
        lambda r: r["input_policy"].update(eval_windowing="random_crop"),
        lambda r: r.update(score_direction="higher_more_human"),
        lambda r: r["model"].update(n_params=1),
    ],
)
def test_inconsistent_run_record_is_refused(bundle, mutate):
    import json

    directory = bundle[0]
    path = directory / "cnn_run_record.json"
    record = json.loads(path.read_text())
    mutate(record)
    path.write_text(json.dumps(record))
    with pytest.raises(CnnBundleError):
        load_cnn_scorer(directory)
