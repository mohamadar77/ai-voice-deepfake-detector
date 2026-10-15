"""Dataset-free tests for CNN training loop (#33).

Uses tiny synthetic audio and a minimal frozen record so the full
train → select → threshold → save path runs in seconds without real data.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from voice_detector.cnn_dataset import MelSpec
from voice_detector.cnn_train import (
    CNN_STUDY_VERSION,
    _aggregate_window_scores,
    _class_weights,
    save_cnn_bundle,
    train_and_save_cnn,
    train_cnn,
)
from voice_detector.reservation import (
    RESERVATION_SCHEMA_VERSION,
    _assign_roles,
)


def _write_wav(path: Path, duration_s: float, sr: int = 16000) -> Path:
    samples = np.random.default_rng(42).standard_normal(int(sr * duration_s)).astype(np.float32)
    sf.write(str(path), samples, sr)
    return path


_TEST_SEED = 99
_SPEAKERS = [f"LA_T{i:03d}" for i in range(6)]
_THRESHOLD_FRACTION = 0.25


def _build_frozen_record() -> dict:
    assignment = _assign_roles(_SPEAKERS, _TEST_SEED, _THRESHOLD_FRACTION)
    select_ids = sorted(s for s, r in assignment.items() if r == "select")
    threshold_ids = sorted(s for s, r in assignment.items() if r == "threshold")
    return {
        "schema_version": RESERVATION_SCHEMA_VERSION,
        "n_speakers": len(_SPEAKERS),
        "speakers": {"select": select_ids, "threshold": threshold_ids},
        "spec": {
            "method": "sha256-rank-v1",
            "seed": _TEST_SEED,
            "threshold_fraction": _THRESHOLD_FRACTION,
        },
    }


def _make_rows(tmpdir: Path, speakers: list[str], clips_per_speaker: int = 4) -> list[dict]:
    rows = []
    for si, speaker in enumerate(speakers):
        for ci in range(clips_per_speaker):
            label = "bonafide" if ci % 2 == 0 else "spoof"
            wav = _write_wav(tmpdir / f"{speaker}_{ci}.flac", duration_s=1.0)
            rows.append({
                "clip_id": f"{speaker}_{ci}",
                "label": label,
                "speaker_id": speaker,
                "official_split": "train" if si < 3 else "development",
                "local_path": str(wav),
            })
    return rows


_STUDY_CONFIG = {
    "version": "cnn-study-v1",
    "mel": {
        "mel_version": "mel-v1",
        "sample_rate_hz": 16000,
        "n_fft": 400,
        "hop_length": 160,
        "n_mels": 32,
        "fmin": 20.0,
        "fmax": 8000.0,
        "window_seconds": 1.0,
    },
    "input_policy": {
        "short_clip_handling": "repeat_pad",
        "training_crop": "random_crop",
        "eval_windowing": "option_d_deterministic",
        "eval_aggregation": "mean_window_logit",
    },
    "training": {
        "loss": "cross_entropy",
        "class_weight": "balanced",
        "optimizer": "adam",
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "batch_size": 4,
        "max_epochs": 2,
        "patience": 2,
        "num_workers": 0,
    },
}


def test_class_weights_balanced() -> None:
    rows = [{"label": "bonafide"}] * 6 + [{"label": "spoof"}] * 4
    w = _class_weights(rows)
    assert w.shape == (2,)
    assert w[0] < w[1]


def test_class_weights_equal() -> None:
    rows = [{"label": "bonafide"}] * 5 + [{"label": "spoof"}] * 5
    w = _class_weights(rows)
    torch.testing.assert_close(w[0], w[1])


def test_aggregate_window_scores() -> None:
    clip_ids = ["a", "a", "b", "b", "b"]
    scores = [0.2, 0.4, 0.6, 0.8, 1.0]
    agg_ids, agg_scores = _aggregate_window_scores(clip_ids, scores)
    assert agg_ids == ["a", "b"]
    np.testing.assert_allclose(agg_scores, [0.3, 0.8])


def test_aggregate_window_scores_preserves_order() -> None:
    clip_ids = ["c", "a", "c"]
    scores = [1.0, 2.0, 3.0]
    agg_ids, _ = _aggregate_window_scores(clip_ids, scores)
    assert agg_ids == ["c", "a"]


def test_train_cnn_returns_tuple() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        frozen = _build_frozen_record()
        rows = _make_rows(tmpdir, _SPEAKERS, clips_per_speaker=4)
        train_rows = [r for r in rows if r["official_split"] == "train"]
        dev_rows = [r for r in rows if r["official_split"] == "development"]

        result = train_cnn(
            train_rows, dev_rows, frozen, _STUDY_CONFIG,
            seed=42, run_label="test",
        )
        assert isinstance(result, tuple)
        assert len(result) == 3

        run_record, best_state, norm_stats = result
        assert run_record["schema_version"] == CNN_STUDY_VERSION
        assert run_record["run_label"] == "test"
        assert run_record["best_epoch"] >= 1
        assert isinstance(best_state, dict)
        assert len(best_state) > 0
        mean, std = norm_stats
        assert mean.shape == (32,)
        assert std.shape == (32,)


def test_train_cnn_run_record_fields() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        frozen = _build_frozen_record()
        rows = _make_rows(tmpdir, _SPEAKERS, clips_per_speaker=4)
        train_rows = [r for r in rows if r["official_split"] == "train"]
        dev_rows = [r for r in rows if r["official_split"] == "development"]

        run_record, _, _ = train_cnn(
            train_rows, dev_rows, frozen, _STUDY_CONFIG,
            seed=42, run_label="test",
        )

        for key in (
            "schema_version", "run_label", "seed", "mel_config", "model",
            "training_config", "input_policy", "normalization",
            "best_epoch", "total_epochs", "epochs", "threshold",
            "dev_full_metrics", "dev_select_metrics", "dev_threshold_metrics",
            "total_seconds", "score_direction",
        ):
            assert key in run_record, f"missing key: {key}"

        assert run_record["score_direction"] == "higher_more_spoof"
        assert run_record["threshold"]["target_fpr"] == 0.05


def test_save_cnn_bundle() -> None:
    from voice_detector.cnn_pilot import CompactMelCNN

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        spec = MelSpec(n_mels=32, window_seconds=1.0)
        model = CompactMelCNN(n_mels=32, n_frames=spec.n_frames)
        mean = torch.zeros(32)
        std = torch.ones(32)
        record = {"test": True}

        outputs = save_cnn_bundle(model, record, (mean, std), tmpdir)
        assert outputs["model"].exists()
        assert outputs["record"].exists()
        assert outputs["norm"].exists()
        loaded_record = json.loads(outputs["record"].read_text())
        assert loaded_record == {"test": True}
        loaded_norm = torch.load(outputs["norm"], weights_only=True)
        torch.testing.assert_close(loaded_norm["mean"], mean)
        torch.testing.assert_close(loaded_norm["std"], std)


def test_train_and_save_cnn() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        frozen = _build_frozen_record()
        rows = _make_rows(tmpdir, _SPEAKERS, clips_per_speaker=4)
        train_rows = [r for r in rows if r["official_split"] == "train"]
        dev_rows = [r for r in rows if r["official_split"] == "development"]
        out_dir = tmpdir / "output"

        run_record, outputs = train_and_save_cnn(
            train_rows, dev_rows, frozen, _STUDY_CONFIG,
            seed=42, run_label="test", out_dir=out_dir,
        )

        assert run_record["schema_version"] == CNN_STUDY_VERSION
        assert outputs["model"].exists()
        assert outputs["record"].exists()
        assert outputs["norm"].exists()

        saved_record = json.loads(outputs["record"].read_text())
        assert saved_record["run_label"] == "test"
