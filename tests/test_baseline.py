"""Tests for the MFCC + Logistic Regression baseline module (#8).

These are dataset-free: they use small synthetic feature arrays so they run
fast without audio, a manifest catalog, or a trained checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.pipeline import Pipeline

from voice_detector.baseline import (
    BASELINE_VERSION,
    TrainedCandidate,
    _aggregate_window_scores,
    _encode_labels,
    save_bundle,
    score_split,
    select_best_candidate,
    train_and_select,
    train_candidate,
)
from voice_detector.features import SplitFeatures
from voice_detector.metrics import evaluate_clip_scores
from voice_detector.scoring import SCORE_DIRECTION

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEED = 42
FEATURE_DIM = 80


def _make_split_features(
    n_clips: int,
    split: str,
    *,
    feature_dim: int = FEATURE_DIM,
    windows_per_clip: int = 2,
    label_ratio: float = 0.5,
    rng: np.random.Generator | None = None,
) -> SplitFeatures:
    """Build a small SplitFeatures with random features for testing."""

    if rng is None:
        rng = np.random.default_rng(SEED)

    n_bonafide = int(n_clips * label_ratio)
    labels = np.array(
        ["bonafide"] * n_bonafide + ["spoof"] * (n_clips - n_bonafide)
    )
    clip_ids = np.array([f"{split}_{i:04d}" for i in range(n_clips)])
    window_counts = np.full(n_clips, windows_per_clip, dtype=np.int64)
    bonafide_features = rng.normal(-1.0, 0.5, (n_bonafide * windows_per_clip, feature_dim))
    spoof_features = rng.normal(1.0, 0.5, ((n_clips - n_bonafide) * windows_per_clip, feature_dim))
    features = np.vstack([bonafide_features, spoof_features]).astype(np.float32)

    return SplitFeatures(
        split=split,
        clip_ids=clip_ids,
        labels=labels,
        window_counts=window_counts,
        features=features,
        cache_key="test-key",
        preprocessing_version="mfcc-v2",
        feature_dim=feature_dim,
        manifest_hash="deadbeef",
    )


def _make_dev_rows(
    dev_features: SplitFeatures,
    select_speakers: list[str],
    threshold_speakers: list[str],
) -> list[dict]:
    """Create manifest-style rows for dev clips with speaker assignments."""

    all_speakers = select_speakers + threshold_speakers
    rows = []
    for i, clip_id in enumerate(dev_features.clip_ids):
        speaker = all_speakers[i % len(all_speakers)]
        rows.append({
            "clip_id": str(clip_id),
            "label": str(dev_features.labels[i]),
            "official_split": "development",
            "speaker_id": speaker,
        })
    return rows


def _make_frozen_record(select_speakers: list[str], threshold_speakers: list[str]) -> dict:
    return {
        "schema_version": "dev-reservation-v1",
        "frozen_utc": "2026-10-14T00:00:00+00:00",
        "git_commit": "abc123",
        "package_version": "0.1.0",
        "spec": {
            "version": "dev-reservation-v1",
            "seed": 20261014,
            "threshold_fraction": 0.25,
            "method": "sha256-rank-v1",
            "coverage": {
                "require_attacks": ["A01"],
                "min_bonafide_clips_per_threshold_speaker": 1,
                "require_both_labels_per_role": True,
            },
        },
        "speakers": {
            "select": select_speakers,
            "threshold": threshold_speakers,
        },
        "n_speakers": {
            "select": len(select_speakers),
            "threshold": len(threshold_speakers),
        },
        "achieved_threshold_fraction": (
            len(threshold_speakers)
            / (len(select_speakers) + len(threshold_speakers))
        ),
        "manifest_hash_before": "aaa",
        "manifest_hash_after": "bbb",
        "coverage": {"failures": [], "roles": [], "per_speaker": [], "requirements": {}},
        "coverage_ok": True,
        "disclosure": "test",
    }


def _make_protocol():
    from voice_detector.pretuning import (
        AggregationRule,
        LrBudget,
        PreTuningProtocol,
        SelectionRule,
        ThresholdRule,
    )

    return PreTuningProtocol(
        version="pre-tuning-v1",
        declared_on="2026-10-14",
        metrics_version="metrics-v1",
        score_direction="higher_more_spoof",
        decision_rule="spoof if score >= threshold, compared in float64",
        selection=SelectionRule(
            development_role="select",
            metric="clip_eer",
            direction="lower_is_better",
            eer_tolerance_absolute=0.005,
            tie_break_order=("simpler_system", "lower_measured_latency"),
        ),
        lr_budget=LrBudget(
            model="logistic_regression",
            regularization_parameter="C",
            values=(0.01, 1.0, 100.0),
            class_weight="balanced",
            max_candidates=3,
        ),
        aggregation=AggregationRule(
            window_to_clip="mean_window_decision_function",
            aggregate_by="clip_id",
        ),
        threshold=ThresholdRule(
            development_role="threshold",
            rule="lowest_threshold_with_bonafide_fpr_at_most_target",
            target_fpr=0.05,
            ties_count_as="spoof",
        ),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEncodeLabels:
    def test_bonafide_and_spoof(self):
        labels = np.array(["bonafide", "spoof", "spoof", "bonafide"])
        encoded = _encode_labels(labels)
        assert list(encoded) == [0, 1, 1, 0]


class TestAggregateWindowScores:
    def test_mean_aggregation(self):
        window_scores = np.array([1.0, 3.0, 2.0, 4.0, 6.0])
        window_counts = np.array([2, 3])
        clip_scores = _aggregate_window_scores(window_scores, window_counts)
        np.testing.assert_allclose(clip_scores, [2.0, 4.0])

    def test_single_window_per_clip(self):
        window_scores = np.array([5.0, 10.0])
        window_counts = np.array([1, 1])
        clip_scores = _aggregate_window_scores(window_scores, window_counts)
        np.testing.assert_allclose(clip_scores, [5.0, 10.0])


class TestTrainCandidate:
    def test_returns_pipeline(self):
        train = _make_split_features(20, "train")
        pipeline = train_candidate(train, C=1.0, class_weight="balanced", seed=SEED)
        assert isinstance(pipeline, Pipeline)
        assert hasattr(pipeline, "predict")
        assert hasattr(pipeline, "decision_function")

    def test_deterministic(self):
        train = _make_split_features(20, "train")
        p1 = train_candidate(train, C=1.0, class_weight="balanced", seed=SEED)
        p2 = train_candidate(train, C=1.0, class_weight="balanced", seed=SEED)
        x = train.features[:5]
        np.testing.assert_array_equal(
            p1.decision_function(x), p2.decision_function(x)
        )


class TestScoreSplit:
    def test_shapes(self):
        train = _make_split_features(20, "train")
        dev = _make_split_features(10, "development")
        pipeline = train_candidate(train, C=1.0, class_weight="balanced", seed=SEED)
        clip_ids, labels, scores = score_split(pipeline, dev)
        assert len(clip_ids) == dev.n_clips
        assert len(labels) == dev.n_clips
        assert len(scores) == dev.n_clips

    def test_scores_are_finite(self):
        train = _make_split_features(20, "train")
        dev = _make_split_features(10, "development")
        pipeline = train_candidate(train, C=1.0, class_weight="balanced", seed=SEED)
        _, _, scores = score_split(pipeline, dev)
        assert np.all(np.isfinite(scores))


class TestSelectBestCandidate:
    def test_picks_lowest_eer(self):
        protocol = _make_protocol()
        c1 = TrainedCandidate(
            C=1.0, pipeline=None,
            select_metrics=evaluate_clip_scores(
                ["a", "b"], ["bonafide", "spoof"], [0.0, 1.0],
                threshold=None, score_direction=SCORE_DIRECTION,
            ),
        )
        c2 = TrainedCandidate(
            C=100.0, pipeline=None,
            select_metrics=evaluate_clip_scores(
                ["c", "d"], ["bonafide", "spoof"], [0.5, 0.5],
                threshold=None, score_direction=SCORE_DIRECTION,
            ),
        )
        best = select_best_candidate([c1, c2], protocol)
        assert best.C == 1.0

    def test_tie_break_simpler_system(self):
        protocol = _make_protocol()
        metrics = evaluate_clip_scores(
            ["a", "b"], ["bonafide", "spoof"], [0.0, 1.0],
            threshold=None, score_direction=SCORE_DIRECTION,
        )
        c1 = TrainedCandidate(C=100.0, pipeline=None, select_metrics=metrics)
        c2 = TrainedCandidate(C=0.01, pipeline=None, select_metrics=metrics)
        best = select_best_candidate([c1, c2], protocol)
        assert best.C == 0.01


class TestTrainAndSelect:
    @patch("voice_detector.baseline.validate_frozen_record")
    @patch("voice_detector.baseline.git_commit", return_value="abc123")
    @patch("voice_detector.baseline.package_version", return_value="0.1.0")
    def test_produces_valid_record(self, _pkg, _git, mock_validate):
        select_speakers = ["S01", "S02", "S03"]
        threshold_speakers = ["S04"]

        mock_validate.return_value = {
            s: "select" for s in select_speakers
        } | {s: "threshold" for s in threshold_speakers}

        rng = np.random.default_rng(SEED)
        train = _make_split_features(40, "train", rng=rng)
        dev = _make_split_features(20, "development", rng=rng)
        dev_rows = _make_dev_rows(dev, select_speakers, threshold_speakers)
        protocol = _make_protocol()
        frozen_record = _make_frozen_record(select_speakers, threshold_speakers)

        record = train_and_select(
            train, dev, dev_rows, protocol, frozen_record, seed=SEED,
        )

        assert record["schema_version"] == BASELINE_VERSION
        assert record["selected_C"] in protocol.lr_budget.values
        assert "threshold" in record
        assert record["threshold"]["target_fpr"] == 0.05
        assert "dev_full_metrics" in record
        assert "dev_select_metrics" in record
        assert "dev_threshold_metrics" in record
        assert len(record["candidates"]) == 3

        full = record["dev_full_metrics"]
        assert full["n_clips"] == dev.n_clips
        assert full["eer"] is not None
        assert full["roc_auc"] is not None
        assert full["macro_f1"] is not None
        assert full["balanced_accuracy"] is not None
        assert full["human_false_positive_rate"] is not None
        assert full["synthetic_detection_rate"] is not None
        assert full["confusion"] is not None


class TestSaveBundle:
    def test_writes_files(self, tmp_path):
        train = _make_split_features(20, "train")
        pipeline = train_candidate(train, C=1.0, class_weight="balanced", seed=SEED)
        record = {"schema_version": BASELINE_VERSION, "selected_C": 1.0}
        outputs = save_bundle(pipeline, record, tmp_path / "bundle")
        assert outputs["model"].exists()
        assert outputs["record"].exists()
        loaded = json.loads(outputs["record"].read_text())
        assert loaded["selected_C"] == 1.0


class TestTrainLrCli:
    @patch("voice_detector.baseline.validate_frozen_record")
    @patch("voice_detector.baseline.git_commit", return_value="abc123")
    @patch("voice_detector.baseline.package_version", return_value="0.1.0")
    def test_cli_end_to_end(self, _pkg, _git, mock_validate, tmp_path, capsys):
        from voice_detector.features import write_split_features

        select_speakers = ["S01", "S02", "S03"]
        threshold_speakers = ["S04"]

        mock_validate.return_value = {
            s: "select" for s in select_speakers
        } | {s: "threshold" for s in threshold_speakers}

        rng = np.random.default_rng(SEED)
        train = _make_split_features(40, "train", rng=rng)
        dev = _make_split_features(20, "development", rng=rng)

        features_dir = tmp_path / "features"
        features_dir.mkdir()
        write_split_features(train, features_dir / "features_train.npz")
        write_split_features(dev, features_dir / "features_development.npz")

        manifest_dir = tmp_path / "manifest"
        manifest_dir.mkdir()
        dev_rows = _make_dev_rows(dev, select_speakers, threshold_speakers)
        import sqlite3

        db_path = manifest_dir / "manifest.sqlite"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE clips (clip_id TEXT, label TEXT, official_split TEXT, "
            "speaker_id TEXT, development_role TEXT, checksum_sha256 TEXT)"
        )
        for row in dev_rows:
            role = "select" if row["speaker_id"] in select_speakers else "threshold"
            conn.execute(
                "INSERT INTO clips VALUES (?, ?, ?, ?, ?, ?)",
                (row["clip_id"], row["label"], "development", row["speaker_id"], role, "abc"),
            )
        for i, clip_id in enumerate(train.clip_ids):
            conn.execute(
                "INSERT INTO clips VALUES (?, ?, ?, ?, ?, ?)",
                (str(clip_id), str(train.labels[i]), "train", f"T{i:02d}", None, "abc"),
            )
        conn.execute("CREATE TABLE catalog_meta (key TEXT, value TEXT)")
        conn.execute("INSERT INTO catalog_meta VALUES ('manifest_hash', 'test')")
        conn.commit()
        conn.close()

        frozen_record = _make_frozen_record(select_speakers, threshold_speakers)
        frozen_path = tmp_path / "frozen.json"
        frozen_path.write_text(json.dumps(frozen_record))

        protocol_path = Path("configs/pre_tuning_protocol.yaml")
        config_path = Path("configs/default.yaml")

        from voice_detector.cli import main

        out_dir = tmp_path / "lr_out"
        rc = main([
            "train-lr",
            "--manifest-dir", str(manifest_dir),
            "--features-dir", str(features_dir),
            "--config", str(config_path),
            "--protocol", str(protocol_path),
            "--frozen-record", str(frozen_path),
            "--out-dir", str(out_dir),
        ])

        assert rc == 0
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["status"] == "trained"
        assert (out_dir / "lr_pipeline.joblib").exists()
        assert (out_dir / "lr_run_record.json").exists()
