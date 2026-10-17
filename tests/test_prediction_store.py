"""Tests for the prediction store (issue #35)."""

from __future__ import annotations

from pathlib import Path

import pytest

from voice_detector.prediction_store import (
    PredictionRow,
    compute_predictions_hash,
    create_prediction_store,
    read_predictions,
    validate_prediction_completeness,
    write_predictions,
)


def _row(
    clip_id: str = "clip1",
    candidate_id: str = "lr_C1.0",
    condition: str = "clean",
    score: float = 0.8,
    label: str = "spoof",
) -> PredictionRow:
    return PredictionRow(
        clip_id=clip_id,
        candidate_id=candidate_id,
        condition=condition,
        clip_score=score,
        label=label,
        speaker_id="SPK1",
        attack_id="A07",
        official_split="evaluation",
        duration_seconds=3.5,
        n_windows=2,
        threshold=0.5,
        decision=score >= 0.5,
    )


class TestWriteAndRead:
    def test_round_trip(self, tmp_path: Path):
        store = create_prediction_store(tmp_path / "predictions.db")
        rows = [_row("clip1"), _row("clip2", score=0.3, label="bonafide")]
        written = write_predictions(store, rows, freeze_record_hash="abc")
        assert written == 2

        read_back = read_predictions(store)
        assert len(read_back) == 2
        assert {r.clip_id for r in read_back} == {"clip1", "clip2"}

    def test_refuses_duplicate_candidate_condition(self, tmp_path: Path):
        store = create_prediction_store(tmp_path / "predictions.db")
        rows = [_row("clip1")]
        write_predictions(store, rows, freeze_record_hash="abc")

        with pytest.raises(ValueError, match="already exist"):
            write_predictions(store, [_row("clip2")], freeze_record_hash="abc")

    def test_write_records_freeze_hash(self, tmp_path: Path):
        import sqlite3

        store = create_prediction_store(tmp_path / "predictions.db")
        write_predictions(store, [_row()], freeze_record_hash="myhash")

        conn = sqlite3.connect(store)
        cursor = conn.execute(
            "SELECT value FROM metadata WHERE key = 'freeze_record_hash'"
        )
        assert cursor.fetchone()[0] == "myhash"
        conn.close()


class TestFilters:
    def test_read_filters_by_candidate_and_condition(self, tmp_path: Path):
        store = create_prediction_store(tmp_path / "predictions.db")
        rows_a = [_row("clip1", candidate_id="a", condition="clean")]
        rows_b = [_row("clip1", candidate_id="b", condition="noise_20db")]
        write_predictions(store, rows_a, freeze_record_hash="h1")
        write_predictions(store, rows_b, freeze_record_hash="h1")

        result = read_predictions(store, candidate_id="a")
        assert len(result) == 1
        assert result[0].candidate_id == "a"

        result = read_predictions(store, condition="noise_20db")
        assert len(result) == 1
        assert result[0].condition == "noise_20db"


class TestCompleteness:
    def test_detects_missing_clips(self, tmp_path: Path):
        store = create_prediction_store(tmp_path / "predictions.db")
        write_predictions(store, [_row("clip1")], freeze_record_hash="h")

        missing = validate_prediction_completeness(
            store,
            expected_clip_ids=["clip1", "clip2"],
            candidate_ids=["lr_C1.0"],
            conditions=["clean"],
        )
        assert len(missing) == 1
        assert "clip2" in missing[0]


class TestHash:
    def test_predictions_hash_is_deterministic(self, tmp_path: Path):
        store1 = create_prediction_store(tmp_path / "s1.db")
        store2 = create_prediction_store(tmp_path / "s2.db")
        rows = [_row("clip1"), _row("clip2", score=0.3, label="bonafide")]
        write_predictions(store1, rows, freeze_record_hash="h")
        write_predictions(store2, rows, freeze_record_hash="h")

        assert compute_predictions_hash(store1) == compute_predictions_hash(store2)

    def test_predictions_hash_changes_on_content_change(self, tmp_path: Path):
        store1 = create_prediction_store(tmp_path / "s1.db")
        store2 = create_prediction_store(tmp_path / "s2.db")
        write_predictions(store1, [_row("clip1", score=0.8)], freeze_record_hash="h")
        write_predictions(store2, [_row("clip1", score=0.9)], freeze_record_hash="h")

        assert compute_predictions_hash(store1) != compute_predictions_hash(store2)
