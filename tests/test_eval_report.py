"""Tests for eval_report (issue #35)."""

from __future__ import annotations

from pathlib import Path

from voice_detector.eval_report import (
    compute_report_hash,
    compute_slice_metrics,
    generate_report,
)
from voice_detector.evaluation import (
    EVALUATION_SCHEMA_VERSION,
    BootstrapSpec,
    CandidateDeclaration,
    FreezeRecord,
    FreezeState,
    SliceDefinition,
)
from voice_detector.metrics import METRICS_VERSION
from voice_detector.prediction_store import PredictionRow
from voice_detector.scoring import DECISION_RULE, SCORE_DIRECTION


def _pred(
    clip_id: str,
    label: str,
    score: float,
    *,
    candidate_id: str = "lr_C1.0",
    condition: str = "clean",
    speaker_id: str = "SPK1",
    attack_id: str | None = None,
    official_split: str = "evaluation",
    duration_seconds: float = 3.0,
) -> PredictionRow:
    return PredictionRow(
        clip_id=clip_id,
        candidate_id=candidate_id,
        condition=condition,
        clip_score=score,
        label=label,
        speaker_id=speaker_id,
        attack_id=attack_id,
        official_split=official_split,
        duration_seconds=duration_seconds,
        n_windows=2,
        threshold=0.5,
        decision=score >= 0.5,
    )


def _freeze() -> FreezeRecord:
    return FreezeRecord(
        schema_version=EVALUATION_SCHEMA_VERSION,
        state=FreezeState.FROZEN.value,
        created_utc="2026-10-15T00:00:00+00:00",
        frozen_utc="2026-10-15T00:01:00+00:00",
        scored_utc=None,
        reported_utc=None,
        candidates=[CandidateDeclaration(
            candidate_id="lr_C1.0",
            model_family="logistic_regression",
            bundle_hash="abc",
            threshold=0.5,
            seed=20261004,
            run_label="C1.0",
            aggregation_rule="mean_window_decision_function",
        )],
        slices=[
            SliceDefinition(
                slice_id="attack_A07",
                description="A07 unseen",
                clip_filter={"attack_id": "A07"},
                bonafide_pool="evaluation_all",
                partition="evaluation",
            ),
            SliceDefinition(
                slice_id="duration_lt_1s",
                description="Duration <1s",
                clip_filter={
                    "duration_lower": 0,
                    "duration_upper": 1,
                    "duration_upper_inclusive": False,
                },
                bonafide_pool="same_partition",
                partition="all",
            ),
        ],
        conditions=["clean"],
        manifest_hash="deadbeef",
        config_hashes={},
        bootstrap=BootstrapSpec(
            confidence_level=0.95,
            n_resamples=100,
            seed=42,
            resampling_unit="speaker_id",
            method="percentile",
        ),
        metrics_version=METRICS_VERSION,
        score_direction=SCORE_DIRECTION,
        decision_rule=DECISION_RULE,
        git_commit="abc",
        package_version="0.1.0",
    )


class TestSliceMetrics:
    def test_with_both_classes(self):
        preds = [
            _pred("c1", "bonafide", 0.1),
            _pred("c2", "bonafide", 0.2),
            _pred("c3", "spoof", 0.8, attack_id="A07"),
            _pred("c4", "spoof", 0.9, attack_id="A07"),
        ]
        result = compute_slice_metrics(
            preds, preds,
            "overall", {}, "same_partition", "evaluation",
            threshold=0.5,
            score_direction=SCORE_DIRECTION,
        )
        assert result["n_clips"] == 4
        metrics = result["metrics"]
        assert metrics is not None
        assert metrics["eer"] is not None
        assert metrics["roc_auc"] is not None

    def test_attack_slice_with_bonafide_pool(self):
        preds = [
            _pred("c1", "bonafide", 0.1),
            _pred("c2", "bonafide", 0.2),
            _pred("c3", "spoof", 0.8, attack_id="A07"),
            _pred("c4", "spoof", 0.9, attack_id="A08"),
        ]
        result = compute_slice_metrics(
            preds, preds,
            "attack_A07",
            {"attack_id": "A07"},
            "evaluation_all",
            "evaluation",
            threshold=0.5,
            score_direction=SCORE_DIRECTION,
        )
        assert result["n_clips"] == 3
        assert result["metrics"]["eer"] is not None

    def test_duration_bin_assignment(self):
        preds = [
            _pred("c1", "bonafide", 0.1, duration_seconds=0.5),
            _pred("c2", "spoof", 0.9, duration_seconds=0.8, attack_id="A07"),
            _pred("c3", "bonafide", 0.2, duration_seconds=2.0),
            _pred("c4", "spoof", 0.7, duration_seconds=5.0, attack_id="A07"),
        ]
        result_lt1 = compute_slice_metrics(
            preds, preds,
            "duration_lt_1s",
            {"duration_lower": 0, "duration_upper": 1, "duration_upper_inclusive": False},
            "same_partition", "all",
            threshold=0.5,
            score_direction=SCORE_DIRECTION,
        )
        assert result_lt1["n_clips"] == 2

        result_2_4 = compute_slice_metrics(
            preds, preds,
            "duration_2s_to_4s",
            {"duration_lower": 2, "duration_upper": 4, "duration_upper_inclusive": False},
            "same_partition", "all",
            threshold=0.5,
            score_direction=SCORE_DIRECTION,
        )
        assert result_2_4["n_clips"] == 1

    def test_empty_slice(self):
        preds = [_pred("c1", "bonafide", 0.1, duration_seconds=5.0)]
        result = compute_slice_metrics(
            preds, preds,
            "duration_lt_1s",
            {"duration_lower": 0, "duration_upper": 1, "duration_upper_inclusive": False},
            "same_partition", "all",
            threshold=0.5,
            score_direction=SCORE_DIRECTION,
        )
        assert result["n_clips"] == 0
        assert result["metrics"] is None


class TestGenerateReport:
    def test_produces_json(self, tmp_path: Path):
        frozen = _freeze()
        preds = [
            _pred("c1", "bonafide", 0.1),
            _pred("c2", "spoof", 0.9, attack_id="A07"),
        ]
        report = generate_report(frozen, preds, output_dir=tmp_path)

        assert report["report_version"] == "report-v1"
        assert len(report["results"]) == 1
        assert report["results"][0]["candidate_id"] == "lr_C1.0"
        assert (tmp_path / "evaluation_report.json").exists()

    def test_report_hash_is_deterministic(self, tmp_path: Path):
        frozen = _freeze()
        preds = [
            _pred("c1", "bonafide", 0.1),
            _pred("c2", "spoof", 0.9, attack_id="A07"),
        ]
        dir1 = tmp_path / "r1"
        dir2 = tmp_path / "r2"
        generate_report(frozen, preds, output_dir=dir1)
        generate_report(frozen, preds, output_dir=dir2)

        h1 = compute_report_hash(dir1 / "evaluation_report.json")
        h2 = compute_report_hash(dir2 / "evaluation_report.json")
        assert h1 == h2
