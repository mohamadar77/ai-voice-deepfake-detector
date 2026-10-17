"""Tests for the freeze record lifecycle (issue #35)."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import manifest_row

from voice_detector.evaluation import (
    EVALUATION_SCHEMA_VERSION,
    BootstrapSpec,
    CandidateDeclaration,
    FreezeRecord,
    FreezeState,
    SliceDefinition,
    freeze_record,
    load_freeze_record,
    mark_reported,
    mark_scored,
    save_freeze_record,
    validate_freeze_record,
    verify_freeze_against_artifacts,
)
from voice_detector.metrics import METRICS_VERSION
from voice_detector.scoring import DECISION_RULE, SCORE_DIRECTION


def _candidate(candidate_id: str = "lr_C1.0", bundle_hash: str = "abc123") -> CandidateDeclaration:
    return CandidateDeclaration(
        candidate_id=candidate_id,
        model_family="logistic_regression",
        bundle_hash=bundle_hash,
        threshold=0.5,
        seed=20261004,
        run_label="C1.0",
        aggregation_rule="mean_window_decision_function",
    )


def _bootstrap() -> BootstrapSpec:
    return BootstrapSpec(
        confidence_level=0.95,
        n_resamples=100,
        seed=20261015,
        resampling_unit="speaker_id",
        method="percentile",
    )


def _draft_record(candidates: list[CandidateDeclaration] | None = None) -> FreezeRecord:
    return FreezeRecord(
        schema_version=EVALUATION_SCHEMA_VERSION,
        state=FreezeState.DRAFT.value,
        created_utc="2026-10-15T00:00:00+00:00",
        frozen_utc=None,
        scored_utc=None,
        reported_utc=None,
        candidates=candidates if candidates is not None else [_candidate()],
        slices=[SliceDefinition(
            slice_id="test_slice",
            description="test",
            clip_filter={"attack_id": "A07"},
            bonafide_pool="evaluation_all",
            partition="evaluation",
        )],
        conditions=["clean"],
        manifest_hash="deadbeef",
        config_hashes={"robustness_spec.yaml": "cafecafe"},
        bootstrap=_bootstrap(),
        metrics_version=METRICS_VERSION,
        score_direction=SCORE_DIRECTION,
        decision_rule=DECISION_RULE,
        git_commit="abc123",
        package_version="0.1.0",
    )


class TestStateTransitions:
    def test_draft_to_frozen_succeeds(self):
        draft = _draft_record()
        frozen = freeze_record(draft)
        assert frozen.state == FreezeState.FROZEN.value
        assert frozen.frozen_utc is not None

    def test_freeze_rejects_non_draft(self):
        draft = _draft_record()
        frozen = freeze_record(draft)
        with pytest.raises(ValueError, match="DRAFT"):
            freeze_record(frozen)

    def test_freeze_rejects_empty_candidates(self):
        draft = _draft_record(candidates=[])
        with pytest.raises(ValueError, match="no candidates"):
            freeze_record(draft)

    def test_scored_rejects_non_frozen(self):
        draft = _draft_record()
        with pytest.raises(ValueError, match="FROZEN"):
            mark_scored(draft, "hash123")

    def test_reported_rejects_non_scored(self):
        draft = _draft_record()
        frozen = freeze_record(draft)
        with pytest.raises(ValueError, match="SCORED"):
            mark_reported(frozen, "hash456")

    def test_full_lifecycle(self):
        draft = _draft_record()
        frozen = freeze_record(draft)
        scored = mark_scored(frozen, "pred_hash")
        reported = mark_reported(scored, "report_hash")

        assert reported.state == FreezeState.REPORTED.value
        assert reported.predictions_hash == "pred_hash"
        assert reported.report_hash == "report_hash"
        assert reported.scored_utc is not None
        assert reported.reported_utc is not None


class TestConventionValidation:
    def test_freeze_rejects_wrong_metrics_version(self):
        draft = FreezeRecord(
            **{**_draft_record().as_dict(),
               "metrics_version": "metrics-v99",
               "candidates": _draft_record().candidates,
               "slices": _draft_record().slices,
               "bootstrap": _draft_record().bootstrap}
        )
        with pytest.raises(ValueError, match="metrics_version mismatch"):
            freeze_record(draft)

    def test_freeze_rejects_wrong_score_direction(self):
        draft = FreezeRecord(
            **{**_draft_record().as_dict(),
               "score_direction": "lower_more_spoof",
               "candidates": _draft_record().candidates,
               "slices": _draft_record().slices,
               "bootstrap": _draft_record().bootstrap}
        )
        with pytest.raises(ValueError, match="score_direction mismatch"):
            freeze_record(draft)


class TestPersistence:
    def test_record_round_trip(self, tmp_path: Path):
        draft = _draft_record()
        frozen = freeze_record(draft)
        path = tmp_path / "freeze.json"
        save_freeze_record(frozen, path)
        loaded = load_freeze_record(path)

        assert loaded.state == frozen.state
        assert loaded.manifest_hash == frozen.manifest_hash
        assert loaded.candidates[0].candidate_id == frozen.candidates[0].candidate_id
        assert loaded.bootstrap.n_resamples == frozen.bootstrap.n_resamples

    def test_save_allows_forward_transition(self, tmp_path: Path):
        draft = _draft_record()
        frozen = freeze_record(draft)
        path = tmp_path / "freeze.json"
        save_freeze_record(frozen, path)

        scored = mark_scored(frozen, "hash")
        save_freeze_record(scored, path)

        loaded = load_freeze_record(path)
        assert loaded.state == FreezeState.SCORED.value

    def test_save_refuses_backward_transition(self, tmp_path: Path):
        draft = _draft_record()
        frozen = freeze_record(draft)
        scored = mark_scored(frozen, "hash")
        path = tmp_path / "freeze.json"
        save_freeze_record(scored, path)

        with pytest.raises(ValueError, match="refusing to overwrite"):
            save_freeze_record(frozen, path)

    def test_save_allows_idempotent_rewrite(self, tmp_path: Path):
        draft = _draft_record()
        frozen = freeze_record(draft)
        path = tmp_path / "freeze.json"
        save_freeze_record(frozen, path)
        save_freeze_record(frozen, path)

    def test_load_rejects_corrupt_record(self, tmp_path: Path):
        path = tmp_path / "bad.json"
        path.write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError):
            load_freeze_record(path)


class TestValidateRecord:
    def test_rejects_wrong_schema(self):
        raw = _draft_record().as_dict()
        raw["schema_version"] = "wrong-v1"
        with pytest.raises(ValueError, match="schema_version"):
            validate_freeze_record(raw)

    def test_rejects_invalid_state(self):
        raw = _draft_record().as_dict()
        raw["state"] = "INVALID"
        with pytest.raises(ValueError, match="state"):
            validate_freeze_record(raw)


class TestVerifyAgainstArtifacts:
    def test_detects_manifest_change(self, tmp_path: Path):
        draft = _draft_record()
        frozen = freeze_record(draft)
        rows = [manifest_row("clip1", speaker_id="SPK1", official_split="evaluation")]
        with pytest.raises(ValueError, match="manifest hash mismatch"):
            verify_freeze_against_artifacts(frozen, rows, tmp_path, tmp_path)

    def test_detects_config_change(self, tmp_path: Path):
        from voice_detector.manifest import compute_manifest_hash

        rows = [manifest_row("clip1", speaker_id="SPK1", official_split="evaluation")]
        real_hash = compute_manifest_hash(rows)

        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        (config_dir / "robustness_spec.yaml").write_text("v1", encoding="utf-8")

        draft = FreezeRecord(
            **{**_draft_record().as_dict(),
               "manifest_hash": real_hash,
               "config_hashes": {"robustness_spec.yaml": "wronghash"},
               "candidates": [],
               "slices": _draft_record().slices,
               "bootstrap": _draft_record().bootstrap}
        )

        with pytest.raises(ValueError, match="hash mismatch"):
            verify_freeze_against_artifacts(draft, rows, config_dir, tmp_path)


class TestSliceDefinitions:
    def test_attack_groups_are_disjoint(self):
        import yaml

        spec_path = Path("configs/evaluation_protocol.yaml")
        if not spec_path.exists():
            pytest.skip("evaluation_protocol.yaml not found")
        with open(spec_path) as fh:
            spec = yaml.safe_load(fh)

        all_attacks: list[str] = []
        for group in spec["attack_groups"].values():
            all_attacks.extend(group["attacks"])

        assert len(all_attacks) == len(set(all_attacks)), "attack groups overlap"
        assert "A16" not in spec["attack_groups"]["evaluation_unseen"]["attacks"]
        assert "A19" not in spec["attack_groups"]["evaluation_unseen"]["attacks"]

    def test_duration_bins_cover_all_durations(self):
        import yaml

        spec_path = Path("configs/evaluation_protocol.yaml")
        if not spec_path.exists():
            pytest.skip("evaluation_protocol.yaml not found")
        with open(spec_path) as fh:
            spec = yaml.safe_load(fh)

        bins = spec["duration_bins"]
        assert bins[0]["lower"] == 0
        assert bins[-1]["upper"] is None

        for i in range(len(bins) - 1):
            assert bins[i]["upper"] == bins[i + 1]["lower"], (
                f"gap between bin {bins[i]['bin_id']} and {bins[i + 1]['bin_id']}"
            )
