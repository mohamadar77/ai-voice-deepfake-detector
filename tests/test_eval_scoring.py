"""Torch-free tests for eval-score orchestration, descriptors and conditions (#35)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from test_evaluation import _candidate, _draft_record

from voice_detector.eval_scoring import (
    ParityTolerances,
    ScoringPreflightError,
    StoreStateError,
    UnsupportedModelFamily,
    apply_condition,
    check_bundle_files,
    check_mel_matches_study,
    check_store_for_report,
    describe_bundle,
    load_robustness_spec,
    parity_passed,
    score_partition,
    validate_scoring_rows,
)
from voice_detector.evaluation import freeze_identity_hash, freeze_record, mark_scored
from voice_detector.metrics import METRICS_VERSION
from voice_detector.prediction_store import (
    completed_pairs,
    compute_predictions_hash,
    create_prediction_store,
    get_metadata,
    read_predictions,
    set_metadata_once,
)
from voice_detector.scoring import (
    CNN_AGGREGATION_RULE,
    DECISION_RULE,
    LR_AGGREGATION_RULE,
    SCORE_DIRECTION,
    WaveformScore,
)

# ---------------------------------------------------------------------------
# Bundles
# ---------------------------------------------------------------------------


def _lr_record(**over: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": "lr-baseline-v1",
        "metrics_version": METRICS_VERSION,
        "score_direction": SCORE_DIRECTION,
        "aggregation": {"window_to_clip": LR_AGGREGATION_RULE},
        "seed": 7,
        "threshold": {"value": 0.25, "decision_rule": DECISION_RULE},
    }
    record.update(over)
    return record


def _cnn_record(**over: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": "cnn-study-v1",
        "metrics_version": METRICS_VERSION,
        "score_direction": SCORE_DIRECTION,
        "input_policy": {"eval_aggregation": CNN_AGGREGATION_RULE},
        "seed": 20261004,
        "run_label": "clean",
        "mel_config": {"n_mels": 64, "n_fft": 400},
        "threshold": {"value": -1.5, "decision_rule": DECISION_RULE},
    }
    record.update(over)
    return record


def _bundle(root: Path, name: str, filename: str, record: dict[str, Any]) -> Path:
    directory = root / name
    directory.mkdir()
    (directory / filename).write_text(json.dumps(record), encoding="utf-8")
    return directory


class TestDescribeBundle:
    def test_lr_declaration_comes_from_record_content(self, tmp_path: Path):
        d = describe_bundle(_bundle(tmp_path, "lr_baseline", "lr_run_record.json", _lr_record()))
        assert d.model_family == "logistic_regression"
        assert d.aggregation_rule == LR_AGGREGATION_RULE
        assert d.threshold == 0.25
        assert d.candidate_id == "lr_baseline"
        assert d.run_label == "lr_baseline"

    def test_cnn_declaration(self, tmp_path: Path):
        d = describe_bundle(_bundle(tmp_path, "cnn_a", "cnn_run_record.json", _cnn_record()))
        assert d.model_family == "cnn"
        assert d.aggregation_rule == CNN_AGGREGATION_RULE
        assert d.run_label == "clean"

    def test_unknown_schema_is_unsupported(self, tmp_path: Path):
        bundle = _bundle(
            tmp_path, "x", "cnn_run_record.json", _cnn_record(schema_version="mystery-v9")
        )
        with pytest.raises(UnsupportedModelFamily):
            describe_bundle(bundle)

    def test_filename_must_agree_with_schema(self, tmp_path: Path):
        bundle = _bundle(tmp_path, "x", "lr_run_record.json", _cnn_record())
        with pytest.raises(ScoringPreflightError, match="implies"):
            describe_bundle(bundle)

    def test_two_run_records_are_refused(self, tmp_path: Path):
        bundle = _bundle(tmp_path, "x", "lr_run_record.json", _lr_record())
        (bundle / "cnn_run_record.json").write_text(json.dumps(_cnn_record()))
        with pytest.raises(ScoringPreflightError, match="exactly one"):
            describe_bundle(bundle)

    def test_wrong_aggregation_rule_is_refused(self, tmp_path: Path):
        record = _cnn_record(input_policy={"eval_aggregation": "max_window"})
        with pytest.raises(ScoringPreflightError, match="aggregation"):
            describe_bundle(_bundle(tmp_path, "x", "cnn_run_record.json", record))

    def test_missing_seed_is_refused_not_zeroed(self, tmp_path: Path):
        record = _lr_record()
        del record["seed"]
        with pytest.raises(ScoringPreflightError, match="seed"):
            describe_bundle(_bundle(tmp_path, "x", "lr_run_record.json", record))

    @pytest.mark.parametrize(
        "over",
        [
            {"score_direction": "higher_more_human"},
            {"metrics_version": "metrics-v0"},
            {"threshold": {"value": 0.1, "decision_rule": "other"}},
            {"threshold": {"value": float("nan"), "decision_rule": DECISION_RULE}},
        ],
    )
    def test_convention_mismatch_is_refused(self, tmp_path: Path, over: dict[str, Any]):
        with pytest.raises(ScoringPreflightError):
            describe_bundle(_bundle(tmp_path, "x", "lr_run_record.json", _lr_record(**over)))

    def test_mel_must_match_study_config(self, tmp_path: Path):
        d = describe_bundle(_bundle(tmp_path, "c", "cnn_run_record.json", _cnn_record()))
        check_mel_matches_study(d, {"mel": {"n_mels": 64, "n_fft": 400}})
        with pytest.raises(ScoringPreflightError, match="n_mels"):
            check_mel_matches_study(d, {"mel": {"n_mels": 80}})


# ---------------------------------------------------------------------------
# Robustness spec and conditions
# ---------------------------------------------------------------------------

_SPEC: dict[str, Any] = {
    "version": "robustness-v1",
    "global_seed": 11,
    "conditions": {
        "clean": {"description": "x"},
        "codec_opus_24k": {
            "transform": "codec",
            "codec": "libopus",
            "container": "ogg",
            "bitrate_kbps": 24,
            "sample_rate_hz": 16000,
        },
        "noise_20db": {
            "transform": "noise",
            "noise_type": "awgn",
            "snr_db": 20,
            "reference": "whole_clip_rms",
            "clip_range": [-1.0, 1.0],
        },
        "noise_10db": {
            "transform": "noise",
            "noise_type": "awgn",
            "snr_db": 10,
            "reference": "whole_clip_rms",
            "clip_range": [-1.0, 1.0],
        },
    },
}
_SPEC["silence_trim"] = {
    "top_db": 30,
    "min_duration_seconds": 0.3,
    "fallback": "keep_original_with_flag",
}
_ALL = ["clean", "codec_opus_24k", "noise_20db", "noise_10db"]
_ALL5 = [*_ALL, "silence_trim"]


def _spec_file(tmp_path: Path, mutate=None) -> Path:
    raw = json.loads(json.dumps(_SPEC))
    if mutate:
        mutate(raw)
    path = tmp_path / "robustness_spec.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


class TestRobustnessSpec:
    def test_loads_required_conditions(self, tmp_path: Path):
        spec = load_robustness_spec(_spec_file(tmp_path), required=_ALL)
        assert spec.global_seed == 11
        assert list(spec.conditions) == _ALL

    def test_missing_file_is_refused(self, tmp_path: Path):
        with pytest.raises(ScoringPreflightError, match="not found"):
            load_robustness_spec(tmp_path / "nope.yaml", required=["clean"])

    def test_wrong_version_is_refused(self, tmp_path: Path):
        path = _spec_file(tmp_path, lambda r: r.update(version="robustness-v0"))
        with pytest.raises(ScoringPreflightError, match="version"):
            load_robustness_spec(path, required=_ALL)

    def test_missing_seed_is_refused(self, tmp_path: Path):
        path = _spec_file(tmp_path, lambda r: r.pop("global_seed"))
        with pytest.raises(ScoringPreflightError, match="global_seed"):
            load_robustness_spec(path, required=_ALL)

    def test_missing_condition_is_refused(self, tmp_path: Path):
        path = _spec_file(tmp_path, lambda r: r["conditions"].pop("noise_10db"))
        with pytest.raises(ScoringPreflightError, match="noise_10db"):
            load_robustness_spec(path, required=_ALL)

    def test_unknown_transform_is_refused(self, tmp_path: Path):
        path = _spec_file(
            tmp_path, lambda r: r["conditions"]["noise_10db"].update(transform="reverb")
        )
        with pytest.raises(ScoringPreflightError, match="unsupported transform"):
            load_robustness_spec(path, required=_ALL)

    def test_clean_with_transform_is_refused(self, tmp_path: Path):
        path = _spec_file(tmp_path, lambda r: r["conditions"]["clean"].update(transform="noise"))
        with pytest.raises(ScoringPreflightError, match="clean"):
            load_robustness_spec(path, required=_ALL)

    def test_unimplemented_codec_is_refused(self, tmp_path: Path):
        path = _spec_file(
            tmp_path, lambda r: r["conditions"]["codec_opus_24k"].update(codec="libmp3lame")
        )
        with pytest.raises(ScoringPreflightError, match="libopus"):
            load_robustness_spec(path, required=_ALL)

    def test_missing_snr_is_refused(self, tmp_path: Path):
        path = _spec_file(tmp_path, lambda r: r["conditions"]["noise_20db"].pop("snr_db"))
        with pytest.raises(ScoringPreflightError, match="snr_db"):
            load_robustness_spec(path, required=_ALL)


class TestSilenceTrimCondition:
    def test_loads_from_the_silence_trim_section(self, tmp_path: Path):
        spec = load_robustness_spec(_spec_file(tmp_path), required=_ALL5)
        assert spec.conditions["silence_trim"] == {
            "transform": "silence_trim",
            "top_db": 30.0,
            "min_duration_seconds": 0.3,
        }

    def test_missing_section_is_refused(self, tmp_path: Path):
        path = _spec_file(tmp_path, lambda r: r.pop("silence_trim"))
        with pytest.raises(ScoringPreflightError, match="silence_trim"):
            load_robustness_spec(path, required=_ALL5)

    def test_other_fallback_is_refused(self, tmp_path: Path):
        path = _spec_file(tmp_path, lambda r: r["silence_trim"].update(fallback="drop_clip"))
        with pytest.raises(ScoringPreflightError, match="fallback"):
            load_robustness_spec(path, required=_ALL5)

    def test_apply_uses_frozen_parameters_and_flags_fallback(self, tmp_path: Path):
        spec = load_robustness_spec(_spec_file(tmp_path), required=_ALL5)
        seen: dict[str, Any] = {}

        class _Trim:
            waveform = np.zeros(2, dtype=np.float32)
            kept_original = True

        def trim_fn(wave, **kw):
            seen.update(kw)
            return _Trim()

        out = apply_condition(
            np.zeros(8, dtype=np.float32), clip_id="c", condition="silence_trim", spec=spec,
            trim_fn=trim_fn, sample_rate_hz=16000,
        )
        assert seen == {"sample_rate": 16000, "top_db": 30.0, "min_duration_seconds": 0.3}
        assert out.kept_original

    def test_apply_needs_the_sample_rate(self, tmp_path: Path):
        spec = load_robustness_spec(_spec_file(tmp_path), required=_ALL5)
        with pytest.raises(ScoringPreflightError, match="sample rate"):
            apply_condition(
                np.zeros(8, dtype=np.float32), clip_id="c", condition="silence_trim", spec=spec,
            )


class _NoiseResult:
    def __init__(self, waveform: np.ndarray, n_clipped: int) -> None:
        self.waveform = waveform
        self.n_clipped = n_clipped


class _CodecResult:
    def __init__(self, waveform: np.ndarray, samples_delta: int) -> None:
        self.waveform = waveform
        self.samples_delta = samples_delta


class TestApplyCondition:
    def _spec(self, tmp_path: Path):
        return load_robustness_spec(_spec_file(tmp_path), required=_ALL)

    def test_clean_returns_the_same_array(self, tmp_path: Path):
        wave = np.ones(10, dtype=np.float32)
        out = apply_condition(wave, clip_id="c", condition="clean", spec=self._spec(tmp_path))
        assert out.waveform is wave

    def test_noise_uses_frozen_seed_and_snr(self, tmp_path: Path):
        calls: list[dict[str, Any]] = []

        def noise_fn(wave, **kw):
            calls.append(kw)
            return _NoiseResult(wave + 1, 3)

        out = apply_condition(
            np.zeros(4, dtype=np.float32),
            clip_id="LA_E_1",
            condition="noise_10db",
            spec=self._spec(tmp_path),
            noise_fn=noise_fn,
        )
        assert calls == [{"snr_db": 10.0, "clip_id": "LA_E_1", "global_seed": 11}]
        assert out.n_clipped == 3

    def test_codec_uses_spec_parameters(self, tmp_path: Path):
        seen: dict[str, Any] = {}

        def codec_fn(wave, **kw):
            seen.update(kw)
            return _CodecResult(wave[:-1], -1)

        out = apply_condition(
            np.zeros(4, dtype=np.float32),
            clip_id="c",
            condition="codec_opus_24k",
            spec=self._spec(tmp_path),
            codec_fn=codec_fn,
        )
        assert seen == {"sample_rate": 16000, "bitrate_kbps": 24}
        assert out.samples_delta == -1

    def test_unknown_condition_is_refused(self, tmp_path: Path):
        with pytest.raises(ScoringPreflightError, match="unknown condition"):
            apply_condition(
                np.zeros(4, dtype=np.float32),
                clip_id="c",
                condition="reverb",
                spec=self._spec(tmp_path),
            )


# ---------------------------------------------------------------------------
# Row validation
# ---------------------------------------------------------------------------


def _row(i: int, **over: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "clip_id": f"LA_D_{i:03d}",
        "label": "bonafide" if i % 2 == 0 else "spoof",
        "speaker_id": f"LA_{i % 3:04d}",
        "attack_id": None if i % 2 == 0 else "A07",
        "official_split": "development",
        "duration_seconds": 1.0 + i,
        "local_path": f"clip_{i:03d}.flac",
    }
    row.update(over)
    return row


def _always(_: str) -> bool:
    return True


class TestValidateScoringRows:
    def test_sorted_by_clip_id_and_filtered_by_partition(self):
        rows = [_row(2), _row(0), _row(1, official_split="evaluation")]
        out = validate_scoring_rows(rows, "development", path_exists=_always)
        assert [r["clip_id"] for r in out] == ["LA_D_000", "LA_D_002"]

    def test_bad_partition_name(self):
        with pytest.raises(ScoringPreflightError, match="partition"):
            validate_scoring_rows([_row(0)], "dev", path_exists=_always)

    def test_empty_partition(self):
        with pytest.raises(ScoringPreflightError, match="no clips"):
            validate_scoring_rows([_row(0)], "evaluation", path_exists=_always)

    def test_missing_local_path_is_refused_not_dropped(self):
        with pytest.raises(ScoringPreflightError, match="no local_path"):
            validate_scoring_rows([_row(0), _row(1, local_path="")], "development",
                                  path_exists=_always)

    def test_missing_file_is_refused(self):
        with pytest.raises(ScoringPreflightError, match="not found"):
            validate_scoring_rows([_row(0)], "development", path_exists=lambda _: False)

    def test_duplicate_clip_ids_are_refused(self):
        with pytest.raises(ScoringPreflightError, match="duplicate"):
            validate_scoring_rows([_row(0), _row(0)], "development", path_exists=_always)

    def test_empty_speaker_is_refused(self):
        with pytest.raises(ScoringPreflightError, match="speaker_id"):
            validate_scoring_rows([_row(0, speaker_id="")], "development", path_exists=_always)

    def test_bad_label_is_refused(self):
        with pytest.raises(ScoringPreflightError, match="label"):
            validate_scoring_rows([_row(0, label="human")], "development", path_exists=_always)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class _StubScorer:
    model_family = "stub"
    aggregation_rule = "stub"
    sample_rate_hz = 16000

    def __init__(self, threshold: float = 0.5, offset: float = 0.0) -> None:
        self.threshold = threshold
        self.offset = offset
        self.seen_ids: list[int] = []

    def score_waveform(self, waveform: np.ndarray) -> WaveformScore:
        self.seen_ids.append(id(waveform))
        return WaveformScore(clip_score=float(waveform.mean()) + self.offset, n_windows=1)


class _Env:
    """Decoder and noise stubs with call counters."""

    def __init__(self) -> None:
        self.decodes = 0
        self.noise_calls = 0
        self.fail_after: int | None = None

    def decode(self, local_path: str, sample_rate: int) -> np.ndarray:
        self.decodes += 1
        if self.fail_after is not None and self.decodes > self.fail_after:
            raise OSError("decode failed")
        value = float(local_path.split("_")[1].split(".")[0]) / 10.0
        wave = np.full(32, value, dtype=np.float32)
        wave.flags.writeable = False
        return wave

    def noise(self, wave: np.ndarray, **kw: Any) -> _NoiseResult:
        self.noise_calls += 1
        return _NoiseResult(wave + np.float32(0.01), 0)

    def codec(self, wave: np.ndarray, **kw: Any) -> _CodecResult:
        return _CodecResult(wave[:-1], -1)


def _run(tmp_path: Path, store: str = "p.db", **over: Any):
    env = over.pop("env", None) or _Env()
    spec = load_robustness_spec(_spec_file(tmp_path), required=_ALL)
    scorers = over.pop("scorers", None) or {
        "a": _StubScorer(threshold=0.2),
        "b": _StubScorer(threshold=0.2, offset=1.0),
    }
    kwargs: dict[str, Any] = {
        "partition": "development",
        "store_path": tmp_path / store,
        "freeze_identity": "ident",
        "decode": env.decode,
        "path_exists": _always,
        "codec_fn": env.codec,
        "noise_fn": env.noise,
        "provenance": {"git_commit": "abc", "torch_threads": "4"},
    }
    kwargs.update(over)
    rows = over.pop("rows", None) or [_row(i) for i in range(4)]
    kwargs.pop("rows", None)
    summary = score_partition(rows, ["a", "b"], _ALL, scorers, spec, **kwargs)
    return summary, env, scorers


class TestScorePartition:
    def test_full_matrix_is_stored(self, tmp_path: Path):
        summary, _, _ = _run(tmp_path)
        assert summary.complete
        assert summary.rows_written == 4 * 2 * 4
        rows = read_predictions(tmp_path / "p.db")
        assert len(rows) == 32
        assert {r.official_split for r in rows} == {"development"}

    def test_labels_and_metadata_propagate(self, tmp_path: Path):
        _run(tmp_path)
        rows = read_predictions(tmp_path / "p.db", candidate_id="a", condition="clean")
        by_id = {r.clip_id: r for r in rows}
        assert by_id["LA_D_001"].label == "spoof"
        assert by_id["LA_D_001"].attack_id == "A07"
        assert by_id["LA_D_000"].speaker_id == "LA_0000"
        assert by_id["LA_D_002"].threshold == 0.2

    def test_decision_is_inclusive_at_threshold(self, tmp_path: Path):
        scorer = _StubScorer(threshold=0.2)
        # clip 2 decodes to value 0.2 exactly in float32; use a float64-exact scorer.
        scorer.score_waveform = lambda w: WaveformScore(clip_score=0.2, n_windows=1)  # type: ignore[method-assign]
        _run(tmp_path, scorers={"a": scorer, "b": _StubScorer(threshold=0.2)})
        rows = read_predictions(tmp_path / "p.db", candidate_id="a", condition="clean")
        assert all(r.decision for r in rows)

    def test_conditioned_waveform_is_shared_across_candidates(self, tmp_path: Path):
        _, env, scorers = _run(tmp_path)
        assert scorers["a"].seen_ids == scorers["b"].seen_ids
        # noise_20db and noise_10db, once per clip each, regardless of candidate count
        assert env.noise_calls == 4 * 2

    def test_decode_once_per_clip_and_condition(self, tmp_path: Path):
        _, env, _ = _run(tmp_path)
        assert env.decodes == 4 * len(_ALL)

    def test_readonly_inputs_survive(self, tmp_path: Path):
        summary, _, _ = _run(tmp_path)  # the decoder returns read-only arrays
        assert summary.complete

    def test_non_empty_store_refused_without_resume_and_no_compute(self, tmp_path: Path):
        _run(tmp_path)
        env = _Env()
        with pytest.raises(StoreStateError, match="--resume"):
            _run(tmp_path, env=env)
        assert env.decodes == 0

    def test_interrupted_run_resumes_to_identical_store(self, tmp_path: Path):
        clean_summary, _, _ = _run(tmp_path, store="whole.db")

        crashing = _Env()
        crashing.fail_after = 4 * 2 + 1  # clean and codec done, crash inside noise_20db
        with pytest.raises(RuntimeError, match="noise_20db"):
            _run(tmp_path, store="resumed.db", env=crashing)
        pairs = completed_pairs(tmp_path / "resumed.db")
        assert {cond for _, cond in pairs} == {"clean", "codec_opus_24k"}

        summary, env, _ = _run(tmp_path, store="resumed.db", resume=True)
        assert summary.complete
        assert len(summary.skipped_pairs) == 4
        assert env.decodes == 4 * 2  # only the two remaining conditions are recomputed
        assert compute_predictions_hash(tmp_path / "resumed.db") == clean_summary.predictions_hash

    def test_resume_into_other_freeze_is_refused(self, tmp_path: Path):
        _run(tmp_path)
        with pytest.raises(StoreStateError, match="freeze_identity"):
            _run(tmp_path, resume=True, freeze_identity="other")

    def test_resume_into_other_partition_is_refused(self, tmp_path: Path):
        _run(tmp_path)
        rows = [_row(i, official_split="evaluation") for i in range(4)]
        with pytest.raises(StoreStateError, match="partition"):
            _run(tmp_path, resume=True, partition="evaluation", rows=rows)

    def test_resume_into_other_scoring_code_is_refused(self, tmp_path: Path):
        _run(tmp_path)
        with pytest.raises(StoreStateError, match="provenance_git_commit"):
            _run(tmp_path, resume=True, provenance={"git_commit": "def", "torch_threads": "4"})

    def test_resume_into_store_without_provenance_is_refused_untouched(self, tmp_path: Path):
        from voice_detector.prediction_store import PredictionRow, write_predictions

        store = create_prediction_store(tmp_path / "legacy.db")
        write_predictions(
            store,
            [PredictionRow(
                clip_id="LA_D_000", candidate_id="a", condition="clean", clip_score=0.1,
                label="bonafide", speaker_id="s", attack_id=None, official_split="development",
                duration_seconds=1.0, n_windows=1, threshold=0.2, decision=False,
            )],
            freeze_record_hash="old",
        )
        with pytest.raises(StoreStateError, match="provenance") as info:
            _run(tmp_path, store="legacy.db", resume=True)
        assert info.value.reason == "store_without_provenance"
        assert "freeze_identity" not in get_metadata(store)

    def test_after_condition_runs_after_each_commit_and_can_abort(self, tmp_path: Path):
        seen: list[tuple[str, int]] = []

        def hook(condition: str) -> None:
            seen.append((condition, sum(completed_pairs(tmp_path / "p.db").values())))

        _run(tmp_path, after_condition=hook)
        assert [c for c, _ in seen] == _ALL
        assert [n for _, n in seen] == [8, 16, 24, 32]

        def abort(condition: str) -> None:
            raise RuntimeError("stop")

        with pytest.raises(RuntimeError, match="stop"):
            _run(tmp_path, store="q.db", after_condition=abort)
        assert {cond for _, cond in completed_pairs(tmp_path / "q.db")} == {"clean"}

    def test_after_condition_also_runs_for_conditions_skipped_on_resume(self, tmp_path: Path):
        _run(tmp_path)
        seen: list[str] = []
        _run(tmp_path, resume=True, after_condition=seen.append)
        assert seen == _ALL

    def test_silence_trim_condition_is_scored_and_diagnosed(self, tmp_path: Path):
        class _Trim:
            def __init__(self, wave):
                self.waveform = wave[:-4]
                self.kept_original = int(wave[0] * 10) % 2 == 0

        spec = load_robustness_spec(_spec_file(tmp_path), required=_ALL5)
        env = _Env()
        scorers = {"a": _StubScorer(threshold=0.2), "b": _StubScorer(threshold=0.2)}
        summary = score_partition(
            [_row(i) for i in range(4)], ["a", "b"], _ALL5, scorers, spec,
            partition="development", store_path=tmp_path / "t.db", freeze_identity="i",
            decode=env.decode, path_exists=_always, codec_fn=env.codec, noise_fn=env.noise,
            trim_fn=lambda wave, **kw: _Trim(wave),
        )
        assert summary.complete
        assert summary.diagnostics["silence_trim"]["silence_trim_kept_original"] == 2
        assert len(read_predictions(tmp_path / "t.db", condition="silence_trim")) == 8

    def test_max_clips_is_development_only_and_recorded(self, tmp_path: Path):
        summary, _, _ = _run(tmp_path, max_clips=2)
        assert summary.n_clips == 2
        assert get_metadata(tmp_path / "p.db")["subset"] == "max_clips=2"
        rows = [_row(i, official_split="evaluation") for i in range(3)]
        with pytest.raises(ScoringPreflightError, match="development"):
            _run(tmp_path, store="e.db", partition="evaluation", rows=rows, max_clips=2)

    def test_non_finite_score_is_an_error(self, tmp_path: Path):
        bad = _StubScorer()
        bad.score_waveform = lambda w: WaveformScore(clip_score=float("nan"), n_windows=1)  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="non-finite"):
            _run(tmp_path, scorers={"a": bad, "b": _StubScorer()})

    def test_missing_scorer_is_unsupported_not_skipped(self, tmp_path: Path):
        with pytest.raises(UnsupportedModelFamily):
            _run(tmp_path, scorers={"a": _StubScorer()})
        assert not (tmp_path / "p.db").exists()

    def test_mixed_sample_rates_are_refused(self, tmp_path: Path):
        odd = _StubScorer()
        odd.sample_rate_hz = 8000
        with pytest.raises(ScoringPreflightError, match="sample rate"):
            _run(tmp_path, scorers={"a": _StubScorer(), "b": odd})

    def test_missing_audio_stops_before_store_creation(self, tmp_path: Path):
        with pytest.raises(ScoringPreflightError, match="not found"):
            _run(tmp_path, path_exists=lambda _: False)
        assert not (tmp_path / "p.db").exists()


# ---------------------------------------------------------------------------
# Store helpers and freeze identity
# ---------------------------------------------------------------------------


class TestStoreHelpers:
    def test_set_metadata_once(self, tmp_path: Path):
        store = create_prediction_store(tmp_path / "s.db")
        set_metadata_once(store, "k", "v")
        set_metadata_once(store, "k", "v")
        with pytest.raises(ValueError, match="already set"):
            set_metadata_once(store, "k", "w")

    def test_completed_pairs_empty(self, tmp_path: Path):
        store = create_prediction_store(tmp_path / "s.db")
        assert completed_pairs(store) == {}


class TestFreezeIdentity:
    def test_stable_across_state_transitions(self):
        frozen = freeze_record(_draft_record())
        scored = mark_scored(frozen, "p" * 64)
        assert freeze_identity_hash(frozen) == freeze_identity_hash(scored)

    def test_changes_with_threshold(self):
        base = _draft_record([_candidate()])
        other = _draft_record(
            [
                _candidate().__class__(
                    **{**_candidate().as_dict(), "threshold": 0.6},
                )
            ]
        )
        assert freeze_identity_hash(base) != freeze_identity_hash(other)


_TOL = ParityTolerances(roc_auc_abs=1e-5, eer_abs=1e-4, max_abs_confusion_delta_total=2)


class TestDevParity:
    def _store(self, tmp_path: Path, scores: list[float], labels: list[str]) -> Path:
        from voice_detector.prediction_store import PredictionRow, write_predictions

        store = create_prediction_store(tmp_path / "parity.db")
        rows = [
            PredictionRow(
                clip_id=f"c{i}", candidate_id="lr_baseline", condition="clean",
                clip_score=s, label=lab, speaker_id="spk", attack_id=None,
                official_split="development", duration_seconds=1.0, n_windows=1,
                threshold=0.0, decision=s >= 0.0,
            )
            for i, (s, lab) in enumerate(zip(scores, labels, strict=True))
        ]
        write_predictions(store, rows, freeze_record_hash="h")
        return store

    def _record(self, scores: list[float], labels: list[str]) -> dict[str, Any]:
        from voice_detector.metrics import evaluate_clip_scores

        metrics = evaluate_clip_scores(
            [f"c{i}" for i in range(len(scores))], labels, np.array(scores),
            threshold=0.0, score_direction=SCORE_DIRECTION,
        )
        return _lr_record(
            threshold={"value": 0.0, "decision_rule": DECISION_RULE},
            dev_full_metrics=metrics.as_dict(),
        )

    def test_matching_predictions_are_within_tolerance(self, tmp_path: Path):
        from voice_detector.eval_scoring import dev_parity

        scores = [-2.0, -1.0, 1.0, 2.0, 0.5, -0.5]
        labels = ["bonafide", "bonafide", "spoof", "spoof", "spoof", "bonafide"]
        _bundle(tmp_path, "lr_baseline", "lr_run_record.json", self._record(scores, labels))
        store = self._store(tmp_path, scores, labels)
        out = dev_parity(store, tmp_path, ["lr_baseline"], _TOL)["lr_baseline"]
        assert out["checked"] and out["within_tolerance"]
        assert out["confusion_delta"] == {"tn": 0, "fp": 0, "fn": 0, "tp": 0}

    def test_flipped_scores_are_detected(self, tmp_path: Path):
        from voice_detector.eval_scoring import dev_parity

        scores = [-2.0, -1.0, 1.0, 2.0, 0.5, -0.5]
        labels = ["bonafide", "bonafide", "spoof", "spoof", "spoof", "bonafide"]
        _bundle(tmp_path, "lr_baseline", "lr_run_record.json", self._record(scores, labels))
        store = self._store(tmp_path, [-s for s in scores], labels)
        out = dev_parity(store, tmp_path, ["lr_baseline"], _TOL)["lr_baseline"]
        assert not out["within_tolerance"]

    def test_unchecked_candidate_fails_the_gate(self, tmp_path: Path):
        from voice_detector.eval_scoring import dev_parity

        record = _lr_record()  # no dev_full_metrics recorded
        _bundle(tmp_path, "lr_baseline", "lr_run_record.json", record)
        store = self._store(tmp_path, [-1.0, 1.0], ["bonafide", "spoof"])
        report = dev_parity(store, tmp_path, ["lr_baseline"], _TOL)
        assert report["lr_baseline"]["checked"] is False
        assert not parity_passed(report)

    def test_gate_needs_every_candidate(self):
        assert parity_passed({"a": {"within_tolerance": True}})
        mixed = {"a": {"within_tolerance": True}, "b": {"within_tolerance": False}}
        assert not parity_passed(mixed)
        assert not parity_passed({})

    def test_tolerances_come_from_the_protocol_yaml(self):
        spec = yaml.safe_load(Path("configs/evaluation_protocol.yaml").read_text())
        tol = ParityTolerances.from_spec(spec)
        assert (tol.roc_auc_abs, tol.eer_abs, tol.max_abs_confusion_delta_total) == (
            1e-5, 1e-4, 2,
        )
        with pytest.raises(ScoringPreflightError, match="dev_parity"):
            ParityTolerances.from_spec({})


class TestBundleFiles:
    def test_missing_model_files_are_reported(self, tmp_path: Path):
        bundle = _bundle(tmp_path, "cnn_a", "cnn_run_record.json", _cnn_record())
        descriptor = describe_bundle(bundle)
        with pytest.raises(ScoringPreflightError, match="cnn_checkpoint.pt"):
            check_bundle_files(descriptor, bundle)
        (bundle / "cnn_checkpoint.pt").write_bytes(b"x")
        (bundle / "cnn_norm_stats.pt").write_bytes(b"x")
        check_bundle_files(descriptor, bundle)

    def test_lr_needs_its_pipeline(self, tmp_path: Path):
        bundle = _bundle(tmp_path, "lr_a", "lr_run_record.json", _lr_record())
        with pytest.raises(ScoringPreflightError, match="lr_pipeline.joblib"):
            check_bundle_files(describe_bundle(bundle), bundle)


class TestCheckStoreForReport:
    def _store(self, tmp_path: Path, *, partition="evaluation", subset="full", parity=None,
               identity=None, complete=True) -> Path:
        from voice_detector.evaluation import freeze_identity_hash
        from voice_detector.prediction_store import PredictionRow, write_predictions

        record = freeze_record(_draft_record())
        store = create_prediction_store(tmp_path / f"{partition}.db")
        set_metadata_once(store, "freeze_identity", identity or freeze_identity_hash(record))
        set_metadata_once(store, "partition", partition)
        set_metadata_once(store, "subset", subset)
        if parity:
            set_metadata_once(store, "dev_parity", parity)
        if complete:
            write_predictions(
                store,
                [PredictionRow(
                    clip_id="c0", candidate_id="lr_C1.0", condition="clean", clip_score=0.0,
                    label="bonafide", speaker_id="s", attack_id=None, official_split=partition,
                    duration_seconds=1.0, n_windows=1, threshold=0.5, decision=False,
                )],
                freeze_record_hash="h",
            )
        return store

    def test_accepts_a_complete_matching_store(self, tmp_path: Path):
        record = freeze_record(_draft_record())
        meta = check_store_for_report(self._store(tmp_path), record, partition="evaluation")
        assert meta["subset"] == "full"

    @pytest.mark.parametrize(
        ("kwargs", "reason"),
        [
            ({"identity": "other"}, "store_freeze_mismatch"),
            ({"partition": "development"}, "store_partition_mismatch"),
            ({"subset": "max_clips=3"}, "subset_store"),
            ({"complete": False}, "incomplete_store"),
        ],
    )
    def test_refusals(self, tmp_path: Path, kwargs, reason):
        record = freeze_record(_draft_record())
        with pytest.raises(StoreStateError) as info:
            check_store_for_report(self._store(tmp_path, **kwargs), record, partition="evaluation")
        assert info.value.reason == reason

    def test_parity_is_required_when_asked(self, tmp_path: Path):
        record = freeze_record(_draft_record())
        store = self._store(tmp_path, partition="development")
        with pytest.raises(StoreStateError) as info:
            check_store_for_report(store, record, partition="development", require_parity=True)
        assert info.value.reason == "parity_not_passed"
        set_metadata_once(store, "dev_parity", "passed")
        check_store_for_report(store, record, partition="development", require_parity=True)
