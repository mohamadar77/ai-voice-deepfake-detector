"""Tests for the supplementary post-report analysis (issue #35).

Torch-free and fast: synthetic prediction rows, no audio, no real stores.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from test_eval_report import _freeze, _multi_pair_setup, _pred

from voice_detector.cli import main
from voice_detector.eval_supplementary import (
    PARTITION_ALL_SLICE,
    Comparison,
    SupplementarySpec,
    _eer_plan,
    compute_supplementary_hash,
    load_supplementary_spec,
    resample_weights,
    resampled_metrics,
    run_supplementary,
    weighted_eer,
    write_supplementary_report,
)
from voice_detector.evaluation import (
    freeze_identity_hash,
    load_freeze_record,
    save_freeze_record,
)
from voice_detector.metrics import compute_eer
from voice_detector.prediction_store import (
    compute_predictions_hash,
    create_prediction_store,
    set_metadata_once,
    write_predictions,
)

REPO = Path(__file__).resolve().parents[1]
ALL_METRICS = (
    "eer",
    "human_false_positive_rate",
    "synthetic_detection_rate",
    "balanced_accuracy",
    "macro_f1",
)


def _spec(comparisons: tuple[Comparison, ...] = ()) -> SupplementarySpec:
    return SupplementarySpec(
        evaluation_slice_ids=("attack_A07", "duration_lt_1s"),
        development_slice_ids=("attack_A07",),
        metrics=ALL_METRICS,
        comparisons=comparisons,
        low_support_min_class_count=5,
        spec_sha256="0" * 64,
    )


def _setup():
    record, preds = _multi_pair_setup()
    spec = _spec((Comparison("a_vs_b", "cand_a", "cand_b"),))
    return record, preds, spec


class TestWeights:
    def test_rows_sum_to_the_number_of_speakers_and_are_deterministic(self):
        weights = resample_weights(7, 50, seed=3)
        assert weights.shape == (50, 7)
        assert (weights.sum(axis=1) == 7).all()
        assert np.array_equal(weights, resample_weights(7, 50, seed=3))
        assert not np.array_equal(weights, resample_weights(7, 50, seed=4))

    def test_needs_two_speakers(self):
        with pytest.raises(ValueError, match="at least 2 speakers"):
            resample_weights(1, 10, seed=1)


class TestWeightedEer:
    def test_equals_compute_eer_on_repeated_clips(self):
        rng = np.random.default_rng(11)
        n = 60
        # Rounded scores create ties, which the EER must handle like compute_eer.
        scores = np.round(rng.normal(size=n), 1)
        is_spoof = rng.random(n) < 0.5
        is_spoof[:2] = [True, False]
        speaker_idx = rng.integers(0, 6, size=n).astype(np.intp)
        plan = _eer_plan(scores, is_spoof, speaker_idx)
        for draw in resample_weights(6, 20, seed=5):
            clip_weight = draw[speaker_idx]
            repeated_scores = np.repeat(scores, clip_weight)
            repeated_spoof = np.repeat(is_spoof, clip_weight)
            if repeated_spoof.all() or not repeated_spoof.any():
                assert np.isnan(weighted_eer(plan, draw))
                continue
            expected = compute_eer(
                repeated_scores[~repeated_spoof], repeated_scores[repeated_spoof]
            ).eer
            assert weighted_eer(plan, draw) == pytest.approx(expected, abs=1e-12)

    def test_nan_when_a_class_has_no_weight(self):
        scores = np.array([0.1, 0.9, 0.2, 0.8])
        is_spoof = np.array([False, True, False, True])
        speaker_idx = np.array([0, 0, 1, 1], dtype=np.intp)
        plan = _eer_plan(scores, is_spoof, speaker_idx)
        assert np.isnan(weighted_eer(plan, np.array([0, 0])))


class TestResampledMetrics:
    def test_rates_match_brute_force_on_repeated_clips(self):
        rng = np.random.default_rng(2)
        n = 80
        scores = rng.normal(size=n)
        is_spoof = rng.random(n) < 0.5
        speaker_idx = rng.integers(0, 8, size=n).astype(np.intp)
        weights = resample_weights(8, 15, seed=9)
        threshold = 0.1
        got = resampled_metrics(scores, is_spoof, speaker_idx, threshold, weights, ALL_METRICS)
        for b in range(weights.shape[0]):
            w = weights[b][speaker_idx]
            s = np.repeat(scores, w)
            sp = np.repeat(is_spoof, w)
            flagged = s >= threshold
            n_b, n_s = (~sp).sum(), sp.sum()
            if n_b == 0 or n_s == 0:
                assert np.isnan(got["balanced_accuracy"][b])
                continue
            fp, tp = (~sp & flagged).sum(), (sp & flagged).sum()
            tn, fn = n_b - fp, n_s - tp
            assert got["human_false_positive_rate"][b] == pytest.approx(fp / n_b)
            assert got["synthetic_detection_rate"][b] == pytest.approx(tp / n_s)
            assert got["balanced_accuracy"][b] == pytest.approx((tn / n_b + tp / n_s) / 2)
            f1 = (2 * tn / (2 * tn + fn + fp) + 2 * tp / (2 * tp + fp + fn)) / 2
            assert got["macro_f1"][b] == pytest.approx(f1)
            assert got["eer"][b] == pytest.approx(compute_eer(s[~sp], s[sp]).eer, abs=1e-12)

    def test_threshold_ties_count_as_spoof(self):
        scores = np.array([0.5, 0.5, 0.4, 0.9])
        is_spoof = np.array([False, True, False, True])
        speaker_idx = np.array([0, 0, 1, 1], dtype=np.intp)
        weights = np.array([[1, 1]], dtype=np.int64)
        got = resampled_metrics(
            scores, is_spoof, speaker_idx, 0.5, weights, ("human_false_positive_rate",)
        )
        assert got["human_false_positive_rate"][0] == pytest.approx(0.5)


class TestRunSupplementary:
    def test_structure_and_evaluation_only_counts(self):
        record, preds, spec = _setup()
        # Development rows must not leak into an evaluation-partition analysis.
        dev = [
            _pred(
                f"dev{i}", "spoof" if i % 2 else "bonafide", 0.9 if i % 2 else 0.1,
                candidate_id=c, condition=cond, speaker_id=f"DSPK{i % 3}",
                attack_id="A01" if i % 2 else None, official_split="development",
                duration_seconds=0.5,
            )
            for c in ("cand_a", "cand_b") for cond in ("clean", "noise_10db") for i in range(6)
        ]
        report = run_supplementary(record, preds + dev, spec, partition="evaluation")
        assert report["label"] == "supplementary_post_report"
        assert report["descriptive_only"] is True
        assert [r["condition"] for r in report["results"]] == ["clean", "noise_10db"]
        clean = report["results"][0]
        assert clean["n_clips"] == 24
        assert clean["n_speakers"] == 6
        slices = {s["slice_id"]: s for s in clean["slices"]}
        assert list(slices) == [PARTITION_ALL_SLICE, "attack_A07", "duration_lt_1s"]
        assert slices[PARTITION_ALL_SLICE]["n_clips"] == 24
        assert slices["attack_A07"]["n_bonafide"] == 12 and slices["attack_A07"]["n_spoof"] == 12
        assert slices["duration_lt_1s"]["n_clips"] == 2
        assert slices["duration_lt_1s"]["low_support"] is True
        candidate = slices[PARTITION_ALL_SLICE]["candidates"]["cand_a"]
        assert set(candidate["bootstrap"]) == set(ALL_METRICS)
        eer = candidate["bootstrap"]["eer"]
        # Every fixture speaker holds one class only, so a few resamples are single-class.
        assert 0 < eer["n_valid_resamples"] <= record.bootstrap.n_resamples
        assert eer["ci_lower"] <= eer["ci_upper"]

    def test_paired_difference_uses_the_same_resamples(self):
        record, preds, _ = _setup()
        # cand_b gets cand_a's scores and threshold: a paired difference must be exactly 0.
        clone = [
            dataclasses.replace(p, candidate_id="cand_b")
            for p in preds if p.candidate_id == "cand_a"
        ]
        preds = [p for p in preds if p.candidate_id == "cand_a"] + clone
        candidates = [
            record.candidates[0],
            dataclasses.replace(
                record.candidates[1], threshold=record.candidates[0].threshold
            ),
        ]
        record = dataclasses.replace(record, candidates=candidates)
        spec = _spec((Comparison("same", "cand_a", "cand_b"),))
        report = run_supplementary(record, preds, spec, partition="evaluation")
        for condition in report["results"]:
            for entry in condition["slices"]:
                for comparison in entry["comparisons"]:
                    for diff in comparison["difference"].values():
                        if "ci_lower" in diff:
                            assert diff["ci_lower"] == 0.0 and diff["ci_upper"] == 0.0
                        assert diff["point_difference"] in (0.0, None)

    def test_difference_point_estimate_is_a_minus_b(self):
        record, preds, spec = _setup()
        report = run_supplementary(record, preds, spec, partition="evaluation")
        entry = report["results"][0]["slices"][0]
        a = entry["candidates"]["cand_a"]["metrics"]["eer"]
        b = entry["candidates"]["cand_b"]["metrics"]["eer"]
        diff = entry["comparisons"][0]["difference"]["eer"]["point_difference"]
        assert diff == pytest.approx(a - b)

    def test_workers_give_an_identical_report(self, tmp_path: Path):
        record, preds, spec = _setup()
        serial = run_supplementary(record, preds, spec, partition="evaluation")
        parallel = run_supplementary(record, preds, spec, partition="evaluation", workers=2)
        strip = lambda rep: {k: v for k, v in rep.items() if k != "generated_utc"}  # noqa: E731
        assert json.dumps(strip(serial), sort_keys=True) == json.dumps(
            strip(parallel), sort_keys=True
        )
        p1 = write_supplementary_report(serial, tmp_path / "s")
        p2 = write_supplementary_report(parallel, tmp_path / "p")
        assert compute_supplementary_hash(p1) == compute_supplementary_hash(p2)

    def test_development_rehearsal_uses_the_development_slices(self):
        record, _, spec = _setup()
        rng = np.random.default_rng(1)
        preds = [
            _pred(
                f"d{i:02d}", "spoof" if i % 2 else "bonafide",
                float(np.clip(rng.normal(0.7 if i % 2 else 0.3, 0.2), 0, 1)),
                candidate_id=c, condition=cond, speaker_id=f"S{i % 4}",
                attack_id="A07" if i % 2 else None, official_split="development",
            )
            for c in ("cand_a", "cand_b") for cond in ("clean", "noise_10db")
            for i in range(20)
        ]
        report = run_supplementary(record, preds, spec, partition="development")
        assert report["label"] == "supplementary_rehearsal"
        ids = [s["slice_id"] for s in report["results"][0]["slices"]]
        assert ids == [PARTITION_ALL_SLICE, "attack_A07"]

    def test_partial_matrix_is_refused(self):
        record, preds, spec = _setup()
        partial = [p for p in preds if not (p.candidate_id == "cand_b" and p.condition == "clean")]
        with pytest.raises(ValueError, match="cand_b"):
            run_supplementary(record, partial, spec, partition="evaluation")

    def test_candidates_must_hold_the_same_clips(self):
        record, preds, spec = _setup()
        broken = [
            dataclasses.replace(p, clip_id="x" + p.clip_id)
            if p.candidate_id == "cand_b" and p.condition == "clean" and p.clip_id == "clip00"
            else p
            for p in preds
        ]
        with pytest.raises(ValueError, match="disagree"):
            run_supplementary(record, broken, spec, partition="evaluation")

    def test_unknown_slice_or_candidate_is_refused(self):
        record, preds, _ = _setup()
        bad_slice = dataclasses.replace(_spec(), evaluation_slice_ids=("no_such_slice",))
        with pytest.raises(ValueError, match="no_such_slice"):
            run_supplementary(record, preds, bad_slice, partition="evaluation")
        bad_pair = _spec((Comparison("x", "cand_a", "ghost"),))
        with pytest.raises(ValueError, match="ghost"):
            run_supplementary(record, preds, bad_pair, partition="evaluation")

    def test_unsupported_bootstrap_unit_is_refused(self):
        record, preds, spec = _setup()
        record = dataclasses.replace(
            record, bootstrap=dataclasses.replace(record.bootstrap, resampling_unit="clip_id")
        )
        with pytest.raises(ValueError, match="speaker_id"):
            run_supplementary(record, preds, spec, partition="evaluation")


class TestSpecAndOutput:
    def test_repo_spec_matches_the_g2_freeze_record(self):
        spec = load_supplementary_spec(REPO / "configs" / "supplementary_analysis.yaml")
        record = load_freeze_record(REPO / "reports" / "evaluation_freeze_g2.json")
        slice_ids = {s.slice_id for s in record.slices}
        candidate_ids = {c.candidate_id for c in record.candidates}
        assert set(spec.evaluation_slice_ids) <= slice_ids
        assert set(spec.development_slice_ids) <= slice_ids
        assert {x for c in spec.comparisons for x in (c.a, c.b)} <= candidate_ids
        assert len(spec.comparisons) == 8
        assert len(spec.spec_sha256) == 64

    def test_spec_validation(self, tmp_path: Path):
        path = tmp_path / "spec.yaml"
        path.write_text(yaml.safe_dump({"version": "nope"}), encoding="utf-8")
        with pytest.raises(ValueError, match="version"):
            load_supplementary_spec(path)
        path.write_text(
            yaml.safe_dump({"version": "supplementary-v1", "metrics": ["accuracy"]}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="metrics"):
            load_supplementary_spec(path)

    def test_report_is_never_overwritten(self, tmp_path: Path):
        record, preds, spec = _setup()
        report = run_supplementary(record, preds, spec, partition="evaluation")
        write_supplementary_report(report, tmp_path)
        with pytest.raises(FileExistsError):
            write_supplementary_report(report, tmp_path)


def _make_store(tmp_path: Path, record, preds, *, partition: str, parity: bool = False) -> Path:
    store = create_prediction_store(tmp_path / f"predictions_{partition}.db")
    set_metadata_once(store, "freeze_identity", freeze_identity_hash(record))
    set_metadata_once(store, "partition", partition)
    set_metadata_once(store, "subset", "full")
    if parity:
        set_metadata_once(store, "dev_parity", "passed")
    write_predictions(store, preds, freeze_record_hash=freeze_identity_hash(record))
    return store


def _spec_file(tmp_path: Path) -> Path:
    path = tmp_path / "spec.yaml"
    path.write_text(
        yaml.safe_dump({
            "version": "supplementary-v1",
            "evaluation_slice_ids": ["attack_A07", "duration_lt_1s"],
            "development_slice_ids": ["attack_A07"],
            "metrics": list(ALL_METRICS),
            "comparisons": [{"id": "a_vs_b", "a": "cand_a", "b": "cand_b"}],
            "low_support_min_class_count": 5,
        }),
        encoding="utf-8",
    )
    return path


def _cli(tmp_path: Path, record_path: Path, store: Path, *extra: str) -> list[str]:
    return [
        "eval-supplement", "--freeze-record", str(record_path), "--store", str(store),
        "--out-dir", str(tmp_path / "out"), "--spec", str(_spec_file(tmp_path)), *extra,
    ]


class TestCli:
    def _reported(self, tmp_path: Path, store_hash: str | None):
        record, preds = _multi_pair_setup()
        store = _make_store(tmp_path, record, preds, partition="evaluation")
        reported = dataclasses.replace(
            record,
            state="REPORTED",
            scored_utc="2026-10-18T18:22:01+00:00",
            reported_utc="2026-10-18T22:00:00+00:00",
            predictions_hash=store_hash or compute_predictions_hash(store),
            report_hash="r" * 64,
        )
        record_path = tmp_path / "record.json"
        save_freeze_record(reported, record_path)
        return record_path, store

    def test_missing_store_is_refused(self, tmp_path: Path, capsys):
        record_path = tmp_path / "record.json"
        save_freeze_record(_freeze(), record_path)
        assert main(_cli(tmp_path, record_path, tmp_path / "none.db")) == 1
        assert json.loads(capsys.readouterr().out)["reason"] == "store_not_found"

    def test_record_must_be_reported(self, tmp_path: Path, capsys):
        record, preds = _multi_pair_setup()
        store = _make_store(tmp_path, record, preds, partition="evaluation")
        record_path = tmp_path / "record.json"
        save_freeze_record(record, record_path)
        assert main(_cli(tmp_path, record_path, store, "--authorization", "owner ok")) == 1
        assert json.loads(capsys.readouterr().out)["reason"] == "record_not_reported"

    def test_authorization_is_required_for_the_evaluation_partition(
        self, tmp_path: Path, capsys
    ):
        record_path, store = self._reported(tmp_path, None)
        assert main(_cli(tmp_path, record_path, store)) == 1
        assert json.loads(capsys.readouterr().out)["reason"] == "authorization_required"
        assert main(_cli(tmp_path, record_path, store, "--authorization", "  ")) == 1
        assert json.loads(capsys.readouterr().out)["reason"] == "authorization_required"

    def test_store_must_be_the_one_the_record_was_scored_from(self, tmp_path: Path, capsys):
        record_path, store = self._reported(tmp_path, "f" * 64)
        assert main(_cli(tmp_path, record_path, store, "--authorization", "owner ok")) == 1
        assert json.loads(capsys.readouterr().out)["reason"] == "store_hash_mismatch"

    def test_authorized_run_writes_once_and_records_the_authorization(
        self, tmp_path: Path, capsys
    ):
        record_path, store = self._reported(tmp_path, None)
        args = _cli(tmp_path, record_path, store, "--authorization", "owner go 2026-10-19")
        assert main(args) == 0
        summary = json.loads(capsys.readouterr().out)
        assert summary["status"] == "supplementary_post_report"
        report = json.loads(Path(summary["output"]).read_text(encoding="utf-8"))
        assert report["provenance"]["authorization"] == "owner go 2026-10-19"
        assert report["provenance"]["predictions_hash"] == compute_predictions_hash(store)
        assert main(args) == 1
        assert json.loads(capsys.readouterr().out)["reason"] == "supplementary_exists"
        # The record is untouched by the analysis.
        assert load_freeze_record(record_path).state == "REPORTED"

    def test_development_rehearsal_needs_no_authorization_or_reported_record(
        self, tmp_path: Path, capsys
    ):
        record, _ = _multi_pair_setup()
        rng = np.random.default_rng(3)
        preds = [
            _pred(
                f"d{i:02d}", "spoof" if i % 2 else "bonafide",
                float(np.clip(rng.normal(0.7 if i % 2 else 0.3, 0.2), 0, 1)),
                candidate_id=c, condition=cond, speaker_id=f"S{i % 4}",
                attack_id="A07" if i % 2 else None, official_split="development",
            )
            for c in ("cand_a", "cand_b") for cond in ("clean", "noise_10db")
            for i in range(20)
        ]
        store = _make_store(tmp_path, record, preds, partition="development", parity=True)
        record_path = tmp_path / "record.json"
        save_freeze_record(record, record_path)
        assert main(_cli(tmp_path, record_path, store)) == 0
        assert json.loads(capsys.readouterr().out)["status"] == "supplementary_rehearsal"
        assert load_freeze_record(record_path).state == "FROZEN"
