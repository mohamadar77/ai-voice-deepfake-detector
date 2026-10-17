"""CLI-level tests for eval-freeze, eval-score and eval-report guards (#35).

Torch-free: bundles hold only run records and placeholder model files, and every
path exercised here refuses or reports before any model or audio is touched.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from conftest import full_manifest_rows
from test_eval_scoring import _bundle, _cnn_record, _lr_record

from voice_detector.cli import main
from voice_detector.evaluation import (
    freeze_identity_hash,
    load_freeze_record,
    mark_scored,
)
from voice_detector.manifest import (
    build_catalog_meta,
    compute_manifest_hash,
    manifest_summary,
    write_manifest_sqlite,
)
from voice_detector.prediction_store import (
    PredictionRow,
    create_prediction_store,
    set_metadata_once,
    write_predictions,
)

EVALUATION_CONFIG = Path("configs/evaluation_protocol.yaml").resolve()
PROJECT_CONFIG = Path("configs/default.yaml").resolve()


@pytest.fixture(autouse=True)
def artifacts_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the project config's artifacts root at the test directory."""

    root = tmp_path / "artifacts"
    root.mkdir()
    monkeypatch.setenv("VOICE_DETECTOR_ARTIFACTS_DIR", str(root))
    return root


def _catalog(tmp_path: Path) -> Path:
    rows = full_manifest_rows()
    manifest_dir = tmp_path / "manifest"
    summary = manifest_summary(rows, manifest_hash=compute_manifest_hash(rows), missing=[])
    meta = build_catalog_meta(summary, json.dumps(summary, sort_keys=True), with_checksum=True)
    write_manifest_sqlite(rows, manifest_dir / "manifest.sqlite", meta=meta)
    return manifest_dir


def _lr_bundle(root: Path, name: str = "lr_baseline") -> Path:
    bundle = _bundle(root, name, "lr_run_record.json", _lr_record())
    (bundle / "lr_pipeline.joblib").write_bytes(b"placeholder")
    return bundle


def _cnn_bundle(root: Path, name: str = "cnn_clean_seed1", **over: Any) -> Path:
    study = yaml.safe_load(Path("configs/cnn_study.yaml").read_text(encoding="utf-8"))
    record = _cnn_record(mel_config=study["mel"], **over)
    bundle = _bundle(root, name, "cnn_run_record.json", record)
    (bundle / "cnn_checkpoint.pt").write_bytes(b"placeholder")
    (bundle / "cnn_norm_stats.pt").write_bytes(b"placeholder")
    return bundle


def _freeze_args(manifest_dir: Path, out: Path, *bundles: Path, dry_run: bool = True) -> list[str]:
    args = [
        "eval-freeze",
        "--manifest-dir", str(manifest_dir),
        "--config", str(PROJECT_CONFIG),
        "--evaluation-config", str(EVALUATION_CONFIG),
        "--out", str(out),
    ]
    for bundle in bundles:
        args += ["--bundle", str(bundle)]
    return args + (["--dry-run"] if dry_run else [])


def _freeze(tmp_path: Path, artifacts_root: Path) -> Path:
    out = tmp_path / "freeze.json"
    code = main(
        _freeze_args(
            _catalog(tmp_path), out, _lr_bundle(artifacts_root), _cnn_bundle(artifacts_root),
            dry_run=False,
        )
    )
    assert code == 0
    return out


class TestFreeze:
    def test_dry_run_declares_family_and_rule_from_bundle_content(
        self, tmp_path: Path, artifacts_root: Path, capsys
    ):
        code = main(
            _freeze_args(
                _catalog(tmp_path), tmp_path / "f.json",
                _lr_bundle(artifacts_root), _cnn_bundle(artifacts_root),
            )
        )
        assert code == 0
        draft = json.loads(capsys.readouterr().out)
        declared = {c["candidate_id"]: c for c in draft["candidates"]}
        assert declared["lr_baseline"]["model_family"] == "logistic_regression"
        assert declared["lr_baseline"]["aggregation_rule"] == "mean_window_decision_function"
        assert declared["cnn_clean_seed1"]["model_family"] == "cnn"
        assert declared["cnn_clean_seed1"]["aggregation_rule"] == "mean_window_logit"
        assert declared["cnn_clean_seed1"]["seed"] == 20261004

    def test_silence_trim_is_a_frozen_condition(
        self, tmp_path: Path, artifacts_root: Path, capsys
    ):
        main(_freeze_args(_catalog(tmp_path), tmp_path / "f.json", _lr_bundle(artifacts_root)))
        draft = json.loads(capsys.readouterr().out)
        assert draft["conditions"] == [
            "clean", "codec_opus_24k", "noise_20db", "noise_10db", "silence_trim",
        ]

    @pytest.mark.parametrize(
        "make",
        [
            lambda root: _bundle(
                root, "cnn_x", "cnn_run_record.json", _cnn_record(schema_version="mystery-v9")
            ),
            lambda root: _bundle(
                root, "lr_x", "lr_run_record.json",
                {k: v for k, v in _lr_record().items() if k != "seed"},
            ),
        ],
    )
    def test_invalid_bundle_is_refused(self, tmp_path: Path, artifacts_root: Path, capsys, make):
        code = main(_freeze_args(_catalog(tmp_path), tmp_path / "f.json", make(artifacts_root)))
        assert code == 1
        assert json.loads(capsys.readouterr().out)["reason"] == "bundle_invalid"

    def test_bundle_without_model_files_is_refused(
        self, tmp_path: Path, artifacts_root: Path, capsys
    ):
        bare = _bundle(artifacts_root, "lr_bare", "lr_run_record.json", _lr_record())
        code = main(_freeze_args(_catalog(tmp_path), tmp_path / "f.json", bare))
        assert code == 1
        assert "lr_pipeline.joblib" in json.loads(capsys.readouterr().out)["detail"]

    def test_bundle_outside_artifacts_root_is_refused(
        self, tmp_path: Path, artifacts_root: Path, capsys
    ):
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        code = main(
            _freeze_args(_catalog(tmp_path), tmp_path / "f.json", _lr_bundle(elsewhere))
        )
        assert code == 1
        assert json.loads(capsys.readouterr().out)["reason"] == "bundle_outside_artifacts_root"

    def test_cnn_bundle_with_other_mel_than_study_is_refused(
        self, tmp_path: Path, artifacts_root: Path, capsys
    ):
        bad = _bundle(artifacts_root, "cnn_x", "cnn_run_record.json",
                      _cnn_record(mel_config={"n_mels": 80}))
        code = main(_freeze_args(_catalog(tmp_path), tmp_path / "f.json", bad))
        assert code == 1
        assert "cnn_study.yaml" in json.loads(capsys.readouterr().out)["detail"]


class TestScoreGuards:
    def _score(self, tmp_path: Path, record: Path, *extra: str) -> list[str]:
        return [
            "eval-score",
            "--manifest-dir", str(tmp_path / "manifest"),
            "--config", str(PROJECT_CONFIG),
            "--freeze-record", str(record),
            "--store", str(tmp_path / "p.db"),
            *extra,
        ]

    def test_without_catalog_is_refused(self, tmp_path: Path, capsys):
        code = main(
            [
                "eval-score",
                "--manifest-dir", str(tmp_path / "nowhere"),
                "--config", str(PROJECT_CONFIG),
                "--freeze-record", str(tmp_path / "f.json"),
                "--store", str(tmp_path / "p.db"),
            ]
        )
        assert code == 1
        assert json.loads(capsys.readouterr().out)["reason"] == "catalog_not_found"

    def test_unknown_partition_is_rejected_by_argparse(self):
        with pytest.raises(SystemExit):
            main(["eval-score", "--partition", "test"])

    def test_evaluation_partition_needs_a_dev_store(
        self, tmp_path: Path, artifacts_root: Path, capsys
    ):
        record = _freeze(tmp_path, artifacts_root)
        capsys.readouterr()
        code = main(self._score(tmp_path, record, "--partition", "evaluation"))
        assert code == 1
        assert json.loads(capsys.readouterr().out)["reason"] == "dev_store_required"
        assert not (tmp_path / "p.db").exists()

    def test_evaluation_partition_needs_a_passed_parity(
        self, tmp_path: Path, artifacts_root: Path, capsys
    ):
        record_path = _freeze(tmp_path, artifacts_root)
        capsys.readouterr()
        record = load_freeze_record(record_path)
        dev = _full_store(tmp_path, record, "development", parity=False)
        code = main(
            self._score(tmp_path, record_path, "--partition", "evaluation", "--dev-store", str(dev))
        )
        assert code == 1
        assert json.loads(capsys.readouterr().out)["reason"] == "parity_not_passed"

    def test_zero_torch_threads_is_refused(self, tmp_path: Path, artifacts_root: Path, capsys):
        record = _freeze(tmp_path, artifacts_root)
        capsys.readouterr()
        code = main(self._score(tmp_path, record, "--torch-threads", "0"))
        assert code == 1
        assert json.loads(capsys.readouterr().out)["reason"] == "invalid_torch_threads"


def _full_store(
    tmp_path: Path, record, partition: str, *, parity: bool = True, subset: str = "full"
) -> Path:
    """A complete synthetic store: every candidate x condition over four clips."""

    store = create_prediction_store(tmp_path / f"{partition}.db")
    set_metadata_once(store, "freeze_identity", freeze_identity_hash(record))
    set_metadata_once(store, "partition", partition)
    set_metadata_once(store, "subset", subset)
    if parity:
        set_metadata_once(store, "dev_parity", "passed")
    rows = [
        PredictionRow(
            clip_id=f"{partition[:3]}_{i}", candidate_id=c.candidate_id, condition=cond,
            clip_score=float(i) - 1.5 + 0.1 * k, label="bonafide" if i < 2 else "spoof",
            speaker_id=f"spk{i}", attack_id=None if i < 2 else "A07",
            official_split=partition, duration_seconds=2.0, n_windows=1,
            threshold=c.threshold, decision=i >= 2,
        )
        for c in record.candidates
        for k, cond in enumerate(record.conditions)
        for i in range(4)
    ]
    write_predictions(store, rows, freeze_record_hash=freeze_identity_hash(record))
    return store


class TestReport:
    def _report(self, record: Path, store: Path, out: Path, *extra: str) -> list[str]:
        return [
            "eval-report",
            "--freeze-record", str(record),
            "--store", str(store),
            "--out-dir", str(out),
            *extra,
        ]

    @pytest.fixture
    def record_path(self, tmp_path: Path, artifacts_root: Path, capsys) -> Path:
        path = _freeze(tmp_path, artifacts_root)
        capsys.readouterr()
        return path

    def test_missing_store(self, tmp_path: Path, record_path: Path, capsys):
        code = main(self._report(record_path, tmp_path / "none.db", tmp_path / "out"))
        assert code == 1
        assert json.loads(capsys.readouterr().out)["reason"] == "store_not_found"

    def test_store_without_partition_metadata_is_refused(
        self, tmp_path: Path, record_path: Path, capsys
    ):
        store = create_prediction_store(tmp_path / "legacy.db")
        code = main(self._report(record_path, store, tmp_path / "out"))
        assert code == 1
        assert json.loads(capsys.readouterr().out)["reason"] == "store_partition_unknown"

    def test_store_scored_under_another_freeze_is_refused(
        self, tmp_path: Path, record_path: Path, capsys
    ):
        store = create_prediction_store(tmp_path / "p.db")
        set_metadata_once(store, "partition", "evaluation")
        set_metadata_once(store, "freeze_identity", "someone-else")
        code = main(self._report(record_path, store, tmp_path / "out"))
        assert code == 1
        assert json.loads(capsys.readouterr().out)["reason"] == "store_freeze_mismatch"

    def test_smoke_subset_store_is_refused(self, tmp_path: Path, record_path: Path, capsys):
        record = load_freeze_record(record_path)
        store = _full_store(tmp_path, record, "development", subset="max_clips=5")
        code = main(self._report(record_path, store, tmp_path / "out"))
        assert code == 1
        assert json.loads(capsys.readouterr().out)["reason"] == "subset_store"

    def test_development_store_gives_a_rehearsal_and_leaves_the_record_alone(
        self, tmp_path: Path, record_path: Path, capsys
    ):
        record = load_freeze_record(record_path)
        store = _full_store(tmp_path, record, "development")
        before = record_path.read_text()
        code = main(self._report(record_path, store, tmp_path / "out"))
        assert code == 0
        assert json.loads(capsys.readouterr().out)["status"] == "rehearsal_report"
        assert record_path.read_text() == before
        assert load_freeze_record(record_path).state == "FROZEN"
        report = json.loads((tmp_path / "out" / "evaluation_report.json").read_text())
        assert report["provenance"]["rehearsal"] is True
        assert report["primary_partition"] == "development"

    def test_evaluation_store_needs_the_dev_store(
        self, tmp_path: Path, record_path: Path, capsys
    ):
        record = load_freeze_record(record_path)
        store = _full_store(tmp_path, record, "evaluation")
        code = main(self._report(record_path, store, tmp_path / "out"))
        assert code == 1
        assert json.loads(capsys.readouterr().out)["reason"] == "dev_store_required"
        assert load_freeze_record(record_path).state == "FROZEN"

    def test_dev_store_without_parity_cannot_support_a_locked_report(
        self, tmp_path: Path, record_path: Path, capsys
    ):
        record = load_freeze_record(record_path)
        evaluation = _full_store(tmp_path, record, "evaluation")
        dev = _full_store(tmp_path, record, "development", parity=False)
        code = main(self._report(record_path, evaluation, tmp_path / "out",
                                 "--dev-store", str(dev)))
        assert code == 1
        assert json.loads(capsys.readouterr().out)["reason"] == "parity_not_passed"
        assert load_freeze_record(record_path).state == "FROZEN"

    def test_locked_report_advances_the_record_and_uses_both_stores(
        self, tmp_path: Path, record_path: Path, capsys
    ):
        record = load_freeze_record(record_path)
        evaluation = _full_store(tmp_path, record, "evaluation")
        dev = _full_store(tmp_path, record, "development")
        code = main(self._report(record_path, evaluation, tmp_path / "out",
                                 "--dev-store", str(dev)))
        assert code == 0
        assert json.loads(capsys.readouterr().out)["status"] == "reported"
        assert load_freeze_record(record_path).state == "REPORTED"
        report = json.loads((tmp_path / "out" / "evaluation_report.json").read_text())
        assert report["primary_partition"] == "evaluation"
        assert report["provenance"]["rehearsal"] is False
        assert report["provenance"]["dev_predictions_hash"]
        first = report["results"][0]
        assert first["overall_metrics"]["n_clips"] == 4  # evaluation clips only, not 8

    def test_scored_record_rejects_a_different_store(
        self, tmp_path: Path, record_path: Path, capsys
    ):
        from voice_detector.evaluation import save_freeze_record

        record = load_freeze_record(record_path)
        save_freeze_record(mark_scored(record, "f" * 64), record_path)
        evaluation = _full_store(tmp_path, record, "evaluation")
        dev = _full_store(tmp_path, record, "development")
        code = main(self._report(record_path, evaluation, tmp_path / "out",
                                 "--dev-store", str(dev)))
        assert code == 1
        assert json.loads(capsys.readouterr().out)["reason"] == "store_hash_mismatch"
