"""CLI-level tests for eval-freeze, eval-score and eval-report guards (#35).

Torch-free: bundles hold only run records, and every scoring path exercised
here refuses before any model or audio is touched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from conftest import full_manifest_rows
from test_eval_scoring import _bundle, _cnn_record, _lr_record

from voice_detector.cli import main
from voice_detector.evaluation import (
    CandidateDeclaration,
    create_draft,
    freeze_record,
    save_freeze_record,
)
from voice_detector.manifest import (
    build_catalog_meta,
    compute_manifest_hash,
    manifest_summary,
    write_manifest_sqlite,
)
from voice_detector.prediction_store import create_prediction_store, set_metadata_once

EVALUATION_CONFIG = Path("configs/evaluation_protocol.yaml").resolve()
PROJECT_CONFIG = Path("configs/default.yaml").resolve()


def _catalog(tmp_path: Path) -> Path:
    rows = full_manifest_rows()
    manifest_dir = tmp_path / "manifest"
    summary = manifest_summary(rows, manifest_hash=compute_manifest_hash(rows), missing=[])
    meta = build_catalog_meta(summary, json.dumps(summary, sort_keys=True), with_checksum=True)
    write_manifest_sqlite(rows, manifest_dir / "manifest.sqlite", meta=meta)
    return manifest_dir


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


def test_dry_run_declares_family_and_rule_from_bundle_content(tmp_path: Path, capsys):
    lr = _bundle(tmp_path, "lr_baseline", "lr_run_record.json", _lr_record())
    study = yaml.safe_load(Path("configs/cnn_study.yaml").read_text(encoding="utf-8"))
    cnn = _bundle(
        tmp_path, "cnn_clean_seed1", "cnn_run_record.json", _cnn_record(mel_config=study["mel"])
    )
    code = main(_freeze_args(_catalog(tmp_path), tmp_path / "f.json", lr, cnn))
    assert code == 0
    draft = json.loads(capsys.readouterr().out)
    declared = {c["candidate_id"]: c for c in draft["candidates"]}
    assert declared["lr_baseline"]["model_family"] == "logistic_regression"
    assert declared["lr_baseline"]["aggregation_rule"] == "mean_window_decision_function"
    assert declared["cnn_clean_seed1"]["model_family"] == "cnn"
    assert declared["cnn_clean_seed1"]["aggregation_rule"] == "mean_window_logit"
    assert declared["cnn_clean_seed1"]["seed"] == 20261004


def test_unknown_bundle_schema_is_refused(tmp_path: Path, capsys):
    bad = _bundle(
        tmp_path, "cnn_x", "cnn_run_record.json", _cnn_record(schema_version="mystery-v9")
    )
    code = main(_freeze_args(_catalog(tmp_path), tmp_path / "f.json", bad))
    assert code == 1
    assert json.loads(capsys.readouterr().out)["reason"] == "bundle_invalid"


def test_bundle_missing_seed_is_refused(tmp_path: Path, capsys):
    record = _lr_record()
    del record["seed"]
    bad = _bundle(tmp_path, "lr_x", "lr_run_record.json", record)
    code = main(_freeze_args(_catalog(tmp_path), tmp_path / "f.json", bad))
    assert code == 1
    assert json.loads(capsys.readouterr().out)["reason"] == "bundle_invalid"


def test_eval_score_without_catalog_is_refused(tmp_path: Path, capsys):
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


def test_eval_score_rejects_unknown_partition(tmp_path: Path):
    with pytest.raises(SystemExit):
        main(["eval-score", "--partition", "test"])


def _frozen_record(tmp_path: Path) -> Path:
    candidate = CandidateDeclaration(
        candidate_id="lr_baseline",
        model_family="logistic_regression",
        bundle_hash="x",
        threshold=0.25,
        seed=7,
        run_label="lr_baseline",
        aggregation_rule="mean_window_decision_function",
    )
    draft = create_draft(
        [candidate], full_manifest_rows(), EVALUATION_CONFIG.parent,
        evaluation_spec_path=EVALUATION_CONFIG,
    )
    path = tmp_path / "freeze.json"
    save_freeze_record(freeze_record(draft), path)
    return path


def _report_args(record: Path, store: Path, out: Path) -> list[str]:
    return [
        "eval-report",
        "--freeze-record", str(record),
        "--store", str(store),
        "--out-dir", str(out),
    ]


def test_report_refuses_a_store_scored_under_another_freeze(tmp_path: Path, capsys):
    record = _frozen_record(tmp_path)
    store = create_prediction_store(tmp_path / "p.db")
    set_metadata_once(store, "freeze_identity", "someone-else")
    code = main(_report_args(record, store, tmp_path / "out"))
    assert code == 1
    assert json.loads(capsys.readouterr().out)["reason"] == "store_freeze_mismatch"


def test_report_refuses_a_smoke_subset_store(tmp_path: Path, capsys):
    from voice_detector.evaluation import freeze_identity_hash, load_freeze_record

    record = _frozen_record(tmp_path)
    store = create_prediction_store(tmp_path / "p.db")
    set_metadata_once(store, "freeze_identity", freeze_identity_hash(load_freeze_record(record)))
    set_metadata_once(store, "subset", "max_clips=5")
    code = main(_report_args(record, store, tmp_path / "out"))
    assert code == 1
    assert json.loads(capsys.readouterr().out)["reason"] == "subset_store"


def test_report_refuses_an_incomplete_store(tmp_path: Path, capsys):
    from voice_detector.evaluation import freeze_identity_hash, load_freeze_record

    record = _frozen_record(tmp_path)
    store = create_prediction_store(tmp_path / "p.db")
    set_metadata_once(store, "freeze_identity", freeze_identity_hash(load_freeze_record(record)))
    set_metadata_once(store, "subset", "full")
    code = main(_report_args(record, store, tmp_path / "out"))
    assert code == 1
    assert json.loads(capsys.readouterr().out)["reason"] == "incomplete_store"


def test_cnn_bundle_with_other_mel_than_study_is_refused(tmp_path: Path, capsys):
    bad = _bundle(tmp_path, "cnn_x", "cnn_run_record.json", _cnn_record(mel_config={"n_mels": 80}))
    code = main(_freeze_args(_catalog(tmp_path), tmp_path / "f.json", bad))
    assert code == 1
    assert "cnn_study.yaml" in json.loads(capsys.readouterr().out)["detail"]
