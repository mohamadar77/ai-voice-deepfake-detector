"""Functional test of scripts/archive_artifacts.sh on a fake artifact tree (no real data)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "archive_artifacts.sh"
BUNDLES = ["lr_baseline", "cnn_clean_seed1", "cnn_clean_seed2",
           "cnn_augmented_seed1", "cnn_augmented_seed2"]

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("sha256sum") is None,
    reason="needs bash and GNU sha256sum",
)


def _tree(root: Path) -> tuple[Path, Path, Path]:
    art = root / "artifacts"
    (art / "manifest").mkdir(parents=True)
    (art / "manifest" / "manifest.sqlite").write_bytes(b"catalog")
    (art / "evaluation" / "report_locked").mkdir(parents=True)
    (art / "evaluation" / "predictions_eval.db").write_bytes(b"rows")
    (art / "evaluation" / "report_locked" / "evaluation_report.json").write_text("{}")
    for bundle in BUNDLES:
        (art / bundle).mkdir()
        (art / bundle / "model.bin").write_bytes(bundle.encode())
        (art / bundle / f"{bundle}_run_record.json").write_text("{}")
    reported = root / "reported.json"
    reported.write_text('{"state": "REPORTED"}')
    frozen = root / "frozen.json"
    frozen.write_text('{"state": "FROZEN"}')
    return art, reported, frozen


def _run(art: Path, reported: Path, frozen: Path, dest: Path, *extra: str):
    return subprocess.run(
        ["bash", str(SCRIPT), "--artifacts", str(art), "--reported-record", str(reported),
         "--frozen-record", str(frozen), "--dest", str(dest), *extra],
        capture_output=True, text=True, check=False,
    )


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    art, reported, frozen = _tree(tmp_path)
    dest = tmp_path / "archive"
    result = _run(art, reported, frozen, dest, "--dry-run")
    assert result.returncode == 0, result.stderr
    assert not dest.exists()


def test_archive_copies_and_checksums_and_refuses_overwrite(tmp_path: Path) -> None:
    art, reported, frozen = _tree(tmp_path)
    dest = tmp_path / "archive"
    result = _run(art, reported, frozen, dest)
    try:
        assert result.returncode == 0, result.stderr
        assert (dest / "SHA256SUMS").is_file()
        assert (dest / "bundles" / "cnn_augmented_seed1" / "model.bin").read_bytes() == (
            b"cnn_augmented_seed1"
        )
        assert (dest / "freeze" / "evaluation_freeze_g2.REPORTED.json").is_file()
        assert (dest / "freeze" / "evaluation_freeze_g2.FROZEN.json").is_file()
        record = dest / "run_records" / "cnn_clean_seed1" / "cnn_clean_seed1_run_record.json"
        assert record.is_file()
        again = _run(art, reported, frozen, dest)
        assert again.returncode == 1 and "not empty" in again.stderr
    finally:
        subprocess.run(["chmod", "-R", "u+w", str(dest)], check=False)


def test_refuses_wrong_record_state_and_live_database(tmp_path: Path) -> None:
    art, reported, frozen = _tree(tmp_path)
    swapped = _run(art, frozen, reported, tmp_path / "a1")
    assert swapped.returncode == 1 and "not in state REPORTED" in swapped.stderr
    (art / "evaluation" / "predictions_eval.db-wal").write_bytes(b"x")
    live = _run(art, reported, frozen, tmp_path / "a2")
    assert live.returncode == 1 and "-wal" in live.stderr
    assert not (tmp_path / "a2").exists()
