"""End-to-end tests for ``voice-detector reserve-development`` (#31).

A synthetic manifest catalog is written to ``tmp_path``, so these tests exercise
the real read/plan/refuse/write path without any ASVspoof audio: the CLI reads
the catalog rather than the dataset, which is the whole point of freezing the
reservation without re-hashing 121k files.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Any

import pytest

from voice_detector import reservation as r
from voice_detector.cli import main
from voice_detector.manifest import (
    build_catalog_meta,
    compute_manifest_hash,
    manifest_summary,
    read_manifest_sqlite,
    write_manifest_sqlite,
)

RESERVATION_CONFIG = Path("configs/dev_reservation.yaml").resolve()
Rows = Callable[..., list[dict[str, Any]]]


def _catalog(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    """Write a manifest catalog with consistent meta, as the builder would."""

    manifest_dir = tmp_path / "manifest"
    summary = manifest_summary(rows, manifest_hash=compute_manifest_hash(rows), missing=[])
    summary_json = json.dumps(summary, indent=2, sort_keys=True)
    meta = build_catalog_meta(summary, summary_json, with_checksum=True)
    write_manifest_sqlite(rows, manifest_dir / "manifest.sqlite", meta=meta)
    return manifest_dir


def _run(
    manifest_dir: Path,
    record_out: Path,
    *extra: str,
    config: Path = RESERVATION_CONFIG,
) -> int:
    return main(
        [
            "reserve-development",
            "--manifest-dir",
            str(manifest_dir),
            "--reservation-config",
            str(config),
            "--record-out",
            str(record_out),
            *extra,
        ]
    )


def _config_with_fraction(tmp_path: Path, fraction: str) -> Path:
    """The committed reservation config with only its fraction changed."""

    text = RESERVATION_CONFIG.read_text(encoding="utf-8")
    assert "threshold_fraction: 0.25" in text, "fixture drifted from the committed config"
    path = tmp_path / "edited_reservation.yaml"
    path.write_text(
        text.replace("threshold_fraction: 0.25", f"threshold_fraction: {fraction}", 1),
        encoding="utf-8",
    )
    return path


def test_freeze_writes_roles_record_and_a_new_hash(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], make_rows: Rows
) -> None:
    manifest_dir = _catalog(tmp_path, make_rows())
    record_out = tmp_path / "reports" / "dev_reservation_frozen.json"
    before, _ = read_manifest_sqlite(manifest_dir / "manifest.sqlite")

    assert _run(manifest_dir, record_out) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "frozen"
    assert payload["changed"] is True
    assert payload["manifest_hash_before"] == compute_manifest_hash(before)
    assert payload["manifest_hash_after"] != payload["manifest_hash_before"]
    assert payload["n_speakers"] == {r.ROLE_SELECT: 15, r.ROLE_THRESHOLD: 5}

    # The catalog now carries the roles and satisfies the committed record.
    after, meta = read_manifest_sqlite(manifest_dir / "manifest.sqlite")
    record = json.loads(record_out.read_text(encoding="utf-8"))
    r.verify_frozen_reservation(after, record)
    assert meta["manifest_hash"] == payload["manifest_hash_after"]
    assert meta["reservation_seed"] == str(
        r.load_reservation_spec(RESERVATION_CONFIG).seed
    )
    assert meta["reservation_manifest_hash_before"] == payload["manifest_hash_before"]
    # The original build provenance is preserved, not overwritten.
    assert "build_utc" in meta and "reservation_frozen_utc" in meta

    # CSV and summary are rewritten in step with the catalog.
    csv_text = (manifest_dir / "manifest.csv").read_text(encoding="utf-8")
    assert "development_role" in csv_text.splitlines()[0]
    assert "local_path" not in csv_text.splitlines()[0]
    summary = json.loads((manifest_dir / "manifest_summary.json").read_text(encoding="utf-8"))
    roles = summary["splits"]["development"]["development_role_counts"]
    assert roles["unassigned"] == 0
    assert roles["select"] + roles["threshold"] == summary["splits"]["development"]["clips"]
    assert summary["splits"]["train"]["development_role_counts"]["unassigned"] > 0

    # No temporary files are left behind by the atomic writes.
    assert not list(manifest_dir.glob("*.tmp"))


def test_dry_run_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], make_rows: Rows
) -> None:
    manifest_dir = _catalog(tmp_path, make_rows())
    record_out = tmp_path / "reports" / "record.json"
    before = (manifest_dir / "manifest.sqlite").read_bytes()

    assert _run(manifest_dir, record_out, "--dry-run") == 0
    record = json.loads(capsys.readouterr().out)
    assert record["schema_version"] == r.RESERVATION_SCHEMA_VERSION
    assert not record_out.exists()
    assert (manifest_dir / "manifest.sqlite").read_bytes() == before
    assert not (manifest_dir / "manifest.csv").exists()


def test_rerunning_is_idempotent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], make_rows: Rows
) -> None:
    manifest_dir = _catalog(tmp_path, make_rows())
    record_out = tmp_path / "record.json"

    assert _run(manifest_dir, record_out) == 0
    first = json.loads(capsys.readouterr().out)
    assert _run(manifest_dir, record_out) == 0
    second = json.loads(capsys.readouterr().out)

    assert second["changed"] is False
    assert second["speakers"] == first["speakers"]
    assert second["manifest_hash_after"] == first["manifest_hash_after"]
    # A second freeze starts from the role-bearing manifest, so nothing moves.
    assert second["manifest_hash_before"] == second["manifest_hash_after"]


def test_a_different_existing_reservation_is_refused_without_force(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    make_rows: Rows,
    reservation_spec: r.ReservationSpec,
) -> None:
    # Freeze with a different seed first, then try the committed rule.
    other = r.plan_reservation(
        make_rows(), dc_replace(reservation_spec, seed=reservation_spec.seed + 1)
    )
    # Deterministic precondition: both seeds and the speaker list are fixed, so
    # asserting beats skipping, which could silently retire this test.
    assert not r.plan_reservation(other.rows, reservation_spec).already_frozen
    manifest_dir = _catalog(tmp_path, other.rows)
    record_out = tmp_path / "record.json"

    assert _run(manifest_dir, record_out) == 1
    refusal = json.loads(capsys.readouterr().out)
    assert refusal["status"] == "refused"
    assert refusal["reason"] == "already_frozen_differently"
    assert not record_out.exists()

    assert _run(manifest_dir, record_out, "--force") == 0
    assert json.loads(capsys.readouterr().out)["changed"] is True


def test_failed_coverage_is_refused_and_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], make_rows: Rows
) -> None:
    # A development set missing A06 cannot satisfy the coverage requirement.
    rows = make_rows(attacks=("A01", "A02", "A03", "A04", "A05"))
    manifest_dir = _catalog(tmp_path, rows)
    record_out = tmp_path / "record.json"

    assert _run(manifest_dir, record_out) == 1
    refusal = json.loads(capsys.readouterr().out)
    assert refusal["status"] == "refused"
    assert refusal["reason"] == "coverage_failed"
    assert any("A06" in failure for failure in refusal["failures"])
    # The evidence is reported, so the fraction can be adjusted knowingly.
    assert refusal["record"]["coverage_ok"] is False
    assert not record_out.exists()
    assert not (manifest_dir / "manifest.csv").exists()


def test_a_catalog_whose_rows_drifted_from_its_meta_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], make_rows: Rows
) -> None:
    manifest_dir = _catalog(tmp_path, make_rows())
    catalog = manifest_dir / "manifest.sqlite"
    conn = sqlite3.connect(str(catalog))
    try:
        conn.execute("DELETE FROM clips WHERE clip_id = 'LA_E_1'")
        conn.commit()
    finally:
        conn.close()

    assert _run(manifest_dir, tmp_path / "record.json") == 1
    refusal = json.loads(capsys.readouterr().out)
    assert refusal["reason"] == "catalog_hash_mismatch"


def test_a_missing_catalog_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _run(tmp_path / "absent", tmp_path / "record.json")


# --- the committed record outranks the rebuildable manifest -----------------


def test_a_rebuilt_manifest_cannot_silently_move_a_committed_membership(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], make_rows: Rows
) -> None:
    """The guard must survive a manifest rebuild, which is a routine event.

    The manifest is Git-ignored and costs hours to rebuild, so after a rebuild
    its ``development_role`` column is empty and has no membership to disagree
    with. If the conflict check read only the manifest, a re-run under an edited
    fraction would move a frozen protocol boundary at ``rc=0`` -- and if that
    happened after #8 had tuned, the contamination would be unrecoverable.
    """

    rows = make_rows()
    manifest_dir = _catalog(tmp_path, rows)
    record_out = tmp_path / "reports" / "dev_reservation_frozen.json"

    assert _run(manifest_dir, record_out) == 0
    frozen = json.loads(capsys.readouterr().out)
    committed = record_out.read_text(encoding="utf-8")

    # The host rebuilds the manifest: same clips, no roles.
    _catalog(tmp_path, rows)
    assert not r.extract_role_assignment(read_manifest_sqlite(manifest_dir / "manifest.sqlite")[0])

    # A re-run under a different fraction is refused on the record alone.
    edited = _config_with_fraction(tmp_path, "0.5")
    assert _run(manifest_dir, record_out, config=edited) == 1
    refusal = json.loads(capsys.readouterr().out)
    assert refusal["status"] == "refused"
    assert refusal["reason"] == "record_disagrees"
    assert refusal["committed_assignment"] != refusal["proposed_assignment"]
    assert record_out.read_text(encoding="utf-8") == committed

    # The same rule, however, restores the roles the record already froze.
    assert _run(manifest_dir, record_out) == 0
    restored = json.loads(capsys.readouterr().out)
    assert restored["matched_committed_record"] is True
    assert restored["speakers"] == frozen["speakers"]
    assert restored["manifest_hash_after"] == frozen["manifest_hash_after"]
    assert record_out.read_text(encoding="utf-8") == committed

    # Moving it needs an explicit, recorded override.
    assert _run(manifest_dir, record_out, "--force", config=edited) == 0
    forced = json.loads(capsys.readouterr().out)
    assert forced["forced"] is True
    assert forced["matched_committed_record"] is False
    assert forced["speakers"] != frozen["speakers"]


def test_an_unreadable_committed_record_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], make_rows: Rows
) -> None:
    """A corrupt authoritative record is never quietly overwritten."""

    manifest_dir = _catalog(tmp_path, make_rows())
    record_out = tmp_path / "record.json"
    record_out.write_text('{"speakers": {"select": ["LA_0001"]}}', encoding="utf-8")

    assert _run(manifest_dir, record_out) == 1
    refusal = json.loads(capsys.readouterr().out)
    assert refusal["reason"] == "unreadable_frozen_record"
    assert not (manifest_dir / "manifest.csv").exists()


# --- catalog provenance and a half-written column ---------------------------


def test_a_catalog_without_recorded_provenance_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], make_rows: Rows
) -> None:
    """Without ``catalog_meta`` the drift check has nothing to compare against.

    Freezing anyway would both skip the integrity check and, worse, let the
    freeze stamp a build commit and checksum coverage it cannot vouch for.
    """

    rows = make_rows()
    manifest_dir = tmp_path / "manifest"
    write_manifest_sqlite(rows, manifest_dir / "manifest.sqlite")
    record_out = tmp_path / "record.json"

    assert _run(manifest_dir, record_out) == 1
    refusal = json.loads(capsys.readouterr().out)
    assert refusal["reason"] == "catalog_provenance_missing"
    assert not record_out.exists()

    assert _run(manifest_dir, record_out, "--force") == 0
    assert json.loads(capsys.readouterr().out)["status"] == "frozen"
    _, meta = read_manifest_sqlite(manifest_dir / "manifest.sqlite")
    # Provenance this run cannot establish is recorded as unknown, not invented.
    assert meta["with_checksum"] == "unknown"
    assert meta["build_provenance"] == "unknown"
    assert "git_commit" not in meta
    assert meta["reservation_forced"] == "true"


def test_a_half_written_role_column_is_refused_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], make_rows: Rows
) -> None:
    """An unreadable reservation is a refusal with evidence, not a traceback.

    One clip of one speaker carrying a role is a partially written freeze. Even
    ``--dry-run``, which should always be safe to run, used to raise through the
    CLI with no way to repair the catalog.
    """

    # The catalog's own meta agrees with these rows, so the hash-drift check
    # passes and the unreadable role column is what the CLI has to handle.
    rows = make_rows()
    half_written = [
        {**row, "development_role": r.ROLE_SELECT}
        if row["clip_id"] == "LA_0002_D_B0"
        else row
        for row in rows
    ]
    assert half_written != rows
    manifest_dir = _catalog(tmp_path, half_written)
    catalog = manifest_dir / "manifest.sqlite"

    for extra in ((), ("--dry-run",)):
        assert _run(manifest_dir, tmp_path / "record.json", *extra) == 1
        refusal = json.loads(capsys.readouterr().out)
        assert refusal["reason"] == "inconsistent_existing_reservation"
        assert "more than one development_role" in refusal["error"]

    # --force repairs the column and records that it did.
    assert _run(manifest_dir, tmp_path / "record.json", "--force") == 0
    assert json.loads(capsys.readouterr().out)["status"] == "frozen"
    _, meta = read_manifest_sqlite(catalog)
    assert "reservation_overrode_inconsistent_column" in meta


def test_a_catalog_without_local_path_keeps_its_columns(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], make_rows: Rows
) -> None:
    """The freeze must not add an all-null column the catalog never had."""

    rows = make_rows()
    manifest_dir = tmp_path / "manifest"
    summary = manifest_summary(rows, manifest_hash=compute_manifest_hash(rows), missing=[])
    summary_json = json.dumps(summary, indent=2, sort_keys=True)
    write_manifest_sqlite(
        rows,
        manifest_dir / "manifest.sqlite",
        meta=build_catalog_meta(summary, summary_json, with_checksum=True),
        include_local_path=False,
    )

    assert _run(manifest_dir, tmp_path / "record.json") == 0
    capsys.readouterr()
    after, _ = read_manifest_sqlite(manifest_dir / "manifest.sqlite")
    assert "local_path" not in after[0]
    assert "development_role" in after[0]
