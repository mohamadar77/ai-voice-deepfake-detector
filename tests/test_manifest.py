"""Dataset-free tests for the manifest validators and builder.

The builder tests synthesise tiny FLAC files and mini CM protocol files in a
``tmp_path`` tree, so they never touch the real ASVspoof dataset, network, GPU
or trained checkpoints.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voice_detector import manifest as m


def _write_flac(
    path: Path,
    *,
    seconds: float = 0.5,
    sample_rate_hz: int = 16000,
    channels: int = 1,
    subtype: str = "PCM_16",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(sample_rate_hz * seconds)
    if channels == 1:
        data = np.linspace(-0.1, 0.1, frames, dtype=np.float32)
    else:
        data = np.zeros((frames, channels), dtype=np.float32)
    sf.write(str(path), data, sample_rate_hz, subtype=subtype, format="FLAC")


_PROTOCOLS = {
    "ASVspoof2019.LA.cm.train.trn.txt": (
        "LA_0001 LA_T_1 - - bonafide\n"
        "LA_0001 LA_T_2 - A01 spoof\n"
        "LA_0002 LA_T_3 - A02 spoof\n"
    ),
    "ASVspoof2019.LA.cm.dev.trl.txt": (
        "LA_0003 LA_D_1 - - bonafide\nLA_0003 LA_D_2 - A01 spoof\n"
    ),
    "ASVspoof2019.LA.cm.eval.trl.txt": (
        "LA_0004 LA_E_1 - - bonafide\nLA_0004 LA_E_2 - A07 spoof\n"
    ),
}
_SPLIT_CLIPS = {
    "ASVspoof2019_LA_train": ["LA_T_1", "LA_T_2", "LA_T_3"],
    "ASVspoof2019_LA_dev": ["LA_D_1", "LA_D_2"],
    "ASVspoof2019_LA_eval": ["LA_E_1", "LA_E_2"],
}


def _make_la_tree(root: Path, **flac_kwargs: object) -> Path:
    protocols = root / m.CM_PROTOCOL_SUBDIR
    protocols.mkdir(parents=True, exist_ok=True)
    for name, text in _PROTOCOLS.items():
        (protocols / name).write_text(text, encoding="utf-8")
    for split_dir, clips in _SPLIT_CLIPS.items():
        for clip_id in clips:
            _write_flac(root / split_dir / "flac" / f"{clip_id}.flac", **flac_kwargs)
    return root


# --- pure helpers -----------------------------------------------------------


def test_file_checksum_matches_hashlib(tmp_path: Path) -> None:
    target = tmp_path / "blob.bin"
    target.write_bytes(b"deepfake-detector")
    assert m.file_checksum(target) == hashlib.sha256(b"deepfake-detector").hexdigest()


def test_audio_header_reports_native_values(tmp_path: Path) -> None:
    clip = tmp_path / "clip.flac"
    _write_flac(clip, seconds=0.25, sample_rate_hz=16000, channels=1)
    header = m.audio_header(clip)
    assert header.sample_rate_hz == 16000
    assert header.channels == 1
    assert header.duration_seconds == pytest.approx(0.25, abs=0.01)


def test_clip_audio_path_layout() -> None:
    path = m.clip_audio_path("/data/LA", "train", "LA_T_9")
    assert path == Path("/data/LA/ASVspoof2019_LA_train/flac/LA_T_9.flac")


# --- validation -------------------------------------------------------------


def _row(**overrides: object) -> dict[str, object]:
    row = {
        "clip_id": "LA_T_1",
        "speaker_id": "LA_0001",
        "source_id": None,
        "attack_id": None,
        "label": "bonafide",
        "official_split": "train",
        "development_role": None,
        "duration_seconds": 3.0,
        "sample_rate_hz": 16000,
        "channels": 1,
        "checksum_sha256": "a" * 64,
        "provenance": "test",
    }
    row.update(overrides)
    return row


def test_validate_row_requires_hex_checksum_by_default() -> None:
    with pytest.raises(ValueError, match="checksum"):
        m.validate_manifest_row(_row(checksum_sha256=""))


def test_validate_row_dry_run_allows_empty_checksum_but_not_nonstring() -> None:
    m.validate_manifest_row(_row(checksum_sha256=""), require_checksum=False)
    with pytest.raises(ValueError, match="checksum"):
        m.validate_manifest_row(_row(checksum_sha256=None), require_checksum=False)


def test_validate_row_rejects_zero_duration_even_in_dry_run() -> None:
    with pytest.raises(ValueError, match="duration"):
        m.validate_manifest_row(
            _row(duration_seconds=0.0, checksum_sha256=""), require_checksum=False
        )


# --- assertions -------------------------------------------------------------


def test_native_audio_assert_flags_wrong_rate_or_channels() -> None:
    rows = [_row(), _row(clip_id="LA_T_2", sample_rate_hz=8000)]
    with pytest.raises(ValueError, match="LA_T_2"):
        m.assert_native_audio(rows)


def test_speaker_disjointness_passes_and_catches_overlap() -> None:
    disjoint = [
        _row(clip_id="a", speaker_id="S1", official_split="train"),
        _row(clip_id="b", speaker_id="S2", official_split="development", development_role=None),
    ]
    m.assert_speakers_disjoint_across_splits(disjoint)

    leaking = [
        _row(clip_id="a", speaker_id="S1", official_split="train"),
        _row(clip_id="b", speaker_id="S1", official_split="development"),
    ]
    with pytest.raises(ValueError, match="Speaker overlap"):
        m.assert_speakers_disjoint_across_splits(leaking)


def test_speaker_disjointness_ignores_none_speaker() -> None:
    rows = [
        _row(clip_id="a", speaker_id=None, official_split="train"),
        _row(clip_id="b", speaker_id=None, official_split="development"),
    ]
    m.assert_speakers_disjoint_across_splits(rows)


def test_expected_counts_guard_with_custom_spec() -> None:
    rows = [
        _row(clip_id="t1", speaker_id="S1", official_split="train", attack_id=None),
        _row(clip_id="t2", speaker_id="S1", official_split="train", attack_id="A01", label="spoof"),
    ]
    spec = {"train": {"clips": 2, "speakers": 1, "attacks": {"A01": 1}}}
    m.assert_expected_la_counts(rows, expected=spec)
    wrong = {"train": {"clips": 3, "speakers": 1, "attacks": {"A01": 1}}}
    with pytest.raises(ValueError, match="clips"):
        m.assert_expected_la_counts(rows, expected=wrong)


# --- hashing ----------------------------------------------------------------


def test_manifest_hash_is_order_independent_and_role_sensitive() -> None:
    a = _row(clip_id="a")
    b = _row(clip_id="b", official_split="development")
    assert m.compute_manifest_hash([a, b]) == m.compute_manifest_hash([b, a])

    reserved = _row(clip_id="b", official_split="development", development_role="select")
    assert m.compute_manifest_hash([a, b]) != m.compute_manifest_hash([a, reserved])


# --- summary ----------------------------------------------------------------


def test_summary_reports_out_of_gate_durations() -> None:
    rows = [
        _row(clip_id="short", duration_seconds=0.5),
        _row(clip_id="ok", duration_seconds=4.0),
        _row(clip_id="long", duration_seconds=45.0),
    ]
    summary = m.manifest_summary(rows)
    gate = summary["splits"]["train"]["duration_out_of_gate"]
    assert gate == {"below_min": 1, "above_max": 1}


# --- end-to-end builder -----------------------------------------------------


def test_build_la_manifest_end_to_end(tmp_path: Path) -> None:
    root = _make_la_tree(tmp_path / "LA")
    rows, summary = m.build_la_manifest(root, enforce_expected_counts=False)
    assert summary["total_clips"] == 7
    assert {r["official_split"] for r in rows} == set(m.ALL_SPLITS)
    assert summary["manifest_hash"] is not None
    assert summary["expected_counts_enforced"] is False
    # speaker disjointness and native-format assertions ran without raising
    assert summary["splits"]["train"]["unique_speakers"] == 2
    # determinism: a second build of the same tree yields the same hash
    _, summary2 = m.build_la_manifest(root, enforce_expected_counts=False)
    assert summary2["manifest_hash"] == summary["manifest_hash"]


def test_build_missing_file_raises_unless_allowed(tmp_path: Path) -> None:
    root = _make_la_tree(tmp_path / "LA")
    (root / "ASVspoof2019_LA_train" / "flac" / "LA_T_3.flac").unlink()
    with pytest.raises(FileNotFoundError):
        m.build_la_manifest(root, enforce_expected_counts=False)
    rows, summary = m.build_la_manifest(root, allow_missing=True, enforce_expected_counts=False)
    assert summary["missing_files"] == 1
    assert summary["total_clips"] == 6


def test_dry_run_build_has_no_hash_and_empty_checksums(tmp_path: Path) -> None:
    root = _make_la_tree(tmp_path / "LA")
    rows, summary = m.build_la_manifest(
        root, with_checksum=False, enforce_expected_counts=False
    )
    assert summary["manifest_hash"] is None
    assert all(r["checksum_sha256"] == "" for r in rows)


# --- writers ----------------------------------------------------------------


def test_csv_omits_local_path_and_sqlite_keeps_it(tmp_path: Path) -> None:
    root = _make_la_tree(tmp_path / "LA")
    rows, summary = m.build_la_manifest(root, enforce_expected_counts=False)

    csv_path = m.write_manifest_csv(rows, tmp_path / "manifest.csv")
    header = csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert "local_path" not in header
    assert "clip_id" in header

    meta = m.build_catalog_meta(summary, "{}", with_checksum=True)
    db_path = m.write_manifest_sqlite(rows, tmp_path / "manifest.sqlite", meta=meta)
    conn = sqlite3.connect(str(db_path))
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(clips)")}
        assert "local_path" in cols
        (count,) = conn.execute("SELECT COUNT(*) FROM clips").fetchone()
        assert count == 7
        meta_rows = dict(conn.execute("SELECT key, value FROM catalog_meta"))
        assert meta_rows["schema_version"] == m.MANIFEST_SCHEMA_VERSION
        assert meta_rows["manifest_hash"] == summary["manifest_hash"]
    finally:
        conn.close()


def test_sqlite_check_rejects_short_checksum(tmp_path: Path) -> None:
    rows = [_row(checksum_sha256="tooshort")]
    with pytest.raises(sqlite3.IntegrityError):
        m.write_manifest_sqlite(rows, tmp_path / "bad.sqlite", include_local_path=False)
