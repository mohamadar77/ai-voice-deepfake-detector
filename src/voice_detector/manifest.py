"""Validation and construction of the canonical clip-level manifest.

Two layers live here:

* Pure, dataset-free validation of a single manifest row and of a list of rows
  against the data contract in ``docs/data.md`` (used since G0).
* The manifest *builder*, which turns parsed ASVspoof 2019 LA CM protocol
  entries into one row per original clip. For each clip it reads the audio file
  *header* (duration, native sample rate, channel count) and hashes the file
  bytes. It reads headers, not full waveforms, so it stays fast over the whole
  dataset, and it records the file's *native* rate and channel count so the
  16 kHz / mono contract can be asserted rather than silently normalised away
  (which is what the shared decoder in ``preprocessing.load_waveform`` would do).

The builder never fits a model, inspects labels for selection, or crosses the
official partition boundaries: it assigns every row to its protocol-declared
``official_split`` and asserts that speakers do not leak across splits. The
per-file SHA-256 is a drift anchor for later caching, not verification against
an authority — the authoritative archive hashes live in ``docs/data.md``.
"""

from __future__ import annotations

import csv
import hashlib
import sqlite3
import subprocess
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from voice_detector.protocol import ProtocolEntry

REQUIRED_FIELDS = frozenset(
    {
        "clip_id",
        "label",
        "official_split",
        "duration_seconds",
        "sample_rate_hz",
        "channels",
        "checksum_sha256",
        "provenance",
    }
)
VALID_LABELS = frozenset({"bonafide", "spoof"})
VALID_SPLITS = frozenset({"train", "development", "evaluation"})
VALID_DEVELOPMENT_ROLES = frozenset({None, "select", "threshold"})


def validate_manifest_row(row: Mapping[str, Any], *, require_checksum: bool = True) -> None:
    """Raise ``ValueError`` when one manifest row violates the data contract.

    With ``require_checksum`` false (a dry-run build without hashing) the
    checksum must still be a string, but its length/hex form is not enforced;
    every other structural check always runs.
    """

    missing = sorted(field for field in REQUIRED_FIELDS if field not in row)
    if missing:
        raise ValueError(f"Manifest row is missing fields: {', '.join(missing)}")

    if not isinstance(row["clip_id"], str) or not row["clip_id"]:
        raise ValueError("clip_id must be a non-empty string")
    if row["label"] not in VALID_LABELS:
        raise ValueError(f"label must be one of {sorted(VALID_LABELS)}")
    if row["official_split"] not in VALID_SPLITS:
        raise ValueError(f"official_split must be one of {sorted(VALID_SPLITS)}")
    if row.get("development_role") not in VALID_DEVELOPMENT_ROLES:
        raise ValueError("development_role must be null, select, or threshold")
    if row["official_split"] != "development" and row.get("development_role") is not None:
        raise ValueError("development_role is allowed only on development rows")
    if not isinstance(row["duration_seconds"], int | float) or row["duration_seconds"] <= 0:
        raise ValueError("duration_seconds must be positive")
    if not isinstance(row["sample_rate_hz"], int) or row["sample_rate_hz"] <= 0:
        raise ValueError("sample_rate_hz must be a positive integer")
    if not isinstance(row["channels"], int) or row["channels"] <= 0:
        raise ValueError("channels must be a positive integer")
    checksum = row["checksum_sha256"]
    if not isinstance(checksum, str):
        raise ValueError("checksum_sha256 must be a string")
    if require_checksum:
        if len(checksum) != 64:
            raise ValueError("checksum_sha256 must be a 64-character hexadecimal string")
        try:
            int(checksum, 16)
        except ValueError as error:
            raise ValueError("checksum_sha256 must be hexadecimal") from error
    if not isinstance(row["provenance"], str) or not row["provenance"]:
        raise ValueError("provenance must be a non-empty string")


def validate_manifest(rows: list[Mapping[str, Any]], *, require_checksum: bool = True) -> None:
    """Validate rows and reject duplicate clip identifiers."""

    seen: set[str] = set()
    for row in rows:
        validate_manifest_row(row, require_checksum=require_checksum)
        clip_id = row["clip_id"]
        if clip_id in seen:
            raise ValueError(f"Duplicate clip_id: {clip_id}")
        seen.add(clip_id)


# ---------------------------------------------------------------------------
# Manifest builder (reads audio headers; runs on the host where data lives)
# ---------------------------------------------------------------------------

EXPECTED_SAMPLE_RATE_HZ = 16000
EXPECTED_CHANNELS = 1
AUDIO_SUFFIX = ".flac"

# Supported decoded-duration gate, mirrored from the serving API's
# DEFAULT_MIN_DURATION_SECONDS / DEFAULT_MAX_DURATION_SECONDS in ``api.py``.
# Duplicated as a constant (not imported) to keep this data tool off the
# FastAPI/scoring import chain; the summary reports clips outside this range so
# train/serve duration skew is visible before training.
MIN_DURATION_SECONDS = 1.0
MAX_DURATION_SECONDS = 30.0

MANIFEST_SCHEMA_VERSION = "manifest-v1"

# Release identity recorded once in the catalog metadata (see docs/data.md).
DATASET_NAME = "ASVspoof 2019 LA"
DATASET_LICENSE = "ODC-By 1.0"
DATASET_DATASHARE_HANDLE = "https://datashare.ed.ac.uk/handle/10283/3336"
LA_ZIP_SHA256 = "208a7e4e3913f8c75ae1afd19bf32a5b29ae68435e9e30e23e5e98b6a155e4ec"

CM_PROTOCOL_SUBDIR = "ASVspoof2019_LA_cm_protocols"
CM_PROTOCOL_FILES = {
    "train": "ASVspoof2019.LA.cm.train.trn.txt",
    "development": "ASVspoof2019.LA.cm.dev.trl.txt",
    "evaluation": "ASVspoof2019.LA.cm.eval.trl.txt",
}
SPLIT_AUDIO_SUBDIR = {
    "train": "ASVspoof2019_LA_train",
    "development": "ASVspoof2019_LA_dev",
    "evaluation": "ASVspoof2019_LA_eval",
}
ALL_SPLITS = ("train", "development", "evaluation")

# Published ASVspoof 2019 LA figures (verified in docs/data.md). The integrity
# guard asserts these for a full three-split build so a partial extraction or a
# swapped protocol cannot produce a "valid" manifest.
_KNOWN_ATTACKS = {attack: 1 for attack in (f"A0{i}" for i in range(1, 7))}
_UNKNOWN_ATTACKS = tuple(f"A{i:02d}" for i in range(7, 20))
EXPECTED_LA = {
    "train": {"clips": 25380, "speakers": 20, "attacks": {a: 3800 for a in _KNOWN_ATTACKS}},
    "development": {"clips": 24844, "speakers": 20, "attacks": {a: 3716 for a in _KNOWN_ATTACKS}},
    "evaluation": {"clips": 71237, "speakers": 67, "attacks": {a: 4914 for a in _UNKNOWN_ATTACKS}},
}

# Exported column order. ``local_path`` is kept only in the SQLite catalog (which
# lives in an ignored directory); the canonical CSV omits it per docs/data.md.
MANIFEST_COLUMNS = (
    "clip_id",
    "speaker_id",
    "source_id",
    "attack_id",
    "label",
    "official_split",
    "development_role",
    "duration_seconds",
    "sample_rate_hz",
    "channels",
    "checksum_sha256",
    "provenance",
)

_SQLITE_TYPES = {
    "clip_id": "TEXT PRIMARY KEY",
    "speaker_id": "TEXT",
    "source_id": "TEXT",
    "attack_id": "TEXT",
    "label": "TEXT NOT NULL",
    "official_split": "TEXT NOT NULL",
    "development_role": "TEXT",
    "duration_seconds": "REAL NOT NULL",
    "sample_rate_hz": "INTEGER NOT NULL",
    "channels": "INTEGER NOT NULL",
    "checksum_sha256": "TEXT NOT NULL CHECK(length(checksum_sha256) = 64)",
    "provenance": "TEXT NOT NULL",
    "local_path": "TEXT",
}


class AudioHeader(NamedTuple):
    """Native audio metadata read from a file header."""

    duration_seconds: float
    sample_rate_hz: int
    channels: int


def file_checksum(path: Path | str, *, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest of a file, read in chunks."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def audio_header(path: Path | str) -> AudioHeader:
    """Return native ``(duration_seconds, sample_rate_hz, channels)``.

    Reads only the audio file header via :func:`soundfile.info`; it does not
    decode samples, so it is fast over the whole dataset and reports the file's
    *native* rate and channel count. ``soundfile`` is imported lazily so the
    validators and the metadata-only protocol path stay importable without it.
    """

    import soundfile as sf

    info = sf.info(str(path))
    return AudioHeader(float(info.duration), int(info.samplerate), int(info.channels))


def clip_audio_path(la_root: Path | str, official_split: str, clip_id: str) -> Path:
    """Resolve the FLAC path for one clip inside an ASVspoof 2019 LA root."""

    if official_split not in SPLIT_AUDIO_SUBDIR:
        raise ValueError(f"official_split must be one of {sorted(SPLIT_AUDIO_SUBDIR)}")
    return Path(la_root) / SPLIT_AUDIO_SUBDIR[official_split] / "flac" / f"{clip_id}{AUDIO_SUFFIX}"


def _manifest_row(
    entry: ProtocolEntry,
    *,
    duration_seconds: float,
    sample_rate_hz: int,
    channels: int,
    checksum_sha256: str,
    local_path: str,
) -> dict[str, Any]:
    return {
        "clip_id": entry.clip_id,
        "speaker_id": entry.speaker_id,
        # LA CM publishes no source-recording grouping; never inferred.
        "source_id": None,
        "attack_id": entry.attack_id,
        "label": entry.label,
        "official_split": entry.official_split,
        "development_role": None,
        "duration_seconds": duration_seconds,
        "sample_rate_hz": sample_rate_hz,
        "channels": channels,
        "checksum_sha256": checksum_sha256,
        "provenance": (
            f"ASVspoof 2019 LA CM; split={entry.official_split}; "
            "source_id unavailable (no source-recording grouping published) "
            "-- clip-level aggregation/bootstrap only"
        ),
        "local_path": local_path,
    }


def build_manifest_rows(
    entries: Iterable[ProtocolEntry],
    la_root: Path | str,
    *,
    with_checksum: bool = True,
    allow_missing: bool = False,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Build manifest rows for parsed protocol entries.

    For each entry the clip's audio file is located, its header is read for
    duration/rate/channels, and (when ``with_checksum``) its bytes are hashed.
    Returns the rows and the list of ``clip_id`` values whose audio file was
    missing. A non-empty missing list raises unless ``allow_missing`` is set.
    """

    la_root = Path(la_root)
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for entry in entries:
        path = clip_audio_path(la_root, entry.official_split, entry.clip_id)
        if not path.is_file():
            missing.append(entry.clip_id)
            continue
        header = audio_header(path)
        rows.append(
            _manifest_row(
                entry,
                duration_seconds=header.duration_seconds,
                sample_rate_hz=header.sample_rate_hz,
                channels=header.channels,
                checksum_sha256=file_checksum(path) if with_checksum else "",
                local_path=str(path.relative_to(la_root)),
            )
        )
    if missing and not allow_missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} referenced audio file(s) are missing (e.g. {preview})"
        )
    return rows, missing


def assert_unique_clip_ids(rows: Iterable[Mapping[str, Any]]) -> None:
    """Raise when any ``clip_id`` appears more than once across the manifest."""

    counts = Counter(row["clip_id"] for row in rows)
    duplicates = sorted(clip_id for clip_id, count in counts.items() if count > 1)
    if duplicates:
        preview = ", ".join(duplicates[:5])
        raise ValueError(f"{len(duplicates)} duplicate clip_id(s) (e.g. {preview})")


def assert_native_audio(
    rows: Iterable[Mapping[str, Any]],
    *,
    sample_rate_hz: int = EXPECTED_SAMPLE_RATE_HZ,
    channels: int = EXPECTED_CHANNELS,
) -> None:
    """Raise when any clip is not the expected native rate and channel count."""

    offenders = [
        row["clip_id"]
        for row in rows
        if row["sample_rate_hz"] != sample_rate_hz or row["channels"] != channels
    ]
    if offenders:
        preview = ", ".join(offenders[:5])
        raise ValueError(
            f"{len(offenders)} clip(s) are not {sample_rate_hz} Hz / {channels}-channel "
            f"(e.g. {preview})"
        )


def assert_speakers_disjoint_across_splits(rows: Iterable[Mapping[str, Any]]) -> None:
    """Raise when a speaker appears in more than one official split.

    Speaker-disjoint partitions are the precondition for a leakage-free,
    speaker-grouped development reservation, so this is asserted before any
    manifest is frozen. Rows without a speaker id cannot establish overlap and
    are ignored here.
    """

    by_split: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        speaker_id = row["speaker_id"]
        if speaker_id is not None:
            by_split[row["official_split"]].add(speaker_id)
    splits = sorted(by_split)
    for index, left in enumerate(splits):
        for right in splits[index + 1 :]:
            overlap = by_split[left] & by_split[right]
            if overlap:
                preview = ", ".join(sorted(overlap)[:5])
                raise ValueError(f"Speaker overlap between {left} and {right}: {preview}")


def assert_expected_la_counts(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected: Mapping[str, Mapping[str, Any]] = EXPECTED_LA,
) -> None:
    """Raise when row/speaker/attack counts do not match the published figures.

    The highest-value evaluation-integrity guard: a partial extraction or a
    swapped/mislabelled protocol cannot yield a manifest that passes.
    """

    by_split: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_split[row["official_split"]].append(row)
    for split, spec in expected.items():
        split_rows = by_split.get(split, [])
        if len(split_rows) != spec["clips"]:
            raise ValueError(f"{split}: expected {spec['clips']} clips, got {len(split_rows)}")
        speakers = len({row["speaker_id"] for row in split_rows})
        if speakers != spec["speakers"]:
            raise ValueError(f"{split}: expected {spec['speakers']} speakers, got {speakers}")
        attacks = Counter(
            row["attack_id"] for row in split_rows if row["attack_id"] is not None
        )
        if dict(attacks) != dict(spec["attacks"]):
            raise ValueError(
                f"{split}: attack distribution mismatch "
                f"(got {dict(sorted(attacks.items()))})"
            )


def compute_manifest_hash(rows: Iterable[Mapping[str, Any]]) -> str:
    """Return a stable content hash over the identity-defining fields.

    Covers ``(clip_id, label, official_split, development_role, checksum)`` per
    clip. Order-independent (rows are sorted by ``clip_id``). ``development_role``
    is included now, while it is uniformly null, so the formula never changes
    after a hash is recorded; it distinguishes two manifests that differ only in
    the frozen select/threshold reservation. Duration/rate/channels are omitted
    (byte-determined by the checksum) and the preprocessing version is omitted
    (the manifest is pre-feature).
    """

    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda row: row["clip_id"]):
        role = row.get("development_role") or ""
        line = (
            f"{row['clip_id']}|{row['label']}|{row['official_split']}|"
            f"{role}|{row['checksum_sha256']}\n"
        )
        digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def manifest_summary(
    rows: list[Mapping[str, Any]],
    *,
    manifest_hash: str | None = None,
    missing: list[str] | None = None,
    expected_counts_enforced: bool = False,
) -> dict[str, Any]:
    """Produce a small JSON-serialisable audit summary of built manifest rows."""

    per_split: dict[str, Any] = {}
    for split in sorted({row["official_split"] for row in rows}):
        split_rows = [row for row in rows if row["official_split"] == split]
        labels = Counter(row["label"] for row in split_rows)
        durations = [row["duration_seconds"] for row in split_rows]
        per_split[split] = {
            "clips": len(split_rows),
            "label_counts": {label: labels.get(label, 0) for label in sorted(VALID_LABELS)},
            "unique_speakers": len({row["speaker_id"] for row in split_rows}),
            "duration_seconds": {
                "min": round(min(durations), 3),
                "max": round(max(durations), 3),
                "mean": round(sum(durations) / len(durations), 3),
            }
            if durations
            else None,
            "duration_out_of_gate": {
                "below_min": sum(1 for value in durations if value < MIN_DURATION_SECONDS),
                "above_max": sum(1 for value in durations if value > MAX_DURATION_SECONDS),
            },
        }
    return {
        "total_clips": len(rows),
        "splits": per_split,
        "manifest_hash": manifest_hash,
        "missing_files": len(missing or []),
        "expected_counts_enforced": expected_counts_enforced,
        "duration_gate_seconds": [MIN_DURATION_SECONDS, MAX_DURATION_SECONDS],
    }


def build_la_manifest(
    la_root: Path | str,
    *,
    splits: Iterable[str] | None = None,
    with_checksum: bool = True,
    allow_missing: bool = False,
    enforce_expected_counts: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the combined LA manifest and assert the integrity invariants.

    Parses each split's CM protocol under ``la_root``, builds one row per clip,
    then asserts clip-id uniqueness, native 16 kHz/mono audio, and speaker
    disjointness across splits. For a full three-split build the published
    ASVspoof 2019 LA counts are also asserted unless ``enforce_expected_counts``
    is false. With ``with_checksum`` the rows are fully validated and a manifest
    hash is computed.
    """

    from voice_detector.protocol import parse_protocol_file

    la_root = Path(la_root)
    selected = tuple(splits) if splits is not None else ALL_SPLITS

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for split in selected:
        if split not in CM_PROTOCOL_FILES:
            raise ValueError(f"split must be one of {sorted(CM_PROTOCOL_FILES)}")
        protocol_path = la_root / CM_PROTOCOL_SUBDIR / CM_PROTOCOL_FILES[split]
        entries = parse_protocol_file(protocol_path, split)
        split_rows, split_missing = build_manifest_rows(
            entries, la_root, with_checksum=with_checksum, allow_missing=allow_missing
        )
        rows.extend(split_rows)
        missing.extend(split_missing)

    assert_unique_clip_ids(rows)
    assert_native_audio(rows)
    assert_speakers_disjoint_across_splits(rows)

    building_full = set(selected) == set(ALL_SPLITS)
    enforced = enforce_expected_counts and building_full
    if enforced:
        assert_expected_la_counts(rows)

    validate_manifest(rows, require_checksum=with_checksum)

    manifest_hash = compute_manifest_hash(rows) if with_checksum else None
    summary = manifest_summary(
        rows,
        manifest_hash=manifest_hash,
        missing=missing,
        expected_counts_enforced=enforced,
    )
    return rows, summary


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _package_version() -> str:
    try:
        from voice_detector import __version__
    except ImportError:
        return "unknown"
    return __version__


def build_catalog_meta(
    summary: Mapping[str, Any],
    summary_json: str,
    *,
    with_checksum: bool,
) -> dict[str, str]:
    """Build the reproducibility metadata recorded alongside the catalog.

    ``summary_json`` must be the exact text written to the sidecar JSON so the
    ``summary_json`` value in the database and the sidecar file cannot diverge.
    """

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "manifest_hash": str(summary.get("manifest_hash")),
        "build_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "package_version": _package_version(),
        "dataset": DATASET_NAME,
        "license": DATASET_LICENSE,
        "datashare_handle": DATASET_DATASHARE_HANDLE,
        "la_zip_sha256": LA_ZIP_SHA256,
        "with_checksum": str(with_checksum).lower(),
        "expected_counts_enforced": str(summary.get("expected_counts_enforced", False)).lower(),
        "total_clips": str(summary.get("total_clips")),
        "summary_json": summary_json,
    }


def write_manifest_csv(
    rows: Iterable[Mapping[str, Any]],
    path: Path | str,
    *,
    include_local_path: bool = False,
) -> Path:
    """Write the manifest to CSV with a fixed column order (no ``local_path``)."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(MANIFEST_COLUMNS) + (["local_path"] if include_local_path else [])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: "" if row.get(col) is None else row[col] for col in columns})
    return path


def write_manifest_sqlite(
    rows: Iterable[Mapping[str, Any]],
    path: Path | str,
    *,
    include_local_path: bool = True,
    meta: Mapping[str, str] | None = None,
) -> Path:
    """Write the manifest to a fresh SQLite catalog with a documented schema.

    The ``clips`` table keeps ``local_path`` (the catalog lives in an ignored
    directory) and a ``CHECK`` that rejects a non-64-char checksum. Indexes cover
    the query axes (split, speaker). When ``meta`` is given a ``catalog_meta``
    key/value table records the reproducibility metadata.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    columns = list(MANIFEST_COLUMNS) + (["local_path"] if include_local_path else [])
    create = f"CREATE TABLE clips ({', '.join(f'{col} {_SQLITE_TYPES[col]}' for col in columns)})"
    insert = (
        f"INSERT INTO clips ({', '.join(columns)}) VALUES ({', '.join('?' for _ in columns)})"
    )
    payload = [tuple(row.get(col) for col in columns) for row in rows]
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(create)
        conn.execute("CREATE INDEX idx_clips_official_split ON clips(official_split)")
        conn.execute("CREATE INDEX idx_clips_speaker_id ON clips(speaker_id)")
        conn.executemany(insert, payload)
        if meta is not None:
            conn.execute("CREATE TABLE catalog_meta (key TEXT PRIMARY KEY, value TEXT)")
            conn.executemany(
                "INSERT INTO catalog_meta (key, value) VALUES (?, ?)", list(meta.items())
            )
        conn.commit()
    finally:
        conn.close()
    return path
