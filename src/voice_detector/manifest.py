"""Validation helpers for the canonical clip-level manifest."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

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


def validate_manifest_row(row: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` when one manifest row violates the data contract."""

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
    if not isinstance(row["duration_seconds"], (int, float)) or row["duration_seconds"] <= 0:
        raise ValueError("duration_seconds must be positive")
    if not isinstance(row["sample_rate_hz"], int) or row["sample_rate_hz"] <= 0:
        raise ValueError("sample_rate_hz must be a positive integer")
    if not isinstance(row["channels"], int) or row["channels"] <= 0:
        raise ValueError("channels must be a positive integer")
    checksum = row["checksum_sha256"]
    if not isinstance(checksum, str) or len(checksum) != 64:
        raise ValueError("checksum_sha256 must be a 64-character hexadecimal string")
    try:
        int(checksum, 16)
    except ValueError as error:
        raise ValueError("checksum_sha256 must be hexadecimal") from error
    if not isinstance(row["provenance"], str) or not row["provenance"]:
        raise ValueError("provenance must be a non-empty string")


def validate_manifest(rows: list[Mapping[str, Any]]) -> None:
    """Validate rows and reject duplicate clip identifiers."""

    seen: set[str] = set()
    for row in rows:
        validate_manifest_row(row)
        clip_id = row["clip_id"]
        if clip_id in seen:
            raise ValueError(f"Duplicate clip_id: {clip_id}")
        seen.add(clip_id)
