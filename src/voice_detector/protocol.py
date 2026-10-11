"""Parsing and metadata-only auditing of ASVspoof 2019 LA CM protocol files.

The official countermeasure (CM) protocol files distribute one clip per line in
five whitespace-separated columns::

    SPEAKER_ID  AUDIO_FILE_NAME  -  SYSTEM_ID  KEY

Column three is an unused placeholder. ``SYSTEM_ID`` is ``-`` for bona fide
clips and an attack identifier (for example ``A01``) for spoofed clips, and
``KEY`` is ``bonafide`` or ``spoof``.

This module turns those lines into structured :class:`ProtocolEntry` records and
summarises them. It never reads audio, so it is safe to run before the dataset
is present and inside fast, dataset-free unit tests. Duration, sample rate,
channel count and checksum are intentionally absent here: they belong to the
later manifest builder that decodes the audio referenced by ``clip_id``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from voice_detector.manifest import VALID_LABELS, VALID_SPLITS

_NULL_TOKEN = "-"
_EXPECTED_COLUMNS = 5


@dataclass(frozen=True)
class ProtocolEntry:
    """One clip as declared by the official ASVspoof 2019 LA CM protocol.

    ``attack_id`` is ``None`` for bona fide clips and the published attack
    identifier (such as ``A07``) for spoofed clips. Fields map onto the
    canonical manifest contract documented in ``docs/data.md``.
    """

    speaker_id: str
    clip_id: str
    attack_id: str | None
    label: str
    official_split: str


def _require_valid_split(official_split: str) -> None:
    if official_split not in VALID_SPLITS:
        raise ValueError(f"official_split must be one of {sorted(VALID_SPLITS)}")


def parse_protocol_line(
    line: str,
    official_split: str,
    *,
    line_number: int | None = None,
) -> ProtocolEntry | None:
    """Parse one protocol line, or return ``None`` for a blank line.

    ``line_number`` is used only to make error messages locatable; pass it when
    iterating a file so malformed rows are easy to find.
    """

    _require_valid_split(official_split)
    where = f" (line {line_number})" if line_number is not None else ""

    stripped = line.strip()
    if not stripped:
        return None

    fields = stripped.split()
    if len(fields) != _EXPECTED_COLUMNS:
        raise ValueError(
            f"Expected {_EXPECTED_COLUMNS} columns, got {len(fields)}{where}: {stripped!r}"
        )

    speaker_id, clip_id, _placeholder, system_id, label = fields
    if not speaker_id or not clip_id:
        raise ValueError(f"speaker_id and clip_id must be non-empty{where}: {stripped!r}")
    if label not in VALID_LABELS:
        raise ValueError(f"label must be one of {sorted(VALID_LABELS)}{where}: {label!r}")

    attack_id = None if system_id == _NULL_TOKEN else system_id
    if label == "bonafide" and attack_id is not None:
        raise ValueError(f"bona fide clip must not declare an attack id{where}: {system_id!r}")
    if label == "spoof" and attack_id is None:
        raise ValueError(f"spoof clip must declare an attack id{where}: {stripped!r}")

    return ProtocolEntry(
        speaker_id=speaker_id,
        clip_id=clip_id,
        attack_id=attack_id,
        label=label,
        official_split=official_split,
    )


def iter_protocol_entries(
    lines: Iterable[str],
    official_split: str,
) -> Iterator[ProtocolEntry]:
    """Yield a :class:`ProtocolEntry` for each non-blank line."""

    for index, line in enumerate(lines, start=1):
        entry = parse_protocol_line(line, official_split, line_number=index)
        if entry is not None:
            yield entry


def parse_protocol_file(path: Path | str, official_split: str) -> list[ProtocolEntry]:
    """Parse an official CM protocol file, rejecting duplicate ``clip_id`` values."""

    _require_valid_split(official_split)
    text = Path(path).read_text(encoding="utf-8")

    entries: list[ProtocolEntry] = []
    seen: set[str] = set()
    for entry in iter_protocol_entries(text.splitlines(), official_split):
        if entry.clip_id in seen:
            raise ValueError(f"Duplicate clip_id in protocol: {entry.clip_id}")
        seen.add(entry.clip_id)
        entries.append(entry)
    return entries


def summarize_protocol(entries: Iterable[ProtocolEntry]) -> dict[str, Any]:
    """Produce a metadata-only audit summary of parsed protocol entries.

    The summary is small and JSON-serialisable: class balance, speaker coverage
    and the attack-id distribution. It reads no audio and is the basis for the
    ``protocol-audit`` CLI command.
    """

    entries = list(entries)
    label_counts = Counter(entry.label for entry in entries)
    attack_counts = Counter(entry.attack_id for entry in entries if entry.attack_id is not None)
    speakers = {entry.speaker_id for entry in entries}
    splits = sorted({entry.official_split for entry in entries})

    return {
        "total_clips": len(entries),
        "official_splits": splits,
        "label_counts": {label: label_counts.get(label, 0) for label in sorted(VALID_LABELS)},
        "unique_speakers": len(speakers),
        "unique_attacks": len(attack_counts),
        "attack_counts": dict(sorted(attack_counts.items())),
    }
