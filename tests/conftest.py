"""Shared dataset-free fixtures.

The reservation and CLI tests both need a synthetic manifest: real ASVspoof
audio is never read, so the row builders live here rather than being imported
across test modules.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from voice_detector import reservation as r

# Twenty development speakers, matching the real LA development partition.
DEV_SPEAKERS = tuple(f"LA_{index:04d}" for index in range(1, 21))


def _checksum(clip_id: str) -> str:
    return hashlib.sha256(clip_id.encode()).hexdigest()


def manifest_row(
    clip_id: str,
    *,
    speaker_id: str | None,
    official_split: str,
    label: str = "bonafide",
    attack_id: str | None = None,
    development_role: str | None = None,
) -> dict[str, Any]:
    """One manifest row that satisfies the data contract in ``docs/data.md``."""

    return {
        "clip_id": clip_id,
        "speaker_id": speaker_id,
        "source_id": None,
        "attack_id": attack_id,
        "label": label,
        "official_split": official_split,
        "development_role": development_role,
        "duration_seconds": 3.0,
        "sample_rate_hz": 16000,
        "channels": 1,
        "checksum_sha256": _checksum(clip_id),
        "provenance": "synthetic test row",
    }


def development_rows(
    speakers: tuple[str, ...] = DEV_SPEAKERS,
    *,
    n_bonafide: int = 2,
    attacks: tuple[str, ...] = r.KNOWN_ATTACKS,
) -> list[dict[str, Any]]:
    """Per speaker: ``n_bonafide`` bona fide clips plus one clip per attack."""

    rows: list[dict[str, Any]] = []
    for speaker in speakers:
        for index in range(n_bonafide):
            rows.append(
                manifest_row(
                    f"{speaker}_D_B{index}",
                    speaker_id=speaker,
                    official_split="development",
                )
            )
        for attack in attacks:
            rows.append(
                manifest_row(
                    f"{speaker}_D_{attack}",
                    speaker_id=speaker,
                    official_split="development",
                    label="spoof",
                    attack_id=attack,
                )
            )
    return rows


def full_manifest_rows(**kwargs: Any) -> list[dict[str, Any]]:
    """Development rows plus train and evaluation rows, as a manifest has."""

    return [
        manifest_row("LA_T_1", speaker_id="LA_0101", official_split="train"),
        manifest_row(
            "LA_T_2",
            speaker_id="LA_0101",
            official_split="train",
            label="spoof",
            attack_id="A01",
        ),
        *development_rows(**kwargs),
        manifest_row("LA_E_1", speaker_id="LA_0201", official_split="evaluation"),
        manifest_row(
            "LA_E_2",
            speaker_id="LA_0201",
            official_split="evaluation",
            label="spoof",
            attack_id="A07",
        ),
    ]


@pytest.fixture
def make_rows() -> Callable[..., list[dict[str, Any]]]:
    """Factory for a synthetic three-split manifest."""

    return full_manifest_rows


@pytest.fixture
def make_row() -> Callable[..., dict[str, Any]]:
    """Factory for one manifest row."""

    return manifest_row


@pytest.fixture
def make_development_rows() -> Callable[..., list[dict[str, Any]]]:
    """Factory for development rows only."""

    return development_rows


@pytest.fixture
def reservation_spec() -> r.ReservationSpec:
    """The committed reservation specification, loaded as written."""

    return r.load_reservation_spec(Path("configs/dev_reservation.yaml"))
