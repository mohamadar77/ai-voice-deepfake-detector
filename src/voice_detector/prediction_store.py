"""Prediction persistence for locked evaluation (issue #35).

One row per (clip_id, condition, candidate_id). The store is SQLite, following
the manifest.py pattern. Predictions are written once per (candidate, condition)
pair — duplicates are refused, not silently overwritten.

The deterministic hash over all rows supports the FROZEN → SCORED transition
in the freeze record.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PREDICTION_STORE_VERSION = "predictions-v1"

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    clip_id         TEXT NOT NULL,
    candidate_id    TEXT NOT NULL,
    condition       TEXT NOT NULL,
    clip_score      REAL NOT NULL,
    label           TEXT NOT NULL,
    speaker_id      TEXT NOT NULL,
    attack_id       TEXT,
    official_split  TEXT NOT NULL,
    duration_seconds REAL NOT NULL,
    n_windows       INTEGER NOT NULL,
    threshold       REAL NOT NULL,
    decision        INTEGER NOT NULL,
    PRIMARY KEY (clip_id, candidate_id, condition)
);

CREATE TABLE IF NOT EXISTS metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class PredictionRow:
    clip_id: str
    candidate_id: str
    condition: str
    clip_score: float
    label: str
    speaker_id: str
    attack_id: str | None
    official_split: str
    duration_seconds: float
    n_windows: int
    threshold: float
    decision: bool

    def as_tuple(self) -> tuple[Any, ...]:
        return (
            self.clip_id,
            self.candidate_id,
            self.condition,
            self.clip_score,
            self.label,
            self.speaker_id,
            self.attack_id,
            self.official_split,
            self.duration_seconds,
            self.n_windows,
            self.threshold,
            int(self.decision),
        )


def create_prediction_store(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(_CREATE_SQL)
        conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            ("version", PREDICTION_STORE_VERSION),
        )
        conn.commit()
    finally:
        conn.close()
    return path


def write_predictions(
    path: Path,
    rows: Sequence[PredictionRow],
    *,
    freeze_record_hash: str,
) -> int:
    if not rows:
        return 0

    pairs = {(r.candidate_id, r.condition) for r in rows}
    conn = sqlite3.connect(path)
    try:
        for candidate_id, condition in pairs:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM predictions WHERE candidate_id = ? AND condition = ?",
                (candidate_id, condition),
            )
            count = cursor.fetchone()[0]
            if count > 0:
                raise ValueError(
                    f"predictions already exist for candidate={candidate_id!r}, "
                    f"condition={condition!r}; refusing duplicate write"
                )

        conn.executemany(
            "INSERT INTO predictions "
            "(clip_id, candidate_id, condition, clip_score, label, speaker_id, "
            "attack_id, official_split, duration_seconds, n_windows, threshold, decision) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [r.as_tuple() for r in rows],
        )
        conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            ("freeze_record_hash", freeze_record_hash),
        )
        conn.commit()
        return len(rows)
    finally:
        conn.close()


def read_predictions(
    path: Path,
    *,
    candidate_id: str | None = None,
    condition: str | None = None,
    official_split: str | None = None,
) -> list[PredictionRow]:
    conn = sqlite3.connect(path)
    try:
        clauses: list[str] = []
        params: list[str] = []
        if candidate_id is not None:
            clauses.append("candidate_id = ?")
            params.append(candidate_id)
        if condition is not None:
            clauses.append("condition = ?")
            params.append(condition)
        if official_split is not None:
            clauses.append("official_split = ?")
            params.append(official_split)

        where = " AND ".join(clauses) if clauses else "1=1"
        cursor = conn.execute(
            f"SELECT clip_id, candidate_id, condition, clip_score, label, "  # noqa: S608
            f"speaker_id, attack_id, official_split, duration_seconds, "
            f"n_windows, threshold, decision "
            f"FROM predictions WHERE {where} "
            f"ORDER BY clip_id, candidate_id, condition",
            params,
        )
        return [
            PredictionRow(
                clip_id=row[0],
                candidate_id=row[1],
                condition=row[2],
                clip_score=row[3],
                label=row[4],
                speaker_id=row[5],
                attack_id=row[6],
                official_split=row[7],
                duration_seconds=row[8],
                n_windows=row[9],
                threshold=row[10],
                decision=bool(row[11]),
            )
            for row in cursor.fetchall()
        ]
    finally:
        conn.close()


def compute_predictions_hash(path: Path) -> str:
    conn = sqlite3.connect(path)
    try:
        cursor = conn.execute(
            "SELECT clip_id, candidate_id, condition, clip_score, label, "
            "speaker_id, attack_id, official_split, duration_seconds, "
            "n_windows, threshold, decision "
            "FROM predictions "
            "ORDER BY candidate_id, condition, clip_id"
        )
        digest = hashlib.sha256()
        for row in cursor:
            line = "|".join(str(v) for v in row) + "\n"
            digest.update(line.encode("utf-8"))
        return digest.hexdigest()
    finally:
        conn.close()


def validate_prediction_completeness(
    path: Path,
    expected_clip_ids: Sequence[str],
    candidate_ids: Sequence[str],
    conditions: Sequence[str],
) -> list[str]:
    conn = sqlite3.connect(path)
    try:
        cursor = conn.execute(
            "SELECT DISTINCT candidate_id, condition, clip_id FROM predictions"
        )
        present = {(row[0], row[1], row[2]) for row in cursor.fetchall()}
    finally:
        conn.close()

    missing: list[str] = []
    for cid in candidate_ids:
        for cond in conditions:
            for clip_id in expected_clip_ids:
                if (cid, cond, clip_id) not in present:
                    missing.append(f"{cid}/{cond}/{clip_id}")
    return missing
