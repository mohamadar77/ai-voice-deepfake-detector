"""Freeze record lifecycle for locked evaluation (issue #35).

The evaluation protocol follows a strict lifecycle:

    DRAFT  →  FROZEN  →  SCORED  →  REPORTED

Each transition is guarded: the record can only move forward and never back.
Every transition validates that the record's declared conventions (metrics
version, score direction, decision rule) agree with the code that will compute
them, so a stale record cannot silently produce numbers under different rules.

The freeze record captures SHA-256 hashes of every artifact that feeds the
evaluation (model bundles, manifest, configs, robustness spec) so any drift
is detected before scoring begins.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

from voice_detector.manifest import compute_manifest_hash, git_commit, package_version
from voice_detector.metrics import METRICS_VERSION
from voice_detector.scoring import DECISION_RULE, SCORE_DIRECTION

EVALUATION_SCHEMA_VERSION = "evaluation-v1"


class FreezeState(StrEnum):
    DRAFT = "DRAFT"
    FROZEN = "FROZEN"
    SCORED = "SCORED"
    REPORTED = "REPORTED"


_VALID_TRANSITIONS = {
    FreezeState.DRAFT: FreezeState.FROZEN,
    FreezeState.FROZEN: FreezeState.SCORED,
    FreezeState.SCORED: FreezeState.REPORTED,
}


@dataclass(frozen=True)
class BootstrapSpec:
    confidence_level: float
    n_resamples: int
    seed: int
    resampling_unit: str
    method: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateDeclaration:
    candidate_id: str
    model_family: str
    bundle_hash: str
    threshold: float
    seed: int
    run_label: str
    aggregation_rule: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SliceDefinition:
    slice_id: str
    description: str
    clip_filter: dict[str, Any]
    bonafide_pool: str
    partition: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FreezeRecord:
    schema_version: str
    state: str
    created_utc: str
    frozen_utc: str | None
    scored_utc: str | None
    reported_utc: str | None
    candidates: list[CandidateDeclaration]
    slices: list[SliceDefinition]
    conditions: list[str]
    manifest_hash: str
    config_hashes: dict[str, str]
    bootstrap: BootstrapSpec
    metrics_version: str
    score_direction: str
    decision_rule: str
    git_commit: str
    package_version: str
    predictions_hash: str | None = None
    report_hash: str | None = None

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["candidates"] = [c.as_dict() for c in self.candidates]
        d["slices"] = [s.as_dict() for s in self.slices]
        d["bootstrap"] = self.bootstrap.as_dict()
        return d


def _now_utc() -> str:
    return datetime.now(UTC).isoformat()


def _validate_conventions(record: FreezeRecord) -> None:
    if record.metrics_version != METRICS_VERSION:
        raise ValueError(
            f"metrics_version mismatch: record says {record.metrics_version!r}, "
            f"code says {METRICS_VERSION!r}"
        )
    if record.score_direction != SCORE_DIRECTION:
        raise ValueError(
            f"score_direction mismatch: record says {record.score_direction!r}, "
            f"code says {SCORE_DIRECTION!r}"
        )
    if record.decision_rule != DECISION_RULE:
        raise ValueError(
            f"decision_rule mismatch: record says {record.decision_rule!r}, "
            f"code says {DECISION_RULE!r}"
        )


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_directory(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
            digest.update(b"|")
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(65536), b""):
                    digest.update(chunk)
            digest.update(b"\n")
    return digest.hexdigest()


def _build_slices_from_config(config: dict[str, Any]) -> list[SliceDefinition]:
    slices: list[SliceDefinition] = []

    for group_id, group in config.get("attack_groups", {}).items():
        partition = group["partition"]
        bonafide_pool = group["bonafide_pool"]
        for attack in group["attacks"]:
            slices.append(SliceDefinition(
                slice_id=f"attack_{attack}",
                description=f"{attack} ({group.get('description', group_id)})",
                clip_filter={"attack_id": attack, "partition": partition},
                bonafide_pool=bonafide_pool,
                partition=partition,
            ))
        slices.append(SliceDefinition(
            slice_id=f"group_{group_id}",
            description=group.get("description", group_id),
            clip_filter={"attack_ids": group["attacks"], "partition": partition},
            bonafide_pool=bonafide_pool,
            partition=partition,
        ))

    for bin_def in config.get("duration_bins", []):
        slices.append(SliceDefinition(
            slice_id=f"duration_{bin_def['bin_id']}",
            description=f"Duration bin {bin_def['bin_id']}",
            clip_filter={
                "duration_lower": bin_def["lower"],
                "duration_upper": bin_def["upper"],
                "duration_upper_inclusive": bin_def["upper_inclusive"],
            },
            bonafide_pool="same_partition",
            partition="all",
        ))

    return slices


def load_evaluation_spec(path: Path) -> dict[str, Any]:
    with open(path) as fh:
        return yaml.safe_load(fh)


def compute_config_hashes(config_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for config_file in sorted(config_dir.glob("*.yaml")):
        hashes[config_file.name] = _hash_file(config_file)
    return hashes


def create_draft(
    candidates: list[CandidateDeclaration],
    manifest_rows: Sequence[Mapping[str, Any]],
    config_dir: Path,
    *,
    evaluation_spec: dict[str, Any] | None = None,
    evaluation_spec_path: Path | None = None,
) -> FreezeRecord:
    if evaluation_spec is None:
        if evaluation_spec_path is None:
            evaluation_spec_path = config_dir / "evaluation_protocol.yaml"
        evaluation_spec = load_evaluation_spec(evaluation_spec_path)

    bootstrap_raw = evaluation_spec["bootstrap"]
    bootstrap = BootstrapSpec(
        confidence_level=bootstrap_raw["confidence_level"],
        n_resamples=bootstrap_raw["n_resamples"],
        seed=bootstrap_raw["seed"],
        resampling_unit=bootstrap_raw["resampling_unit"],
        method=bootstrap_raw["method"],
    )

    slices = _build_slices_from_config(evaluation_spec)
    conditions = list(evaluation_spec.get("conditions", ["clean"]))

    record = FreezeRecord(
        schema_version=EVALUATION_SCHEMA_VERSION,
        state=FreezeState.DRAFT.value,
        created_utc=_now_utc(),
        frozen_utc=None,
        scored_utc=None,
        reported_utc=None,
        candidates=candidates,
        slices=slices,
        conditions=conditions,
        manifest_hash=compute_manifest_hash(manifest_rows),
        config_hashes=compute_config_hashes(config_dir),
        bootstrap=bootstrap,
        metrics_version=METRICS_VERSION,
        score_direction=SCORE_DIRECTION,
        decision_rule=DECISION_RULE,
        git_commit=git_commit(),
        package_version=package_version(),
    )

    _validate_conventions(record)
    return record


def freeze_record(draft: FreezeRecord) -> FreezeRecord:
    if draft.state != FreezeState.DRAFT.value:
        raise ValueError(
            f"can only freeze a DRAFT record, got state={draft.state!r}"
        )
    _validate_conventions(draft)
    if not draft.candidates:
        raise ValueError("cannot freeze a record with no candidates")

    return FreezeRecord(
        **{
            **asdict(draft),
            "state": FreezeState.FROZEN.value,
            "frozen_utc": _now_utc(),
            "candidates": draft.candidates,
            "slices": draft.slices,
            "bootstrap": draft.bootstrap,
        }
    )


def mark_scored(record: FreezeRecord, predictions_hash: str) -> FreezeRecord:
    if record.state != FreezeState.FROZEN.value:
        raise ValueError(
            f"can only mark a FROZEN record as scored, got state={record.state!r}"
        )
    _validate_conventions(record)
    return FreezeRecord(
        **{
            **asdict(record),
            "state": FreezeState.SCORED.value,
            "scored_utc": _now_utc(),
            "predictions_hash": predictions_hash,
            "candidates": record.candidates,
            "slices": record.slices,
            "bootstrap": record.bootstrap,
        }
    )


def mark_reported(record: FreezeRecord, report_hash: str) -> FreezeRecord:
    if record.state != FreezeState.SCORED.value:
        raise ValueError(
            f"can only mark a SCORED record as reported, got state={record.state!r}"
        )
    _validate_conventions(record)
    return FreezeRecord(
        **{
            **asdict(record),
            "state": FreezeState.REPORTED.value,
            "reported_utc": _now_utc(),
            "report_hash": report_hash,
            "candidates": record.candidates,
            "slices": record.slices,
            "bootstrap": record.bootstrap,
        }
    )


def validate_freeze_record(raw: Mapping[str, Any]) -> FreezeRecord:
    if raw.get("schema_version") != EVALUATION_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema_version: {raw.get('schema_version')!r}, "
            f"expected {EVALUATION_SCHEMA_VERSION!r}"
        )
    try:
        state = FreezeState(raw["state"])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"invalid state: {raw.get('state')!r}") from exc

    candidates = [
        CandidateDeclaration(**c) for c in raw.get("candidates", [])
    ]
    slices = [
        SliceDefinition(**s) for s in raw.get("slices", [])
    ]
    bootstrap = BootstrapSpec(**raw["bootstrap"])

    record = FreezeRecord(
        schema_version=raw["schema_version"],
        state=state.value,
        created_utc=raw["created_utc"],
        frozen_utc=raw.get("frozen_utc"),
        scored_utc=raw.get("scored_utc"),
        reported_utc=raw.get("reported_utc"),
        candidates=candidates,
        slices=slices,
        conditions=list(raw.get("conditions", [])),
        manifest_hash=raw["manifest_hash"],
        config_hashes=dict(raw.get("config_hashes", {})),
        bootstrap=bootstrap,
        metrics_version=raw["metrics_version"],
        score_direction=raw["score_direction"],
        decision_rule=raw["decision_rule"],
        git_commit=raw.get("git_commit", "unknown"),
        package_version=raw.get("package_version", "unknown"),
        predictions_hash=raw.get("predictions_hash"),
        report_hash=raw.get("report_hash"),
    )
    _validate_conventions(record)
    return record


def verify_freeze_against_artifacts(
    record: FreezeRecord,
    manifest_rows: Sequence[Mapping[str, Any]],
    config_dir: Path,
    bundle_dir: Path,
) -> None:
    current_manifest_hash = compute_manifest_hash(manifest_rows)
    if current_manifest_hash != record.manifest_hash:
        raise ValueError(
            f"manifest hash mismatch: record={record.manifest_hash[:16]}…, "
            f"current={current_manifest_hash[:16]}…"
        )

    current_config_hashes = compute_config_hashes(config_dir)
    for name, expected in record.config_hashes.items():
        actual = current_config_hashes.get(name)
        if actual is None:
            raise ValueError(f"config file {name!r} missing from {config_dir}")
        if actual != expected:
            raise ValueError(
                f"config {name!r} hash mismatch: record={expected[:16]}…, "
                f"current={actual[:16]}…"
            )

    for candidate in record.candidates:
        candidate_dir = bundle_dir / candidate.candidate_id
        if not candidate_dir.is_dir():
            raise ValueError(
                f"candidate bundle directory not found: {candidate_dir}"
            )
        actual_hash = _hash_directory(candidate_dir)
        if actual_hash != candidate.bundle_hash:
            raise ValueError(
                f"bundle hash mismatch for {candidate.candidate_id!r}: "
                f"record={candidate.bundle_hash[:16]}…, "
                f"current={actual_hash[:16]}…"
            )


def save_freeze_record(record: FreezeRecord, path: Path) -> Path:
    if path.exists():
        with open(path) as fh:
            existing = json.load(fh)
        if existing.get("state") != record.state:
            raise ValueError(
                f"refusing to overwrite {path.name} (state={existing['state']!r}) "
                f"with a record in state={record.state!r}; "
                "only idempotent rewrites of the same state are allowed"
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    data = record.as_dict()
    text = json.dumps(data, indent=2, sort_keys=True) + "\n"

    tmp = path.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return path


def load_freeze_record(path: Path) -> FreezeRecord:
    with open(path) as fh:
        raw = json.load(fh)
    return validate_freeze_record(raw)
