"""Evaluation report generation for locked evaluation (issue #35).

Computes per-slice metrics from the prediction store, optionally with
speaker-grouped bootstrap confidence intervals. Produces a JSON report
covering every candidate x condition x slice combination.

The actual scoring of clips (loading bundles, applying conditions, extracting
features) is wired in the CLI or rehearsal command. This module consumes
PredictionRows and produces structured metrics.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from voice_detector.bootstrap import BootstrapSpec, speaker_grouped_bootstrap
from voice_detector.evaluation import FreezeRecord
from voice_detector.manifest import git_commit, package_version
from voice_detector.metrics import (
    METRICS_VERSION,
    compute_eer,
    evaluate_clip_scores,
)
from voice_detector.prediction_store import PredictionRow
from voice_detector.scoring import SCORE_DIRECTION

REPORT_VERSION = "report-v1"


def _filter_by_slice(
    predictions: Sequence[PredictionRow],
    clip_filter: dict[str, Any],
    partition: str,
) -> list[PredictionRow]:
    result: list[PredictionRow] = []
    for p in predictions:
        if partition != "all" and p.official_split != partition:
            continue

        if (
            "attack_id" in clip_filter
            and p.attack_id != clip_filter["attack_id"]
            and p.label != "bonafide"
        ):
            continue

        if (
            "attack_ids" in clip_filter
            and p.label == "spoof"
            and p.attack_id not in clip_filter["attack_ids"]
        ):
            continue

        if "duration_lower" in clip_filter:
            lower = clip_filter["duration_lower"]
            upper = clip_filter.get("duration_upper")
            upper_inclusive = clip_filter.get("duration_upper_inclusive", False)
            if p.duration_seconds < lower:
                continue
            if upper is not None:
                if upper_inclusive and p.duration_seconds > upper:
                    continue
                if not upper_inclusive and p.duration_seconds >= upper:
                    continue

        result.append(p)
    return result


def _add_bonafide_pool(
    spoof_predictions: list[PredictionRow],
    all_predictions: Sequence[PredictionRow],
    bonafide_pool: str,
    partition: str,
) -> list[PredictionRow]:
    if bonafide_pool == "same_partition":
        pool_partition = partition
    elif bonafide_pool == "development_all":
        pool_partition = "development"
    elif bonafide_pool == "evaluation_all":
        pool_partition = "evaluation"
    else:
        return spoof_predictions

    existing_ids = {p.clip_id for p in spoof_predictions}
    combined = list(spoof_predictions)
    for p in all_predictions:
        if (
            p.label == "bonafide"
            and p.official_split == pool_partition
            and p.clip_id not in existing_ids
        ):
            combined.append(p)
            existing_ids.add(p.clip_id)
    return combined


def _eer_metric_fn(
    clip_ids: list[str],
    labels: list[str],
    scores: np.ndarray,
    *,
    score_direction: str,
    threshold: float | None = None,
) -> float:
    bonafide_scores = scores[np.array([lab == "bonafide" for lab in labels])]
    spoof_scores = scores[np.array([lab == "spoof" for lab in labels])]
    if len(bonafide_scores) == 0 or len(spoof_scores) == 0:
        return float("nan")
    return compute_eer(bonafide_scores, spoof_scores).eer


def compute_slice_metrics(
    predictions: Sequence[PredictionRow],
    all_predictions: Sequence[PredictionRow],
    slice_id: str,
    clip_filter: dict[str, Any],
    bonafide_pool: str,
    partition: str,
    *,
    threshold: float,
    score_direction: str,
    bootstrap_spec: BootstrapSpec | None = None,
) -> dict[str, Any]:
    filtered = _filter_by_slice(predictions, clip_filter, partition)
    if bonafide_pool != "same_partition":
        filtered = _add_bonafide_pool(filtered, all_predictions, bonafide_pool, partition)

    if not filtered:
        return {
            "slice_id": slice_id,
            "n_clips": 0,
            "metrics": None,
            "bootstrap_ci": None,
        }

    clip_ids = [p.clip_id for p in filtered]
    labels = [p.label for p in filtered]
    scores = np.array([p.clip_score for p in filtered], dtype=np.float64)

    metrics = evaluate_clip_scores(
        clip_ids, labels, scores,
        threshold=threshold,
        score_direction=score_direction,
    )

    bootstrap_ci: dict[str, Any] | None = None
    if bootstrap_spec is not None and metrics.eer is not None:
        speaker_ids = [p.speaker_id for p in filtered]
        unique_speakers = set(speaker_ids)
        if len(unique_speakers) >= 2:
            try:
                ci = speaker_grouped_bootstrap(
                    clip_ids, labels, scores, speaker_ids,
                    metric_fn=_eer_metric_fn,
                    metric_name="eer",
                    spec=bootstrap_spec,
                    score_direction=score_direction,
                )
                bootstrap_ci = ci.as_dict()
            except ValueError:
                pass

    return {
        "slice_id": slice_id,
        "n_clips": len(filtered),
        "metrics": metrics.as_dict(),
        "bootstrap_ci": bootstrap_ci,
    }


def generate_report(
    freeze_record: FreezeRecord,
    predictions: Sequence[PredictionRow],
    *,
    output_dir: Path,
    bootstrap_spec: BootstrapSpec | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []

    for candidate in freeze_record.candidates:
        for condition in freeze_record.conditions:
            cond_preds = [
                p for p in predictions
                if p.candidate_id == candidate.candidate_id
                and p.condition == condition
            ]

            if not cond_preds:
                raise ValueError(
                    f"no predictions for candidate {candidate.candidate_id!r} "
                    f"under condition {condition!r}; refusing to report a partial matrix"
                )

            clip_ids = [p.clip_id for p in cond_preds]
            labels = [p.label for p in cond_preds]
            scores = np.array([p.clip_score for p in cond_preds], dtype=np.float64)

            overall_metrics = evaluate_clip_scores(
                clip_ids, labels, scores,
                threshold=candidate.threshold,
                score_direction=SCORE_DIRECTION,
            )

            slice_results: list[dict[str, Any]] = []
            for slice_def in freeze_record.slices:
                slice_result = compute_slice_metrics(
                    cond_preds, cond_preds,
                    slice_def.slice_id,
                    slice_def.clip_filter,
                    slice_def.bonafide_pool,
                    slice_def.partition,
                    threshold=candidate.threshold,
                    score_direction=SCORE_DIRECTION,
                    bootstrap_spec=bootstrap_spec,
                )
                slice_results.append(slice_result)

            results.append({
                "candidate_id": candidate.candidate_id,
                "condition": condition,
                "overall_metrics": overall_metrics.as_dict(),
                "slices": slice_results,
            })

    report = {
        "report_version": REPORT_VERSION,
        "schema_version": freeze_record.schema_version,
        "metrics_version": METRICS_VERSION,
        "score_direction": SCORE_DIRECTION,
        "generated_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "package_version": package_version(),
        "freeze_record_state": freeze_record.state,
        "n_candidates": len(freeze_record.candidates),
        "n_conditions": len(freeze_record.conditions),
        "n_slices": len(freeze_record.slices),
        "results": results,
    }

    report_path = output_dir / "evaluation_report.json"
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    report_path.write_text(text, encoding="utf-8")

    return report


def compute_report_hash(report_path: Path) -> str:
    with open(report_path) as fh:
        data = json.load(fh)
    data.pop("generated_utc", None)
    canonical = json.dumps(data, indent=2, sort_keys=True) + "\n"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
