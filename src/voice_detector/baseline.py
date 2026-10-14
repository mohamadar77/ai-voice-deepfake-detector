"""MFCC + Logistic Regression baseline: train, select, threshold, report (#8).

This module trains a StandardScaler + LogisticRegression pipeline on the
training split's window-level features, selects the best regularization value
on ``dev_select`` (clip-level EER, per the pre-tuning protocol), picks the
operating threshold on ``dev_threshold`` (target FPR from the protocol), and
produces a versioned dev report with every required metric.

The window-to-clip aggregation rule is the mean of the per-window decision
function (``decision_function``, the raw margin, not ``predict_proba``), taken
over the windows of one clip. The aggregated score follows
``SCORE_DIRECTION`` (higher = more synthetic) because sklearn's
``LogisticRegression`` with ``classes_=[0, 1]`` puts the positive class
(spoof = 1) on the positive side of the decision boundary.

Nothing here touches the evaluation partition.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from voice_detector.manifest import git_commit, package_version

if TYPE_CHECKING:
    from voice_detector.features import SplitFeatures
from voice_detector.metrics import (
    METRICS_VERSION,
    ClipMetrics,
    evaluate_clip_scores,
    threshold_at_target_fpr,
)
from voice_detector.pretuning import PreTuningProtocol
from voice_detector.reservation import (
    ROLE_SELECT,
    ROLE_THRESHOLD,
    validate_frozen_record,
)
from voice_detector.scoring import SCORE_DIRECTION

BASELINE_VERSION = "lr-baseline-v1"
LABEL_MAP = {"bonafide": 0, "spoof": 1}


@dataclass(frozen=True)
class TrainedCandidate:
    """One fitted LR pipeline + its dev_select clip-level metrics."""

    C: float
    pipeline: Pipeline
    select_metrics: ClipMetrics


@dataclass(frozen=True)
class BaselineBundle:
    """Everything needed to reproduce and serve the chosen baseline."""

    version: str
    pipeline: Pipeline
    C: float
    threshold: float
    target_fpr: float
    achieved_fpr: float
    preprocessing_version: str
    aggregation_rule: str


def _encode_labels(labels: np.ndarray) -> np.ndarray:
    return np.array([LABEL_MAP[str(label)] for label in labels], dtype=np.int32)


def _aggregate_window_scores(
    window_scores: np.ndarray,
    window_counts: np.ndarray,
) -> np.ndarray:
    """Mean of per-window decision-function values, one score per clip."""

    clip_scores = np.empty(len(window_counts), dtype=np.float64)
    offset = 0
    for i, count in enumerate(window_counts):
        clip_scores[i] = window_scores[offset : offset + count].mean()
        offset += count
    return clip_scores


def train_candidate(
    train_features: SplitFeatures,
    C: float,
    *,
    class_weight: str,
    seed: int,
) -> Pipeline:
    """Fit a StandardScaler + LogisticRegression on training windows."""

    y = np.repeat(_encode_labels(train_features.labels), train_features.window_counts)
    weight = class_weight if class_weight != "none" else None
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=C,
            class_weight=weight,
            solver="lbfgs",
            max_iter=1000,
            random_state=seed,
        )),
    ])
    pipeline.fit(train_features.features, y)
    return pipeline


def score_split(
    pipeline: Pipeline,
    split_features: SplitFeatures,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score a split's windows, aggregate to clips, return (clip_ids, labels, clip_scores)."""

    window_scores = pipeline.decision_function(split_features.features).astype(np.float64)
    clip_scores = _aggregate_window_scores(window_scores, split_features.window_counts)
    return split_features.clip_ids, split_features.labels, clip_scores


def _eer_value(candidate: TrainedCandidate) -> float:
    eer = candidate.select_metrics.eer
    return float("inf") if eer is None else eer


def select_best_candidate(
    candidates: Sequence[TrainedCandidate],
    protocol: PreTuningProtocol,
) -> TrainedCandidate:
    """Pick the best candidate by clip-level EER on dev_select, with tie-break."""

    tolerance = protocol.selection.eer_tolerance_absolute
    ranked = sorted(candidates, key=lambda c: (_eer_value(c), c.C))
    best_eer = _eer_value(ranked[0])
    tied = [c for c in ranked if abs(_eer_value(c) - best_eer) <= tolerance]
    if len(tied) == 1:
        return tied[0]
    for criterion in protocol.selection.tie_break_order:
        if criterion == "simpler_system":
            tied.sort(key=lambda c: abs(c.C))
            return tied[0]
        if criterion == "lower_measured_latency":
            return tied[0]
    return tied[0]


def train_and_select(
    train_features: SplitFeatures,
    dev_features: SplitFeatures,
    dev_rows: Sequence[Mapping[str, Any]],
    protocol: PreTuningProtocol,
    frozen_record: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    """Full LR baseline pipeline: train candidates, select, threshold, report.

    Returns a JSON-serializable run record with all required fields.
    """

    assignment = validate_frozen_record(frozen_record)
    select_speakers = {s for s, r in assignment.items() if r == ROLE_SELECT}
    threshold_speakers = {s for s, r in assignment.items() if r == ROLE_THRESHOLD}

    dev_clip_ids = dev_features.clip_ids
    dev_labels = dev_features.labels
    dev_speaker_map = {
        str(row["clip_id"]): str(row["speaker_id"]) for row in dev_rows
        if row["official_split"] == "development"
    }

    select_mask = np.array([
        dev_speaker_map.get(str(cid), "") in select_speakers
        for cid in dev_clip_ids
    ])
    threshold_mask = np.array([
        dev_speaker_map.get(str(cid), "") in threshold_speakers
        for cid in dev_clip_ids
    ])

    candidates: list[TrainedCandidate] = []
    candidate_records: list[dict[str, Any]] = []
    started = time.perf_counter()

    for C in protocol.lr_budget.values:
        t0 = time.perf_counter()
        pipeline = train_candidate(
            train_features, C,
            class_weight=protocol.lr_budget.class_weight,
            seed=seed,
        )
        train_time = time.perf_counter() - t0

        clip_ids_all, labels_all, scores_all = score_split(pipeline, dev_features)

        select_ids = clip_ids_all[select_mask]
        select_labels = labels_all[select_mask]
        select_scores = scores_all[select_mask]
        select_metrics = evaluate_clip_scores(
            list(select_ids), list(select_labels), select_scores,
            threshold=None, score_direction=SCORE_DIRECTION,
        )

        candidate = TrainedCandidate(C=C, pipeline=pipeline, select_metrics=select_metrics)
        candidates.append(candidate)
        candidate_records.append({
            "C": C,
            "train_seconds": round(train_time, 3),
            "dev_select_metrics": select_metrics.as_dict(),
        })

    best = select_best_candidate(candidates, protocol)

    _, _, best_scores_all = score_split(best.pipeline, dev_features)
    threshold_ids = dev_clip_ids[threshold_mask]
    threshold_labels = dev_labels[threshold_mask]
    threshold_scores = best_scores_all[threshold_mask]

    threshold_result = threshold_at_target_fpr(
        list(threshold_ids), list(threshold_labels), threshold_scores,
        target_fpr=protocol.threshold.target_fpr,
        score_direction=SCORE_DIRECTION,
    )

    dev_full_metrics = evaluate_clip_scores(
        list(dev_clip_ids), list(dev_labels), best_scores_all,
        threshold=threshold_result.threshold, score_direction=SCORE_DIRECTION,
    )

    select_at_threshold = evaluate_clip_scores(
        list(dev_clip_ids[select_mask]),
        list(dev_labels[select_mask]),
        best_scores_all[select_mask],
        threshold=threshold_result.threshold,
        score_direction=SCORE_DIRECTION,
    )

    threshold_at_threshold = evaluate_clip_scores(
        list(threshold_ids),
        list(threshold_labels),
        threshold_scores,
        threshold=threshold_result.threshold,
        score_direction=SCORE_DIRECTION,
    )

    total_time = time.perf_counter() - started

    return {
        "schema_version": BASELINE_VERSION,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "package_version": package_version(),
        "seed": seed,
        "pre_tuning_protocol": protocol.as_dict(),
        "metrics_version": METRICS_VERSION,
        "score_direction": SCORE_DIRECTION,
        "aggregation": asdict(protocol.aggregation),
        "train_split": train_features.split,
        "train_n_clips": train_features.n_clips,
        "train_n_windows": train_features.n_windows,
        "train_feature_dim": train_features.feature_dim,
        "train_preprocessing_version": train_features.preprocessing_version,
        "train_manifest_hash": train_features.manifest_hash,
        "dev_split": dev_features.split,
        "dev_n_clips": dev_features.n_clips,
        "dev_n_windows": dev_features.n_windows,
        "dev_feature_dim": dev_features.feature_dim,
        "selected_C": best.C,
        "candidates": candidate_records,
        "threshold": {
            "value": threshold_result.threshold,
            "boundary_score": threshold_result.boundary_score,
            "decision_rule": threshold_result.decision_rule,
            "target_fpr": threshold_result.target_fpr,
            "achieved_fpr": threshold_result.achieved_fpr,
            "n_bonafide": threshold_result.n_bonafide,
            "n_false_alarms": threshold_result.n_false_alarms,
        },
        "dev_full_metrics": dev_full_metrics.as_dict(),
        "dev_select_metrics": select_at_threshold.as_dict(),
        "dev_threshold_metrics": threshold_at_threshold.as_dict(),
        "total_seconds": round(total_time, 3),
    }


def save_bundle(
    pipeline: Pipeline,
    run_record: Mapping[str, Any],
    out_dir: Path | str,
) -> dict[str, Path]:
    """Save the chosen pipeline and run record to an artifacts directory."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "lr_pipeline.joblib"
    record_path = out_dir / "lr_run_record.json"

    temp_model = model_path.with_name(f"{model_path.name}.tmp")
    joblib.dump(pipeline, temp_model)
    os.replace(temp_model, model_path)

    temp_record = record_path.with_name(f"{record_path.name}.tmp")
    temp_record.write_text(
        json.dumps(dict(run_record), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temp_record, record_path)

    return {"model": model_path, "record": record_path}
