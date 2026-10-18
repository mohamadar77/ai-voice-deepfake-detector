"""Supplementary post-report analysis for the locked evaluation (issue #35).

Declared in ``configs/supplementary_analysis.yaml`` and described in
``docs/evaluation-protocol.md`` section 6. The locked report gives a bootstrap
interval for slice EER only. This module adds, for pre-declared slices and for
every candidate x condition:

* EER, human false-positive rate, synthetic detection rate, balanced accuracy and
  macro F1 with speaker-grouped percentile intervals,
* paired differences between pre-declared candidate pairs on the same resamples,
* evaluation-only duration slices (the report's duration slices pool both
  partitions).

Speakers are resampled once per condition from the whole partition, using the
freeze record's bootstrap seed, and every slice, candidate and comparison in that
condition uses the same resamples. A clip's weight in a resample is the number of
times its speaker was drawn, which is the same as repeating the speaker's clips.
The intervals therefore differ slightly from the report's, which resamples
speakers inside each slice. The analysis is descriptive: it uses the frozen
thresholds and changes nothing in the freeze record.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import multiprocessing
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from voice_detector.eval_report import _add_bonafide_pool, _filter_by_slice
from voice_detector.eval_scoring import pinned_thread_env
from voice_detector.evaluation import FreezeRecord
from voice_detector.manifest import git_commit, package_version
from voice_detector.metrics import evaluate_clip_scores
from voice_detector.prediction_store import PredictionRow
from voice_detector.scoring import SCORE_DIRECTION, decide_spoof

SUPPLEMENTARY_VERSION = "supplementary-v1"
REPORT_FILENAME = "supplementary_report.json"
PARTITION_ALL_SLICE = "partition_all"

METRIC_NAMES = (
    "eer",
    "human_false_positive_rate",
    "synthetic_detection_rate",
    "balanced_accuracy",
    "macro_f1",
)
_PARTITIONS = ("evaluation", "development")


@dataclass(frozen=True)
class Comparison:
    comparison_id: str
    a: str
    b: str


@dataclass(frozen=True)
class SupplementarySpec:
    evaluation_slice_ids: tuple[str, ...]
    development_slice_ids: tuple[str, ...]
    metrics: tuple[str, ...]
    comparisons: tuple[Comparison, ...]
    low_support_min_class_count: int
    spec_sha256: str


def load_supplementary_spec(path: Path) -> SupplementarySpec:
    """Load and validate the pre-declared analysis specification."""

    raw = Path(path).read_bytes()
    data = yaml.safe_load(raw)
    if not isinstance(data, dict) or data.get("version") != SUPPLEMENTARY_VERSION:
        raise ValueError(f"supplementary spec version must be {SUPPLEMENTARY_VERSION!r}")

    metrics = tuple(data.get("metrics", ()))
    unknown = sorted(set(metrics) - set(METRIC_NAMES))
    if not metrics or unknown:
        raise ValueError(f"metrics must be a non-empty subset of {METRIC_NAMES}, got {unknown}")

    comparisons = []
    for entry in data.get("comparisons", ()):
        try:
            comparisons.append(Comparison(str(entry["id"]), str(entry["a"]), str(entry["b"])))
        except KeyError as exc:
            raise ValueError(f"comparison entry is missing {exc}") from exc
    ids = [c.comparison_id for c in comparisons]
    if len(set(ids)) != len(ids):
        raise ValueError("comparison ids must be unique")
    if any(c.a == c.b for c in comparisons):
        raise ValueError("a comparison needs two different candidates")

    min_count = int(data.get("low_support_min_class_count", 100))
    if min_count < 1:
        raise ValueError("low_support_min_class_count must be at least 1")

    return SupplementarySpec(
        evaluation_slice_ids=tuple(data.get("evaluation_slice_ids", ())),
        development_slice_ids=tuple(data.get("development_slice_ids", ())),
        metrics=metrics,
        comparisons=tuple(comparisons),
        low_support_min_class_count=min_count,
        spec_sha256=hashlib.sha256(raw).hexdigest(),
    )


def resample_weights(n_speakers: int, n_resamples: int, seed: int) -> np.ndarray:
    """Speaker draw counts, shape ``(n_resamples, n_speakers)``.

    Each row is one bootstrap resample: ``n_speakers`` draws with replacement from
    the sorted speaker list, as counts per speaker. A clip's weight in the
    resample is the count of its speaker.
    """

    if n_speakers < 2:
        raise ValueError(f"bootstrap needs at least 2 speakers, got {n_speakers}")
    if n_resamples < 1:
        raise ValueError("n_resamples must be at least 1")
    rng = np.random.default_rng(seed)
    weights = np.empty((n_resamples, n_speakers), dtype=np.int64)
    for b in range(n_resamples):
        chosen = rng.choice(n_speakers, size=n_speakers, replace=True)
        weights[b] = np.bincount(chosen, minlength=n_speakers)
    return weights


@dataclass(frozen=True)
class _EerPlan:
    """Score ordering reused across resamples (it does not depend on the weights)."""

    group_end: np.ndarray
    is_spoof: np.ndarray
    speaker: np.ndarray


def _eer_plan(scores: np.ndarray, is_spoof: np.ndarray, speaker_idx: np.ndarray) -> _EerPlan:
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    group_end = np.append(np.flatnonzero(np.diff(sorted_scores)), scores.size - 1)
    return _EerPlan(
        group_end=group_end,
        is_spoof=is_spoof[order].astype(np.int64),
        speaker=speaker_idx[order],
    )


def weighted_eer(plan: _EerPlan, speaker_weights: np.ndarray) -> float:
    """EER of the clips weighted by their speaker's draw count.

    The same construction as ``metrics.compute_eer`` on integer counts, so for
    integer weights it equals ``compute_eer`` on the clips repeated by weight.
    Returns NaN when either class has zero weight.
    """

    weight = speaker_weights[plan.speaker]
    spoof_cum = np.cumsum(weight * plan.is_spoof)[plan.group_end]
    all_cum = np.cumsum(weight)[plan.group_end]
    tp = np.concatenate(([0], spoof_cum))
    fp = np.concatenate(([0], all_cum - spoof_cum))
    n_spoof = int(tp[-1])
    n_bonafide = int(fp[-1])
    if n_spoof == 0 or n_bonafide == 0:
        return float("nan")

    gap = (n_spoof - tp) * n_bonafide - fp * n_spoof
    k = int(np.argmax(gap <= 0))
    fpr = fp / n_bonafide
    if gap[k] == 0:
        return float(fpr[k])
    fnr = (n_spoof - tp) / n_spoof
    before, after = fnr[k - 1] - fpr[k - 1], fnr[k] - fpr[k]
    weight_k = before / (before - after)
    return float(fpr[k - 1] + weight_k * (fpr[k] - fpr[k - 1]))


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    out = np.full(numerator.shape, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=out, where=denominator > 0)
    return out


def resampled_metrics(
    scores: np.ndarray,
    is_spoof: np.ndarray,
    speaker_idx: np.ndarray,
    threshold: float,
    weights: np.ndarray,
    metrics: Sequence[str],
) -> dict[str, np.ndarray]:
    """Every requested metric on every resample, as arrays of length ``n_resamples``.

    NaN marks a resample in which a needed class has no clips.
    """

    n_speakers = weights.shape[1]
    flagged = np.asarray(decide_spoof(scores, threshold), dtype=bool)

    def per_speaker(mask: np.ndarray) -> np.ndarray:
        return np.bincount(speaker_idx[mask], minlength=n_speakers).astype(np.int64)

    bonafide = ~is_spoof
    n_b = weights @ per_speaker(bonafide)
    n_s = weights @ per_speaker(is_spoof)
    fp = weights @ per_speaker(bonafide & flagged)
    tp = weights @ per_speaker(is_spoof & flagged)

    hfpr = _safe_ratio(fp, n_b)
    sdr = _safe_ratio(tp, n_s)
    result: dict[str, np.ndarray] = {}
    if "human_false_positive_rate" in metrics:
        result["human_false_positive_rate"] = hfpr
    if "synthetic_detection_rate" in metrics:
        result["synthetic_detection_rate"] = sdr
    if "balanced_accuracy" in metrics:
        result["balanced_accuracy"] = ((1.0 - hfpr) + sdr) / 2.0
    if "macro_f1" in metrics:
        tn, fn = n_b - fp, n_s - tp
        valid = (n_b > 0) & (n_s > 0)
        f1_bonafide = _safe_ratio(2 * tn, np.where(valid, 2 * tn + fn + fp, 0))
        f1_spoof = _safe_ratio(2 * tp, np.where(valid, 2 * tp + fp + fn, 0))
        result["macro_f1"] = (f1_bonafide + f1_spoof) / 2.0
    if "eer" in metrics:
        plan = _eer_plan(scores, is_spoof, speaker_idx)
        result["eer"] = np.array([weighted_eer(plan, w) for w in weights], dtype=np.float64)
    return result


def _interval(values: np.ndarray, confidence_level: float) -> dict[str, Any] | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    alpha = 1.0 - confidence_level
    return {
        "ci_lower": float(np.percentile(finite, 100 * alpha / 2)),
        "ci_upper": float(np.percentile(finite, 100 * (1 - alpha / 2))),
        "n_valid_resamples": int(finite.size),
    }


def _slice_members(
    reference: Sequence[PredictionRow],
    clip_filter: dict[str, Any],
    bonafide_pool: str,
    partition: str,
) -> np.ndarray:
    """Indices into ``reference`` of the clips in a slice, with the report's semantics."""

    filtered = _filter_by_slice(reference, clip_filter, partition)
    if bonafide_pool != "same_partition":
        filtered = _add_bonafide_pool(filtered, reference, bonafide_pool, partition)
    members = {p.clip_id for p in filtered}
    return np.array(
        [i for i, p in enumerate(reference) if p.clip_id in members], dtype=np.intp
    )


def _aligned_rows(
    rows: Sequence[PredictionRow], partition: str
) -> list[PredictionRow]:
    chosen = [p for p in rows if p.official_split == partition]
    chosen.sort(key=lambda p: p.clip_id)
    return chosen


def _condition_task(task: dict[str, Any]) -> dict[str, Any]:
    """All supplementary results for one condition (a pure function of its task)."""

    condition: str = task["condition"]
    partition: str = task["partition"]
    metrics: tuple[str, ...] = task["metrics"]
    confidence: float = task["confidence_level"]
    min_count: int = task["low_support_min_class_count"]
    thresholds: dict[str, float] = task["thresholds"]
    candidate_ids: list[str] = task["candidate_ids"]

    rows = {c: _aligned_rows(task["rows"][c], partition) for c in candidate_ids}
    reference = rows[candidate_ids[0]]
    if not reference:
        raise ValueError(f"no {partition} predictions for condition {condition!r}")
    ref_ids = [p.clip_id for p in reference]
    for c in candidate_ids:
        if [p.clip_id for p in rows[c]] != ref_ids:
            raise ValueError(
                f"candidates disagree on the {partition} clip set under {condition!r}"
            )
        if [p.label for p in rows[c]] != [p.label for p in reference] or [
            p.speaker_id for p in rows[c]
        ] != [p.speaker_id for p in reference]:
            raise ValueError(f"candidates disagree on labels or speakers under {condition!r}")

    speakers = sorted({p.speaker_id for p in reference})
    speaker_index = {s: i for i, s in enumerate(speakers)}
    speaker_idx = np.array([speaker_index[p.speaker_id] for p in reference], dtype=np.intp)
    labels = [p.label for p in reference]
    is_spoof_all = np.array([lab == "spoof" for lab in labels], dtype=bool)
    scores_all = {
        c: np.array([p.clip_score for p in rows[c]], dtype=np.float64) for c in candidate_ids
    }
    weights = resample_weights(len(speakers), task["n_resamples"], task["seed"])

    slice_specs: list[tuple[str, np.ndarray]] = [
        (PARTITION_ALL_SLICE, np.arange(len(reference), dtype=np.intp))
    ]
    for slice_id, clip_filter, bonafide_pool in task["slices"]:
        slice_specs.append(
            (slice_id, _slice_members(reference, clip_filter, bonafide_pool, partition))
        )

    slice_results: list[dict[str, Any]] = []
    for slice_id, members in slice_specs:
        n_clips = int(members.size)
        entry: dict[str, Any] = {"slice_id": slice_id, "n_clips": n_clips}
        if n_clips == 0:
            entry.update(
                n_bonafide=0, n_spoof=0, n_speakers=0, low_support=True,
                candidates={}, comparisons=[],
            )
            slice_results.append(entry)
            continue

        is_spoof = is_spoof_all[members]
        slice_speakers = speaker_idx[members]
        n_spoof = int(np.count_nonzero(is_spoof))
        n_bonafide = n_clips - n_spoof
        entry.update(
            n_bonafide=n_bonafide,
            n_spoof=n_spoof,
            n_speakers=int(np.unique(slice_speakers).size),
            low_support=bool(min(n_bonafide, n_spoof) < min_count),
        )

        point: dict[str, dict[str, Any]] = {}
        draws: dict[str, dict[str, np.ndarray]] = {}
        candidates_out: dict[str, Any] = {}
        member_ids = [ref_ids[i] for i in members]
        member_labels = [labels[i] for i in members]
        for c in candidate_ids:
            scores = scores_all[c][members]
            point[c] = evaluate_clip_scores(
                member_ids, member_labels, scores,
                threshold=thresholds[c], score_direction=SCORE_DIRECTION,
            ).as_dict()
            draws[c] = resampled_metrics(
                scores, is_spoof, slice_speakers, thresholds[c], weights, metrics
            )
            candidates_out[c] = {
                "metrics": point[c],
                "bootstrap": {
                    m: {
                        "point_estimate": point[c][m],
                        **(_interval(draws[c][m], confidence) or {}),
                    }
                    for m in metrics
                },
            }
        entry["candidates"] = candidates_out

        comparisons_out = []
        for comparison in task["comparisons"]:
            a, b = comparison["a"], comparison["b"]
            per_metric = {}
            for m in metrics:
                point_a, point_b = point[a][m], point[b][m]
                per_metric[m] = {
                    "point_difference": (
                        None if point_a is None or point_b is None else point_a - point_b
                    ),
                    **(_interval(draws[a][m] - draws[b][m], confidence) or {}),
                }
            comparisons_out.append(
                {"comparison_id": comparison["id"], "a": a, "b": b, "difference": per_metric}
            )
        entry["comparisons"] = comparisons_out
        slice_results.append(entry)

    return {
        "condition": condition,
        "n_speakers": len(speakers),
        "n_clips": len(reference),
        "slices": slice_results,
    }


def run_supplementary(
    freeze_record: FreezeRecord,
    predictions: Sequence[PredictionRow],
    spec: SupplementarySpec,
    *,
    partition: str,
    workers: int = 1,
    provenance: dict[str, Any] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Build the supplementary report for one partition.

    ``partition`` is ``evaluation`` for the post-report analysis, or
    ``development`` to build and test the tooling on the development store. The
    report is a pure function of its inputs apart from ``generated_utc``, and is
    identical for any ``workers``.
    """

    if partition not in _PARTITIONS:
        raise ValueError(f"partition must be one of {_PARTITIONS}, got {partition!r}")
    bootstrap = freeze_record.bootstrap
    if bootstrap.resampling_unit != "speaker_id" or bootstrap.method != "percentile":
        raise ValueError(
            "the supplementary analysis supports speaker_id resampling with percentile "
            f"intervals, got {bootstrap.resampling_unit!r}/{bootstrap.method!r}"
        )

    slice_ids = (
        spec.evaluation_slice_ids if partition == "evaluation" else spec.development_slice_ids
    )
    definitions = {s.slice_id: s for s in freeze_record.slices}
    missing = [s for s in slice_ids if s not in definitions]
    if missing:
        raise ValueError(f"slice ids not in the freeze record: {missing}")
    candidate_ids = [c.candidate_id for c in freeze_record.candidates]
    unknown = sorted(
        {x for c in spec.comparisons for x in (c.a, c.b)} - set(candidate_ids)
    )
    if unknown:
        raise ValueError(f"comparison candidates not in the freeze record: {unknown}")

    by_pair: dict[tuple[str, str], list[PredictionRow]] = {}
    for p in predictions:
        if p.official_split == partition:
            by_pair.setdefault((p.candidate_id, p.condition), []).append(p)

    tasks = []
    for condition in freeze_record.conditions:
        for c in candidate_ids:
            if not by_pair.get((c, condition)):
                raise ValueError(
                    f"no {partition} predictions for candidate {c!r} under {condition!r}; "
                    "refusing a partial matrix"
                )
        tasks.append({
            "condition": condition,
            "partition": partition,
            "metrics": spec.metrics,
            "confidence_level": bootstrap.confidence_level,
            "n_resamples": bootstrap.n_resamples,
            "seed": bootstrap.seed,
            "low_support_min_class_count": spec.low_support_min_class_count,
            "thresholds": {c.candidate_id: c.threshold for c in freeze_record.candidates},
            "candidate_ids": candidate_ids,
            "rows": {c: by_pair[(c, condition)] for c in candidate_ids},
            "slices": [
                (s, definitions[s].clip_filter, definitions[s].bonafide_pool) for s in slice_ids
            ],
            "comparisons": [
                {"id": c.comparison_id, "a": c.a, "b": c.b} for c in spec.comparisons
            ],
        })

    outputs: list[dict[str, Any]] = []

    def done(index: int) -> None:
        if progress is not None:
            progress(f"{tasks[index]['condition']}: {index + 1}/{len(tasks)} conditions done")

    if workers > 1 and len(tasks) > 1:
        with contextlib.ExitStack() as stack:
            stack.enter_context(pinned_thread_env())
            pool = ProcessPoolExecutor(
                max_workers=min(workers, len(tasks)),
                mp_context=multiprocessing.get_context("spawn"),
            )
            stack.callback(pool.shutdown, wait=True, cancel_futures=True)
            for index, output in enumerate(pool.map(_condition_task, tasks)):
                outputs.append(output)
                done(index)
    else:
        for index, task in enumerate(tasks):
            outputs.append(_condition_task(task))
            done(index)

    return {
        "supplementary_version": SUPPLEMENTARY_VERSION,
        "label": (
            "supplementary_post_report" if partition == "evaluation"
            else "supplementary_rehearsal"
        ),
        "descriptive_only": True,
        "partition": partition,
        "score_direction": SCORE_DIRECTION,
        "freeze_record_state": freeze_record.state,
        "spec_sha256": spec.spec_sha256,
        "bootstrap": {
            "confidence_level": bootstrap.confidence_level,
            "n_resamples": bootstrap.n_resamples,
            "seed": bootstrap.seed,
            "resampling_unit": bootstrap.resampling_unit,
            "method": bootstrap.method,
            "note": (
                "speakers resampled once per condition from the whole partition; every "
                "slice, candidate and comparison in the condition uses the same resamples"
            ),
        },
        "metrics": list(spec.metrics),
        "candidates": candidate_ids,
        "slice_ids": [PARTITION_ALL_SLICE, *slice_ids],
        "git_commit": git_commit(),
        "package_version": package_version(),
        "generated_utc": datetime.now(UTC).isoformat(),
        "provenance": provenance or {},
        "results": outputs,
    }


def write_supplementary_report(report: dict[str, Any], output_dir: Path) -> Path:
    """Write the report; refuses to overwrite an existing one."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / REPORT_FILENAME
    if path.exists():
        raise FileExistsError(f"{path} already exists; refusing to overwrite")
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def compute_supplementary_hash(report_path: Path) -> str:
    """Hash of the report without ``generated_utc``."""

    with open(report_path) as fh:
        data = json.load(fh)
    data.pop("generated_utc", None)
    canonical = json.dumps(data, indent=2, sort_keys=True) + "\n"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
