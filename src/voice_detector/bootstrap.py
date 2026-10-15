"""Speaker-grouped bootstrap for locked evaluation (issue #35).

Resampling unit is ``speaker_id`` (67 evaluation speakers in ASVspoof 2019 LA,
no source recording mapping available). Each resample draws speakers with
replacement; all clips of each drawn speaker are included.

Paired differences use the same speaker resamples so confidence intervals on
differences between candidates or conditions are valid paired comparisons.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

BOOTSTRAP_VERSION = "bootstrap-v1"


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
class BootstrapCI:
    metric: str
    point_estimate: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    n_resamples: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PairedDifference:
    metric: str
    candidate_a: str
    candidate_b: str
    point_difference: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    n_resamples: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _group_by_speaker(
    speaker_ids: Sequence[str],
) -> tuple[list[str], dict[str, list[int]]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, spk in enumerate(speaker_ids):
        groups[spk].append(idx)
    unique_speakers = sorted(groups.keys())
    return unique_speakers, groups


def _resample_indices(
    speakers: list[str],
    groups: dict[str, list[int]],
    rng: np.random.Generator,
) -> np.ndarray:
    chosen = rng.choice(len(speakers), size=len(speakers), replace=True)
    indices: list[int] = []
    for i in chosen:
        indices.extend(groups[speakers[i]])
    return np.array(indices, dtype=np.intp)


def _percentile_ci(
    distribution: np.ndarray,
    confidence_level: float,
) -> tuple[float, float]:
    alpha = 1.0 - confidence_level
    lower = float(np.percentile(distribution, 100 * alpha / 2))
    upper = float(np.percentile(distribution, 100 * (1 - alpha / 2)))
    return lower, upper


def speaker_grouped_bootstrap(
    clip_ids: Sequence[str],
    labels: Sequence[str],
    scores: ArrayLike,
    speaker_ids: Sequence[str],
    *,
    metric_fn: Callable[..., float],
    metric_name: str,
    spec: BootstrapSpec,
    score_direction: str,
    threshold: float | None = None,
) -> BootstrapCI:
    scores_arr = np.asarray(scores, dtype=np.float64)
    labels_arr = np.array(labels)
    clip_ids_arr = np.array(clip_ids)
    speaker_ids_list = list(speaker_ids)
    n = len(clip_ids)

    if n == 0:
        raise ValueError("cannot bootstrap with zero clips")

    unique_speakers, groups = _group_by_speaker(speaker_ids_list)
    if len(unique_speakers) < 2:
        raise ValueError(
            f"bootstrap needs at least 2 speakers, got {len(unique_speakers)}"
        )

    kwargs: dict[str, Any] = {"score_direction": score_direction}
    if threshold is not None:
        kwargs["threshold"] = threshold

    point_estimate = float(metric_fn(
        list(clip_ids_arr), list(labels_arr), scores_arr, **kwargs,
    ))

    rng = np.random.default_rng(spec.seed)
    boot_values = np.empty(spec.n_resamples, dtype=np.float64)

    for b in range(spec.n_resamples):
        idx = _resample_indices(unique_speakers, groups, rng)
        boot_values[b] = metric_fn(
            list(clip_ids_arr[idx]),
            list(labels_arr[idx]),
            scores_arr[idx],
            **kwargs,
        )

    ci_lower, ci_upper = _percentile_ci(boot_values, spec.confidence_level)

    return BootstrapCI(
        metric=metric_name,
        point_estimate=point_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence_level=spec.confidence_level,
        n_resamples=spec.n_resamples,
    )


def paired_bootstrap(
    clip_ids: Sequence[str],
    labels: Sequence[str],
    scores_a: ArrayLike,
    scores_b: ArrayLike,
    speaker_ids: Sequence[str],
    *,
    metric_fn: Callable[..., float],
    metric_name: str,
    candidate_a: str,
    candidate_b: str,
    spec: BootstrapSpec,
    score_direction: str,
    threshold_a: float | None = None,
    threshold_b: float | None = None,
) -> PairedDifference:
    scores_a_arr = np.asarray(scores_a, dtype=np.float64)
    scores_b_arr = np.asarray(scores_b, dtype=np.float64)
    labels_arr = np.array(labels)
    clip_ids_arr = np.array(clip_ids)
    speaker_ids_list = list(speaker_ids)
    n = len(clip_ids)

    if n == 0:
        raise ValueError("cannot bootstrap with zero clips")

    unique_speakers, groups = _group_by_speaker(speaker_ids_list)
    if len(unique_speakers) < 2:
        raise ValueError(
            f"paired bootstrap needs at least 2 speakers, got {len(unique_speakers)}"
        )

    kwargs_a: dict[str, Any] = {"score_direction": score_direction}
    kwargs_b: dict[str, Any] = {"score_direction": score_direction}
    if threshold_a is not None:
        kwargs_a["threshold"] = threshold_a
    if threshold_b is not None:
        kwargs_b["threshold"] = threshold_b

    point_a = float(metric_fn(
        list(clip_ids_arr), list(labels_arr), scores_a_arr, **kwargs_a,
    ))
    point_b = float(metric_fn(
        list(clip_ids_arr), list(labels_arr), scores_b_arr, **kwargs_b,
    ))
    point_diff = point_a - point_b

    rng = np.random.default_rng(spec.seed)
    diffs = np.empty(spec.n_resamples, dtype=np.float64)

    for b in range(spec.n_resamples):
        idx = _resample_indices(unique_speakers, groups, rng)
        val_a = metric_fn(
            list(clip_ids_arr[idx]), list(labels_arr[idx]),
            scores_a_arr[idx], **kwargs_a,
        )
        val_b = metric_fn(
            list(clip_ids_arr[idx]), list(labels_arr[idx]),
            scores_b_arr[idx], **kwargs_b,
        )
        diffs[b] = val_a - val_b

    ci_lower, ci_upper = _percentile_ci(diffs, spec.confidence_level)

    return PairedDifference(
        metric=metric_name,
        candidate_a=candidate_a,
        candidate_b=candidate_b,
        point_difference=point_diff,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence_level=spec.confidence_level,
        n_resamples=spec.n_resamples,
    )
