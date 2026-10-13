"""Clip-level detection metrics shared by development reports and locked evaluation.

One implementation serves the LR baseline, the CNN study and the locked
evaluation, so every reported number is computed the same way (``METRICS_VERSION``).
Frozen project conventions:

- **One score per original clip.** Window scores are aggregated upstream by the
  scorer. Window-level rows are a bug here, so the public entry points reject
  duplicate clip IDs.
- **Synthetic speech is the positive class.** Larger scores mean more synthetic
  (``scoring.SCORE_DIRECTION``, which callers must state explicitly). The
  operating decision is ``scoring.decide_spoof``: spoof if ``score >= threshold``,
  compared in ``float64``, so a tie at the threshold counts as spoof.
- **Scores, not probabilities.** Nothing here calibrates a score.
- **Thresholds come from reserved development data** (``threshold_at_target_fpr``
  on ``dev_threshold``). EER is descriptive and never selects a threshold.
- **No invented numbers.** A metric a slice cannot support, such as EER on an
  attack-only slice with no bona fide clips, is ``None``, and the class counts are
  always reported.

Precision depends on class prevalence (ASVspoof 2019 LA is about 1:8.8 bona fide
to spoof), so it is returned with ``spoof_prevalence`` and is not transferable to
a population with a different mix.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import roc_auc_score

from voice_detector.manifest import VALID_LABELS
from voice_detector.scoring import DECISION_RULE, SCORE_DIRECTION, decide_spoof

# Bump when any metric definition changes. Recorded in every ClipMetrics, so
# numbers from different definitions are never compared silently. metrics-v1 EER:
# tie-grouped ROC with linear interpolation at FNR == FPR (see compute_eer).
METRICS_VERSION = "metrics-v1"
POSITIVE_LABEL = "spoof"


def _as_scores(values: ArrayLike, name: str) -> np.ndarray:
    """Return finite, one-dimensional ``float64`` scores or raise."""

    raw = np.asarray(values)
    if raw.size and raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be numeric (int or float), got dtype {raw.dtype}")
    scores = raw.astype(np.float64)
    if scores.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {scores.shape}")
    if not np.all(np.isfinite(scores)):
        raise ValueError(f"{name} must contain only finite values")
    return scores


def _check_direction(score_direction: str) -> None:
    if score_direction != SCORE_DIRECTION:
        raise ValueError(
            f"scores must follow {SCORE_DIRECTION!r}, got {score_direction!r}; "
            "convert them before computing metrics"
        )


def _validated_labels(clip_ids: Sequence[str], labels: Sequence[str], n_scores: int) -> np.ndarray:
    """Check the one-score-per-clip contract and return a boolean spoof mask."""

    if not len(clip_ids) == len(labels) == n_scores:
        raise ValueError(
            f"clip_ids, labels and scores must have equal length, got "
            f"{len(clip_ids)}, {len(labels)} and {n_scores}"
        )
    if n_scores == 0:
        raise ValueError("no clips to evaluate")
    duplicates = sorted(clip_id for clip_id, count in Counter(clip_ids).items() if count > 1)
    if duplicates:
        raise ValueError(
            f"expected one score per clip; {len(duplicates)} duplicated clip IDs "
            f"(aggregate window scores first), e.g. {duplicates[:3]}"
        )
    unknown = sorted({label for label in labels if label not in VALID_LABELS})
    if unknown:
        raise ValueError(f"labels must be one of {sorted(VALID_LABELS)}, got {unknown}")
    return np.array([label == POSITIVE_LABEL for label in labels], dtype=bool)


@dataclass(frozen=True)
class EerResult:
    """Equal error rate and an achievable threshold at the crossing."""

    eer: float
    threshold: float


def compute_eer(bonafide_scores: ArrayLike, spoof_scores: ArrayLike) -> EerResult:
    """Equal error rate, where the miss rate equals the false-alarm rate.

    This is a low-level helper on bare score arrays that follow
    ``SCORE_DIRECTION``. It does not check the one-score-per-clip contract.
    Reports go through ``evaluate_clip_scores``, which does.

    The ROC is built over distinct score values, so tied scores never create an
    operating point that no threshold can reach. The crossing is located on
    integer counts, so an exact crossing is never missed to float rounding. The
    EER is where the linearly interpolated curve crosses ``FNR == FPR``.

    For tie-free scores this is within a quarter score step,
    ``max(1/n_bonafide, 1/n_spoof) / 4``, of the ASVspoof reference step
    implementation. With ties it can differ more; fully tied scores give 0.5,
    where the reference gives 1.0.

    ``threshold`` is the first achievable threshold at or past the crossing. It
    is diagnostic only and never an operating threshold.
    """

    bonafide = _as_scores(bonafide_scores, "bonafide_scores")
    spoof = _as_scores(spoof_scores, "spoof_scores")
    n_b, n_s = bonafide.size, spoof.size
    if n_b == 0 or n_s == 0:
        raise ValueError("EER needs at least one bona fide and one spoof score")

    scores = np.concatenate([bonafide, spoof])
    is_spoof = np.concatenate([np.zeros(n_b, dtype=np.int64), np.ones(n_s, dtype=np.int64)])
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    # Last index of each group of tied scores, from highest to lowest score.
    group_end = np.append(np.flatnonzero(np.diff(sorted_scores)), scores.size - 1)
    tp = np.concatenate([[0], np.cumsum(is_spoof[order])[group_end]])
    fp = np.concatenate([[0], group_end + 1 - tp[1:]])
    thresholds = np.concatenate([[np.inf], sorted_scores[group_end]])

    # Sign of FNR - FPR on integers: (n_s - tp)/n_s - fp/n_b, scaled by n_s * n_b.
    # It falls strictly from +n_s*n_b (nothing flagged) to -n_s*n_b (all flagged).
    gap = (n_s - tp) * n_b - fp * n_s
    k = int(np.argmax(gap <= 0))
    fpr = fp / n_b
    if gap[k] == 0:
        return EerResult(eer=float(fpr[k]), threshold=float(thresholds[k]))
    fnr = (n_s - tp) / n_s
    before, after = fnr[k - 1] - fpr[k - 1], fnr[k] - fpr[k]
    weight = before / (before - after)
    eer = fpr[k - 1] + weight * (fpr[k] - fpr[k - 1])
    return EerResult(eer=float(eer), threshold=float(thresholds[k]))


@dataclass(frozen=True)
class ThresholdResult:
    """An operating threshold chosen from bona fide clips, with its evidence."""

    threshold: float
    boundary_score: float
    decision_rule: str
    target_fpr: float
    achieved_fpr: float
    n_bonafide: int
    n_false_alarms: int


def _max_allowed_false_alarms(target_fpr: float, n: int) -> int:
    """Largest ``k`` with ``k / n <= target_fpr``, robust to float rounding."""

    k = int(np.floor(target_fpr * n))
    if (k + 1) / n <= target_fpr:
        k += 1
    while k > 0 and k / n > target_fpr:
        k -= 1
    return k


def threshold_at_target_fpr(
    clip_ids: Sequence[str],
    labels: Sequence[str],
    scores: ArrayLike,
    *,
    target_fpr: float,
    score_direction: str,
) -> ThresholdResult:
    """Lowest threshold whose human false-positive rate is at most ``target_fpr``.

    The project rule, applied to the clips of the reserved ``dev_threshold``
    speakers: flag as much synthetic speech as possible while at most
    ``target_fpr`` of bona fide clips are flagged. Input is validated like a
    report (one score per clip, known labels). Only bona fide clips determine
    the threshold.

    Ties count as spoof (``decide_spoof``). If several bona fide clips share the
    boundary score, they are all excluded, and the achieved rate falls below the
    target rather than above it. The threshold is the next ``float64`` above the
    boundary score, the ``(k + 1)``-th highest bona fide score, where ``k`` is the
    largest allowed number of false alarms. ``decide_spoof`` compares in
    ``float64``, so ``float32`` model scores reproduce it exactly. The target
    holds on the data it was chosen on, not as a guarantee on other data.
    """

    _check_direction(score_direction)
    values = _as_scores(scores, "scores")
    is_spoof = _validated_labels(clip_ids, labels, values.size)
    bonafide = values[~is_spoof]
    if bonafide.size == 0:
        raise ValueError("threshold selection needs at least one bona fide clip")
    if isinstance(target_fpr, bool) or not 0.0 <= target_fpr < 1.0:
        raise ValueError(f"target_fpr must be in [0, 1), got {target_fpr!r}")

    allowed = _max_allowed_false_alarms(float(target_fpr), bonafide.size)
    boundary = float(np.sort(bonafide)[::-1][allowed])
    threshold = float(np.nextafter(boundary, np.inf))
    false_alarms = int(np.count_nonzero(decide_spoof(bonafide, threshold)))
    return ThresholdResult(
        threshold=threshold,
        boundary_score=boundary,
        decision_rule=DECISION_RULE,
        target_fpr=float(target_fpr),
        achieved_fpr=false_alarms / bonafide.size,
        n_bonafide=int(bonafide.size),
        n_false_alarms=false_alarms,
    )


@dataclass(frozen=True)
class ConfusionCounts:
    """Counts at one threshold. Spoof is positive, so ``fp`` is a human false alarm."""

    tn: int
    fp: int
    fn: int
    tp: int


@dataclass(frozen=True)
class ClassScores:
    """Per-class precision, recall and F1. ``None`` where the slice cannot support it."""

    support: int
    precision: float | None
    recall: float | None
    f1: float | None


@dataclass(frozen=True)
class ClipMetrics:
    """The required clip-level metrics for one set of clips.

    With ``threshold=None`` only the threshold-free metrics are filled (counts,
    EER, ROC AUC); that is the form used for model selection on ``dev_select``.
    """

    metrics_version: str
    n_clips: int
    n_bonafide: int
    n_spoof: int
    spoof_prevalence: float
    score_direction: str
    decision_rule: str
    threshold: float | None
    confusion: ConfusionCounts | None
    eer: float | None
    roc_auc: float | None
    balanced_accuracy: float | None
    macro_f1: float | None
    human_false_positive_rate: float | None
    synthetic_detection_rate: float | None
    bonafide: ClassScores
    spoof: ClassScores

    def as_dict(self) -> dict[str, Any]:
        """Plain, JSON-serializable form for run records and reports."""

        return asdict(self)


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _class_scores(tp: int, fp: int, fn: int, support: int, *, both_classes: bool) -> ClassScores:
    """Scores for one class, treating that class as positive.

    Precision and F1 need both classes in the slice; with one class they are
    trivially perfect or undefined and would mislead, so they are ``None``.
    """

    if not both_classes:
        return ClassScores(support=support, precision=None, recall=_ratio(tp, support), f1=None)
    return ClassScores(
        support=support,
        precision=_ratio(tp, tp + fp),
        recall=_ratio(tp, support),
        f1=_ratio(2 * tp, 2 * tp + fp + fn),
    )


def evaluate_clip_scores(
    clip_ids: Sequence[str],
    labels: Sequence[str],
    scores: ArrayLike,
    *,
    threshold: float | None,
    score_direction: str,
) -> ClipMetrics:
    """Compute every required clip-level metric for one set of clips.

    ``threshold`` must be a frozen operating threshold from reserved development
    data, or ``None`` for threshold-free metrics (model selection on
    ``dev_select``). This function never chooses a threshold. EER and ROC AUC are
    threshold-free and descriptive.
    """

    _check_direction(score_direction)
    values = _as_scores(scores, "scores")
    is_spoof = _validated_labels(clip_ids, labels, values.size)
    n_spoof = int(np.count_nonzero(is_spoof))
    n_bonafide = int(values.size - n_spoof)
    both = n_spoof > 0 and n_bonafide > 0

    eer: float | None = None
    roc_auc: float | None = None
    if both:
        eer = compute_eer(values[~is_spoof], values[is_spoof]).eer
        roc_auc = float(roc_auc_score(is_spoof, values))

    common = {
        "metrics_version": METRICS_VERSION,
        "n_clips": int(values.size),
        "n_bonafide": n_bonafide,
        "n_spoof": n_spoof,
        "spoof_prevalence": n_spoof / values.size,
        "score_direction": SCORE_DIRECTION,
        "decision_rule": DECISION_RULE,
        "eer": eer,
        "roc_auc": roc_auc,
    }
    if threshold is None:
        return ClipMetrics(
            **common,
            threshold=None,
            confusion=None,
            balanced_accuracy=None,
            macro_f1=None,
            human_false_positive_rate=None,
            synthetic_detection_rate=None,
            bonafide=ClassScores(support=n_bonafide, precision=None, recall=None, f1=None),
            spoof=ClassScores(support=n_spoof, precision=None, recall=None, f1=None),
        )

    flagged = decide_spoof(values, threshold)
    tp = int(np.count_nonzero(is_spoof & flagged))
    fn = n_spoof - tp
    fp = int(np.count_nonzero(~is_spoof & flagged))
    tn = n_bonafide - fp

    balanced_accuracy: float | None = None
    macro_f1: float | None = None
    if both:
        balanced_accuracy = (tn / n_bonafide + tp / n_spoof) / 2.0
        macro_f1 = (2 * tn / (2 * tn + fn + fp) + 2 * tp / (2 * tp + fp + fn)) / 2.0

    spoof = _class_scores(tp, fp, fn, n_spoof, both_classes=both)
    return ClipMetrics(
        **common,
        threshold=float(threshold),
        confusion=ConfusionCounts(tn=tn, fp=fp, fn=fn, tp=tp),
        balanced_accuracy=balanced_accuracy,
        macro_f1=macro_f1,
        human_false_positive_rate=_ratio(fp, n_bonafide),
        synthetic_detection_rate=spoof.recall,
        bonafide=_class_scores(tn, fn, fp, n_bonafide, both_classes=both),
        spoof=spoof,
    )
