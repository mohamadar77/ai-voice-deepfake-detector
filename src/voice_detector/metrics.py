"""Clip-level detection metrics shared by development reports and locked evaluation.

One implementation serves the LR baseline, the CNN study and the locked
evaluation, so every reported number is computed the same way. Conventions are
frozen project-wide:

- **One score per original clip.** Window scores are aggregated upstream by the
  scorer. Window-level rows are a bug here, so duplicate clip IDs are rejected.
- **Synthetic speech is the positive class.** ``spoof`` is positive and larger
  scores mean more synthetic (``SCORE_DIRECTION``). A clip is labeled spoof when
  ``score >= threshold``, so a tie at the threshold counts as spoof.
- **Scores, not probabilities.** Nothing here calibrates a score.
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
from sklearn.metrics import roc_auc_score, roc_curve

from voice_detector.manifest import VALID_LABELS

POSITIVE_LABEL = "spoof"
NEGATIVE_LABEL = "bonafide"
SCORE_DIRECTION = "higher_more_spoof"


def _as_scores(values: ArrayLike, name: str) -> np.ndarray:
    """Return finite, one-dimensional ``float64`` scores or raise."""

    scores = np.asarray(values, dtype=np.float64)
    if scores.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {scores.shape}")
    if not np.all(np.isfinite(scores)):
        raise ValueError(f"{name} must contain only finite values")
    return scores


@dataclass(frozen=True)
class EerResult:
    """Equal error rate and an achievable threshold at the crossing."""

    eer: float
    threshold: float


def compute_eer(bonafide_scores: ArrayLike, spoof_scores: ArrayLike) -> EerResult:
    """Equal error rate, where the miss rate equals the false-alarm rate.

    The ROC is built over distinct score values, so tied scores never create an
    operating point no threshold can reach. The EER is where the linearly
    interpolated curve crosses ``FNR == FPR``. For tie-free scores this agrees
    with the ASVspoof reference step implementation to within half a score step,
    ``max(1/n_bonafide, 1/n_spoof) / 2``.

    ``threshold`` is the first achievable threshold (``score >= threshold`` means
    spoof) at or past the crossing. It is descriptive only: operating thresholds
    come from ``threshold_at_target_fpr`` on reserved development data, never
    from an EER.
    """

    bonafide = _as_scores(bonafide_scores, "bonafide_scores")
    spoof = _as_scores(spoof_scores, "spoof_scores")
    if bonafide.size == 0 or spoof.size == 0:
        raise ValueError("EER needs at least one bona fide and one spoof score")

    y_true = np.concatenate([np.zeros(bonafide.size), np.ones(spoof.size)])
    y_score = np.concatenate([bonafide, spoof])
    fpr, tpr, thresholds = roc_curve(y_true, y_score, drop_intermediate=False)

    # gap = FNR - FPR falls monotonically from +1 (nothing flagged) to -1
    # (everything flagged), so it crosses zero exactly once.
    gap = (1.0 - tpr) - fpr
    k = int(np.argmax(gap <= 0.0))
    if gap[k] == 0.0:
        return EerResult(eer=float(fpr[k]), threshold=float(thresholds[k]))
    weight = gap[k - 1] / (gap[k - 1] - gap[k])
    eer = fpr[k - 1] + weight * (fpr[k] - fpr[k - 1])
    return EerResult(eer=float(eer), threshold=float(thresholds[k]))


@dataclass(frozen=True)
class ThresholdResult:
    """An operating threshold chosen from bona fide scores, with its evidence."""

    threshold: float
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


def threshold_at_target_fpr(bonafide_scores: ArrayLike, *, target_fpr: float) -> ThresholdResult:
    """Lowest threshold whose human false-positive rate is at most ``target_fpr``.

    The project rule, applied on the reserved ``dev_threshold`` speakers: flag
    as much synthetic speech as possible while at most ``target_fpr`` of bona
    fide clips score ``>= threshold``. Ties count as spoof. If several bona fide
    clips share the boundary score, they are all excluded, and the achieved rate
    falls below the target rather than above it.

    The threshold is the next ``float64`` above the ``(k + 1)``-th highest bona
    fide score, where ``k`` is the largest allowed number of false alarms.
    Compare scores in ``float64`` to reproduce it. The target holds on the data
    it was chosen on, not as a guarantee on other data.
    """

    bonafide = _as_scores(bonafide_scores, "bonafide_scores")
    if bonafide.size == 0:
        raise ValueError("threshold selection needs at least one bona fide score")
    if not 0.0 <= target_fpr < 1.0:
        raise ValueError(f"target_fpr must be in [0, 1), got {target_fpr}")

    allowed = _max_allowed_false_alarms(target_fpr, bonafide.size)
    boundary = np.sort(bonafide)[::-1][allowed]
    threshold = float(np.nextafter(boundary, np.inf))
    false_alarms = int(np.count_nonzero(bonafide >= threshold))
    return ThresholdResult(
        threshold=threshold,
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
    """The required clip-level metrics for one set of clips at one threshold."""

    n_clips: int
    n_bonafide: int
    n_spoof: int
    spoof_prevalence: float
    score_direction: str
    threshold: float
    confusion: ConfusionCounts
    eer: float | None
    eer_threshold: float | None
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


def evaluate_clip_scores(
    clip_ids: Sequence[str],
    labels: Sequence[str],
    scores: ArrayLike,
    *,
    threshold: float,
) -> ClipMetrics:
    """Compute every required clip-level metric at a fixed operating threshold.

    The threshold must come from reserved development data; this function never
    chooses one. EER and ROC AUC are threshold-free and descriptive.
    """

    values = _as_scores(scores, "scores")
    is_spoof = _validated_labels(clip_ids, labels, values.size)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")

    flagged = values >= threshold
    tp = int(np.count_nonzero(is_spoof & flagged))
    fn = int(np.count_nonzero(is_spoof & ~flagged))
    fp = int(np.count_nonzero(~is_spoof & flagged))
    tn = int(np.count_nonzero(~is_spoof & ~flagged))
    n_spoof = tp + fn
    n_bonafide = tn + fp
    both = n_spoof > 0 and n_bonafide > 0

    bonafide = _class_scores(tn, fn, fp, n_bonafide, both_classes=both)
    spoof = _class_scores(tp, fp, fn, n_spoof, both_classes=both)

    eer: EerResult | None = None
    roc_auc: float | None = None
    balanced_accuracy: float | None = None
    macro_f1: float | None = None
    if both:
        eer = compute_eer(values[~is_spoof], values[is_spoof])
        roc_auc = float(roc_auc_score(is_spoof, values))
        balanced_accuracy = (tn / n_bonafide + tp / n_spoof) / 2.0
        macro_f1 = (2 * tn / (2 * tn + fn + fp) + 2 * tp / (2 * tp + fp + fn)) / 2.0

    human_fpr = _ratio(fp, n_bonafide)
    return ClipMetrics(
        n_clips=int(values.size),
        n_bonafide=n_bonafide,
        n_spoof=n_spoof,
        spoof_prevalence=n_spoof / values.size,
        score_direction=SCORE_DIRECTION,
        threshold=float(threshold),
        confusion=ConfusionCounts(tn=tn, fp=fp, fn=fn, tp=tp),
        eer=None if eer is None else eer.eer,
        eer_threshold=None if eer is None else eer.threshold,
        roc_auc=roc_auc,
        balanced_accuracy=balanced_accuracy,
        macro_f1=macro_f1,
        human_false_positive_rate=human_fpr,
        synthetic_detection_rate=spoof.recall,
        bonafide=bonafide,
        spoof=spoof,
    )
