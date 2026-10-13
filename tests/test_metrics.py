import json

import numpy as np
import pytest

from voice_detector.metrics import (
    SCORE_DIRECTION,
    compute_eer,
    evaluate_clip_scores,
    threshold_at_target_fpr,
)

# Hand-checkable fixture. Sorted: 1b 2b 3b 4s 5s 6b 7s 8s.
# At threshold 5 (score >= 5 means spoof): TP=3 (5, 7, 8), FN=1 (4), FP=1 (6), TN=3.
_BONAFIDE = [1.0, 2.0, 3.0, 6.0]
_SPOOF = [4.0, 5.0, 7.0, 8.0]


def _clips(bonafide: list[float], spoof: list[float]) -> tuple[list[str], list[str], list[float]]:
    ids = [f"b{i}" for i in range(len(bonafide))] + [f"s{i}" for i in range(len(spoof))]
    labels = ["bonafide"] * len(bonafide) + ["spoof"] * len(spoof)
    return ids, labels, bonafide + spoof


def _reference_step_eer(bonafide: np.ndarray, spoof: np.ndarray) -> float:
    """Step-DET EER in the ASVspoof reference style, written independently.

    Walk the thresholds over every sorted score. Take the operating point with
    the smallest |miss - false alarm| and average the two rates there.
    """

    best_gap, best_eer = np.inf, np.nan
    for t in np.concatenate([[-np.inf], np.sort(np.concatenate([bonafide, spoof]))]):
        miss = np.mean(spoof <= t)
        false_alarm = np.mean(bonafide > t)
        if abs(miss - false_alarm) < best_gap:
            best_gap, best_eer = abs(miss - false_alarm), (miss + false_alarm) / 2
    return float(best_eer)


def test_eer_of_perfect_separation_is_zero() -> None:
    result = compute_eer([0.0, 1.0], [2.0, 3.0])
    assert result.eer == 0.0
    assert 1.0 < result.threshold <= 2.0


def test_eer_of_inverted_scores_is_one() -> None:
    assert compute_eer([2.0, 3.0], [0.0, 1.0]).eer == pytest.approx(1.0)


def test_eer_of_fully_tied_scores_is_one_half() -> None:
    # A stable-sort step DET reports 1.0 here; grouping ties gives the honest 0.5.
    assert compute_eer([0.5, 0.5, 0.5], [0.5, 0.5]).eer == pytest.approx(0.5)


def test_eer_hand_computed_exact_crossing() -> None:
    result = compute_eer(_BONAFIDE, _SPOOF)
    assert result.eer == pytest.approx(0.25)
    assert result.threshold == 5.0


def test_eer_interpolates_across_a_miss_rate_step() -> None:
    # Sorted 1b 2s 3s 4b 5s. FPR stays at 0.5 while FNR drops 2/3 -> 1/3,
    # so the curves cross at 0.5. The step reference would report 0.583.
    result = compute_eer([1.0, 4.0], [2.0, 3.0, 5.0])
    assert result.eer == pytest.approx(0.5)
    assert result.threshold == 3.0


def test_eer_of_identical_distributions_is_near_one_half() -> None:
    rng = np.random.default_rng(20261004)
    result = compute_eer(rng.normal(size=4000), rng.normal(size=4000))
    assert result.eer == pytest.approx(0.5, abs=0.03)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_eer_matches_step_reference_within_half_a_step(seed: int) -> None:
    rng = np.random.default_rng(seed)
    bonafide = rng.normal(0.0, 1.0, size=257)
    spoof = rng.normal(1.2, 1.0, size=83)
    tolerance = max(1 / bonafide.size, 1 / spoof.size) / 2 + 1e-12
    ours = compute_eer(bonafide, spoof).eer
    assert abs(ours - _reference_step_eer(bonafide, spoof)) <= tolerance


def test_eer_rejects_empty_class_and_non_finite_scores() -> None:
    with pytest.raises(ValueError, match="at least one"):
        compute_eer([], [1.0])
    with pytest.raises(ValueError, match="finite"):
        compute_eer([np.nan], [1.0])


def test_threshold_is_the_lowest_meeting_the_target() -> None:
    bonafide = np.arange(20, dtype=float)  # 0..19; 5% of 20 allows one false alarm
    result = threshold_at_target_fpr(bonafide, target_fpr=0.05)
    assert 18.0 < result.threshold <= 19.0
    assert result.n_false_alarms == 1
    assert result.achieved_fpr == pytest.approx(0.05)
    # Any lower threshold that reaches the next bona fide score breaks the target.
    assert np.mean(bonafide >= 18.0) > 0.05


def test_threshold_ties_at_the_boundary_count_as_spoof() -> None:
    # Three clips tie at the top, but only one false alarm is allowed: all three are excluded.
    bonafide = [5.0, 5.0, 5.0] + [v / 10 for v in range(17)]  # the rest stay below 5
    result = threshold_at_target_fpr(bonafide, target_fpr=0.05)
    assert result.threshold > 5.0
    assert result.n_false_alarms == 0
    assert result.achieved_fpr <= 0.05


def test_zero_target_flags_no_bona_fide_clip() -> None:
    result = threshold_at_target_fpr([0.1, 0.7, 0.3], target_fpr=0.0)
    assert result.threshold > 0.7
    assert result.n_false_alarms == 0


def test_allowed_false_alarms_survive_float_rounding() -> None:
    # 0.29 * 100 is 28.999999999999996 in float64; 29 false alarms are allowed.
    result = threshold_at_target_fpr(np.arange(100, dtype=float), target_fpr=0.29)
    assert result.n_false_alarms == 29


@pytest.mark.parametrize("target", [-0.01, 1.0, 1.5])
def test_threshold_rejects_out_of_range_target(target: float) -> None:
    with pytest.raises(ValueError, match="target_fpr"):
        threshold_at_target_fpr([0.0, 1.0], target_fpr=target)


def test_evaluate_hand_computed_metrics() -> None:
    metrics = evaluate_clip_scores(*_clips(_BONAFIDE, _SPOOF), threshold=5.0)

    assert (metrics.n_clips, metrics.n_bonafide, metrics.n_spoof) == (8, 4, 4)
    assert metrics.spoof_prevalence == 0.5
    assert metrics.score_direction == SCORE_DIRECTION
    assert (metrics.confusion.tn, metrics.confusion.fp) == (3, 1)
    assert (metrics.confusion.fn, metrics.confusion.tp) == (1, 3)
    assert metrics.human_false_positive_rate == pytest.approx(0.25)
    assert metrics.synthetic_detection_rate == pytest.approx(0.75)
    assert metrics.spoof.precision == pytest.approx(0.75)
    assert metrics.bonafide.precision == pytest.approx(0.75)
    assert metrics.bonafide.recall == pytest.approx(0.75)
    assert metrics.balanced_accuracy == pytest.approx(0.75)
    assert metrics.macro_f1 == pytest.approx(0.75)
    assert metrics.eer == pytest.approx(0.25)
    assert metrics.roc_auc == pytest.approx(14 / 16)  # 14 of 16 spoof/bona fide pairs ordered


def test_score_equal_to_threshold_counts_as_spoof() -> None:
    metrics = evaluate_clip_scores(*_clips(_BONAFIDE, _SPOOF), threshold=6.0)
    assert metrics.confusion.fp == 1  # the bona fide clip scoring exactly 6.0


def test_precision_reflects_prevalence() -> None:
    # Same per-class rates as the hand fixture, but 4x more bona fide clips.
    bonafide = _BONAFIDE * 4
    metrics = evaluate_clip_scores(*_clips(bonafide, _SPOOF), threshold=5.0)
    assert metrics.human_false_positive_rate == pytest.approx(0.25)
    assert metrics.synthetic_detection_rate == pytest.approx(0.75)
    assert metrics.spoof.precision == pytest.approx(3 / 7)
    assert metrics.spoof_prevalence == pytest.approx(0.2)


def test_single_class_slice_reports_counts_not_invented_numbers() -> None:
    ids, labels, scores = _clips([], [4.0, 5.0, 7.0])
    metrics = evaluate_clip_scores(ids, labels, scores, threshold=5.0)

    assert (metrics.n_bonafide, metrics.n_spoof) == (0, 3)
    assert metrics.synthetic_detection_rate == pytest.approx(2 / 3)
    for undefined in (
        metrics.eer,
        metrics.eer_threshold,
        metrics.roc_auc,
        metrics.balanced_accuracy,
        metrics.macro_f1,
        metrics.human_false_positive_rate,
        metrics.spoof.precision,
        metrics.spoof.f1,
    ):
        assert undefined is None


def test_duplicate_clip_ids_are_rejected_as_window_level_input() -> None:
    with pytest.raises(ValueError, match="one score per clip"):
        evaluate_clip_scores(["a", "a"], ["spoof", "spoof"], [0.1, 0.2], threshold=0.0)


@pytest.mark.parametrize(
    ("clip_ids", "labels", "scores", "message"),
    [
        (["a"], ["fake"], [0.1], "labels must be one of"),
        (["a", "b"], ["spoof"], [0.1, 0.2], "equal length"),
        ([], [], [], "no clips"),
        (["a"], ["spoof"], [np.inf], "finite"),
    ],
)
def test_invalid_inputs_are_rejected(
    clip_ids: list[str], labels: list[str], scores: list[float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_clip_scores(clip_ids, labels, scores, threshold=0.0)


def test_metrics_serialize_to_json_for_run_records() -> None:
    metrics = evaluate_clip_scores(*_clips(_BONAFIDE, _SPOOF), threshold=5.0)
    record = json.loads(json.dumps(metrics.as_dict()))
    assert record["confusion"] == {"tn": 3, "fp": 1, "fn": 1, "tp": 3}
    assert record["spoof"]["support"] == 4
