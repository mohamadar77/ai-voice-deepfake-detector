import json

import numpy as np
import pytest

from voice_detector.metrics import (
    METRICS_VERSION,
    compute_eer,
    evaluate_clip_scores,
    threshold_at_target_fpr,
)
from voice_detector.scoring import DECISION_RULE, SCORE_DIRECTION, decide_spoof

# Asymmetric hand-checkable fixture, so swapped classes or formulas cannot pass.
# Sorted: 1b 2b 3b 4s 5s 6b 7s 9b. At threshold 5 (score >= 5 means spoof):
# TP=2 (5, 7), FN=1 (4), FP=2 (6, 9), TN=3 (1, 2, 3).
_BONAFIDE = [1.0, 2.0, 3.0, 6.0, 9.0]
_SPOOF = [4.0, 5.0, 7.0]


def _clips(bonafide: list[float], spoof: list[float]) -> tuple[list[str], list[str], list[float]]:
    ids = [f"b{i}" for i in range(len(bonafide))] + [f"s{i}" for i in range(len(spoof))]
    labels = ["bonafide"] * len(bonafide) + ["spoof"] * len(spoof)
    return ids, labels, bonafide + spoof


def _evaluate(bonafide: list[float], spoof: list[float], threshold: float | None):
    return evaluate_clip_scores(
        *_clips(bonafide, spoof), threshold=threshold, score_direction=SCORE_DIRECTION
    )


def _threshold(bonafide, target_fpr: float, spoof=()):
    ids, labels, scores = _clips(list(bonafide), list(spoof))
    return threshold_at_target_fpr(
        ids, labels, scores, target_fpr=target_fpr, score_direction=SCORE_DIRECTION
    )


def _reference_step_eer(bonafide: np.ndarray, spoof: np.ndarray) -> float:
    """Independent step-DET EER for tie-free scores (ASVspoof reference style).

    Walk every sorted score as a threshold, take the operating point with the
    smallest |miss - false alarm| and average the two rates there. Unlike the
    official script it groups ties, so use it only on tie-free scores.
    """

    best_gap, best_eer = np.inf, np.nan
    for t in np.concatenate([[-np.inf], np.sort(np.concatenate([bonafide, spoof]))]):
        miss = np.mean(spoof <= t)
        false_alarm = np.mean(bonafide > t)
        if abs(miss - false_alarm) < best_gap:
            best_gap, best_eer = abs(miss - false_alarm), (miss + false_alarm) / 2
    return float(best_eer)


# --- compute_eer ---------------------------------------------------------------


def test_eer_of_perfect_separation_is_zero() -> None:
    result = compute_eer([0.0, 1.0], [2.0, 3.0])
    assert result.eer == 0.0
    assert result.threshold == 2.0


def test_eer_of_inverted_scores_is_one() -> None:
    assert compute_eer([2.0, 3.0], [0.0, 1.0]).eer == pytest.approx(1.0)


def test_eer_of_fully_tied_scores_is_one_half() -> None:
    # The official stable-sort step DET reports 1.0 here; grouping ties gives 0.5.
    assert compute_eer([0.5, 0.5, 0.5], [0.5, 0.5]).eer == pytest.approx(0.5)


def test_eer_hand_computed_exact_crossing_with_binary_exact_rates() -> None:
    result = compute_eer([1.0, 2.0, 3.0, 6.0], [4.0, 5.0, 7.0, 8.0])
    assert result.eer == pytest.approx(0.25)
    assert result.threshold == 5.0


def test_eer_finds_an_exact_crossing_despite_float_rounding() -> None:
    # At threshold 4, FNR = FPR = 1/3 exactly. In floats, 1 - 2/3 - 1/3 is 5.6e-17,
    # which would push a float comparison one point past the crossing (to 3.0).
    result = compute_eer([1.0, 2.0, 5.0], [3.0, 4.0, 6.0])
    assert result.eer == pytest.approx(1 / 3)
    assert result.threshold == 4.0


def test_eer_interpolation_weight_is_not_fixed() -> None:
    # Sorted 1b 2s 3b 4s 5s. FNR stays at 1/3 while FPR jumps 0 -> 1/2, so the
    # curves cross at 1/3 (weight 2/3). A fixed 0.5 weight would give 0.25 and
    # the step reference 5/12.
    result = compute_eer([1.0, 3.0], [2.0, 4.0, 5.0])
    assert result.eer == pytest.approx(1 / 3)
    assert result.threshold == 3.0


def test_eer_interpolates_across_a_miss_rate_step() -> None:
    # Sorted 1b 2s 3s 4b 5s. FPR stays at 0.5 while FNR drops 2/3 -> 1/3.
    result = compute_eer([1.0, 4.0], [2.0, 3.0, 5.0])
    assert result.eer == pytest.approx(0.5)
    assert result.threshold == 3.0


def test_eer_of_identical_distributions_is_near_one_half() -> None:
    rng = np.random.default_rng(20261004)
    result = compute_eer(rng.normal(size=4000), rng.normal(size=4000))
    assert result.eer == pytest.approx(0.5, abs=0.03)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_eer_matches_step_reference_within_a_quarter_step(seed: int) -> None:
    rng = np.random.default_rng(seed)
    bonafide = rng.normal(0.0, 1.0, size=257)
    spoof = rng.normal(1.2, 1.0, size=83)
    tolerance = max(1 / bonafide.size, 1 / spoof.size) / 4 + 1e-12
    ours = compute_eer(bonafide, spoof).eer
    assert abs(ours - _reference_step_eer(bonafide, spoof)) <= tolerance


def test_eer_rejects_empty_class_and_non_finite_scores() -> None:
    with pytest.raises(ValueError, match="at least one"):
        compute_eer([], [1.0])
    with pytest.raises(ValueError, match="finite"):
        compute_eer([np.nan], [1.0])


# --- threshold_at_target_fpr ------------------------------------------------------


def test_threshold_is_exactly_the_lowest_meeting_the_target() -> None:
    bonafide = np.arange(20, dtype=float)  # 0..19; 5% of 20 allows one false alarm
    result = _threshold(bonafide, 0.05, spoof=[18.5, 3.0])
    assert result.threshold == np.nextafter(18.0, np.inf)
    assert result.boundary_score == 18.0
    assert result.decision_rule == DECISION_RULE
    assert (result.n_bonafide, result.n_false_alarms) == (20, 1)
    assert result.achieved_fpr == pytest.approx(0.05)
    assert decide_spoof([18.5], result.threshold).all()  # a spoof just above is flagged


def test_threshold_ties_at_the_boundary_count_as_spoof() -> None:
    # Three clips tie at the top, but only one false alarm is allowed: all three are excluded.
    bonafide = [5.0, 5.0, 5.0] + [v / 10 for v in range(17)]  # the rest stay below 5
    result = _threshold(bonafide, 0.05)
    assert result.threshold > 5.0
    assert result.n_false_alarms == 0


def test_zero_target_flags_no_bona_fide_clip() -> None:
    result = _threshold([0.1, 0.7, 0.3], 0.0)
    assert result.threshold > 0.7
    assert result.n_false_alarms == 0


def test_allowed_false_alarms_survive_float_rounding() -> None:
    # 0.29 * 100 is 28.999999999999996 in float64; 29 false alarms are allowed.
    assert _threshold(np.arange(100, dtype=float), 0.29).n_false_alarms == 29


def test_threshold_holds_for_float32_model_scores() -> None:
    # Regression: a float64 threshold one step above a float32 boundary score
    # rounds back onto the boundary in a float32 comparison and flips that clip.
    scores = np.random.default_rng(7).normal(size=600).astype(np.float32)
    ids = [f"b{i}" for i in range(scores.size)]
    labels = ["bonafide"] * scores.size
    result = threshold_at_target_fpr(
        ids, labels, scores, target_fpr=0.05, score_direction=SCORE_DIRECTION
    )
    naive = int(np.count_nonzero(scores >= result.threshold))  # float32 comparison
    assert naive > result.n_false_alarms  # the hazard is real in this setup

    assert result.n_false_alarms == 30
    assert int(np.count_nonzero(decide_spoof(scores, result.threshold))) == 30
    metrics = evaluate_clip_scores(
        ids, labels, scores, threshold=result.threshold, score_direction=SCORE_DIRECTION
    )
    assert metrics.confusion is not None and metrics.confusion.fp == 30


def test_threshold_rejects_window_level_input() -> None:
    with pytest.raises(ValueError, match="one score per clip"):
        threshold_at_target_fpr(
            ["a", "a"],
            ["bonafide", "bonafide"],
            [0.1, 0.2],
            target_fpr=0.05,
            score_direction=SCORE_DIRECTION,
        )


def test_threshold_needs_bona_fide_clips() -> None:
    with pytest.raises(ValueError, match="bona fide"):
        _threshold([], 0.05, spoof=[0.4])


@pytest.mark.parametrize("target", [-0.01, 1.0, 1.5, True])
def test_threshold_rejects_out_of_range_target(target: float) -> None:
    with pytest.raises(ValueError, match="target_fpr"):
        _threshold([0.0, 1.0], target)


# --- evaluate_clip_scores ---------------------------------------------------------


def test_evaluate_hand_computed_metrics() -> None:
    metrics = _evaluate(_BONAFIDE, _SPOOF, threshold=5.0)

    assert metrics.metrics_version == METRICS_VERSION
    assert (metrics.n_clips, metrics.n_bonafide, metrics.n_spoof) == (8, 5, 3)
    assert metrics.spoof_prevalence == pytest.approx(3 / 8)
    assert metrics.score_direction == SCORE_DIRECTION
    assert metrics.confusion is not None
    assert (metrics.confusion.tn, metrics.confusion.fp) == (3, 2)
    assert (metrics.confusion.fn, metrics.confusion.tp) == (1, 2)
    assert metrics.human_false_positive_rate == pytest.approx(2 / 5)
    assert metrics.synthetic_detection_rate == pytest.approx(2 / 3)

    assert metrics.bonafide.support == 5
    assert metrics.bonafide.precision == pytest.approx(3 / 4)
    assert metrics.bonafide.recall == pytest.approx(3 / 5)
    assert metrics.bonafide.f1 == pytest.approx(2 / 3)
    assert metrics.spoof.support == 3
    assert metrics.spoof.precision == pytest.approx(1 / 2)
    assert metrics.spoof.recall == pytest.approx(2 / 3)
    assert metrics.spoof.f1 == pytest.approx(4 / 7)

    assert metrics.balanced_accuracy == pytest.approx(19 / 30)
    assert metrics.macro_f1 == pytest.approx(13 / 21)
    assert metrics.roc_auc == pytest.approx(2 / 3)  # 10 of 15 spoof/bona fide pairs ordered
    assert metrics.eer == pytest.approx(0.4)


def test_score_equal_to_threshold_counts_as_spoof() -> None:
    metrics = _evaluate(_BONAFIDE, _SPOOF, threshold=6.0)
    assert metrics.confusion is not None
    assert metrics.confusion.fp == 2  # bona fide 6.0 (exactly at the threshold) and 9.0


def test_precision_reflects_prevalence() -> None:
    # Same per-class rates as the hand fixture, with 4x more bona fide clips.
    metrics = _evaluate(_BONAFIDE * 4, _SPOOF, threshold=5.0)
    assert metrics.human_false_positive_rate == pytest.approx(2 / 5)
    assert metrics.synthetic_detection_rate == pytest.approx(2 / 3)
    assert metrics.spoof.precision == pytest.approx(2 / 10)
    assert metrics.bonafide.precision == pytest.approx(12 / 13)
    assert metrics.spoof_prevalence == pytest.approx(3 / 23)


def test_threshold_free_metrics_for_model_selection() -> None:
    metrics = _evaluate(_BONAFIDE, _SPOOF, threshold=None)
    assert metrics.eer == pytest.approx(0.4)
    assert metrics.roc_auc == pytest.approx(2 / 3)
    assert (metrics.n_bonafide, metrics.n_spoof) == (5, 3)
    assert metrics.threshold is None
    assert metrics.confusion is None
    for undefined in (
        metrics.balanced_accuracy,
        metrics.macro_f1,
        metrics.human_false_positive_rate,
        metrics.synthetic_detection_rate,
        metrics.spoof.precision,
        metrics.bonafide.recall,
    ):
        assert undefined is None


def test_spoof_only_slice_reports_counts_not_invented_numbers() -> None:
    metrics = _evaluate([], [4.0, 5.0, 7.0], threshold=5.0)

    assert (metrics.n_bonafide, metrics.n_spoof) == (0, 3)
    assert metrics.synthetic_detection_rate == pytest.approx(2 / 3)
    for undefined in (
        metrics.eer,
        metrics.roc_auc,
        metrics.balanced_accuracy,
        metrics.macro_f1,
        metrics.human_false_positive_rate,
        metrics.spoof.precision,
        metrics.spoof.f1,
    ):
        assert undefined is None


def test_bona_fide_only_slice_reports_counts_not_invented_numbers() -> None:
    metrics = _evaluate([1.0, 6.0, 9.0], [], threshold=5.0)

    assert (metrics.n_bonafide, metrics.n_spoof) == (3, 0)
    assert metrics.human_false_positive_rate == pytest.approx(2 / 3)
    assert metrics.bonafide.recall == pytest.approx(1 / 3)
    for undefined in (
        metrics.eer,
        metrics.roc_auc,
        metrics.balanced_accuracy,
        metrics.macro_f1,
        metrics.synthetic_detection_rate,
        metrics.bonafide.precision,
        metrics.bonafide.f1,
    ):
        assert undefined is None


def test_duplicate_clip_ids_are_rejected_as_window_level_input() -> None:
    with pytest.raises(ValueError, match="one score per clip"):
        evaluate_clip_scores(
            ["a", "a"],
            ["spoof", "spoof"],
            [0.1, 0.2],
            threshold=0.0,
            score_direction=SCORE_DIRECTION,
        )


def test_other_score_directions_are_rejected() -> None:
    with pytest.raises(ValueError, match="higher_more_spoof"):
        evaluate_clip_scores(
            *_clips(_BONAFIDE, _SPOOF), threshold=None, score_direction="higher_more_bonafide"
        )


@pytest.mark.parametrize(
    ("clip_ids", "labels", "scores", "message"),
    [
        (["a"], ["fake"], [0.1], "labels must be one of"),
        (["a", "b"], ["spoof"], [0.1, 0.2], "equal length"),
        ([], [], [], "no clips"),
        (["a"], ["spoof"], [np.inf], "finite"),
        (["a"], ["spoof"], ["0.5"], "numeric"),
        (["a"], ["spoof"], [True], "numeric"),
    ],
)
def test_invalid_inputs_are_rejected(
    clip_ids: list[str], labels: list[str], scores: list, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_clip_scores(
            clip_ids, labels, scores, threshold=0.0, score_direction=SCORE_DIRECTION
        )


@pytest.mark.parametrize("threshold", [True, "0.5", np.float32(0.5), np.nan])
def test_invalid_thresholds_are_rejected(threshold: object) -> None:
    with pytest.raises(ValueError, match="threshold"):
        _evaluate(_BONAFIDE, _SPOOF, threshold=threshold)  # type: ignore[arg-type]


def test_metrics_serialize_to_json_for_run_records() -> None:
    record = json.loads(json.dumps(_evaluate(_BONAFIDE, _SPOOF, threshold=5.0).as_dict()))
    assert record["metrics_version"] == METRICS_VERSION
    assert record["decision_rule"] == DECISION_RULE
    assert record["confusion"] == {"tn": 3, "fp": 2, "fn": 1, "tp": 2}
    assert record["spoof"]["support"] == 3
    assert "eer_threshold" not in record
