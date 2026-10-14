"""Dataset-free tests for the pre-tuning protocol record (#31).

The committed ``configs/pre_tuning_protocol.yaml`` is loaded as written, so a
silent change to the declared selection rule, candidate budget, aggregation or
threshold rule breaks a test. The mutation tests prove the loader is real
enforcement rather than documentation: a record that merely *claims* a metrics
version or decision rule is rejected when the code disagrees.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from voice_detector import pretuning as pt
from voice_detector.metrics import (
    METRICS_VERSION,
    SELECTION_DIRECTION,
    SELECTION_METRIC,
    THRESHOLD_RULE,
)
from voice_detector.reservation import ROLE_SELECT, ROLE_THRESHOLD
from voice_detector.scoring import DECISION_RULE, SCORE_DIRECTION

CONFIG = Path("configs/pre_tuning_protocol.yaml")


def test_committed_record_declares_the_frozen_protocol() -> None:
    protocol = pt.load_pre_tuning_protocol(CONFIG)

    assert protocol.version == pt.PRE_TUNING_SCHEMA_VERSION
    assert protocol.metrics_version == METRICS_VERSION
    assert protocol.score_direction == SCORE_DIRECTION
    assert protocol.decision_rule == DECISION_RULE

    # Asserted against the metrics module's own constants, not the loader's, so
    # these cannot drift apart without a test failing.
    assert protocol.selection.development_role == ROLE_SELECT
    assert protocol.selection.metric == SELECTION_METRIC
    assert protocol.selection.direction == SELECTION_DIRECTION
    assert protocol.selection.eer_tolerance_absolute == 0.005
    assert protocol.selection.tie_break_order == ("simpler_system", "lower_measured_latency")

    # The declared budget is three regularization values and one fixed
    # class-imbalance policy; extensive optimization is Phase 2.
    assert len(protocol.lr_budget.values) == 3
    assert protocol.lr_budget.max_candidates == 3
    assert len(protocol.lr_budget.values) <= protocol.lr_budget.max_candidates
    assert protocol.lr_budget.max_candidates <= pt.MAX_DECLARED_CANDIDATES
    assert protocol.lr_budget.model in pt.LR_MODELS
    assert protocol.lr_budget.regularization_parameter == "C"
    assert protocol.lr_budget.class_weight == "balanced"

    assert protocol.aggregation.aggregate_by == "clip_id"
    assert protocol.aggregation.window_to_clip == "mean_window_decision_function"

    assert protocol.threshold.development_role == ROLE_THRESHOLD
    assert protocol.threshold.rule == THRESHOLD_RULE
    assert protocol.threshold.target_fpr == 0.05
    assert protocol.threshold.ties_count_as == "spoof"


def test_the_tie_rule_is_read_off_the_decision_function() -> None:
    """The declared tie label must be what ``decide_spoof`` actually does."""

    assert pt._tie_decision_label() == "spoof"
    assert pt.load_pre_tuning_protocol(CONFIG).threshold.ties_count_as == "spoof"


def test_record_round_trips_to_a_plain_dict() -> None:
    payload = pt.load_pre_tuning_protocol(CONFIG).as_dict()
    assert payload["threshold"]["target_fpr"] == 0.05
    assert payload["lr_budget"]["values"] == (0.01, 1.0, 100.0)
    assert payload["lr_budget"]["max_candidates"] == 3


@pytest.mark.parametrize(
    ("replace", "with_", "match"),
    [
        ("version: pre-tuning-v1", "version: pre-tuning-v9", "version"),
        ("metrics_version: metrics-v1", "metrics_version: metrics-v0", "metrics_version"),
        (
            "score_direction: higher_more_spoof",
            "score_direction: higher_more_human",
            "score_direction",
        ),
        (
            'decision_rule: "spoof if score >= threshold, compared in float64"',
            'decision_rule: "spoof if score > threshold"',
            "decision_rule",
        ),
        ("  development_role: select", "  development_role: threshold", "selection"),
        ("  metric: clip_eer", "  metric: accuracy", "selection.metric"),
        ("  direction: lower_is_better", "  direction: higher_is_better", "selection.direction"),
        (
            "    eer_tolerance_absolute: 0.005",
            "    eer_tolerance_absolute: 1.5",
            "eer_tolerance_absolute",
        ),
        (
            "    order: [simpler_system, lower_measured_latency]",
            "    order: [best_test_eer]",
            "tie_break.order",
        ),
        (
            "    order: [simpler_system, lower_measured_latency]",
            "    order: [simpler_system, simpler_system]",
            "repeat",
        ),
        ("  values: [0.01, 1.0, 100.0]", "  values: [1.0, 1.0, 100.0]", "distinct"),
        ("  values: [0.01, 1.0, 100.0]", "  values: [0.0, 1.0, 100.0]", "positive"),
        ("  values: [0.01, 1.0, 100.0]", "  values: []", "non-empty"),
        # The budget is a ceiling, not a label: more candidates than declared,
        # and a declared ceiling above the project's own, are both refused.
        (
            "  values: [0.01, 1.0, 100.0]",
            "  values: [0.001, 0.01, 1.0, 100.0]",
            "budget of 3",
        ),
        ("  max_candidates: 3", "  max_candidates: 200", "max_candidates"),
        ("  max_candidates: 3", "  max_candidates: 0", "max_candidates"),
        (
            "  model: logistic_regression",
            "  model: gradient_boosting_ensemble",
            "lr_budget.model",
        ),
        ("  class_weight: balanced", "  class_weight: tuned_per_candidate", "class_weight"),
        # declared_on carries the record's entire "declared before the scores"
        # claim, so it may not be prose and may not be in the future.
        (
            'declared_on: "2026-10-14"',
            'declared_on: "whenever, after I saw the scores"',
            "ISO date",
        ),
        ('declared_on: "2026-10-14"', 'declared_on: "2099-01-01"', "future"),
        ("  aggregate_by: clip_id", "  aggregate_by: window_id", "aggregate_by"),
        ("  development_role: threshold", "  development_role: select", "threshold"),
        (
            "  rule: lowest_threshold_with_bonafide_fpr_at_most_target",
            "  rule: best_eer_threshold",
            "threshold.rule",
        ),
        ("  target_fpr: 0.05", "  target_fpr: 1.0", "target_fpr"),
        ("  ties_count_as: spoof", "  ties_count_as: bonafide", "ties_count_as"),
    ],
)
def test_a_record_disagreeing_with_the_code_is_rejected(
    tmp_path: Path, replace: str, with_: str, match: str
) -> None:
    text = CONFIG.read_text(encoding="utf-8")
    assert replace in text, f"fixture drifted from the committed record: {replace!r}"
    path = tmp_path / "record.yaml"
    path.write_text(text.replace(replace, with_, 1), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        pt.load_pre_tuning_protocol(path)


def test_a_missing_section_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "record.yaml"
    path.write_text("version: pre-tuning-v1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="declared_on|metrics_version"):
        pt.load_pre_tuning_protocol(path)


def test_a_non_mapping_record_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "record.yaml"
    path.write_text("- not a mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="pre-tuning record"):
        pt.load_pre_tuning_protocol(path)
