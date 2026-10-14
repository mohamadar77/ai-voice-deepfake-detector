"""The pre-tuning protocol record: what is declared before any score is seen.

Model selection is only meaningful if the rule was fixed *before* the numbers
existed. This module loads and validates that declaration from versioned YAML
(``configs/pre_tuning_protocol.yaml``), so the LR baseline (#8) reads its
selection metric, tie-break, candidate budget, window-to-clip aggregation and
threshold rule from a committed record instead of a notebook default or a CLI
flag. The architecture plan is explicit that a configuration flag alone is not
enforcement: the loader therefore *rejects* a record that disagrees with the
frozen code conventions it claims to follow. Every such convention is imported
from the module that implements it, never restated here, so changing the
implementation breaks the committed record instead of silently diverging from
it: :data:`voice_detector.metrics.METRICS_VERSION`,
:data:`~voice_detector.metrics.SELECTION_METRIC`,
:data:`~voice_detector.metrics.SELECTION_DIRECTION`,
:data:`~voice_detector.metrics.THRESHOLD_RULE`,
:data:`voice_detector.scoring.SCORE_DIRECTION`,
:data:`voice_detector.scoring.DECISION_RULE` and the reservation's role names.
The selection metric is additionally checked to name a field
:class:`~voice_detector.metrics.ClipMetrics` actually reports, and the tie rule
is derived by asking :func:`~voice_detector.scoring.decide_spoof` what it does
at the threshold rather than by restating its prose.

Scope: the selection, aggregation and threshold rules are the project's general
development protocol; ``lr_budget`` is the Logistic Regression baseline's own
tuning budget. The CNN declares its own score convention and budget in its own
record (#33), because the two families' raw scores are not comparable.

This module reads no data and fits nothing.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from voice_detector.metrics import (
    METRICS_VERSION,
    POSITIVE_LABEL,
    SELECTION_DIRECTION,
    SELECTION_METRIC,
    SELECTION_METRIC_FIELD,
    THRESHOLD_RULE,
    ClipMetrics,
)
from voice_detector.reservation import ROLE_SELECT, ROLE_THRESHOLD
from voice_detector.scoring import DECISION_RULE, SCORE_DIRECTION, decide_spoof

PRE_TUNING_SCHEMA_VERSION = "pre-tuning-v1"

# Recognised tie-break criteria, in the order the record may list them.
TIE_BREAK_CRITERIA = ("simpler_system", "lower_measured_latency")

# Phase 1 keeps a deliberately small development tuning budget (the roadmap's
# "three LR settings and one CNN architecture"), so the record may not declare a
# larger one: an unbounded candidate list is a hyperparameter search, which is
# Phase 2 work and would make the select/threshold split carry more selection
# than it was reserved for.
MAX_DECLARED_CANDIDATES = 5

# The baseline this budget belongs to, and the class weightings it may declare.
LR_MODELS = ("logistic_regression",)
LR_CLASS_WEIGHTS = ("balanced", "none")


@dataclass(frozen=True)
class SelectionRule:
    """How a candidate is chosen on the ``select`` role."""

    development_role: str
    metric: str
    direction: str
    eer_tolerance_absolute: float
    tie_break_order: tuple[str, ...]


@dataclass(frozen=True)
class LrBudget:
    """The Logistic Regression baseline's declared tuning budget."""

    model: str
    regularization_parameter: str
    values: tuple[float, ...]
    class_weight: str
    max_candidates: int


@dataclass(frozen=True)
class AggregationRule:
    """How window scores become one score per original clip."""

    window_to_clip: str
    aggregate_by: str


@dataclass(frozen=True)
class ThresholdRule:
    """How the operating threshold is set on the reserved role."""

    development_role: str
    rule: str
    target_fpr: float
    ties_count_as: str


@dataclass(frozen=True)
class PreTuningProtocol:
    """The complete pre-tuning declaration, validated against frozen code."""

    version: str
    declared_on: str
    metrics_version: str
    score_direction: str
    decision_rule: str
    selection: SelectionRule
    lr_budget: LrBudget
    aggregation: AggregationRule
    threshold: ThresholdRule

    def as_dict(self) -> dict[str, Any]:
        """Plain, JSON-serializable form for run records and reports."""

        return asdict(self)


def _require_mapping(raw: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"pre-tuning record needs a {name!r} mapping")
    return raw


def _require_str(raw: Mapping[str, Any], key: str, name: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name}.{key} must be a non-empty string, got {value!r}")
    return value


def _require_number(raw: Mapping[str, Any], key: str, name: str) -> float:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name}.{key} must be a number, got {value!r}")
    return float(value)


def _load_selection(raw: Mapping[str, Any]) -> SelectionRule:
    section = _require_mapping(raw.get("selection"), "selection")
    role = _require_str(section, "development_role", "selection")
    if role != ROLE_SELECT:
        raise ValueError(
            f"selection.development_role must be {ROLE_SELECT!r} (the reserved role "
            f"for model choices), got {role!r}"
        )
    metric = _require_str(section, "metric", "selection")
    if metric != SELECTION_METRIC:
        raise ValueError(f"selection.metric must be {SELECTION_METRIC!r}, got {metric!r}")
    # The declared metric must still be one the metrics module reports, so
    # renaming the field cannot leave the record naming a metric that is gone.
    reported = {field.name for field in fields(ClipMetrics)}
    if SELECTION_METRIC_FIELD not in reported:
        raise ValueError(
            f"selection.metric {metric!r} names {SELECTION_METRIC_FIELD!r}, which "
            f"{ClipMetrics.__name__} no longer reports; the metrics API changed under "
            "this record"
        )
    direction = _require_str(section, "direction", "selection")
    if direction != SELECTION_DIRECTION:
        raise ValueError(
            f"selection.direction must be {SELECTION_DIRECTION!r}, got {direction!r}"
        )

    tie_break = _require_mapping(section.get("tie_break"), "selection.tie_break")
    tolerance = _require_number(tie_break, "eer_tolerance_absolute", "selection.tie_break")
    if not 0.0 <= tolerance < 1.0:
        raise ValueError(
            f"selection.tie_break.eer_tolerance_absolute must be in [0, 1), got {tolerance}"
        )
    order = tie_break.get("order")
    if not isinstance(order, list) or not order:
        raise ValueError("selection.tie_break.order must be a non-empty list")
    unknown = [item for item in order if item not in TIE_BREAK_CRITERIA]
    if unknown:
        raise ValueError(
            f"selection.tie_break.order may only contain {list(TIE_BREAK_CRITERIA)}, "
            f"got {unknown}"
        )
    if len(set(order)) != len(order):
        raise ValueError("selection.tie_break.order must not repeat a criterion")

    return SelectionRule(
        development_role=role,
        metric=metric,
        direction=direction,
        eer_tolerance_absolute=tolerance,
        tie_break_order=tuple(order),
    )


def _load_lr_budget(raw: Mapping[str, Any]) -> LrBudget:
    section = _require_mapping(raw.get("lr_budget"), "lr_budget")
    model = _require_str(section, "model", "lr_budget")
    if model not in LR_MODELS:
        raise ValueError(f"lr_budget.model must be one of {list(LR_MODELS)}, got {model!r}")
    class_weight = _require_str(section, "class_weight", "lr_budget")
    if class_weight not in LR_CLASS_WEIGHTS:
        raise ValueError(
            f"lr_budget.class_weight must be one of {list(LR_CLASS_WEIGHTS)}, "
            f"got {class_weight!r}"
        )

    max_candidates = section.get("max_candidates")
    if (
        isinstance(max_candidates, bool)
        or not isinstance(max_candidates, int)
        or not 1 <= max_candidates <= MAX_DECLARED_CANDIDATES
    ):
        raise ValueError(
            f"lr_budget.max_candidates must be an integer in [1, {MAX_DECLARED_CANDIDATES}], "
            f"got {max_candidates!r}"
        )

    values = section.get("values")
    if not isinstance(values, list) or not values:
        raise ValueError("lr_budget.values must be a non-empty list")
    if any(isinstance(value, bool) or not isinstance(value, int | float) for value in values):
        raise ValueError(f"lr_budget.values must all be numbers, got {values!r}")
    numbers = tuple(float(value) for value in values)
    if any(value <= 0.0 for value in numbers):
        raise ValueError(f"lr_budget.values must all be positive, got {list(numbers)}")
    if len(set(numbers)) != len(numbers):
        raise ValueError(f"lr_budget.values must be distinct, got {list(numbers)}")
    if len(numbers) > max_candidates:
        raise ValueError(
            f"lr_budget declares {len(numbers)} candidate value(s) against a budget of "
            f"{max_candidates}; widen the declared budget deliberately or drop values"
        )
    return LrBudget(
        model=model,
        regularization_parameter=_require_str(
            section, "regularization_parameter", "lr_budget"
        ),
        values=numbers,
        class_weight=class_weight,
        max_candidates=max_candidates,
    )


def _load_aggregation(raw: Mapping[str, Any]) -> AggregationRule:
    section = _require_mapping(raw.get("aggregation"), "aggregation")
    aggregate_by = _require_str(section, "aggregate_by", "aggregation")
    if aggregate_by != "clip_id":
        raise ValueError(
            f"aggregation.aggregate_by must be 'clip_id' (one score per original clip), "
            f"got {aggregate_by!r}"
        )
    return AggregationRule(
        window_to_clip=_require_str(section, "window_to_clip", "aggregation"),
        aggregate_by=aggregate_by,
    )


def _tie_decision_label() -> str:
    """Which class the frozen decision rule gives a score exactly at the threshold.

    Asked of the implementation rather than restated from its prose:
    ``decide_spoof`` is the single operating decision in the project, so if its
    comparison ever stopped counting a boundary score as the positive class,
    the committed record would stop loading instead of quietly describing
    behaviour the code no longer has.
    """

    tie_is_positive = bool(decide_spoof([0.0], 0.0)[0])
    return POSITIVE_LABEL if tie_is_positive else "bonafide"


def _load_threshold(raw: Mapping[str, Any]) -> ThresholdRule:
    section = _require_mapping(raw.get("threshold"), "threshold")
    role = _require_str(section, "development_role", "threshold")
    if role != ROLE_THRESHOLD:
        raise ValueError(
            f"threshold.development_role must be {ROLE_THRESHOLD!r} (the role reserved "
            f"from model choices), got {role!r}"
        )
    rule = _require_str(section, "rule", "threshold")
    if rule != THRESHOLD_RULE:
        raise ValueError(f"threshold.rule must be {THRESHOLD_RULE!r}, got {rule!r}")
    target_fpr = _require_number(section, "target_fpr", "threshold")
    if not 0.0 <= target_fpr < 1.0:
        raise ValueError(f"threshold.target_fpr must be in [0, 1), got {target_fpr}")
    ties = _require_str(section, "ties_count_as", "threshold")
    expected_ties = _tie_decision_label()
    if ties != expected_ties:
        raise ValueError(
            f"threshold.ties_count_as must be {expected_ties!r} to match {DECISION_RULE!r}, "
            f"got {ties!r}"
        )
    return ThresholdRule(
        development_role=role, rule=rule, target_fpr=target_fpr, ties_count_as=ties
    )


def _require_declared_on(raw: Mapping[str, Any]) -> str:
    """The declaration date, which is the record's whole claim.

    ``declared_on`` is what makes this a *pre*-tuning record, so it is checked
    rather than merely stored: an ISO date, and not one in the future. It cannot
    prove the declaration preceded the scores -- the committing Git history does
    that -- but it can refuse to carry a date that was never a date.

    One day of slack, because the date is a local calendar day while the check
    runs in UTC: a record written east of UTC is legitimately "tomorrow" for a
    few hours. Anything beyond that is not a timezone.
    """

    value = _require_str(raw, "declared_on", "pre-tuning record")
    try:
        declared = date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(
            f"pre-tuning record declared_on must be an ISO date (YYYY-MM-DD), got {value!r}"
        ) from error
    today = datetime.now(UTC).date()
    if declared > today + timedelta(days=1):
        raise ValueError(
            f"pre-tuning record declared_on {declared.isoformat()} is in the future "
            f"(today is {today.isoformat()} UTC); a declaration cannot predate itself"
        )
    return value


def load_pre_tuning_protocol(path: Path | str) -> PreTuningProtocol:
    """Load and validate the committed pre-tuning protocol record.

    Every declared convention is checked against the code that implements it,
    so the record cannot drift from the metrics version, score direction,
    selection metric, threshold rule or decision rule it claims. Raises
    ``ValueError`` on any mismatch.
    """

    raw = _require_mapping(
        yaml.safe_load(Path(path).read_text(encoding="utf-8")), "pre-tuning record"
    )

    version = _require_str(raw, "version", "pre-tuning record")
    if version != PRE_TUNING_SCHEMA_VERSION:
        raise ValueError(
            f"pre-tuning record version must be {PRE_TUNING_SCHEMA_VERSION!r}, got {version!r}"
        )
    metrics_version = _require_str(raw, "metrics_version", "pre-tuning record")
    if metrics_version != METRICS_VERSION:
        raise ValueError(
            f"pre-tuning record declares metrics_version {metrics_version!r} but the code "
            f"computes {METRICS_VERSION!r}; numbers from different definitions are not "
            "comparable"
        )
    score_direction = _require_str(raw, "score_direction", "pre-tuning record")
    if score_direction != SCORE_DIRECTION:
        raise ValueError(
            f"pre-tuning record declares score_direction {score_direction!r}, code uses "
            f"{SCORE_DIRECTION!r}"
        )
    decision_rule = _require_str(raw, "decision_rule", "pre-tuning record")
    if decision_rule != DECISION_RULE:
        raise ValueError(
            f"pre-tuning record declares decision_rule {decision_rule!r}, code uses "
            f"{DECISION_RULE!r}"
        )

    return PreTuningProtocol(
        version=version,
        declared_on=_require_declared_on(raw),
        metrics_version=metrics_version,
        score_direction=score_direction,
        decision_rule=decision_rule,
        selection=_load_selection(raw),
        lr_budget=_load_lr_budget(raw),
        aggregation=_load_aggregation(raw),
        threshold=_load_threshold(raw),
    )
