"""Scoring orchestration for the locked evaluation protocol (#35).

Torch-free at import: CNN bundles are loaded lazily by ``load_scorers``. This
module never sees a label while scoring; labels are copied into prediction rows
only after a clip's scores exist.

Loop order is condition, then clip, then candidate. The conditioned waveform is
built once per (clip, condition) and shared by every candidate, and each
condition is committed to the prediction store in one transaction, so an
interrupted run leaves only whole conditions behind.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from voice_detector.evaluation import FreezeRecord, hash_directory
from voice_detector.metrics import METRICS_VERSION
from voice_detector.prediction_store import (
    PredictionRow,
    completed_pairs,
    compute_predictions_hash,
    create_prediction_store,
    get_metadata,
    set_metadata_once,
    validate_prediction_completeness,
    write_predictions,
)
from voice_detector.scoring import (
    CNN_AGGREGATION_RULE,
    DECISION_RULE,
    LR_AGGREGATION_RULE,
    SCORE_DIRECTION,
    WaveformScore,
    WaveformScorer,
    decide_spoof,
)

PARTITIONS = ("development", "evaluation")
_LABELS = ("bonafide", "spoof")


class UnsupportedModelFamily(ValueError):
    """A candidate's model family or run record schema has no registered scorer."""


class ParityFailedError(ValueError):
    """Development parity with the bundle's recorded metrics failed."""


class ScoringPreflightError(ValueError):
    """The scoring inputs are incomplete or inconsistent; nothing was written."""


class StoreStateError(ValueError):
    """The prediction store is not in a state this run may continue from.

    ``reason`` is a short machine-readable code for the CLI's JSON refusal.
    """

    def __init__(self, message: str, reason: str = "store_state") -> None:
        super().__init__(message)
        self.reason = reason


# ---------------------------------------------------------------------------
# Bundle descriptors
# ---------------------------------------------------------------------------

# schema_version -> (model_family, aggregation rule, run record file name)
_REGISTRY: dict[str, tuple[str, str, str]] = {
    "lr-baseline-v1": ("logistic_regression", LR_AGGREGATION_RULE, "lr_run_record.json"),
    "cnn-study-v1": ("cnn", CNN_AGGREGATION_RULE, "cnn_run_record.json"),
}


@dataclass(frozen=True)
class BundleDescriptor:
    """What a bundle declares about itself, read from its run record."""

    candidate_id: str
    model_family: str
    aggregation_rule: str
    schema_version: str
    seed: int
    run_label: str
    threshold: float
    run_record: dict[str, Any] = field(repr=False, compare=False)


def _declared_rule(schema_version: str, record: Mapping[str, Any]) -> Any:
    if schema_version == "lr-baseline-v1":
        return record.get("aggregation", {}).get("window_to_clip")
    return record.get("input_policy", {}).get("eval_aggregation")


def describe_bundle(bundle_dir: Path) -> BundleDescriptor:
    """Read and validate a bundle's declaration from its run record (JSON only).

    The family and aggregation rule come from record content, never from the
    filename alone, and are cross-checked against the rule the registered
    scorer implements.
    """

    bundle_dir = Path(bundle_dir)
    found = sorted(p for p in bundle_dir.glob("*run_record.json") if p.is_file())
    if len(found) != 1:
        raise ScoringPreflightError(
            f"bundle {bundle_dir.name!r} must contain exactly one *run_record.json, "
            f"found {[p.name for p in found]}"
        )
    record_path = found[0]
    record = json.loads(record_path.read_text(encoding="utf-8"))

    schema_version = record.get("schema_version")
    if schema_version not in _REGISTRY:
        raise UnsupportedModelFamily(
            f"bundle {bundle_dir.name!r} has unknown run record schema {schema_version!r}; "
            f"supported: {sorted(_REGISTRY)}"
        )
    family, rule, record_name = _REGISTRY[schema_version]
    if record_path.name != record_name:
        raise ScoringPreflightError(
            f"bundle {bundle_dir.name!r}: run record is {record_path.name!r} but "
            f"schema {schema_version!r} implies {record_name!r}"
        )
    declared = _declared_rule(schema_version, record)
    if declared != rule:
        raise ScoringPreflightError(
            f"bundle {bundle_dir.name!r} declares aggregation {declared!r}; "
            f"the {family} scorer implements {rule!r}"
        )
    if record.get("metrics_version") != METRICS_VERSION:
        raise ScoringPreflightError(
            f"bundle {bundle_dir.name!r} metrics_version {record.get('metrics_version')!r} "
            f"is not {METRICS_VERSION!r}"
        )
    if record.get("score_direction") != SCORE_DIRECTION:
        raise ScoringPreflightError(
            f"bundle {bundle_dir.name!r} score_direction "
            f"{record.get('score_direction')!r} is not {SCORE_DIRECTION!r}"
        )
    threshold_block = record.get("threshold") or {}
    threshold = threshold_block.get("value")
    if isinstance(threshold, bool) or not isinstance(threshold, int | float):
        raise ScoringPreflightError(f"bundle {bundle_dir.name!r} has no threshold.value")
    if not math.isfinite(threshold):
        raise ScoringPreflightError(f"bundle {bundle_dir.name!r} threshold is not finite")
    if threshold_block.get("decision_rule") != DECISION_RULE:
        raise ScoringPreflightError(
            f"bundle {bundle_dir.name!r} threshold.decision_rule differs from {DECISION_RULE!r}"
        )
    seed = record.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ScoringPreflightError(f"bundle {bundle_dir.name!r} has no integer seed")

    return BundleDescriptor(
        candidate_id=bundle_dir.name,
        model_family=family,
        aggregation_rule=rule,
        schema_version=schema_version,
        seed=seed,
        run_label=str(record.get("run_label", bundle_dir.name)),
        threshold=float(threshold),
        run_record=record,
    )


def check_mel_matches_study(descriptor: BundleDescriptor, study_config: Mapping[str, Any]) -> None:
    """A CNN bundle's mel settings must equal the (frozen) study config."""

    if descriptor.model_family != "cnn":
        return
    recorded = descriptor.run_record.get("mel_config", {})
    declared = study_config.get("mel", {})
    keys = ("sample_rate_hz", "n_fft", "hop_length", "n_mels", "fmin", "fmax", "window_seconds")
    for key in keys:
        if key in declared and recorded.get(key) != declared[key]:
            raise ScoringPreflightError(
                f"bundle {descriptor.candidate_id!r} mel_config.{key}={recorded.get(key)!r} "
                f"differs from cnn_study.yaml ({declared[key]!r})"
            )


# ---------------------------------------------------------------------------
# Robustness conditions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RobustnessSpec:
    version: str
    global_seed: int
    conditions: dict[str, dict[str, Any]]


def load_robustness_spec(path: Path, *, required: Sequence[str]) -> RobustnessSpec:
    """Load ``robustness_spec.yaml`` and refuse anything the code cannot honour.

    There are no fallbacks: a missing file, version, seed, condition or
    parameter is an error, never a silent clean-audio condition.
    """

    import yaml

    from voice_detector.augmentation import AUGMENTATION_VERSION

    path = Path(path)
    if not path.is_file():
        raise ScoringPreflightError(f"robustness spec not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if raw.get("version") != AUGMENTATION_VERSION:
        raise ScoringPreflightError(
            f"robustness spec version {raw.get('version')!r} is not {AUGMENTATION_VERSION!r}"
        )
    seed = raw.get("global_seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ScoringPreflightError("robustness spec needs an integer global_seed")
    conditions = dict(raw.get("conditions") or {})
    if "silence_trim" in required:
        conditions["silence_trim"] = _silence_trim_condition(raw.get("silence_trim"))

    for name in required:
        spec = conditions.get(name)
        if not isinstance(spec, dict):
            raise ScoringPreflightError(f"condition {name!r} is missing from the robustness spec")
        transform = spec.get("transform")
        if name == "clean":
            if transform is not None:
                raise ScoringPreflightError("condition 'clean' must not declare a transform")
        elif transform == "codec":
            _require_values(name, spec, {"codec": "libopus", "container": "ogg"})
            _require_int(name, spec, "bitrate_kbps")
            _require_int(name, spec, "sample_rate_hz")
        elif transform == "silence_trim":
            pass  # validated in _silence_trim_condition
        elif transform == "noise":
            _require_values(
                name,
                spec,
                {"noise_type": "awgn", "reference": "whole_clip_rms", "clip_range": [-1.0, 1.0]},
            )
            if isinstance(spec.get("snr_db"), bool) or not isinstance(
                spec.get("snr_db"), int | float
            ):
                raise ScoringPreflightError(f"condition {name!r} needs a numeric snr_db")
        else:
            raise ScoringPreflightError(
                f"condition {name!r} has unsupported transform {transform!r}"
            )
    return RobustnessSpec(
        version=raw["version"],
        global_seed=seed,
        conditions={name: dict(conditions[name]) for name in required},
    )


def _silence_trim_condition(section: Any) -> dict[str, Any]:
    """The frozen silence-trim diagnostic, declared under ``silence_trim:`` in the spec."""

    if not isinstance(section, dict):
        raise ScoringPreflightError("robustness spec has no silence_trim section")
    for key in ("top_db", "min_duration_seconds"):
        value = section.get(key)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ScoringPreflightError(f"silence_trim needs a numeric {key}")
    if section.get("fallback") != "keep_original_with_flag":
        raise ScoringPreflightError(
            "silence_trim.fallback must be keep_original_with_flag, the implemented behaviour"
        )
    return {
        "transform": "silence_trim",
        "top_db": float(section["top_db"]),
        "min_duration_seconds": float(section["min_duration_seconds"]),
    }


def _require_values(name: str, spec: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    for key, value in expected.items():
        if spec.get(key) != value:
            raise ScoringPreflightError(
                f"condition {name!r}: {key}={spec.get(key)!r}, "
                f"the implementation requires {value!r}"
            )


def _require_int(name: str, spec: Mapping[str, Any], key: str) -> None:
    value = spec.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ScoringPreflightError(f"condition {name!r} needs an integer {key}")


@dataclass(frozen=True)
class ConditionedWaveform:
    waveform: np.ndarray
    n_clipped: int = 0
    samples_delta: int = 0
    kept_original: bool = False


def apply_condition(
    waveform: np.ndarray,
    *,
    clip_id: str,
    condition: str,
    spec: RobustnessSpec,
    codec_fn: Callable[..., Any] | None = None,
    noise_fn: Callable[..., Any] | None = None,
    trim_fn: Callable[..., Any] | None = None,
    sample_rate_hz: int | None = None,
) -> ConditionedWaveform:
    """Apply one frozen condition to a whole decoded clip, before any windowing.

    Label-blind by signature. ``clean`` returns the input array unchanged.
    The noise seed derives from ``global_seed`` and ``clip_id`` only, so
    ``noise_20db`` and ``noise_10db`` share the same unit noise at different scales.
    """

    cond = spec.conditions.get(condition)
    if cond is None:
        raise ScoringPreflightError(f"unknown condition {condition!r}")
    transform = cond.get("transform")
    if transform is None:
        return ConditionedWaveform(waveform)
    if transform == "codec":
        if codec_fn is None:
            from voice_detector.augmentation import codec_transform as codec_fn
        codec = codec_fn(
            waveform,
            sample_rate=int(cond["sample_rate_hz"]),
            bitrate_kbps=int(cond["bitrate_kbps"]),
        )
        return ConditionedWaveform(codec.waveform, samples_delta=int(codec.samples_delta))
    if transform == "noise":
        if noise_fn is None:
            from voice_detector.augmentation import noise_transform as noise_fn
        noise = noise_fn(
            waveform,
            snr_db=float(cond["snr_db"]),
            clip_id=clip_id,
            global_seed=spec.global_seed,
        )
        return ConditionedWaveform(noise.waveform, n_clipped=int(noise.n_clipped))
    if transform == "silence_trim":
        if sample_rate_hz is None:
            raise ScoringPreflightError("silence_trim needs the decode sample rate")
        if trim_fn is None:
            from voice_detector.augmentation import silence_trim as trim_fn
        trim = trim_fn(
            waveform,
            sample_rate=int(sample_rate_hz),
            top_db=float(cond["top_db"]),
            min_duration_seconds=float(cond["min_duration_seconds"]),
        )
        return ConditionedWaveform(trim.waveform, kept_original=bool(trim.kept_original))
    raise ScoringPreflightError(f"condition {condition!r} has unsupported transform {transform!r}")


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------


class LrClipScorer:
    """MFCC + Logistic Regression scorer (the path eval-score always had)."""

    model_family = "logistic_regression"
    aggregation_rule = LR_AGGREGATION_RULE

    def __init__(self, pipeline: Any, pre_config: Any, threshold: float) -> None:
        self._pipeline = pipeline
        self._pre_config = pre_config
        self.threshold = float(threshold)
        self.sample_rate_hz = int(pre_config.audio.sample_rate_hz)

    def score_waveform(self, waveform: np.ndarray) -> WaveformScore:
        from voice_detector.preprocessing import clip_feature_matrix

        features = clip_feature_matrix(waveform, self._pre_config)
        window_scores = self._pipeline.decision_function(features)
        return WaveformScore(
            clip_score=float(window_scores.mean()), n_windows=int(len(window_scores))
        )


def load_scorers(
    record: FreezeRecord,
    bundle_root: Path,
    *,
    pre_config: Any,
) -> dict[str, WaveformScorer]:
    """Load one verified scorer per frozen candidate; refuse unsupported ones.

    Each bundle is re-hashed against the declared hash before deserialization,
    and its run record must agree with the freeze declaration.
    """

    scorers: dict[str, WaveformScorer] = {}
    for candidate in record.candidates:
        bundle_dir = Path(bundle_root) / candidate.candidate_id
        descriptor = describe_bundle(bundle_dir)
        if candidate.candidate_id in scorers:
            raise ScoringPreflightError(f"duplicate candidate {candidate.candidate_id!r}")
        if (
            descriptor.model_family != candidate.model_family
            or descriptor.aggregation_rule != candidate.aggregation_rule
        ):
            raise ScoringPreflightError(
                f"candidate {candidate.candidate_id!r} is declared as "
                f"{candidate.model_family}/{candidate.aggregation_rule} but its bundle is "
                f"{descriptor.model_family}/{descriptor.aggregation_rule}"
            )
        if descriptor.threshold != candidate.threshold:
            raise ScoringPreflightError(
                f"candidate {candidate.candidate_id!r}: freeze threshold "
                f"{candidate.threshold!r} differs from the bundle's {descriptor.threshold!r}"
            )
        if hash_directory(bundle_dir) != candidate.bundle_hash:
            raise ScoringPreflightError(
                f"candidate {candidate.candidate_id!r}: bundle hash differs from the freeze record"
            )

        if descriptor.model_family == "logistic_regression":
            import joblib

            model_path = bundle_dir / "lr_pipeline.joblib"
            if not model_path.is_file():
                raise ScoringPreflightError(
                    f"candidate {candidate.candidate_id!r} has no lr_pipeline.joblib"
                )
            scorers[candidate.candidate_id] = LrClipScorer(
                joblib.load(model_path), pre_config, candidate.threshold
            )
        elif descriptor.model_family == "cnn":
            from voice_detector.cnn_scoring import load_cnn_scorer

            scorers[candidate.candidate_id] = load_cnn_scorer(
                bundle_dir,
                expected_bundle_hash=candidate.bundle_hash,
                expected_threshold=candidate.threshold,
            )
        else:
            raise UnsupportedModelFamily(
                f"candidate {candidate.candidate_id!r}: no scorer for {descriptor.model_family!r}"
            )
    return scorers


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _first(items: Sequence[str], n: int = 5) -> str:
    shown = ", ".join(items[:n])
    return shown + (f", … (+{len(items) - n})" if len(items) > n else "")


def validate_scoring_rows(
    rows: Sequence[Mapping[str, Any]],
    partition: str,
    *,
    path_exists: Callable[[str], bool],
) -> list[Mapping[str, Any]]:
    """Check the partition's rows and return them sorted by ``clip_id``.

    Refuses instead of dropping: a silently shorter test set is an evaluation
    integrity failure.
    """

    if partition not in PARTITIONS:
        raise ScoringPreflightError(f"partition must be one of {PARTITIONS}, got {partition!r}")
    selected = [r for r in rows if r.get("official_split") == partition]
    if not selected:
        raise ScoringPreflightError(f"no clips for partition {partition!r}")

    ids = [str(r["clip_id"]) for r in selected]
    if len(set(ids)) != len(ids):
        raise ScoringPreflightError(f"duplicate clip_id values in partition {partition!r}")
    no_path = [str(r["clip_id"]) for r in selected if not r.get("local_path")]
    if no_path:
        raise ScoringPreflightError(
            f"{len(no_path)} clips have no local_path: {_first(no_path)}"
        )
    missing = [str(r["clip_id"]) for r in selected if not path_exists(str(r["local_path"]))]
    if missing:
        raise ScoringPreflightError(
            f"{len(missing)} audio files not found (is VOICE_DETECTOR_LA_ROOT set?): "
            f"{_first(missing)}"
        )
    bad_label = [str(r["clip_id"]) for r in selected if str(r.get("label")) not in _LABELS]
    if bad_label:
        raise ScoringPreflightError(
            f"{len(bad_label)} clips have an invalid label: {_first(bad_label)}"
        )
    no_speaker = [str(r["clip_id"]) for r in selected if not r.get("speaker_id")]
    if no_speaker:
        raise ScoringPreflightError(
            f"{len(no_speaker)} clips have no speaker_id: {_first(no_speaker)}"
        )
    return sorted(selected, key=lambda r: str(r["clip_id"]))


@dataclass
class ScoringSummary:
    complete: bool
    partition: str
    n_clips: int
    rows_written: int = 0
    scored_pairs: list[str] = field(default_factory=list)
    skipped_pairs: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    condition_seconds: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, dict[str, Any]] = field(default_factory=dict)
    predictions_hash: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "complete": self.complete,
            "partition": self.partition,
            "n_clips": self.n_clips,
            "rows_written": self.rows_written,
            "scored": self.scored_pairs,
            "skipped_complete": self.skipped_pairs,
            "n_missing": len(self.missing),
            "condition_seconds": self.condition_seconds,
            "diagnostics": self.diagnostics,
            "predictions_hash": self.predictions_hash,
        }


def score_partition(
    rows: Sequence[Mapping[str, Any]],
    candidate_ids: Sequence[str],
    conditions: Sequence[str],
    scorers: Mapping[str, WaveformScorer],
    spec: RobustnessSpec,
    *,
    partition: str,
    store_path: Path,
    freeze_identity: str,
    decode: Callable[[str, int], np.ndarray],
    path_exists: Callable[[str], bool],
    resume: bool = False,
    max_clips: int | None = None,
    codec_fn: Callable[..., Any] | None = None,
    noise_fn: Callable[..., Any] | None = None,
    trim_fn: Callable[..., Any] | None = None,
    provenance: Mapping[str, str] | None = None,
    after_condition: Callable[[str], None] | None = None,
    progress: Callable[[str], None] | None = None,
) -> ScoringSummary:
    """Score every clip of ``partition`` under every condition for every candidate.

    ``provenance`` (scoring commit, library versions) is pinned in the store, and a
    resumed run must match it. ``after_condition`` runs after each condition is
    committed (or found complete on resume) and may raise to abort the run.
    """

    if max_clips is not None and partition != "development":
        raise ScoringPreflightError("--max-clips is only allowed on the development partition")
    missing_scorers = [c for c in candidate_ids if c not in scorers]
    if missing_scorers:
        raise UnsupportedModelFamily(f"no scorer loaded for {missing_scorers}")
    rates = {scorers[c].sample_rate_hz for c in candidate_ids}
    if len(rates) != 1:
        raise ScoringPreflightError(f"candidates disagree on sample rate: {sorted(rates)}")
    sample_rate = rates.pop()

    clips = validate_scoring_rows(rows, partition, path_exists=path_exists)
    if max_clips is not None:
        if max_clips < 1:
            raise ScoringPreflightError("--max-clips must be positive")
        clips = clips[:max_clips]
    subset = "full" if max_clips is None else f"max_clips={len(clips)}"

    store_path = Path(store_path)
    existing = completed_pairs(store_path) if store_path.exists() else {}
    if existing and not resume:
        raise StoreStateError(
            f"prediction store {store_path} already holds {len(existing)} scored "
            "(candidate, condition) pairs; use a fresh --store or pass --resume"
        )
    pinned = {
        "freeze_identity": freeze_identity,
        "partition": partition,
        "subset": subset,
        **{f"provenance_{k}": v for k, v in (provenance or {}).items()},
    }
    if existing:
        absent = [k for k in pinned if k not in get_metadata(store_path)]
        if absent:
            raise StoreStateError(
                f"store {store_path} lacks provenance {absent}; it was not written by this "
                "scoring code and cannot be resumed",
                reason="store_without_provenance",
            )
    create_prediction_store(store_path)
    try:
        for key, value in pinned.items():
            set_metadata_once(store_path, key, value)
    except ValueError as exc:
        raise StoreStateError(str(exc), reason="store_mismatch") from exc

    n_clips = len(clips)
    summary = ScoringSummary(complete=False, partition=partition, n_clips=n_clips)

    for condition in conditions:
        done = [(c, condition) in existing for c in candidate_ids]
        if all(done):
            if any(existing[(c, condition)] != n_clips for c in candidate_ids):
                raise StoreStateError(
                    f"condition {condition!r} has a partial row count in the store"
                )
            summary.skipped_pairs.extend(f"{c}/{condition}" for c in candidate_ids)
            if after_condition is not None:
                after_condition(condition)
            continue
        if any(done):
            raise StoreStateError(
                f"condition {condition!r} is complete for only some candidates; "
                "refusing to resume into an inconsistent store"
            )

        started = time.perf_counter()
        pred_rows: list[PredictionRow] = []
        n_clipped = 0
        shorter = 0
        kept_original = 0
        for row in clips:
            clip_id = str(row["clip_id"])
            try:
                waveform = decode(str(row["local_path"]), sample_rate)
                conditioned = apply_condition(
                    waveform,
                    clip_id=clip_id,
                    condition=condition,
                    spec=spec,
                    codec_fn=codec_fn,
                    noise_fn=noise_fn,
                    trim_fn=trim_fn,
                    sample_rate_hz=sample_rate,
                )
                n_clipped += conditioned.n_clipped
                shorter += int(conditioned.samples_delta < 0)
                kept_original += int(conditioned.kept_original)
                scored = {
                    c: scorers[c].score_waveform(conditioned.waveform) for c in candidate_ids
                }
            except Exception as exc:  # noqa: BLE001 - add clip context, then re-raise
                raise RuntimeError(
                    f"scoring failed for clip {clip_id!r} under {condition!r}: {exc}"
                ) from exc
            for c in candidate_ids:
                score = scored[c]
                if not math.isfinite(score.clip_score):
                    raise RuntimeError(
                        f"non-finite score for clip {clip_id!r}, candidate {c!r}, "
                        f"condition {condition!r}"
                    )
                threshold = scorers[c].threshold
                pred_rows.append(
                    PredictionRow(
                        clip_id=clip_id,
                        candidate_id=c,
                        condition=condition,
                        clip_score=score.clip_score,
                        label=str(row["label"]),
                        speaker_id=str(row["speaker_id"]),
                        attack_id=row.get("attack_id"),
                        official_split=partition,
                        duration_seconds=float(row.get("duration_seconds", 0)),
                        n_windows=score.n_windows,
                        threshold=threshold,
                        decision=bool(decide_spoof(np.array([score.clip_score]), threshold)[0]),
                    )
                )
        summary.rows_written += write_predictions(
            store_path, pred_rows, freeze_record_hash=freeze_identity
        )
        elapsed = time.perf_counter() - started
        summary.condition_seconds[condition] = round(elapsed, 3)
        summary.diagnostics[condition] = {
            "noise_samples_clipped": n_clipped,
            "codec_outputs_shorter_than_input": shorter,
            "silence_trim_kept_original": kept_original,
        }
        summary.scored_pairs.extend(f"{c}/{condition}" for c in candidate_ids)
        if progress is not None:
            progress(
                f"{condition}: {n_clips} clips x {len(candidate_ids)} candidates "
                f"in {elapsed:.1f}s"
            )
        if after_condition is not None:
            after_condition(condition)

    summary.missing = validate_prediction_completeness(
        store_path, [str(r["clip_id"]) for r in clips], list(candidate_ids), list(conditions)
    )
    summary.complete = not summary.missing
    summary.predictions_hash = compute_predictions_hash(store_path)
    return summary


@dataclass(frozen=True)
class ParityTolerances:
    """Frozen dev-parity tolerances, read from ``evaluation_protocol.yaml``."""

    roc_auc_abs: float
    eer_abs: float
    max_abs_confusion_delta_total: int

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> ParityTolerances:
        section = spec.get("dev_parity")
        if not isinstance(section, Mapping):
            raise ScoringPreflightError("evaluation protocol has no dev_parity section")
        try:
            return cls(
                roc_auc_abs=float(section["roc_auc_abs_tolerance"]),
                eer_abs=float(section["eer_abs_tolerance"]),
                max_abs_confusion_delta_total=int(section["max_abs_confusion_delta_total"]),
            )
        except KeyError as exc:
            raise ScoringPreflightError(f"dev_parity section is missing {exc}") from exc


def dev_parity(
    store_path: Path,
    bundle_root: Path,
    candidate_ids: Sequence[str],
    tolerances: ParityTolerances,
) -> dict[str, dict[str, Any]]:
    """Compare clean development predictions with each bundle's recorded dev metrics.

    A wiring oracle for the development partition only: it catches wrong
    normalization, windowing, score direction or mel settings that unit tests
    with random weights cannot. A candidate fails when it has no recorded
    metrics or no predictions (``checked`` false), or when it is outside
    ``tolerances``. At the frozen threshold a single boundary clip may differ
    by one (``confusion_delta``) because batch composition can shift a score by
    about 1e-6.
    """

    from voice_detector.metrics import evaluate_clip_scores
    from voice_detector.prediction_store import read_predictions

    report: dict[str, dict[str, Any]] = {}
    for candidate_id in candidate_ids:
        descriptor = describe_bundle(Path(bundle_root) / candidate_id)
        expected = descriptor.run_record.get("dev_full_metrics")
        rows = read_predictions(
            store_path, candidate_id=candidate_id, condition="clean", official_split="development"
        )
        if not expected or not rows:
            report[candidate_id] = {"checked": False, "within_tolerance": False}
            continue
        got = evaluate_clip_scores(
            [r.clip_id for r in rows],
            [r.label for r in rows],
            np.array([r.clip_score for r in rows], dtype=np.float64),
            threshold=descriptor.threshold,
            score_direction=SCORE_DIRECTION,
        ).as_dict()
        confusion_delta = {
            key: int(got["confusion"][key]) - int(expected["confusion"][key])
            for key in ("tn", "fp", "fn", "tp")
        }
        auc_delta = abs(got["roc_auc"] - expected["roc_auc"])
        eer_delta = abs(got["eer"] - expected["eer"])
        report[candidate_id] = {
            "checked": True,
            "n_clips": [got["n_clips"], expected["n_clips"]],
            "roc_auc_abs_delta": auc_delta,
            "eer_abs_delta": eer_delta,
            "confusion_delta": confusion_delta,
            "within_tolerance": bool(
                got["n_clips"] == expected["n_clips"]
                and auc_delta <= tolerances.roc_auc_abs
                and eer_delta <= tolerances.eer_abs
                and sum(abs(v) for v in confusion_delta.values())
                <= tolerances.max_abs_confusion_delta_total
            ),
        }
    return report


def parity_passed(report: Mapping[str, Mapping[str, Any]]) -> bool:
    return bool(report) and all(r.get("within_tolerance") for r in report.values())


def check_store_for_report(
    store_path: Path,
    record: FreezeRecord,
    *,
    partition: str,
    require_parity: bool = False,
) -> dict[str, str]:
    """Validate a prediction store before it feeds a report or an evaluation run.

    The store must exist, have been scored under this freeze record (by identity
    hash) for ``partition`` over a full (non-subset) clip set, hold the same clips
    for every candidate x condition, and, if requested, carry a passed dev parity.
    Returns the store metadata.
    """

    from voice_detector.evaluation import freeze_identity_hash

    store_path = Path(store_path)
    if not store_path.is_file():
        raise StoreStateError(f"store not found: {store_path}", reason="store_not_found")
    meta = get_metadata(store_path)
    if meta.get("freeze_identity") != freeze_identity_hash(record):
        raise StoreStateError(
            f"{store_path} was not scored under this freeze record", reason="store_freeze_mismatch"
        )
    if meta.get("partition") != partition:
        raise StoreStateError(
            f"{store_path} holds partition {meta.get('partition')!r}, expected {partition!r}",
            reason="store_partition_mismatch",
        )
    if meta.get("subset") != "full":
        raise StoreStateError(
            f"{store_path} is a smoke subset ({meta.get('subset')!r})", reason="subset_store"
        )
    counts = completed_pairs(store_path)
    expected = {(c.candidate_id, cond) for c in record.candidates for cond in record.conditions}
    missing = sorted(expected - set(counts))
    if missing or len({counts[p] for p in expected}) > 1:
        raise StoreStateError(
            f"{store_path} does not hold the same clips for every candidate x condition; "
            f"missing {[f'{c}/{k}' for c, k in missing]}",
            reason="incomplete_store",
        )
    if require_parity and meta.get("dev_parity") != "passed":
        raise StoreStateError(
            f"{store_path} has no passed development parity", reason="parity_not_passed"
        )
    return meta


def check_bundle_files(descriptor: BundleDescriptor, bundle_dir: Path) -> None:
    """The model files a scorer needs must exist before a bundle can be frozen."""

    needed = {
        "logistic_regression": ["lr_pipeline.joblib"],
        "cnn": ["cnn_checkpoint.pt", "cnn_norm_stats.pt"],
    }[descriptor.model_family]
    absent = [name for name in needed if not (Path(bundle_dir) / name).is_file()]
    if absent:
        raise ScoringPreflightError(f"bundle {descriptor.candidate_id!r} is missing {absent}")
