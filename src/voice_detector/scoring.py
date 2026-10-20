"""Neutral scoring contract shared by the serving layer and the scorer.

The FastAPI app depends only on these abstractions, never on the feature
extractor, model or torch. That keeps serving importable and testable without
a trained bundle, and lets training/inference code supply a concrete
implementation later without the API importing it. A score is a score, not a
probability of authenticity: ``score_direction`` states what a higher value
means and ``threshold`` is the operating point that maps it to a label.

``decide_spoof`` is the single operating decision. Offline metrics and every
scorer call it, so a threshold frozen on development data labels a clip the same
way everywhere.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, BinaryIO, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike

AudioSource = str | Path | BinaryIO

# Frozen project convention: synthetic speech is the positive class and larger
# scores mean more synthetic.
SCORE_DIRECTION = "higher_more_spoof"
DECISION_RULE = "spoof if score >= threshold, compared in float64"


# Window-to-clip aggregation rules, one per model family. A clip score is the
# arithmetic mean, in float64, of the per-window quantity named here.
LR_AGGREGATION_RULE = "mean_window_decision_function"
# Mean over option-D windows of (logit_spoof - logit_bonafide).
CNN_AGGREGATION_RULE = "mean_window_logit"


def decide_spoof(scores: ArrayLike, threshold: float) -> np.ndarray:
    """Boolean spoof decisions: ``score >= threshold``, compared in ``float64``.

    The comparison is forced to ``float64`` because operating thresholds sit one
    ``float64`` step above a boundary score. Under NumPy 2 promotion, comparing
    ``float32`` model outputs with a Python float happens in ``float32``, which
    rounds such a threshold back onto the boundary and flips tied bona fide
    clips to spoof. For the same reason a reduced-precision threshold
    (``np.float32``) is rejected: store thresholds as ``float64``.
    """

    if isinstance(threshold, bool) or not isinstance(threshold, Real):
        raise ValueError(f"threshold must be a real number, got {threshold!r}")
    if isinstance(threshold, np.floating) and threshold.dtype != np.float64:
        raise ValueError(f"threshold must be float64, got {threshold.dtype}")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    return np.asarray(scores, dtype=np.float64) >= np.float64(threshold)


@dataclass(frozen=True)
class ClipScore:
    """One aggregated score per original clip, plus its operating decision."""

    score: float
    label: str  # "bonafide" or "spoof"
    score_direction: str  # SCORE_DIRECTION
    threshold: float
    n_windows: int


@dataclass(frozen=True)
class WaveformScore:
    """A clip score from a decoded waveform, before any threshold is applied."""

    clip_score: float  # float64, SCORE_DIRECTION
    n_windows: int


class WaveformScorer(Protocol):
    """Scores one decoded, mono, finite waveform at ``sample_rate_hz``.

    Implementations must not mutate the waveform. Labels never reach a scorer.
    """

    sample_rate_hz: int
    model_family: str
    aggregation_rule: str
    threshold: float

    def score_waveform(self, waveform: np.ndarray) -> WaveformScore: ...


@runtime_checkable
class Scorer(Protocol):
    """What the serving layer needs from any clip scorer.

    A concrete implementation (for example an MFCC + Logistic Regression
    bundle) decodes the source with the shared preprocessing, scores each
    window, aggregates to one clip score and applies the operating threshold.
    """

    model_version: str
    preprocessing_version: str

    def score_source(self, source: AudioSource) -> ClipScore: ...


def clip_score_from(result: WaveformScore, threshold: float) -> ClipScore:
    """Apply the operating decision to a clip score.

    The one place a label is derived from a score outside offline metrics, so the
    serving API and the library entry points label a clip exactly as
    ``decide_spoof`` does.
    """

    is_spoof = bool(decide_spoof(np.array([result.clip_score]), threshold)[0])
    return ClipScore(
        score=result.clip_score,
        label="spoof" if is_spoof else "bonafide",
        score_direction=SCORE_DIRECTION,
        threshold=float(threshold),
        n_windows=result.n_windows,
    )


@dataclass(frozen=True)
class ScoredClip:
    """A clip score with the per-window evidence behind it.

    ``window_real_samples`` is the number of real (non-repeated) samples in each
    window, so a caller can warn when a window is mostly repeat padding.
    """

    clip: ClipScore
    window_scores: tuple[float, ...]
    window_real_samples: tuple[int, ...]


@runtime_checkable
class ServableScorer(Protocol):
    """What the serving layer needs from a clip scorer that takes decoded audio.

    The API owns decoding and limits, so it hands the scorer a decoded mono
    waveform at ``sample_rate_hz``. ``describe`` returns an allowlisted summary of
    the model's run record (no filesystem paths).
    """

    model_family: str
    model_version: str
    preprocessing_version: str
    aggregation_rule: str
    sample_rate_hz: int
    threshold: float
    bundle_hash: str | None

    def score_decoded(self, waveform: np.ndarray) -> ScoredClip: ...

    def describe(self) -> Mapping[str, Any]: ...
