"""Neutral scoring contract shared by the serving layer and the scorer.

The FastAPI app depends only on these abstractions, never on the feature
extractor, model or torch. That keeps serving importable and testable without
a trained bundle, and lets training/inference code supply a concrete
implementation later without the API importing it. A score is a score, not a
probability of authenticity: ``score_direction`` states what a higher value
means and ``threshold`` is the operating point that maps it to a label.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol, runtime_checkable

AudioSource = str | Path | BinaryIO


@dataclass(frozen=True)
class ClipScore:
    """One aggregated score per original clip, plus its operating decision."""

    score: float
    label: str  # "bonafide" or "spoof"
    score_direction: str  # e.g. "higher_more_spoof"
    threshold: float
    n_windows: int


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
