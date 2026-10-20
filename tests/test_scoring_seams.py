"""Small shared seams added for serving: labels from scores, hashing, the scorer protocol."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from serving_fakes import FakeScorer

from voice_detector import evaluation, hashing
from voice_detector.scoring import (
    SCORE_DIRECTION,
    ScoredClip,
    ServableScorer,
    WaveformScore,
    clip_score_from,
)


def test_clip_score_from_labels_with_the_float64_rule():
    boundary = 0.1234567890123
    at = clip_score_from(WaveformScore(clip_score=boundary, n_windows=3), boundary)
    above = clip_score_from(
        WaveformScore(clip_score=boundary, n_windows=3), float(np.nextafter(boundary, np.inf))
    )
    assert at.label == "spoof" and above.label == "bonafide"
    assert at.score == boundary and at.n_windows == 3
    assert at.score_direction == SCORE_DIRECTION
    assert at.threshold == boundary


def test_clip_score_from_rejects_a_non_finite_threshold():
    with pytest.raises(ValueError):
        clip_score_from(WaveformScore(clip_score=0.0, n_windows=1), float("nan"))


def test_scored_clip_is_immutable():
    clip = clip_score_from(WaveformScore(clip_score=1.0, n_windows=1), 0.0)
    scored = ScoredClip(clip=clip, window_scores=(1.0,), window_real_samples=(16000,))
    with pytest.raises(dataclasses.FrozenInstanceError):
        scored.window_scores = ()  # type: ignore[misc]


def test_the_fake_scorer_satisfies_the_servable_protocol():
    assert isinstance(FakeScorer(), ServableScorer)


def test_hash_directory_is_shared_and_stable(tmp_path):
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("b", encoding="utf-8")
    assert evaluation.hash_directory is hashing.hash_directory
    first = hashing.hash_directory(tmp_path)
    assert first == hashing.hash_directory(tmp_path)
    (tmp_path / "sub" / "b.txt").write_text("changed", encoding="utf-8")
    assert hashing.hash_directory(tmp_path) != first
