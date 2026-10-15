"""Tests for the speaker-grouped bootstrap (issue #35)."""

from __future__ import annotations

import numpy as np
import pytest

from voice_detector.bootstrap import (
    BootstrapSpec,
    PairedDifference,
    paired_bootstrap,
    speaker_grouped_bootstrap,
)
from voice_detector.scoring import SCORE_DIRECTION


def _spec(n_resamples: int = 500, seed: int = 42) -> BootstrapSpec:
    return BootstrapSpec(
        confidence_level=0.95,
        n_resamples=n_resamples,
        seed=seed,
        resampling_unit="speaker_id",
        method="percentile",
    )


def _eer_metric(
    clip_ids: list[str],
    labels: list[str],
    scores: np.ndarray,
    *,
    score_direction: str,
    threshold: float | None = None,
) -> float:
    from voice_detector.metrics import compute_eer

    bonafide = scores[np.array([lab == "bonafide" for lab in labels])]
    spoof = scores[np.array([lab == "spoof" for lab in labels])]
    if len(bonafide) == 0 or len(spoof) == 0:
        return float("nan")
    return compute_eer(bonafide, spoof).eer


class TestSpeakerGroupedBootstrap:
    def test_produces_ci(self):
        rng = np.random.default_rng(123)
        n_speakers = 10
        clips_per_speaker = 5
        n = n_speakers * clips_per_speaker

        clip_ids = [f"clip_{i}" for i in range(n)]
        speaker_ids = [f"spk_{i // clips_per_speaker}" for i in range(n)]
        labels = ["bonafide" if i % 2 == 0 else "spoof" for i in range(n)]
        scores = rng.standard_normal(n)
        scores[np.array([lab == "spoof" for lab in labels])] += 1.0

        ci = speaker_grouped_bootstrap(
            clip_ids, labels, scores, speaker_ids,
            metric_fn=_eer_metric,
            metric_name="eer",
            spec=_spec(),
            score_direction=SCORE_DIRECTION,
        )

        assert ci.ci_lower <= ci.point_estimate <= ci.ci_upper
        assert ci.confidence_level == 0.95
        assert ci.n_resamples == 500

    def test_seed_is_deterministic(self):
        rng = np.random.default_rng(456)
        n = 30
        clip_ids = [f"c{i}" for i in range(n)]
        speaker_ids = [f"s{i // 3}" for i in range(n)]
        labels = ["bonafide" if i % 2 == 0 else "spoof" for i in range(n)]
        scores = rng.standard_normal(n)

        spec = _spec(n_resamples=100, seed=99)
        ci1 = speaker_grouped_bootstrap(
            clip_ids, labels, scores, speaker_ids,
            metric_fn=_eer_metric, metric_name="eer",
            spec=spec, score_direction=SCORE_DIRECTION,
        )
        ci2 = speaker_grouped_bootstrap(
            clip_ids, labels, scores, speaker_ids,
            metric_fn=_eer_metric, metric_name="eer",
            spec=spec, score_direction=SCORE_DIRECTION,
        )

        assert ci1.ci_lower == ci2.ci_lower
        assert ci1.ci_upper == ci2.ci_upper

    def test_rejects_empty_speakers(self):
        with pytest.raises(ValueError, match="zero clips"):
            speaker_grouped_bootstrap(
                [], [], np.array([]), [],
                metric_fn=_eer_metric, metric_name="eer",
                spec=_spec(), score_direction=SCORE_DIRECTION,
            )

    def test_rejects_single_speaker(self):
        with pytest.raises(ValueError, match="at least 2"):
            speaker_grouped_bootstrap(
                ["c1", "c2"], ["bonafide", "spoof"],
                np.array([0.1, 0.9]), ["spk1", "spk1"],
                metric_fn=_eer_metric, metric_name="eer",
                spec=_spec(), score_direction=SCORE_DIRECTION,
            )


class TestPairedBootstrap:
    def test_uses_same_resamples(self):
        rng = np.random.default_rng(789)
        n = 40
        clip_ids = [f"c{i}" for i in range(n)]
        speaker_ids = [f"s{i // 4}" for i in range(n)]
        labels = ["bonafide" if i % 2 == 0 else "spoof" for i in range(n)]
        scores_a = rng.standard_normal(n)
        scores_b = scores_a + 0.1

        diff = paired_bootstrap(
            clip_ids, labels, scores_a, scores_b, speaker_ids,
            metric_fn=_eer_metric, metric_name="eer",
            candidate_a="model_a", candidate_b="model_b",
            spec=_spec(n_resamples=200), score_direction=SCORE_DIRECTION,
        )

        assert isinstance(diff, PairedDifference)
        assert diff.candidate_a == "model_a"
        assert diff.candidate_b == "model_b"
        assert diff.ci_lower <= diff.point_difference <= diff.ci_upper

    def test_respects_spec_parameters(self):
        rng = np.random.default_rng(101)
        n = 20
        clip_ids = [f"c{i}" for i in range(n)]
        speaker_ids = [f"s{i // 2}" for i in range(n)]
        labels = ["bonafide" if i % 2 == 0 else "spoof" for i in range(n)]
        scores = rng.standard_normal(n)

        spec = _spec(n_resamples=50)
        ci = speaker_grouped_bootstrap(
            clip_ids, labels, scores, speaker_ids,
            metric_fn=_eer_metric, metric_name="eer",
            spec=spec, score_direction=SCORE_DIRECTION,
        )
        assert ci.n_resamples == 50
