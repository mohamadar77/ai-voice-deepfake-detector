"""Tests for the speaker-grouped bootstrap (issue #35)."""

from __future__ import annotations

from collections import Counter

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


# ---------------------------------------------------------------------------
# Resampling unit: whole speakers, never individual clips
# ---------------------------------------------------------------------------

# Unequal group sizes so that a clip-level bootstrap is distinguishable: it would
# always return exactly ``n`` clips, while whole-speaker resampling returns a
# varying number.
_SPEAKER_SIZES = {"spk_a": 1, "spk_b": 2, "spk_c": 3, "spk_d": 4, "spk_e": 5, "spk_f": 6}


def _grouped_data() -> tuple[list[str], list[str], np.ndarray, list[str]]:
    clip_ids: list[str] = []
    speaker_ids: list[str] = []
    for speaker, size in _SPEAKER_SIZES.items():
        for j in range(size):
            clip_ids.append(f"{speaker}_c{j}")
            speaker_ids.append(speaker)
    labels = ["bonafide" if i % 2 == 0 else "spoof" for i in range(len(clip_ids))]
    scores = np.arange(len(clip_ids), dtype=np.float64)  # score identifies its clip
    return clip_ids, labels, scores, speaker_ids


def _recording_metric(calls: list[tuple[list[str], list[str], np.ndarray]]):
    def metric(clip_ids, labels, scores, *, score_direction, threshold=None) -> float:
        calls.append((list(clip_ids), list(labels), np.array(scores)))
        return 0.0

    return metric


def _speaker_draw_counts(resample: list[str]) -> dict[str, int]:
    """How many times each speaker was drawn, requiring all its clips to agree."""

    counts = Counter(resample)
    draws: dict[str, int] = {}
    for speaker, size in _SPEAKER_SIZES.items():
        per_clip = {counts[f"{speaker}_c{j}"] for j in range(size)}
        assert len(per_clip) == 1, (
            f"clips of {speaker} were resampled independently: {sorted(per_clip)}"
        )
        draws[speaker] = per_clip.pop()
    return draws


class TestResamplingUnitIsTheSpeaker:
    def test_every_resample_is_a_whole_speaker_draw_with_replacement(self):
        clip_ids, labels, scores, speaker_ids = _grouped_data()
        calls: list[tuple[list[str], list[str], np.ndarray]] = []

        speaker_grouped_bootstrap(
            clip_ids, labels, scores, speaker_ids,
            metric_fn=_recording_metric(calls), metric_name="spy",
            spec=_spec(n_resamples=200), score_direction=SCORE_DIRECTION,
        )

        point_estimate, resamples = calls[0], calls[1:]
        assert len(resamples) == 200
        assert point_estimate[0] == clip_ids  # the point estimate uses every clip once

        all_draws = []
        for resample_ids, _, _ in resamples:
            draws = _speaker_draw_counts(resample_ids)
            # One draw per speaker in the pool, with replacement.
            assert sum(draws.values()) == len(_SPEAKER_SIZES)
            all_draws.append(draws)

        # The test must be able to see resampling happen: some speakers are drawn
        # twice or more, some not at all, and the clip total varies.
        assert any(max(d.values()) >= 2 for d in all_draws)
        assert any(min(d.values()) == 0 for d in all_draws)
        assert len({len(ids) for ids, _, _ in resamples}) > 1

    def test_resampling_keeps_each_clip_with_its_own_label_and_score(self):
        clip_ids, labels, scores, speaker_ids = _grouped_data()
        calls: list[tuple[list[str], list[str], np.ndarray]] = []

        speaker_grouped_bootstrap(
            clip_ids, labels, scores, speaker_ids,
            metric_fn=_recording_metric(calls), metric_name="spy",
            spec=_spec(n_resamples=50), score_direction=SCORE_DIRECTION,
        )

        by_clip = {cid: (lab, sc) for cid, lab, sc in zip(clip_ids, labels, scores, strict=True)}
        for resample_ids, resample_labels, resample_scores in calls:
            for cid, lab, sc in zip(resample_ids, resample_labels, resample_scores, strict=True):
                assert by_clip[cid] == (lab, sc)

    def test_paired_bootstrap_scores_both_candidates_on_identical_whole_speaker_resamples(self):
        clip_ids, labels, scores, speaker_ids = _grouped_data()
        seen: list[tuple[str, list[str]]] = []

        def metric(ids, labs, scs, *, score_direction, threshold=None) -> float:
            # Candidate B's scores are shifted far above candidate A's.
            seen.append(("b" if np.min(scs) >= 100.0 else "a", list(ids)))
            return 0.0

        paired_bootstrap(
            clip_ids, labels, scores, scores + 100.0, speaker_ids,
            metric_fn=metric, metric_name="spy",
            candidate_a="a", candidate_b="b",
            spec=_spec(n_resamples=100), score_direction=SCORE_DIRECTION,
        )

        resamples_a = [ids for who, ids in seen if who == "a"][1:]  # drop the point estimate
        resamples_b = [ids for who, ids in seen if who == "b"][1:]
        assert len(resamples_a) == len(resamples_b) == 100
        assert resamples_a == resamples_b
        for ids in resamples_a:
            assert sum(_speaker_draw_counts(ids).values()) == len(_SPEAKER_SIZES)
