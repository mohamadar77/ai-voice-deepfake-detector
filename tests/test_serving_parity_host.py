"""Opt-in host test: the served score equals the library scorer's score for real clips.

Run on a host with the real bundle (marker ``parity``). It needs development or demo
clips only; any path that looks like the evaluation partition is refused, and no
prediction store is opened. For a comparison with ``eval-score`` numbers, run a
development smoke scoring separately and compare by hand (see the PR description).

    VOICE_DETECTOR_PARITY_CLIP_DIR=/path/to/dev/flac \
      uv run pytest tests/test_serving_parity_host.py -m parity -q
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from voice_detector.config import load_config
from voice_detector.preprocessing import AudioConfig, load_waveform
from voice_detector.serving.loader import build_scorer
from voice_detector.serving.service import ScoringService
from voice_detector.serving.settings import load_serving_settings

pytestmark = pytest.mark.parity

REPO = Path(__file__).resolve().parents[1]
CLIP_DIR = os.environ.get("VOICE_DETECTOR_PARITY_CLIP_DIR")

if not CLIP_DIR:
    pytest.skip("VOICE_DETECTOR_PARITY_CLIP_DIR is not set", allow_module_level=True)


def _clips() -> list[Path]:
    directory = Path(str(CLIP_DIR))
    if any("eval" in part.lower() for part in directory.parts):
        pytest.fail("refusing a clip directory that looks like the evaluation partition")
    clips = sorted([*directory.glob("*.flac"), *directory.glob("*.wav")])[:8]
    assert clips, "no .flac/.wav clips found"
    return clips


def test_served_score_equals_the_library_score_exactly():
    settings = load_serving_settings(REPO / "configs" / "serving.yaml")
    project = load_config(REPO / "configs" / "default.yaml")
    scorer = build_scorer(settings, project.paths.artifacts_root)
    service = ScoringService(scorer, settings)
    audio = AudioConfig(sample_rate_hz=scorer.sample_rate_hz)

    windows_seen = set()
    for index, clip in enumerate(_clips()):
        data = clip.read_bytes()
        served = service.score_bytes(data, request_id=f"parity-{index}").response
        library = scorer.score_waveform(load_waveform(clip, audio))
        assert served.score == library.clip_score, clip.name
        assert served.n_windows == library.n_windows
        assert served.threshold == settings.model.expected_threshold
        assert served.bundle_hash == settings.model.expected_bundle_hash
        windows_seen.add(min(served.n_windows, 2))
    assert windows_seen, "no clips scored"
