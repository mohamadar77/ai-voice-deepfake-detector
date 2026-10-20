"""Static checks for the latency benchmark script and config."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


def test_benchmark_config_declares_the_expected_mix():
    path = REPO / "configs" / "latency_benchmark.yaml"
    cfg = yaml.safe_load(path.read_text())
    assert isinstance(cfg["seed"], int)
    assert cfg["warmup_rounds"] >= 1
    assert cfg["rounds_per_clip"] >= 1
    mix = cfg["input_mix"]
    assert len(mix) >= 4, "need at least four duration bins"
    durations = [s["duration_seconds"] for s in mix]
    assert 1.0 in durations, "must include the minimum accepted duration"
    assert 30.0 in durations, "must include the maximum accepted duration"
    for spec in mix:
        assert spec["sample_rate_hz"] > 0
        assert spec["channels"] >= 1
        assert spec["label"]


def test_benchmark_script_generates_valid_wav():
    """The synthetic WAV generator produces decodable audio."""

    import io

    import soundfile as sf

    # inline import to avoid needing the bundle
    from scripts.latency_benchmark import _synthetic_wav

    data = _synthetic_wav(2.0, 16000, 1, 42)
    info = sf.info(io.BytesIO(data))
    assert info.samplerate == 16000
    assert info.channels == 1
    assert 1.9 < info.duration < 2.1

    stereo = _synthetic_wav(4.0, 44100, 2, 43)
    info2 = sf.info(io.BytesIO(stereo))
    assert info2.samplerate == 44100
    assert info2.channels == 2


def test_benchmark_config_seed_is_deterministic():
    """Two calls with the same seed produce identical bytes."""

    from scripts.latency_benchmark import _synthetic_wav

    a = _synthetic_wav(1.0, 16000, 1, 99)
    b = _synthetic_wav(1.0, 16000, 1, 99)
    assert a == b
