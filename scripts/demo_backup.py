"""Generate backup demo materials for the presentation (§13).

Produces a set of pre-scored results using the ScoringService with synthetic
test clips, saving JSON responses that can be shown if the live demo fails.
Run on the host where the bundle is available:

    VOICE_DETECTOR_ARTIFACTS_DIR=/path/to/artifacts \
        uv run python scripts/demo_backup.py --output-dir reports/demo_backup

The bundle is read from <artifacts root>/<candidate_id> as configured in
configs/serving.yaml.

The output directory will contain:
  - tone_4s.json: score response for a 4-second generated tone
  - tone_1s.json: score response for a 1-second generated tone
  - model_info.json: the /model endpoint payload
  - hardware.json: host info
"""

from __future__ import annotations

import argparse
import io
import json
import platform
import time
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf


def _synthetic_wav(seconds: float, freq_hz: float, seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    sr = 16_000
    n = int(seconds * sr)
    t = np.arange(n) / sr
    audio = (0.3 * np.sin(2 * np.pi * freq_hz * t) + 0.02 * rng.standard_normal(n)).astype(
        "float32"
    )
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    from voice_detector.config import load_config
    from voice_detector.serving.loader import build_scorer
    from voice_detector.serving.service import ScoringService
    from voice_detector.serving.settings import load_serving_settings

    settings = load_serving_settings(Path("configs/serving.yaml"))
    project = load_config(Path("configs/default.yaml"))
    scorer = build_scorer(settings, project.paths.artifacts_root)
    svc = ScoringService(scorer, settings)
    svc.warmup()

    clips = [
        ("tone_4s", 4.0, 220.0, 42),
        ("tone_1s", 1.0, 440.0, 43),
    ]

    for name, dur, freq, seed in clips:
        wav = _synthetic_wav(dur, freq, seed)
        request_id = str(uuid.uuid4())
        t0 = time.perf_counter()
        outcome = svc.score_bytes(wav, request_id=request_id)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        resp = outcome.response
        payload = {
            "name": name,
            "duration_seconds": resp.duration_seconds,
            "score": resp.score,
            "decision": resp.decision,
            "threshold": resp.threshold,
            "n_windows": resp.n_windows,
            "processing_ms": round(elapsed_ms, 1),
            "model_id": resp.model_id,
            "model_version": resp.model_version,
            "bundle_hash": resp.bundle_hash,
        }
        out = args.output_dir / f"{name}.json"
        out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"{name}: score={resp.score:.4f} decision={resp.decision} ({elapsed_ms:.1f} ms)")

    info = svc.model_info()
    (args.output_dir / "model_info.json").write_text(
        json.dumps(info.model_dump(mode="json"), indent=2) + "\n"
    )

    hw = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (args.output_dir / "hardware.json").write_text(json.dumps(hw, indent=2) + "\n")

    print(f"\nBackup materials written to {args.output_dir}")


if __name__ == "__main__":
    main()
