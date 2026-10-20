"""Generate backup demo materials for the presentation (§13).

Produces a set of pre-scored results using the ScoringService with synthetic
test clips, saving JSON responses that can be shown if the live demo fails.
Run on the host where the bundle is available:

    uv run python scripts/demo_backup.py --bundle-dir artifacts/cnn_augmented_seed1 \
        --output-dir reports/demo_backup

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
    parser.add_argument("--bundle-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    from voice_detector.serving import ScoringService

    svc = ScoringService.from_config()

    clips = [
        ("tone_4s", 4.0, 220.0, 42),
        ("tone_1s", 1.0, 440.0, 43),
    ]

    for name, dur, freq, seed in clips:
        wav = _synthetic_wav(dur, freq, seed)
        t0 = time.perf_counter()
        result = svc.score_bytes(wav, f"{name}.wav")
        elapsed_ms = (time.perf_counter() - t0) * 1000
        payload = {
            "name": name,
            "duration_seconds": dur,
            "score": result.score,
            "decision": result.decision,
            "threshold": result.threshold,
            "n_windows": result.n_windows,
            "processing_ms": round(elapsed_ms, 1),
            "model_id": result.model_id,
        }
        out = args.output_dir / f"{name}.json"
        out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"{name}: score={result.score:.4f} decision={result.decision} ({elapsed_ms:.1f} ms)")

    info = svc.model_info()
    (args.output_dir / "model_info.json").write_text(json.dumps(info, indent=2) + "\n")

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
