"""Compare a running container's /predict scores with the in-process library scorer.

Run on a host where the container is up (compose, port 8000) and the bundle is available.
Development or demo clips only; any path that looks like the evaluation partition is refused.

    VOICE_DETECTOR_ARTIFACTS_DIR=/path/to/artifacts \
      uv run python scripts/container_parity.py --clip-dir /path/to/dev/flac \
      --count 200 --seed 20261021 --output reports/container_parity.json

--reference-csv (optional, columns clip_id,score) adds a comparison with stored scores,
for example development predictions from the locked-evaluation run on CUDA torch.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _multipart(data: bytes, filename: str) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    body = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def _post(url: str, data: bytes, filename: str) -> dict:
    body, content_type = _multipart(data, filename)
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": content_type}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 - local URL
            return json.load(response)
    except urllib.error.HTTPError as error:
        return {"http_status": error.code, "error": error.read().decode(errors="replace")[:300]}


def _pick(clip_dir: Path, count: int, seed: int) -> list[Path]:
    if any("eval" in part.lower() for part in clip_dir.parts):
        raise SystemExit("refusing a clip directory that looks like the evaluation partition")
    clips = sorted([*clip_dir.glob("*.flac"), *clip_dir.glob("*.wav")])
    if len(clips) < count:
        raise SystemExit(f"need at least {count} clips, found {len(clips)}")
    longest = sorted(clips, key=lambda p: p.stat().st_size, reverse=True)[:8]
    rest = [c for c in clips if c not in longest]
    random.Random(seed).shuffle(rest)
    return [*longest, *rest[: count - len(longest)]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-dir", required=True, type=Path)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--reference-csv", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    from voice_detector.config import load_config
    from voice_detector.preprocessing import AudioConfig, load_waveform
    from voice_detector.serving.loader import build_scorer
    from voice_detector.serving.settings import load_serving_settings

    settings = load_serving_settings(REPO / "configs" / "serving.yaml")
    project = load_config(REPO / "configs" / "default.yaml")
    scorer = build_scorer(settings, project.paths.artifacts_root)
    audio = AudioConfig(sample_rate_hz=scorer.sample_rate_hz)
    threshold = settings.model.expected_threshold

    reference: dict[str, float] = {}
    if args.reference_csv:
        with args.reference_csv.open(newline="") as handle:
            reference = {row["clip_id"]: float(row["score"]) for row in csv.DictReader(handle)}

    rows = []
    rejected: list[dict] = []
    for clip in _pick(args.clip_dir, args.count, args.seed):
        served = _post(args.url, clip.read_bytes(), clip.name)
        library = scorer.score_waveform(load_waveform(clip, audio))
        if "http_status" in served:
            rejected.append(
                {
                    "clip_id": clip.stem,
                    "n_windows": library.n_windows,
                    "library_score": library.clip_score,
                    **served,
                }
            )
            continue
        row = {
            "clip_id": clip.stem,
            "n_windows": library.n_windows,
            "served_score": served["score"],
            "library_score": library.clip_score,
            "abs_diff": abs(served["score"] - library.clip_score),
            "served_flag": served["decision"] == "flagged_spoof_like",
            "library_flag": library.clip_score >= threshold,
        }
        if clip.stem in reference:
            row["reference_score"] = reference[clip.stem]
            row["reference_abs_diff"] = abs(served["score"] - reference[clip.stem])
            row["reference_flag"] = reference[clip.stem] >= threshold
        rows.append(row)

    def summary(diff_key: str, flag_key: str, subset: list[dict]) -> dict:
        if not subset:
            return {"n": 0}
        return {
            "n": len(subset),
            "max_abs_diff": max(r[diff_key] for r in subset),
            "n_above_tolerance": sum(r[diff_key] > args.tolerance for r in subset),
            "n_decision_flips": sum(r["served_flag"] != r[flag_key] for r in subset),
        }

    multi = [r for r in rows if r["n_windows"] > 1]
    report = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "url": args.url,
        "seed": args.seed,
        "tolerance": args.tolerance,
        "threshold": threshold,
        "bundle_hash": settings.model.expected_bundle_hash,
        "container_vs_library": summary("abs_diff", "library_flag", rows),
        "container_vs_library_multiwindow": summary("abs_diff", "library_flag", multi),
        "container_vs_reference": summary(
            "reference_abs_diff",
            "reference_flag",
            [r for r in rows if "reference_score" in r],
        ),
        "n_rejected_by_service": len(rejected),
        "rejected": rejected,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("n_rejected_by_service", len(rejected), json.dumps(rejected[:3]))
    keys = ("container_vs_library", "container_vs_library_multiwindow", "container_vs_reference")
    for key in keys:
        print(key, json.dumps(report[key]))


if __name__ == "__main__":
    main()
