#!/usr/bin/env python3
"""p50/p95 latency benchmark for the scoring service (library path).

Generates seeded synthetic WAV clips at declared durations, warms up the
scorer, then measures ``ScoringService.score_bytes`` wall-clock time per
clip over many rounds.  Reports p50, p95, p99, mean, min, max latency per
duration bin and overall, plus failure rate, peak RSS, and hardware info.

Usage (on the host, never the Mac):

    uv run python scripts/latency_benchmark.py \
        [--config configs/latency_benchmark.yaml] \
        [--output reports/latency_benchmark.json]

The script loads the model bundle through the normal serving path and runs
all scoring on the calling thread (no ASGI overhead, no network).  Torch
thread count is whatever ``configs/serving.yaml`` sets.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

REPO = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Latency benchmark")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs" / "latency_benchmark.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write JSON results here (default: stdout only)",
    )
    return parser.parse_args()


def _synthetic_wav(
    seconds: float,
    sample_rate_hz: int,
    channels: int,
    seed: int,
) -> bytes:
    rng = np.random.default_rng(seed)
    frames = int(round(seconds * sample_rate_hz))
    shape = (frames, channels) if channels > 1 else (frames,)
    samples = (0.1 * rng.standard_normal(shape)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate_hz, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _build_service():
    from voice_detector.config import load_config
    from voice_detector.serving.loader import build_scorer
    from voice_detector.serving.service import ScoringService
    from voice_detector.serving.settings import load_serving_settings

    settings = load_serving_settings(REPO / "configs" / "serving.yaml")
    project = load_config(REPO / "configs" / "default.yaml")
    scorer = build_scorer(settings, project.paths.artifacts_root)
    return ScoringService(scorer, settings)


def _peak_rss_mib() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    if sys.platform == "darwin":
        return ru.ru_maxrss / (1024 * 1024)
    return ru.ru_maxrss / 1024


def _hardware_info() -> dict:
    uname = platform.uname()
    info: dict = {
        "hostname": uname.node,
        "system": uname.system,
        "machine": uname.machine,
        "processor": uname.processor or platform.processor(),
        "python": platform.python_version(),
    }
    cpu_count = os.cpu_count()
    if cpu_count:
        info["cpu_count"] = cpu_count
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return info


def run(config_path: Path, output_path: Path | None) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    seed = cfg["seed"]
    warmup_rounds = cfg["warmup_rounds"]
    rounds = cfg["rounds_per_clip"]
    mix = cfg["input_mix"]

    print(f"Config: {config_path}")
    print(f"Seed: {seed}, warmup: {warmup_rounds}, rounds: {rounds}")
    print(f"Input mix: {len(mix)} clip types\n")

    print("Generating clips...")
    clips: list[tuple[str, bytes]] = []
    for i, spec in enumerate(mix):
        data = _synthetic_wav(
            spec["duration_seconds"],
            spec["sample_rate_hz"],
            spec["channels"],
            seed + i,
        )
        clips.append((spec["label"], data))
        print(f"  {spec['label']}: {len(data)} bytes")

    print("\nBuilding scorer...")
    t0 = time.perf_counter()
    service = _build_service()
    build_ms = (time.perf_counter() - t0) * 1000
    print(f"  Scorer built in {build_ms:.0f} ms")

    print(f"\nWarming up ({warmup_rounds} rounds)...")
    for w in range(warmup_rounds):
        for _label, data in clips:
            service.score_bytes(data, request_id=f"warmup-{w}")
    print("  Done")

    print(f"\nBenchmarking ({rounds} rounds per clip)...")
    results_per_bin: list[dict] = []
    all_latencies: list[float] = []
    failures = 0
    total_requests = 0

    for label, data in clips:
        latencies: list[float] = []
        bin_failures = 0
        for r in range(rounds):
            total_requests += 1
            try:
                t_start = time.perf_counter()
                service.score_bytes(data, request_id=f"bench-{label}-{r}")
                elapsed_ms = (time.perf_counter() - t_start) * 1000
                latencies.append(elapsed_ms)
            except Exception as exc:
                bin_failures += 1
                print(f"  FAIL {label} round {r}: {exc}")

        failures += bin_failures
        arr = np.array(latencies)
        if len(arr) > 0:
            stats = {
                "label": label,
                "n": len(arr),
                "failures": bin_failures,
                "p50_ms": float(np.percentile(arr, 50)),
                "p95_ms": float(np.percentile(arr, 95)),
                "p99_ms": float(np.percentile(arr, 99)),
                "mean_ms": float(np.mean(arr)),
                "min_ms": float(np.min(arr)),
                "max_ms": float(np.max(arr)),
                "std_ms": float(np.std(arr)),
            }
        else:
            stats = {"label": label, "n": 0, "failures": bin_failures}
        results_per_bin.append(stats)
        all_latencies.extend(latencies)
        p50 = stats.get("p50_ms", float("nan"))
        p95 = stats.get("p95_ms", float("nan"))
        print(f"  {label}: p50={p50:.1f} ms, p95={p95:.1f} ms ({len(arr)} ok, {bin_failures} fail)")

    all_arr = np.array(all_latencies)
    overall = {
        "n": len(all_arr),
        "failures": failures,
        "failure_rate": failures / total_requests if total_requests else 0.0,
        "p50_ms": float(np.percentile(all_arr, 50)),
        "p95_ms": float(np.percentile(all_arr, 95)),
        "p99_ms": float(np.percentile(all_arr, 99)),
        "mean_ms": float(np.mean(all_arr)),
        "min_ms": float(np.min(all_arr)),
        "max_ms": float(np.max(all_arr)),
    }

    peak_rss = _peak_rss_mib()
    hw = _hardware_info()

    report = {
        "benchmark": "latency_p50_p95",
        "config": str(config_path),
        "seed": seed,
        "warmup_rounds": warmup_rounds,
        "rounds_per_clip": rounds,
        "hardware": hw,
        "peak_rss_mib": round(peak_rss, 1),
        "scorer_build_ms": round(build_ms, 1),
        "per_bin": results_per_bin,
        "overall": overall,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    print(f"\n{'='*60}")
    print(f"Overall: p50={overall['p50_ms']:.1f} ms, p95={overall['p95_ms']:.1f} ms, "
          f"p99={overall['p99_ms']:.1f} ms")
    print(f"Failures: {failures}/{total_requests} "
          f"({overall['failure_rate']*100:.1f}%)")
    print(f"Peak RSS: {peak_rss:.0f} MiB")
    print(f"Hardware: {hw.get('hostname', '?')} / {hw.get('cpu_model', hw.get('processor', '?'))}")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nJSON written to {output_path}")

    return report


if __name__ == "__main__":
    args = _parse_args()
    run(args.config, args.output)
