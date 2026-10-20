#!/usr/bin/env bash
# Container smoke on the presentation host (issue #11). Run from the repository root:
#   VOICE_DETECTOR_BUNDLE_DIR=/path/to/cnn_augmented_seed1 scripts/docker_smoke.sh
# Builds the image, verifies the pinned bundle, starts API + UI, scores a generated tone,
# and prints image size and cold start. Uploads and the bundle are never copied into the image.
set -euo pipefail

: "${VOICE_DETECTOR_BUNDLE_DIR:?set VOICE_DETECTOR_BUNDLE_DIR}"
export VOICE_DETECTOR_BUNDLE_DIR
trap 'docker compose down --remove-orphans >/dev/null 2>&1 || true' EXIT

docker compose build
echo "image_size_bytes=$(docker image inspect voice-detector:local --format '{{.Size}}')"

echo "--- serve --check (pinned bundle, read-only mount)"
docker compose run --rm --no-deps api voice-detector serve --check

echo "--- cold start"
start=$(date +%s)
docker compose up -d --wait
echo "cold_start_seconds=$(( $(date +%s) - start ))"

echo "--- /ready, /model"
curl -fsS http://127.0.0.1:8000/ready; echo
curl -fsS http://127.0.0.1:8000/model | head -c 400; echo

echo "--- score a generated 4 s tone"
docker compose exec -T api python - <<'PY'
import io
import json
import urllib.request

import numpy as np
import soundfile as sf

rng = np.random.default_rng(0)
t = np.arange(64000) / 16000
audio = (0.3 * np.sin(2 * np.pi * 220 * t) + 0.02 * rng.standard_normal(t.size)).astype("float32")
buf = io.BytesIO()
sf.write(buf, audio, 16000, format="WAV", subtype="PCM_16")
boundary = "smoke"
body = (
    f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"tone.wav\"\r\n"
    "Content-Type: audio/wav\r\n\r\n"
).encode() + buf.getvalue() + f"\r\n--{boundary}--\r\n".encode()
req = urllib.request.Request(
    "http://127.0.0.1:8000/predict",
    data=body,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
)
print(json.dumps(json.load(urllib.request.urlopen(req)), indent=2)[:600])
PY

echo "--- read-only root and no audio retained"
docker compose exec -T api sh -c 'touch /probe 2>&1 || echo "root fs is read-only (expected)"; ls -A /tmp'

echo "--- UI health"
curl -fsS http://127.0.0.1:8501/_stcore/health; echo
docker stats --no-stream --format 'mem_usage={{.Name}} {{.MemUsage}}'
echo "SMOKE OK"
