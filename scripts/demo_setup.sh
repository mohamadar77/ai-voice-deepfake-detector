#!/usr/bin/env bash
# Demo setup for the 5 Nov presentation.
# Run on the presenting laptop with Docker Desktop. Its CPU architecture is not
# recorded yet (issue #108): step 1 prints the Docker engine's architecture and
# warns when it is ARM64.
#
# Prerequisites:
#   - Docker Desktop running
#   - SSH access to the dev host (hp-xl420gen9-01)
#   - This repo checked out and up to date (git pull origin main)
#
# The script is meant to be run step-by-step, not all at once.
# Read each section, then copy-paste the commands.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEV_HOST="root@hp-xl420gen9-01"
DEV_REPO="/root/ai-voice-deepfake-detector"
BUNDLE_NAME="cnn_augmented_seed1"
# Files that make up a CNN bundle (src/voice_detector/cnn_scoring.py). The service
# additionally verifies the whole-directory hash against the pin in configs/serving.yaml.
BUNDLE_FILES=(cnn_run_record.json cnn_checkpoint.pt cnn_norm_stats.pt)

cd "$REPO_ROOT"

bundle_complete() {
  local name
  for name in "${BUNDLE_FILES[@]}"; do
    [ -f "artifacts/$BUNDLE_NAME/$name" ] || return 1
  done
}

cat <<'BANNER'
╔══════════════════════════════════════════════════════════════╗
║           AI Voice Deepfake Detector — Demo Setup           ║
║                   Presentation 5 Nov 2026                   ║
╚══════════════════════════════════════════════════════════════╝
BANNER

# ─────────────────────────────────────────────────────────────
# STEP 1: Verify Docker Desktop is running
# ─────────────────────────────────────────────────────────────
echo "=== Step 1: Checking Docker Desktop ==="
if ! docker info >/dev/null 2>&1; then
  echo "ERROR: Docker daemon is not running."
  echo "  → Open Docker Desktop and wait until it says 'Engine running'."
  exit 1
fi
echo "Docker is running."
echo "Engine OS/Arch: $(docker info --format '{{.OSType}}/{{.Architecture}}')"
case "$(docker info --format '{{.Architecture}}')" in
  aarch64|arm64)
    echo "NOTE: ARM64 engine. Step 4 builds a native ARM64 image whose torch wheel is a"
    echo "  different binary from the linux/amd64 one used for the locked evaluation and"
    echo "  the recorded container parity. Before relying on its scores, run"
    echo "  scripts/container_parity.py against this container (issue #108)."
    ;;
esac
echo

# ─────────────────────────────────────────────────────────────
# STEP 2: Copy model bundle from the dev host
# ─────────────────────────────────────────────────────────────
echo "=== Step 2: Copying model bundle from dev host ==="
if bundle_complete; then
  echo "Bundle already present at artifacts/$BUNDLE_NAME — skipping."
elif [ -e "artifacts/$BUNDLE_NAME" ]; then
  echo "ERROR: artifacts/$BUNDLE_NAME exists but is missing one of: ${BUNDLE_FILES[*]}"
  echo "  Inspect it, remove it yourself if it is stale, and re-run. Nothing was deleted."
  exit 1
else
  echo "Downloading bundle from $DEV_HOST..."
  mkdir -p artifacts
  scp -r "$DEV_HOST:$DEV_REPO/artifacts/$BUNDLE_NAME" artifacts/
  echo "Bundle copied."
fi
if ! bundle_complete; then
  echo "ERROR: artifacts/$BUNDLE_NAME is missing one of: ${BUNDLE_FILES[*]}"
  exit 1
fi
echo "Bundle contents:"
ls -lh "artifacts/$BUNDLE_NAME/"
echo

# ─────────────────────────────────────────────────────────────
# STEP 3: Copy demo clips from the dev host
# ─────────────────────────────────────────────────────────────
# demo_clips/ is produced by scripts/generate_demo_clips.py (synthetic, too-short,
# MP3 and digital-silence inputs) plus the presenter's own recording
# (human_speech.wav) and any other clips from the predeclared demo set.
echo "=== Step 3: Copying demo clips from dev host ==="
if [ -d "demo_clips" ] && [ -f "demo_clips/human_speech.wav" ]; then
  echo "Demo clips already present — skipping."
else
  echo "Downloading demo clips from $DEV_HOST..."
  scp -r "$DEV_HOST:$DEV_REPO/demo_clips" .
  echo "Demo clips copied."
fi
echo "Demo clips:"
ls -lh demo_clips/
echo

# ─────────────────────────────────────────────────────────────
# STEP 4: Build the Docker image (native to this engine's architecture)
# ─────────────────────────────────────────────────────────────
echo "=== Step 4: Building Docker image ==="
echo "The first build downloads the CPU torch wheel for this architecture and is slow"
echo "(about 25 minutes was measured on hp-xl420gen9-01)."
VOICE_DETECTOR_BUNDLE_DIR="artifacts/$BUNDLE_NAME" docker compose build
echo "Build complete."
docker images --format '{{.Repository}}:{{.Tag}}  {{.Size}}' | grep voice || true
echo

# ─────────────────────────────────────────────────────────────
# STEP 5: Start the containers
# ─────────────────────────────────────────────────────────────
# --wait blocks until the compose health checks pass. The API check is /ready: the
# pinned bundle hash and threshold were verified and the warm-up scoring succeeded.
echo "=== Step 5: Starting containers ==="
if ! VOICE_DETECTOR_BUNDLE_DIR="artifacts/$BUNDLE_NAME" docker compose up -d --wait --wait-timeout 300; then
  echo "ERROR: the containers did not become healthy. Check logs:"
  echo "  docker compose logs api"
  exit 1
fi
echo

# ─────────────────────────────────────────────────────────────
# STEP 6: Readiness check
# ─────────────────────────────────────────────────────────────
echo "=== Step 6: Readiness check ==="
if curl -sf http://127.0.0.1:8000/ready >/dev/null; then
  echo "API is ready:"
  curl -s http://127.0.0.1:8000/ready
  echo
  echo "Model info (bundle hash and threshold should match configs/serving.yaml):"
  curl -s http://127.0.0.1:8000/model | python3 -c "import sys,json; r=json.load(sys.stdin); print(f'  model_id={r[\"model_id\"]}  threshold={r[\"threshold\"]}  bundle_hash={r[\"bundle_hash\"]}')" || true
else
  echo "ERROR: the API is not ready. Check logs:"
  echo "  docker compose logs api"
  exit 1
fi
echo

# ─────────────────────────────────────────────────────────────
# STEP 7: Smoke test every demo clip
# ─────────────────────────────────────────────────────────────
# Every WAV/FLAC/MP3 in demo_clips/ is sent, so no clip is skipped silently. The
# output is whatever the service returns; nothing here assumes a decision.
echo "=== Step 7: Smoke testing demo clips ==="
echo

shopt -s nullglob
clips=(demo_clips/*.wav demo_clips/*.flac demo_clips/*.mp3)
shopt -u nullglob
if [ "${#clips[@]}" -eq 0 ]; then
  echo "ERROR: no .wav, .flac or .mp3 files in demo_clips/"
  exit 1
fi

for f in "${clips[@]}"; do
  name="$(basename "$f")"
  echo "--- $name ---"
  if ! response=$(curl -s -w "\n%{http_code}" -F "file=@$f" http://127.0.0.1:8000/predict); then
    echo "  curl failed (is the API reachable on 127.0.0.1:8000?)"
    echo
    continue
  fi
  http_code=$(echo "$response" | tail -1)
  body=$(echo "$response" | sed '$d')

  if [ "$http_code" = "200" ]; then
    score=$(echo "$body" | python3 -c "import sys,json; r=json.load(sys.stdin); print(f'score={r[\"score\"]:.4f}  decision={r[\"decision\"]}')" 2>/dev/null || echo "$body")
    echo "  HTTP 200 — $score"
  else
    detail=$(echo "$body" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r.get('code','?'))" 2>/dev/null || echo "$body")
    echo "  HTTP $http_code — $detail"
  fi
  echo
done

# ─────────────────────────────────────────────────────────────
# STEP 8: Verify Streamlit UI
# ─────────────────────────────────────────────────────────────
echo "=== Step 8: Open Streamlit UI ==="
echo "  → http://localhost:8501"
echo
if command -v open >/dev/null 2>&1; then
  open http://localhost:8501
fi

cat <<'SUMMARY'
Demo setup complete.

  Streamlit UI:  http://localhost:8501
  FastAPI docs:  http://localhost:8000/docs
  Model info:    http://localhost:8000/model

Demo flow (slide 10). Scored clips: show the score and decision the service
returns, exactly as shown. Do not promise an outcome in advance:
  - flagged_spoof_like is not "fake"; not_flagged is not "real".
  - A real human recording can be flagged spoof-like (issue #103: both of the
    owner's two phone recordings were). Report n, show misses, and say the demo
    set is illustrative.
Inputs that must be rejected (tested behaviour of the service):
  - too_short.wav (under 1 s)        → rejected, duration
  - invalid.mp3                      → rejected, format
  - digital_silence.wav              → rejected, silence
Then show /model: bundle hash, threshold, limitations.

To stop:  docker compose down
To restart:
  VOICE_DETECTOR_BUNDLE_DIR=artifacts/cnn_augmented_seed1 docker compose up -d --wait
SUMMARY
