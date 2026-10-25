#!/usr/bin/env bash
# Demo setup for the 5 Nov presentation.
# Run on the presentation Mac (M4 Pro, ARM64, Docker Desktop).
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

cd "$REPO_ROOT"

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
docker info --format 'OS/Arch: {{.OSType}}/{{.Architecture}}'
echo

# ─────────────────────────────────────────────────────────────
# STEP 2: Copy model bundle from the dev host
# ─────────────────────────────────────────────────────────────
echo "=== Step 2: Copying model bundle from dev host ==="
if [ -d "artifacts/$BUNDLE_NAME" ] && [ -f "artifacts/$BUNDLE_NAME/model.pt" ]; then
  echo "Bundle already present at artifacts/$BUNDLE_NAME — skipping."
else
  echo "Downloading bundle from $DEV_HOST..."
  mkdir -p artifacts
  scp -r "$DEV_HOST:$DEV_REPO/artifacts/$BUNDLE_NAME" artifacts/
  echo "Bundle copied."
fi
echo "Bundle contents:"
ls -lh "artifacts/$BUNDLE_NAME/"
echo

# ─────────────────────────────────────────────────────────────
# STEP 3: Copy demo clips from the dev host
# ─────────────────────────────────────────────────────────────
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
# STEP 4: Build the Docker image (ARM64 native)
# ─────────────────────────────────────────────────────────────
echo "=== Step 4: Building Docker image ==="
echo "This may take 5-10 minutes on first build (downloading ARM64 torch wheels)."
VOICE_DETECTOR_BUNDLE_DIR="artifacts/$BUNDLE_NAME" docker compose build
echo "Build complete."
docker images --format '{{.Repository}}:{{.Tag}}  {{.Size}}' | grep voice || true
echo

# ─────────────────────────────────────────────────────────────
# STEP 5: Start the containers
# ─────────────────────────────────────────────────────────────
echo "=== Step 5: Starting containers ==="
VOICE_DETECTOR_BUNDLE_DIR="artifacts/$BUNDLE_NAME" docker compose up -d
echo "Waiting 60 seconds for model warmup..."
sleep 60
echo

# ─────────────────────────────────────────────────────────────
# STEP 6: Health check
# ─────────────────────────────────────────────────────────────
echo "=== Step 6: Health check ==="
if curl -sf http://127.0.0.1:8000/health >/dev/null; then
  echo "API is healthy:"
  curl -s http://127.0.0.1:8000/health
  echo
else
  echo "ERROR: API is not responding. Check logs:"
  echo "  docker compose logs api"
  exit 1
fi
echo

# ─────────────────────────────────────────────────────────────
# STEP 7: Smoke test all demo clips
# ─────────────────────────────────────────────────────────────
echo "=== Step 7: Smoke testing demo clips ==="
echo

for f in demo_clips/human_speech.wav demo_clips/spoof_a01.flac demo_clips/too_short.wav demo_clips/invalid.mp3 demo_clips/digital_silence.wav; do
  if [ ! -f "$f" ]; then
    echo "SKIP: $f not found"
    echo
    continue
  fi
  name="$(basename "$f")"
  echo "--- $name ---"
  response=$(curl -s -w "\n%{http_code}" -F "file=@$f" http://127.0.0.1:8000/predict)
  http_code=$(echo "$response" | tail -1)
  body=$(echo "$response" | sed '$d')

  if [ "$http_code" = "200" ]; then
    score=$(echo "$body" | python3 -c "import sys,json; r=json.load(sys.stdin); print(f'score={r[\"score\"]:.4f}  decision={r[\"decision\"]}')")
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
╔══════════════════════════════════════════════════════════════╗
║                    Demo setup complete!                     ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Streamlit UI:  http://localhost:8501                        ║
║  FastAPI docs:  http://localhost:8000/docs                   ║
║  Model info:    http://localhost:8000/model                  ║
║                                                              ║
║  Demo flow (slide 10):                                      ║
║    1. Upload human_speech.wav    → not_flagged               ║
║    2. Upload spoof_a01.flac     → flagged_spoof_like         ║
║    3. Upload too_short.wav      → rejected (duration)        ║
║    4. Upload invalid.mp3        → rejected (format)          ║
║    5. Upload digital_silence.wav → rejected (silence)        ║
║    6. Show /model endpoint      → bundle hash, limitations  ║
║                                                              ║
║  To stop:  docker compose down                              ║
║  To restart:                                                 ║
║    VOICE_DETECTOR_BUNDLE_DIR=artifacts/cnn_augmented_seed1  ║
║    docker compose up -d                                      ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
SUMMARY
