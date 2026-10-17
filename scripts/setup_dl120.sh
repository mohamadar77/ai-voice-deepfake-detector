#!/usr/bin/env bash
# DL120 host setup — run once after cloning or pulling a new branch.
# Installs system dependencies and creates data symlinks.
#
# Usage:  bash scripts/setup_dl120.sh
# Requires: root (for dnf install)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- System dependencies -----------------------------------------------------

echo "==> Installing system dependencies..."
if command -v dnf >/dev/null 2>&1; then
    dnf install -y ffmpeg libsndfile
elif command -v apt-get >/dev/null 2>&1; then
    apt-get update -qq
    apt-get install -y ffmpeg libsndfile1
else
    echo "WARN: unknown package manager — install ffmpeg and libsndfile manually"
fi

command -v ffmpeg >/dev/null 2>&1 && echo "  OK  ffmpeg: $(ffmpeg -version 2>&1 | head -1)" || echo "  FAIL  ffmpeg not found after install"

# --- Data symlinks ------------------------------------------------------------

echo "==> Creating ASVspoof data symlinks in project root..."

DATA_ROOT="/root/data/raw/LA"

declare -A LINKS=(
    ["ASVspoof2019_LA_train"]="$DATA_ROOT/ASVspoof2019_LA_train"
    ["ASVspoof2019_LA_dev"]="$DATA_ROOT/ASVspoof2019_LA_dev"
    ["ASVspoof2019_LA_eval"]="$DATA_ROOT/ASVspoof2019_LA_eval"
)

for name in "${!LINKS[@]}"; do
    target="${LINKS[$name]}"
    link="$PROJECT_ROOT/$name"
    if [ -L "$link" ]; then
        echo "  SKIP  $name (symlink already exists)"
    elif [ -e "$link" ]; then
        echo "  SKIP  $name (path exists and is not a symlink)"
    elif [ ! -d "$target" ]; then
        echo "  WARN  $name target $target does not exist"
    else
        ln -s "$target" "$link"
        echo "  OK    $name -> $target"
    fi
done

# --- Python environment ------------------------------------------------------

echo "==> Syncing Python environment..."
cd "$PROJECT_ROOT"
uv sync --frozen --all-groups

echo ""
echo "==> Setup complete. Run training with:"
echo "  uv run voice-detector train-cnn --augmented --seed 20261004 --run-label augmented_seed1"
echo "  uv run voice-detector train-cnn --augmented --seed 20261005 --run-label augmented_seed2"
