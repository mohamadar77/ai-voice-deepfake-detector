#!/usr/bin/env bash
# Preflight check for the ai-voice-deepfake-detector training/eval host.
# Read-only: inspects the machine, installs nothing, changes nothing.
# Verifies the host can run `uv sync --frozen --all-groups` and the CNN work.
#
# Usage:  bash preflight.sh
# Exit:   0 if no hard failures, 1 if any FAIL (WARN does not fail).

set -u

# --- Targets (edit if your project bar changes) ------------------------------
MIN_CORES=20            # logical CPUs you reserved
MIN_RAM_GB=35           # usable RAM
MIN_DISK_GB=500         # free disk on the checkout/venv filesystem
MIN_VENV_GB=15          # the torch + CUDA venv alone needs ~10-15 GB
GLIBC_MAJOR=2
GLIBC_MINOR=28          # torch wheel is manylinux_2_28 -> needs glibc >= 2.28

fail=0
pass() { printf '  \033[32mPASS\033[0m  %s\n' "$1"; }
warn() { printf '  \033[33mWARN\033[0m  %s\n' "$1"; }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; fail=1; }
hdr()  { printf '\n== %s ==\n' "$1"; }

hdr "OS / architecture"
OS="$(uname -s)"
ARCH="$(uname -m)"
if [ "$OS" = "Linux" ]; then pass "OS is Linux"; else bad "OS is $OS (need Linux)"; fi
if [ "$ARCH" = "x86_64" ]; then
  pass "Arch is x86_64 (install walls do not apply here)"
else
  bad "Arch is $ARCH (need x86_64; aarch64 also has wheels but this project targets x86_64)"
fi
if [ -r /etc/os-release ]; then
  . /etc/os-release
  printf '  info  distro: %s\n' "${PRETTY_NAME:-unknown}"
fi

hdr "glibc (torch wheel needs >= ${GLIBC_MAJOR}.${GLIBC_MINOR})"
GLIBC_VER="$(ldd --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)"
if [ -n "$GLIBC_VER" ]; then
  gmaj="${GLIBC_VER%%.*}"; gmin="${GLIBC_VER##*.}"
  if [ "$gmaj" -gt "$GLIBC_MAJOR" ] 2>/dev/null || { [ "$gmaj" -eq "$GLIBC_MAJOR" ] && [ "$gmin" -ge "$GLIBC_MINOR" ]; }; then
    pass "glibc $GLIBC_VER"
  else
    bad "glibc $GLIBC_VER is too old (torch/torchaudio wheels will refuse to install)"
  fi
else
  warn "could not detect glibc (musl? e.g. Alpine won't take the manylinux wheels)"
fi

hdr "CPU (target >= ${MIN_CORES} logical cores)"
CORES="$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 0)"
MODEL="$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | sed 's/^ *//')"
[ -n "$MODEL" ] && printf '  info  %s\n' "$MODEL"
if [ "$CORES" -ge "$MIN_CORES" ] 2>/dev/null; then
  pass "$CORES logical cores"
elif [ "$CORES" -gt 0 ] 2>/dev/null; then
  warn "$CORES logical cores (< $MIN_CORES target: CNN training will be slower, still correct)"
else
  warn "could not read core count"
fi

hdr "Memory (target >= ${MIN_RAM_GB} GB)"
RAM_KB="$(grep -m1 MemTotal /proc/meminfo 2>/dev/null | awk '{print $2}')"
if [ -n "${RAM_KB:-}" ]; then
  RAM_GB=$(( RAM_KB / 1024 / 1024 ))
  if [ "$RAM_GB" -ge "$MIN_RAM_GB" ]; then pass "${RAM_GB} GB RAM"; else bad "${RAM_GB} GB RAM (< $MIN_RAM_GB target)"; fi
else
  warn "could not read MemTotal"
fi

hdr "Disk free (target >= ${MIN_DISK_GB} GB; venv alone ~${MIN_VENV_GB} GB)"
DISK_GB="$(df -PBG . 2>/dev/null | awk 'NR==2{gsub(/G/,"",$4); print $4}')"
if [ -n "${DISK_GB:-}" ]; then
  if [ "$DISK_GB" -ge "$MIN_DISK_GB" ]; then
    pass "${DISK_GB} GB free on $(df -P . | awk 'NR==2{print $6}')"
  elif [ "$DISK_GB" -ge "$MIN_VENV_GB" ]; then
    warn "${DISK_GB} GB free (enough for the venv, short of the ${MIN_DISK_GB} GB dataset target)"
  else
    bad "${DISK_GB} GB free (not enough even for the ~${MIN_VENV_GB} GB venv)"
  fi
else
  warn "could not read df for current directory"
fi
printf '  info  /tmp is not used for the venv; avoid cloning under /tmp (it gets cleared)\n'

hdr "uv package manager (project standard; >= 0.12 known-good)"
if command -v uv >/dev/null 2>&1; then
  pass "uv present: $(uv --version 2>/dev/null)"
else
  warn "uv not found. Install (no root needed): curl -LsSf https://astral.sh/uv/install.sh | sh"
fi

hdr "Python 3.12 (project pins ==3.12.*)"
PYOK=""
for py in python3.12 python3 python; do
  if command -v "$py" >/dev/null 2>&1; then
    v="$("$py" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null)"
    [ "$v" = "3.12" ] && { pass "$py is $v"; PYOK=1; break; }
  fi
done
if [ -z "$PYOK" ]; then
  warn "no system Python 3.12 found -- fine: run 'uv python install 3.12' and uv provides it"
fi

hdr "Network to PyPI (wheels download from here)"
if command -v curl >/dev/null 2>&1; then
  code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 https://pypi.org/simple/ 2>/dev/null)"
  if [ "$code" = "200" ] || [ "$code" = "301" ] || [ "$code" = "302" ]; then
    pass "reached pypi.org (HTTP $code)"
  else
    warn "could not reach pypi.org (HTTP ${code:-none}) -- check proxy/firewall before sync"
  fi
else
  warn "curl not available to test connectivity"
fi

hdr "GPU (optional -- absence is fine, CPU-only is supported)"
if command -v nvidia-smi >/dev/null 2>&1; then
  gpu="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
  if [ -n "$gpu" ]; then pass "NVIDIA GPU detected: $gpu (CNN training will auto-use it)"; else warn "nvidia-smi present but no GPU reported"; fi
else
  printf '  info  no nvidia-smi -- CPU-only, as planned. CUDA wheels still install, just idle.\n'
fi

hdr "Audio runtime libs (needed later for decoding, not for sync)"
command -v ffmpeg >/dev/null 2>&1 && printf '  info  ffmpeg: present\n' || printf '  info  ffmpeg: absent (install later if a codec needs it)\n'
ldconfig -p 2>/dev/null | grep -q libsndfile && printf '  info  libsndfile: present\n' || printf '  info  libsndfile: not found via ldconfig (soundfile wheel usually bundles it)\n'

printf '\n========================================\n'
if [ "$fail" -eq 0 ]; then
  printf '\033[32mPREFLIGHT OK\033[0m -- no hard failures. Proceed:\n'
  printf '  uv python install 3.12\n'
  printf '  uv sync --frozen --all-groups\n'
  printf '  uv run voice-detector smoke && uv run pytest -q\n'
else
  printf '\033[31mPREFLIGHT FAILED\033[0m -- fix the FAIL item(s) above before syncing.\n'
fi
printf '========================================\n'
exit "$fail"
