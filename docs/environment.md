# Development environment

Originally recorded 4 October 2026 (planning machine). **Reconciled 11 October
2026** when the reserved Linux training host was stood up and the G0 environment
gate passed on it. The earlier macOS/M4 record is preserved below as history, not
silently overwritten: the M4 Pro was the planning reference, not the machine the
environment was ever installed on.

## Verified training/evaluation host (reconciled 11 October 2026)

This is the machine where `uv sync --frozen --all-groups` installs cleanly and the
smoke/lint/test gate passes. It is the recorded compute target for all training,
validation and locked evaluation.

- Host: HPE ProLiant DL120 Gen10 (`hpe-dl120gen10-01`)
- OS: Fedora Linux 44, `x86_64`, glibc 2.43
- Processor: Intel Xeon Gold 6152 @ 2.10 GHz — **44 logical cores**
- Memory: 30 GB RAM
- Accelerator: **none (CPU-only)**; no NVIDIA GPU present
- Local storage: ~209 GB free at setup
- Python: 3.12 (provided via `uv python install 3.12`)
- Environment/dependency manager: uv 0.13.0
- Dependency source of truth: `pyproject.toml` and `uv.lock` (unchanged — the
  locked Linux `x86_64` wheels, including CPU-capable CUDA `torch`, install here)

**G0 environment gate — PASSED (11 October 2026):** `uv sync --frozen --all-groups`
completed, `uv run voice-detector smoke` reported runtime/dependency versions, and
`uv run ruff check .` / `uv run pytest -q` were green on this host.

### Capacity notes against the original reservation targets

The reservation targeted ≥20 cores, ≥35 GB RAM and ≥500 GB disk. The host clears
cores comfortably; RAM (30 GB) and free disk (209 GB) are below the generous
targets but sufficient for the **ASVspoof 2019 LA-only, CPU** scope:

- RAM: features are disk-cached and loaded per batch, so the dataset is never held
  whole in memory; the MFCC/LR feature set is only a few hundred MB and the compact
  CNN has few parameters. Cap DataLoader workers and batch size in YAML. Swap is
  recommended as OOM insurance.
- Disk: LA-only footprint (dataset ~20 GB + venv ~15 GB + log-Mel cache incl. one
  augmented copy ~40–60 GB + small checkpoints) fits within ~209 GB. The unused
  Physical Access (PA) archive is out of scope and should be removed to reclaim
  space.
- Compute: CPU-only is the planned mode. A compact CNN run (~20 epochs on ~25k
  train clips) is estimated at a few hours on 44 threads; the clean + augmented ×
  2-seed matrix is an overnight unattended job. The first real training pilot must
  record actual runtime, peak memory, batch size and completed epochs/steps.

## History — planning machine (recorded 4 October 2026, superseded)

The original setup record described an Apple Silicon laptop that was never the
install target; the project's locked PyTorch resolution has no macOS `x86_64` wheel,
and work was in fact attempted on an Intel macOS machine where the install failed.
Retained for provenance only:

- Machine: MacBook Pro Mac16,8
- Processor: Apple M4 Pro, 12 CPU cores (8 performance and 4 efficiency)
- Memory: 48 GB unified memory
- Accelerator: integrated Apple M4 Pro GPU, 16 cores, Metal support (MPS reported
  unavailable in that execution environment)
- Local storage at setup: 926 GiB total, approximately 540 GiB available

## Local storage configuration

The default paths are `data/` and `artifacts/`. Set `VOICE_DETECTOR_DATA_DIR` and
`VOICE_DETECTOR_ARTIFACTS_DIR` to place large files outside the checkout.
Configuration files must remain credential-free.

## Compatibility check

Repository setup is accepted only after the locked environment installs, the package
imports, the smoke command imports every major runtime dependency, and lint and unit
tests pass. As of 11 October 2026 this holds on the verified host above. This does
not validate data decoding, training runtime or model accuracy, which are recorded
separately as they are measured.
