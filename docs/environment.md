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

**Role (owner, 13 October 2026):** this one machine is used for **development,
training, evaluation and the demo/presentation**. Development and the full test
suite run here natively. The Intel Mac isolated-env workaround is optional.

- The owner states 32 GB RAM and a 270 GB disk. Setup measured 30 GB usable RAM and
  about 209 GB free. These are consistent: installed versus usable memory, total
  versus free disk.
- Re-measure with `bash scripts/preflight.sh` and `df -h` before the CNN runs and
  before G4.

**Access (owner, 13 October 2026):** the host is reached **remotely over SSH from the
owner's laptop** for all work, including the presentation demo. The host is not
moved. Consequences:

- **Long jobs run detached.** Feature extraction, CNN training and locked scoring
  run under `tmux` (or `systemd-run --user`) with logs written to files, so a dropped
  SSH session never kills a run.
- **Long jobs are resumable.** CNN training checkpoints every epoch, including
  optimizer and RNG state. Cache writes are atomic (write to a temp file, then
  rename). A resumed run records that it resumed.
- **No public ports.** API and UI bind to `localhost` on the host. The laptop reaches
  them through an SSH tunnel, for example `ssh -L 8501:localhost:8501 -L 8000:localhost:8000 <host>`.
  Uploads therefore never cross an exposed endpoint.
- **Offline demo fallback.** A remote demo depends on the venue network, while G5
  requires a demo without network dependence. The fallback: `docker save` the
  presentation image on the host, `docker load` it on the laptop (Intel Mac with
  Docker Desktop runs the `linux/amd64` image natively), and copy the frozen bundle
  (mounted read-only) and the permitted demo clips. Verify it offline on the laptop
  in G4, and check that its scores match the host's. The backup recording remains
  the last resort.
- **Latency.** p50/p95 are measured server-side on the host (named hardware). Tunnel
  round-trip time is a separate demo-experience number, never mixed in.

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
- **Phase 1 disk budget (13 October 2026):**

  | Item | Size |
  | --- | --- |
  | LA data | ~20 GB |
  | venv | ~15 GB |
  | Docker image with the CUDA `torch` build, plus build cache | ~10–15 GB |
  | MFCC features | <1 GB |
  | log-Mel train/dev cache | ~4–8 GB (float32) |
  | Checkpoints and predictions | <1 GB |
  | **Total** | **~50–60 GB** |

  Three rules keep it there:
  - evaluation robustness and silence-trim conditions are computed **on the fly** at scoring time and never materialized as transformed audio copies, which would cost about 7 GB per condition;
  - training augmentation is sampled per epoch and never cached;
  - keep at least 30 GB free.

  Record actual cache sizes in #7 and #33.
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
