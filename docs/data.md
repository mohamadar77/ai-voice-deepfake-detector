# Data contract and G0 feasibility record

Status: contract drafted; benchmark acquisition and audio processing are not complete.

Recorded 6 October 2026 from the current workspace:

- Local `data/` contains documentation only; no ASVspoof audio or protocol files are present.
- Workspace volume: approximately 520 GiB available at the time of inspection.
- Shell Python: 3.14.6. The project requires Python 3.12.
- uv installed: 0.12.0. Python 3.12.13 was downloaded into `/tmp/voice-detector-python` for the G0 attempt.
- `uv sync --frozen --all-groups` is currently blocked on this x86_64 macOS environment because the locked PyTorch resolution has no compatible x86_64 wheel. The environment notes describe an M4/ARM64 machine, while this execution environment reports x86_64. Treat architecture as unresolved until rechecked on the intended machine.
- No dataset, model, checkpoint or evaluation prediction was downloaded or inspected during this G0 task.

## Primary data contract

The primary study uses ASVspoof 2019 Logical Access (LA), preserving its official train, development and evaluation partitions. Acquisition must follow the published release terms and be recorded before audio processing.

The canonical manifest is one row per original clip. Local paths are resolution details and are not provenance. The manifest must contain:

| Field | Meaning | Required handling |
| --- | --- | --- |
| `clip_id` | Stable identifier from the official protocol | Required; unique within the manifest |
| `source_id` | Original source recording/group when published | Nullable; never inferred silently |
| `speaker_id` | Speaker identifier when published | Nullable; required for grouped development split when available |
| `attack_id` | Spoof attack identifier when published | Nullable for bona fide clips |
| `label` | `bonafide` or `spoof` | Required; explicit mapping from protocol |
| `official_split` | `train`, `development` or `evaluation` | Required; immutable |
| `development_role` | `select`, `threshold` or null | Required for development rows after reservation is frozen |
| `duration_seconds` | Decoded duration | Required after validation |
| `sample_rate_hz` | Decoded sample rate | Required after validation |
| `channels` | Decoded channel count | Required after validation |
| `checksum_sha256` | Audio checksum | Required before feature caching |
| `provenance` | Release/protocol/source notes | Required; preserve unknowns explicitly |
| `local_path` | Local storage path | Local-only, excluded from committed manifests |

The partition is assigned before windows, augmentation, features or caches. Every derived record carries `clip_id`, `official_split`, source/speaker identifiers when available, transformation name and seed. A window never becomes an independent sample for final clip-level metrics.

## Development reservation

Freeze development speaker membership before tuning. The proposed reservation is approximately 25% of development speakers, adjusted if the available speaker count or class/attack coverage makes that infeasible. `select` is used for features, model settings and checkpoints. `threshold` is opened only after candidate bundles are fixed and sets the operating threshold targeting at most 5% human false positives. Record exact counts and the deterministic assignment in the manifest/run record.

## G0 exit evidence

G0 is complete when the intended machine passes the Python 3.12/uv check, the locked dependency environment is installable, access/terms/storage are recorded, and the manifest/schema fixtures validate official protocol rows. The current x86_64 PyTorch wheel mismatch is an open blocker for this execution environment. It does not justify changing the lockfile or downloading an alternate dataset.

## Artifact policy

Audio, extracted features, checkpoints, prediction files, SQLite catalogs and uploaded recordings remain ignored/local. Commit the schema, small approved summaries, checksums without restricted paths, and reproducibility metadata. Do not commit credentials or `.env` files.
