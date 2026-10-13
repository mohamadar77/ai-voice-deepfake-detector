# Data contract and dataset provenance

Status: the ASVspoof 2019 LA benchmark is acquired and integrity-verified on the
training host (11 October 2026); the G0 environment and data gates have passed.
Audio processing (manifest build, feature extraction) begins in G1.

> The 6 October 2026 entries in this file recorded the G0 feasibility *attempt* on
> the macOS planning machine, where the locked environment was not installable.
> That record is superseded and preserved at the end under **History**. See also
> `docs/environment.md`, which was reconciled the same way.

## Dataset identity

- Database: **ASVspoof 2019** — the 3rd Automatic Speaker Verification Spoofing and
  Countermeasures Challenge database.
- Partition in use: **Logical Access (LA)** only. Physical Access (PA) is out of
  scope (see Exclusions).
- Audio: FLAC, **16 kHz, 16-bit**, single channel. File-name prefixes `LA_T_*`
  (train), `LA_D_*` (development), `LA_E_*` (evaluation).

## Source and release

- Official release: ASVspoof 2019 database, © 2019 The Centre for Speech Technology
  Research (CSTR), University of Edinburgh, published via the University of Edinburgh
  DataShare — handle <https://datashare.ed.ac.uk/handle/10283/3336>.
- Required citation (per the bundled `README.txt`): Todisco, Wang, Sahidullah,
  Delgado, Nautsch, Yamagishi, Evans, Kinnunen and Lee, *ASVspoof 2019: Future
  Horizons in Spoofed and Fake Audio Detection*, Proc. Interspeech 2019. This must be
  cited in any published study or result on the corpus (model card, slides, report).
- Immediate acquisition (what was actually downloaded on the host): a bundled
  `dataset.zip` containing `LA.zip`, `PA.zip`, the challenge PDFs, and
  `LICENSE_text.txt`. This is a redistributed bundle/mirror — the official DataShare
  release ships LA and PA as separate archives.
  - Download URL: _owner to record the exact mirror URL here._
- Authoritative fingerprints of the exact bytes acquired (SHA-256):
  - `dataset.zip`: `397f5dc2fdd261ecd6555975843a8b393b4f4c7278449c380d42519bb9ccca7b`
  - `LA.zip`: `208a7e4e3913f8c75ae1afd19bf32a5b29ae68435e9e30e23e5e98b6a155e4ec`

  The SHA-256 values are the authoritative identifier of what was used; the URL is a
  convenience pointer. Anyone reproducing the work should match these hashes.

## License and attribution obligations

- License: **Open Data Commons Attribution License (ODC-By) v1.0** (per the bundled
  `LICENSE_text.txt`). Worldwide, royalty-free, non-exclusive; commercial use
  permitted; derivative and collective databases permitted; database provided "as
  is" with no warranty.
- The one condition that affects our deliverables is **attribution**. If we publicly
  convey the database/a derivative, or publicly use a "Produced Work" (our results,
  model card, slides, or demo), we must carry an attribution notice. Notice text to
  use in the model card, slides, and demo (G4):

  > Contains information from the ASVspoof 2019 database
  > (<https://datashare.ed.ac.uk/handle/10283/3336>), made available under the ODC
  > Attribution License (<https://opendatacommons.org/licenses/by/1-0/>).

- ODC-By §2.4: the license governs rights in the **database**; rights in the
  individual audio contents may differ. Use stays within research/evaluation under
  the official protocol.

## Integrity verification (11 October 2026)

Verified with `uv run voice-detector protocol-audit` on each official CM protocol
file (reads the label columns only; decodes no audio). Every count matches the
published ASVspoof 2019 LA figures exactly:

| Official split | Protocol file | bonafide | spoof | total | attacks | speakers |
| --- | --- | --- | --- | --- | --- | --- |
| train | `ASVspoof2019.LA.cm.train.trn.txt` | 2,580 | 22,800 | 25,380 | A01–A06 (3,800 each) | 20 |
| development | `ASVspoof2019.LA.cm.dev.trl.txt` | 2,548 | 22,296 | 24,844 | A01–A06 (3,716 each) | 20 |
| evaluation | `ASVspoof2019.LA.cm.eval.trl.txt` | 7,355 | 63,882 | 71,237 | A07–A19 (4,914 each) | 67 |

Consequences for evaluation design (all already required by `AGENTS.md`):

- Train and development share attacks **A01–A06**. Evaluation uses attack IDs
  **A07–A19**, which are disjoint *as IDs* but not entirely as algorithms: **A16 and
  A19 reuse the A04 and A06 algorithms** from training (Wang et al., 2020, *Computer
  Speech & Language*; confirm against the challenge PDFs bundled in `dataset.zip`).
  The unseen-attack headline is therefore A07–A15, A17 and A18. A16/A19 are reported
  as a known-algorithm control on unseen speakers, alongside development A01–A06
  (corrected 13 October 2026; see #35). Do not present known-attack numbers as the
  headline result.
- Class imbalance ≈ **1 : 8.8** (bonafide : spoof). Lead with EER, macro-F1 and
  balanced accuracy; consider class weighting. Never headline raw accuracy.
- Per-split speaker counts 20 / 20 / 67. Cross-split speaker disjointness (a known LA
  property) is asserted by the G1 manifest builder before the speaker-grouped
  development reservation is frozen.

## Local layout

Extracted to `data/raw/LA/` in the working tree (git-ignored via `data/**`).
Contents: `ASVspoof2019_LA_train`, `ASVspoof2019_LA_dev`, `ASVspoof2019_LA_eval`,
`ASVspoof2019_LA_cm_protocols`, `ASVspoof2019_LA_asv_protocols`,
`ASVspoof2019_LA_asv_scores`, `README.LA.txt`.

## Exclusions and transformations

- **Physical Access (PA) excluded**: `PA.zip` was deleted and never extracted; PA is
  out of scope for this project.
- **ASV task files unused**: `ASVspoof2019_LA_asv_protocols` and
  `ASVspoof2019_LA_asv_scores` are for the speaker-verification (ASV) task, not the
  countermeasure (spoof-detection) task. The CM pipeline uses only
  `ASVspoof2019_LA_cm_protocols`. Do not wire ASV scores into CM evaluation.
- **No audio transformation at acquisition**: files are the original 16 kHz / 16-bit
  FLAC as distributed. Any windowing, augmentation, or feature extraction is applied
  downstream, after split assignment, and recorded per experiment.

## Primary data contract

The primary study uses ASVspoof 2019 Logical Access (LA), preserving its official
train, development and evaluation partitions. Acquisition follows the published
release terms (recorded above) and precedes audio processing.

The canonical manifest is one row per original clip. Local paths are resolution
details and are not provenance. The manifest must contain:

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

The partition is assigned before windows, augmentation, features or caches. Every
derived record carries `clip_id`, `official_split`, source/speaker identifiers when
available, transformation name and seed. A window never becomes an independent
sample for final clip-level metrics.

## Development reservation

Freeze development speaker membership before tuning. The proposed reservation is
approximately 25% of development speakers, adjusted if the available speaker count or
class/attack coverage makes that infeasible. `select` is used for features, model
settings and checkpoints. `threshold` is opened only after candidate bundles are
fixed and sets the operating threshold targeting at most 5% human false positives.
Record exact counts and the deterministic assignment in the manifest/run record.

## G0 exit evidence

G0 is complete when the training host passes the Python 3.12 / uv check, the locked
dependency environment installs, dataset access/terms/storage/checksums are recorded,
and the protocol parser validates the official rows.

As of 11 October 2026 all four hold:

- **Environment:** `uv sync --frozen --all-groups`, `uv run voice-detector smoke`,
  `uv run ruff check .` and `uv run pytest` pass on the HPE DL120 Gen10 host (see
  `docs/environment.md`).
- **Access / terms / storage:** recorded above — ODC-By 1.0, SHA-256 checksums, and
  the LA-only footprint fits within the host's ~209 GB.
- **Protocol validation:** the three CM protocols audit to the published counts in
  the table above, with bonafide rows carrying no attack and spoof rows carrying a
  system ID.

The earlier x86_64 macOS wheel mismatch is resolved by using the reserved Linux
x86_64 host; it required no lockfile or dataset change.

## Artifact policy

Audio, extracted features, checkpoints, prediction files, SQLite catalogs and
uploaded recordings remain ignored/local. Commit the schema, small approved
summaries, checksums without restricted paths, and reproducibility metadata. Do not
commit credentials or `.env` files.

## History — G0 feasibility attempt (6 October 2026, superseded)

The following was recorded on 6 October 2026 on the macOS planning machine, before a
Linux x86_64 host was reserved. It is kept for provenance; the blocker it describes
is resolved (see `docs/environment.md` and the G0 exit evidence above).

- Local `data/` contained documentation only; no ASVspoof audio or protocol files
  were present.
- Workspace volume: approximately 520 GiB available at the time of inspection.
- Shell Python: 3.14.6. The project requires Python 3.12.
- uv installed: 0.12.0. Python 3.12.13 was downloaded into
  `/tmp/voice-detector-python` for the G0 attempt.
- `uv sync --frozen --all-groups` was blocked on that x86_64 macOS environment
  because the locked PyTorch resolution has no compatible x86_64 macOS wheel. Treated
  as unresolved until rechecked on the intended machine — which it now has been.
- No dataset, model, checkpoint or evaluation prediction was downloaded or inspected
  during that G0 task.
