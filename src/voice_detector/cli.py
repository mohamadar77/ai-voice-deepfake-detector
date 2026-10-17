"""Command-line utilities for repository and environment validation."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from voice_detector import __version__
from voice_detector.config import load_config
from voice_detector.manifest import (
    VALID_SPLITS,
    build_catalog_meta,
    build_la_manifest,
    compute_manifest_hash,
    git_commit,
    manifest_summary,
    package_version,
    read_manifest_sqlite,
    write_manifest_csv,
    write_manifest_sqlite,
)
from voice_detector.protocol import parse_protocol_file, summarize_protocol
from voice_detector.reservation import (
    RESERVATION_SCHEMA_VERSION,
    build_reservation_record,
    load_reservation_spec,
    plan_reservation,
    read_frozen_record,
    verify_frozen_reservation,
)

_REQUIRED_IMPORTS = {
    "fastapi": "fastapi",
    "librosa": "librosa",
    "numpy": "numpy",
    "pandas": "pandas",
    "scikit-learn": "sklearn",
    "soundfile": "soundfile",
    "streamlit": "streamlit",
    "torch": "torch",
    "torchaudio": "torchaudio",
}


def _dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution, module in _REQUIRED_IMPORTS.items():
        importlib.import_module(module)
        versions[distribution] = importlib.metadata.version(distribution)
    return versions


def _smoke(config_path: Path) -> int:
    config = load_config(config_path)
    payload = {
        "status": "ok",
        "package": "voice_detector",
        "package_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "project": config.name,
        "seed": config.seed,
        "paths": {
            "data_root": str(config.paths.data_root),
            "artifacts_root": str(config.paths.artifacts_root),
        },
        "dependencies": _dependency_versions(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _protocol_audit(protocol_path: Path, official_split: str) -> int:
    entries = parse_protocol_file(protocol_path, official_split)
    summary = summarize_protocol(entries)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _build_manifest(
    la_root: Path,
    config_path: Path,
    out_dir: Path | None,
    *,
    dry_run: bool,
    allow_missing: bool,
    enforce_counts: bool,
) -> int:
    rows, summary = build_la_manifest(
        la_root,
        with_checksum=not dry_run,
        allow_missing=allow_missing,
        enforce_expected_counts=enforce_counts,
    )
    summary_json = json.dumps(summary, indent=2, sort_keys=True)
    if dry_run:
        print(summary_json)
        return 0

    if out_dir is None:
        out_dir = load_config(config_path).paths.artifacts_root / "manifest"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = write_manifest_csv(rows, out_dir / "manifest.csv")
    meta = build_catalog_meta(summary, summary_json, with_checksum=True)
    db_path = write_manifest_sqlite(rows, out_dir / "manifest.sqlite", meta=meta)
    sidecar = out_dir / "manifest_summary.json"
    sidecar.write_text(summary_json, encoding="utf-8")

    print(
        json.dumps(
            {
                "manifest_hash": summary["manifest_hash"],
                "total_clips": summary["total_clips"],
                "outputs": {
                    "csv": str(csv_path),
                    "sqlite": str(db_path),
                    "summary_json": str(sidecar),
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0



def _atomic_write_text(path: Path, text: str) -> Path:
    """Write ``text`` to ``path`` via a temporary file and one rename.

    The manifest costs hours to rebuild (121k SHA-256 hashes), so a failed
    rewrite must never leave a truncated catalog behind.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)
    return path


def _refuse(reason: str, **fields: Any) -> int:
    """Print a refusal as JSON on stdout and return a non-zero exit code.

    Refusing is a normal outcome for a freeze, not a crash: the operator needs
    the reason and its evidence as data they can read and record, so every
    guard reports this way rather than raising.
    """

    payload = {"status": "refused", "reason": reason, **fields}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 1


def _reserve_development(
    manifest_dir: Path | None,
    config_path: Path,
    reservation_config: Path,
    record_out: Path,
    *,
    note: str | None,
    dry_run: bool,
    force: bool,
) -> int:
    """Freeze the speaker-grouped development reservation (#31).

    Reads the built manifest catalog, derives the deterministic membership,
    refuses to write unless coverage holds, then rewrites the manifest with the
    ``development_role`` column and writes the committed frozen record.

    Two independent guards must agree before an existing membership can move:
    the committed record at ``record_out``, which is authoritative and is the
    only one that survives a manifest rebuild, and the manifest's own role
    column. Either disagreement refuses without ``--force``, because moving a
    protocol boundary that tuning may already depend on cannot be undone.
    """

    spec = load_reservation_spec(reservation_config)
    if manifest_dir is None:
        manifest_dir = load_config(config_path).paths.artifacts_root / "manifest"
    catalog_path = manifest_dir / "manifest.sqlite"

    rows, meta = read_manifest_sqlite(catalog_path)
    if not rows:
        return _refuse(
            "empty_catalog",
            catalog=str(catalog_path),
            hint="this catalog holds no clips; rebuild it with build-manifest",
        )

    recorded_hash = meta.get("manifest_hash")
    actual_hash = compute_manifest_hash(rows)
    if not recorded_hash:
        # An absent catalog_meta means the drift check has nothing to compare
        # against, so proceeding would freeze against an unverifiable catalog.
        if not force:
            return _refuse(
                "catalog_provenance_missing",
                catalog=str(catalog_path),
                recomputed_manifest_hash=actual_hash,
                hint=(
                    "this catalog carries no recorded manifest_hash, so neither its build "
                    "commit nor its checksum coverage can be verified. Rebuild it with "
                    "build-manifest, or pass --force to freeze against an unverified catalog"
                ),
            )
    elif recorded_hash != actual_hash:
        return _refuse(
            "catalog_hash_mismatch",
            catalog=str(catalog_path),
            recorded_manifest_hash=recorded_hash,
            recomputed_manifest_hash=actual_hash,
            hint="rebuild the manifest; this catalog's rows no longer match its meta",
        )

    # A role column that cannot be read back as a speaker-grouped reservation
    # (half-written, or holding a value outside the contract) is a refusal with
    # evidence rather than a traceback; --force plans over it to repair it.
    overrode_inconsistent: str | None = None
    try:
        plan = plan_reservation(rows, spec)
    except ValueError as error:
        if not force:
            return _refuse(
                "inconsistent_existing_reservation",
                error=str(error),
                hint=(
                    "the manifest's development_role column is not a readable "
                    "speaker-grouped reservation; pass --force to overwrite it with a "
                    "recorded decision, or rebuild the manifest"
                ),
            )
        overrode_inconsistent = str(error)
        try:
            plan = plan_reservation(rows, spec, ignore_existing=True)
        except ValueError as fatal:
            return _refuse(
                "unusable_manifest",
                error=str(fatal),
                hint="this manifest cannot carry a speaker-grouped reservation at all",
            )

    record = build_reservation_record(plan, note=note)

    if not plan.coverage.ok:
        return _refuse(
            "coverage_failed",
            failures=list(plan.coverage.failures),
            hint=(
                "adjust threshold_fraction in the reservation config -- never the "
                "rule -- and record why"
            ),
            record=record,
        )

    # The committed record outranks the manifest. The manifest is rebuildable
    # and Git-ignored, so after a rebuild it carries no roles to disagree with:
    # without this check a re-run could not tell a first freeze from a silent
    # re-freeze under an edited fraction.
    committed: dict[str, str] | None = None
    if record_out.is_file():
        try:
            _, committed = read_frozen_record(record_out)
        except (OSError, ValueError) as error:
            return _refuse(
                "unreadable_frozen_record",
                record=str(record_out),
                error=str(error),
                hint=(
                    "this file is the authoritative membership and it does not validate; "
                    "restore it from Git rather than overwriting it"
                ),
            )
        if committed != plan.assignment and not force:
            return _refuse(
                "record_disagrees",
                record=str(record_out),
                committed_assignment=committed,
                proposed_assignment=plan.assignment,
                hint=(
                    "a membership is already frozen and committed, and this run derives a "
                    "different one -- check whether the reservation config changed. "
                    "Re-freezing moves a protocol boundary any completed tuning rests on; "
                    "pass --force only with a recorded decision"
                ),
            )

    if plan.conflicts_with_existing and not force:
        return _refuse(
            "already_frozen_differently",
            existing_assignment=plan.previous_assignment,
            proposed_assignment=plan.assignment,
            hint=(
                "this manifest already carries a different reservation; re-freezing "
                "moves a protocol boundary. Pass --force only with a recorded decision"
            ),
        )

    summary = manifest_summary(
        plan.rows, manifest_hash=plan.manifest_hash_after, missing=[],
        expected_counts_enforced=meta.get("expected_counts_enforced") == "true",
    )
    summary_json = json.dumps(summary, indent=2, sort_keys=True)
    record_json = json.dumps(record, indent=2, sort_keys=True)

    if dry_run:
        print(record_json)
        return 0

    # Keep the original build provenance. When the catalog carried none, record
    # that it is unknown rather than asserting a checksum coverage, build time
    # and commit this run cannot vouch for.
    if meta:
        new_meta = dict(meta)
    else:
        new_meta = {
            "with_checksum": "unknown",
            "build_provenance": "unknown",
            "build_provenance_note": (
                "this catalog carried no catalog_meta when the reservation was frozen; its "
                "build commit, build time and checksum coverage are unknown"
            ),
        }
    new_meta.update(
        {
            "manifest_hash": plan.manifest_hash_after,
            "summary_json": summary_json,
            "reservation_schema_version": RESERVATION_SCHEMA_VERSION,
            "reservation_method": spec.method,
            "reservation_seed": str(spec.seed),
            "reservation_frozen_utc": str(record["frozen_utc"]),
            "reservation_manifest_hash_before": plan.manifest_hash_before,
            "reservation_record_sha256": hashlib.sha256(record_json.encode("utf-8")).hexdigest(),
            "reservation_forced": str(force).lower(),
        }
    )
    if overrode_inconsistent:
        new_meta["reservation_overrode_inconsistent_column"] = overrode_inconsistent

    # ``local_path`` follows the catalog that was read, so a freeze never adds
    # an all-null column to a catalog that was built without one.
    include_local_path = "local_path" in rows[0]
    csv_temp = manifest_dir / "manifest.csv.tmp"
    db_temp = manifest_dir / "manifest.sqlite.tmp"
    write_manifest_csv(plan.rows, csv_temp)
    write_manifest_sqlite(
        plan.rows, db_temp, meta=new_meta, include_local_path=include_local_path
    )

    # Self-check the staged catalog *before* the point of no return, so a
    # failure leaves every committed artifact untouched.
    staged_rows, _ = read_manifest_sqlite(db_temp)
    try:
        verify_frozen_reservation(staged_rows, record)
    except ValueError as error:
        csv_temp.unlink(missing_ok=True)
        db_temp.unlink(missing_ok=True)
        return _refuse(
            "self_check_failed",
            error=str(error),
            hint="nothing was written; the staged catalog did not satisfy its own record",
        )

    # SQLite is the authoritative input for later slices, so it moves first: an
    # interruption leaves the CSV stale rather than ahead of the catalog.
    os.replace(db_temp, catalog_path)
    os.replace(csv_temp, manifest_dir / "manifest.csv")
    _atomic_write_text(manifest_dir / "manifest_summary.json", summary_json)
    _atomic_write_text(manifest_dir / "dev_reservation_frozen.json", record_json)
    _atomic_write_text(record_out, record_json)

    matched_record = None if committed is None else committed == plan.assignment
    print(
        json.dumps(
            {
                "status": "frozen",
                "changed": not plan.already_frozen,
                "matched_committed_record": matched_record,
                "forced": force,
                "manifest_hash_before": plan.manifest_hash_before,
                "manifest_hash_after": plan.manifest_hash_after,
                "n_speakers": record["n_speakers"],
                "speakers": record["speakers"],
                "outputs": {
                    "csv": str(manifest_dir / "manifest.csv"),
                    "sqlite": str(catalog_path),
                    "summary_json": str(manifest_dir / "manifest_summary.json"),
                    "frozen_record": str(record_out),
                },
                "commit_this": str(record_out),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _extract_features(
    manifest_dir: Path | None,
    config_path: Path,
    splits: Sequence[str],
    out_dir: Path | None,
    *,
    force: bool,
    dry_run: bool,
) -> int:
    """Extract and cache pooled MFCC features for the requested splits (#7).

    Features come from the one shared preprocessing path, so offline and served
    features cannot diverge. The cache is role-independent (see features.py), so
    the #31 reservation freeze does not invalidate it. The evaluation split is
    refused here: extracting its features belongs to the locked #35 flow, never
    an ad-hoc run.
    """

    # Imported lazily: features -> preprocessing -> librosa, which the host and
    # CI have but the dataset-free laptop runs do not. Keeping the import here
    # lets smoke/build-manifest/reserve-development stay importable without it.
    import time

    from voice_detector.features import (
        FEATURE_SCHEMA_VERSION,
        cache_path,
        extract_split_features,
        feature_cache_key,
        load_cached_if_fresh,
        write_split_features,
    )
    from voice_detector.preprocessing import PreprocessingConfig

    requested = list(dict.fromkeys(splits))  # de-duplicate, keep order
    if "evaluation" in requested:
        return _refuse(
            "evaluation_is_locked",
            hint=(
                "the evaluation partition is locked; its features are extracted only inside "
                "the frozen #35 evaluation flow, never by extract-features. Request "
                "train/development only"
            ),
        )
    unknown = [split for split in requested if split not in VALID_SPLITS]
    if unknown:
        return _refuse(
            "unknown_split",
            unknown=unknown,
            valid=sorted(VALID_SPLITS - {"evaluation"}),
            hint="request one or more of train, development",
        )

    config = load_config(config_path)
    pre_config = PreprocessingConfig.from_mapping(config.raw)
    if manifest_dir is None:
        manifest_dir = config.paths.artifacts_root / "manifest"
    if out_dir is None:
        out_dir = config.paths.artifacts_root / "features"
    catalog_path = manifest_dir / "manifest.sqlite"

    rows, meta = read_manifest_sqlite(catalog_path)
    if not rows:
        return _refuse(
            "empty_catalog",
            catalog=str(catalog_path),
            hint="this catalog holds no clips; rebuild it with build-manifest",
        )
    manifest_hash = meta.get("manifest_hash") or compute_manifest_hash(rows)

    by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in requested}
    for row in rows:
        split = str(row["official_split"])
        if split in by_split:
            by_split[split].append(row)

    results: list[dict[str, Any]] = []
    for split in requested:
        split_rows = by_split[split]
        if not split_rows:
            return _refuse(
                "split_absent_from_manifest",
                split=split,
                catalog=str(catalog_path),
                hint="the manifest has no clips for this split; rebuild it",
            )
        destination = cache_path(out_dir, split)
        key = feature_cache_key(split_rows, pre_config)
        if dry_run:
            fresh = load_cached_if_fresh(split_rows, pre_config, destination) is not None
            results.append(
                {
                    "split": split,
                    "n_clips": len(split_rows),
                    "cache_key": key,
                    "output": str(destination),
                    "cache_fresh": fresh,
                    "would_reuse": fresh and not force,
                }
            )
            continue

        reused = None if force else load_cached_if_fresh(split_rows, pre_config, destination)
        if reused is not None:
            results.append(
                {
                    "split": split,
                    "reused_cache": True,
                    "n_clips": reused.n_clips,
                    "n_windows": reused.n_windows,
                    "feature_dim": reused.feature_dim,
                    "cache_key": reused.cache_key,
                    "output": str(destination),
                    "cache_bytes": destination.stat().st_size,
                }
            )
            continue

        started = time.perf_counter()
        extracted = extract_split_features(split_rows, pre_config, manifest_hash=manifest_hash)
        elapsed = time.perf_counter() - started
        write_split_features(extracted, destination)
        results.append(
            {
                "split": split,
                "reused_cache": False,
                "n_clips": extracted.n_clips,
                "n_windows": extracted.n_windows,
                "feature_dim": extracted.feature_dim,
                "cache_key": extracted.cache_key,
                "output": str(destination),
                "cache_bytes": destination.stat().st_size,
                "elapsed_seconds": round(elapsed, 3),
            }
        )

    payload = {
        "status": "planned" if dry_run else "extracted",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "preprocessing_version": pre_config.version,
        "manifest_hash": manifest_hash,
        "git_commit": git_commit(),
        "package_version": package_version(),
        "splits": results,
    }
    payload_json = json.dumps(payload, indent=2, sort_keys=True)
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(out_dir / "features_summary.json", payload_json)
    print(payload_json)
    return 0


def _find_run_record(bundle_dir: Path) -> Path | None:
    """Locate the run record in a bundle, accepting prefixed names."""
    for name in ("run_record.json", "lr_run_record.json", "cnn_run_record.json"):
        candidate = bundle_dir / name
        if candidate.exists():
            return candidate
    return None


def _infer_model_family(record_filename: str) -> str:
    if record_filename.startswith("lr_"):
        return "logistic_regression"
    if record_filename.startswith("cnn_"):
        return "cnn"
    return "unknown"


def _eval_freeze(
    manifest_dir: Path | None,
    config_path: Path,
    evaluation_config_path: Path,
    bundle_dirs: list[Path],
    out_path: Path,
    *,
    dry_run: bool,
) -> int:
    """Create and freeze an evaluation protocol record (#35).

    Lazy imports keep cli.py importable without yaml on dataset-free runs.
    """

    from voice_detector.evaluation import (
        CandidateDeclaration,
        _hash_directory,
        create_draft,
        freeze_record,
        save_freeze_record,
    )

    config = load_config(config_path)
    if manifest_dir is None:
        manifest_dir = config.paths.artifacts_root / "manifest"
    catalog_path = manifest_dir / "manifest.sqlite"

    rows, meta = read_manifest_sqlite(catalog_path)
    if not rows:
        return _refuse(
            "empty_catalog",
            catalog=str(catalog_path),
            hint="rebuild the manifest with build-manifest",
        )

    candidates: list[CandidateDeclaration] = []
    for bundle_dir in bundle_dirs:
        if not bundle_dir.is_dir():
            return _refuse(
                "bundle_not_found",
                bundle=str(bundle_dir),
                hint="each --bundle must be a directory containing a run record and model",
            )
        run_record_path = _find_run_record(bundle_dir)
        if run_record_path is None:
            return _refuse(
                "run_record_missing",
                bundle=str(bundle_dir),
                hint="bundle directory must contain *run_record.json",
            )
        run_record = json.loads(run_record_path.read_text(encoding="utf-8"))
        bundle_hash = _hash_directory(bundle_dir)
        threshold = run_record.get("threshold", {}).get("value")
        if threshold is None:
            return _refuse(
                "threshold_missing",
                bundle=str(bundle_dir),
                hint="run_record.json must contain threshold.value",
            )
        candidate_id = bundle_dir.name
        model_family = run_record.get(
            "model_family",
            _infer_model_family(run_record_path.name),
        )
        seed = run_record.get("seed", 0)
        run_label = run_record.get("run_label", candidate_id)
        aggregation = run_record.get("aggregation_rule", "mean_window_decision_function")
        candidates.append(CandidateDeclaration(
            candidate_id=candidate_id,
            model_family=model_family,
            bundle_hash=bundle_hash,
            threshold=float(threshold),
            seed=int(seed),
            run_label=str(run_label),
            aggregation_rule=str(aggregation),
        ))

    if not candidates:
        return _refuse(
            "no_candidates",
            hint="provide at least one --bundle directory",
        )

    config_dir = Path(evaluation_config_path).parent
    draft = create_draft(
        candidates, rows, config_dir,
        evaluation_spec_path=Path(evaluation_config_path),
    )

    if dry_run:
        print(json.dumps(draft.as_dict(), indent=2, sort_keys=True))
        return 0

    frozen = freeze_record(draft)
    save_freeze_record(frozen, out_path)

    print(
        json.dumps(
            {
                "status": "frozen",
                "state": frozen.state,
                "n_candidates": len(frozen.candidates),
                "n_slices": len(frozen.slices),
                "n_conditions": len(frozen.conditions),
                "manifest_hash": frozen.manifest_hash,
                "output": str(out_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _eval_score(
    manifest_dir: Path | None,
    config_path: Path,
    freeze_record_path: Path,
    store_path: Path,
    partition: str,
) -> int:
    """Score clips under all conditions for a frozen evaluation record (#35).

    Currently supports LR candidates only. CNN support will be added once
    CNN bundles are available.
    """

    import hashlib

    import numpy as np

    from voice_detector.evaluation import (
        load_freeze_record as load_eval_record,
    )
    from voice_detector.evaluation import (
        verify_freeze_against_artifacts,
    )
    from voice_detector.prediction_store import (
        PredictionRow,
        compute_predictions_hash,
        create_prediction_store,
        write_predictions,
    )
    from voice_detector.scoring import decide_spoof

    config = load_config(config_path)
    if manifest_dir is None:
        manifest_dir = config.paths.artifacts_root / "manifest"
    catalog_path = manifest_dir / "manifest.sqlite"

    rows, meta = read_manifest_sqlite(catalog_path)
    if not rows:
        return _refuse(
            "empty_catalog",
            catalog=str(catalog_path),
            hint="rebuild the manifest with build-manifest",
        )

    record = load_eval_record(freeze_record_path)
    if record.state != "FROZEN":
        return _refuse(
            "not_frozen",
            state=record.state,
            hint="the evaluation record must be in FROZEN state to score",
        )

    config_dir = Path(config_path).parent
    bundle_root = Path(config.paths.artifacts_root)
    verify_freeze_against_artifacts(record, rows, config_dir, bundle_root)

    split_rows = [r for r in rows if r["official_split"] == partition]
    if not split_rows:
        return _refuse(
            "no_clips_for_partition",
            partition=partition,
            hint=f"no clips found for partition {partition!r}",
        )

    create_prediction_store(store_path)
    freeze_hash = hashlib.sha256(
        json.dumps(record.as_dict(), sort_keys=True).encode()
    ).hexdigest()

    from voice_detector.augmentation import codec_transform, noise_transform
    from voice_detector.preprocessing import AudioConfig, load_waveform

    robustness_spec_path = config_dir / "robustness_spec.yaml"
    robustness_spec: dict = {}
    if robustness_spec_path.exists():
        import yaml

        robustness_spec = yaml.safe_load(
            robustness_spec_path.read_text(encoding="utf-8")
        )
    spec_global_seed = int(robustness_spec.get("global_seed", 20261004))
    spec_conditions = robustness_spec.get("conditions", {})

    scored_pairs: list[str] = []

    for candidate in record.candidates:
        if candidate.model_family == "logistic_regression":
            import joblib

            from voice_detector.manifest import resolve_local_path
            from voice_detector.preprocessing import PreprocessingConfig, clip_feature_matrix

            pre_config = PreprocessingConfig.from_mapping(
                load_config(config_path).raw,
            )
            bundle_dir = bundle_root / candidate.candidate_id
            model_path = bundle_dir / "lr_pipeline.joblib"
            if not model_path.exists():
                model_path = bundle_dir / "pipeline.joblib"
            pipeline = joblib.load(model_path)

            for condition in record.conditions:
                pred_rows: list[PredictionRow] = []
                cond_spec = spec_conditions.get(condition, {})
                for row in split_rows:
                    clip_id = str(row["clip_id"])
                    local_path = row.get("local_path")
                    if not local_path:
                        continue

                    audio_config = AudioConfig(sample_rate_hz=16000)
                    waveform = load_waveform(
                        resolve_local_path(local_path), audio_config,
                    )

                    transform = cond_spec.get("transform")
                    if transform == "codec":
                        waveform = codec_transform(
                            waveform,
                            sample_rate=cond_spec.get("sample_rate_hz", 16000),
                            bitrate_kbps=cond_spec.get("bitrate_kbps", 24),
                        ).waveform
                    elif transform == "noise":
                        waveform = noise_transform(
                            waveform,
                            snr_db=float(cond_spec["snr_db"]),
                            clip_id=clip_id,
                            global_seed=spec_global_seed,
                        ).waveform

                    features = clip_feature_matrix(waveform, pre_config)
                    window_scores = pipeline.decision_function(features)
                    clip_score = float(window_scores.mean())
                    n_windows = len(window_scores)

                    decision = bool(
                        decide_spoof(np.array([clip_score]), candidate.threshold)[0]
                    )

                    pred_rows.append(PredictionRow(
                        clip_id=clip_id,
                        candidate_id=candidate.candidate_id,
                        condition=condition,
                        clip_score=clip_score,
                        label=str(row["label"]),
                        speaker_id=str(row.get("speaker_id", "")),
                        attack_id=row.get("attack_id"),
                        official_split=partition,
                        duration_seconds=float(row.get("duration_seconds", 0)),
                        n_windows=n_windows,
                        threshold=candidate.threshold,
                        decision=decision,
                    ))

                write_predictions(store_path, pred_rows, freeze_record_hash=freeze_hash)
                scored_pairs.append(f"{candidate.candidate_id}/{condition}")
        else:
            scored_pairs.append(f"{candidate.candidate_id}/skipped (unsupported family)")

    predictions_hash = compute_predictions_hash(store_path)
    print(
        json.dumps(
            {
                "status": "scored",
                "partition": partition,
                "scored": scored_pairs,
                "predictions_hash": predictions_hash,
                "store": str(store_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _eval_report(
    freeze_record_path: Path,
    store_path: Path,
    out_dir: Path,
    *,
    with_bootstrap: bool,
) -> int:
    """Generate the evaluation report from scored predictions (#35)."""

    from voice_detector.bootstrap import BootstrapSpec as BootSpec
    from voice_detector.eval_report import compute_report_hash, generate_report
    from voice_detector.evaluation import (
        load_freeze_record as load_eval_record,
    )
    from voice_detector.evaluation import (
        mark_reported,
        mark_scored,
        save_freeze_record,
    )
    from voice_detector.prediction_store import (
        compute_predictions_hash,
        read_predictions,
    )

    record = load_eval_record(freeze_record_path)

    if record.state == "FROZEN":
        predictions_hash = compute_predictions_hash(store_path)
        record = mark_scored(record, predictions_hash)
        save_freeze_record(record, freeze_record_path)

    if record.state != "SCORED":
        return _refuse(
            "not_scored",
            state=record.state,
            hint="the evaluation record must be SCORED before generating a report",
        )

    predictions = read_predictions(store_path)
    if not predictions:
        return _refuse(
            "no_predictions",
            store=str(store_path),
            hint="the prediction store is empty; run eval-score first",
        )

    bootstrap_spec: BootSpec | None = None
    if with_bootstrap:
        bs = record.bootstrap
        bootstrap_spec = BootSpec(
            confidence_level=bs.confidence_level,
            n_resamples=bs.n_resamples,
            seed=bs.seed,
            resampling_unit=bs.resampling_unit,
            method=bs.method,
        )

    report = generate_report(
        record, predictions,
        output_dir=out_dir,
        bootstrap_spec=bootstrap_spec,
    )

    report_hash = compute_report_hash(out_dir / "evaluation_report.json")
    record = mark_reported(record, report_hash)
    save_freeze_record(record, freeze_record_path)

    n_results = len(report.get("results", []))
    print(
        json.dumps(
            {
                "status": "reported",
                "state": record.state,
                "n_results": n_results,
                "report_hash": report_hash,
                "output": str(out_dir / "evaluation_report.json"),
                "freeze_record": str(freeze_record_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _train_cnn(
    manifest_dir: Path | None,
    config_path: Path,
    study_config_path: Path,
    frozen_record_path: Path,
    out_dir: Path | None,
    seed: int | None,
    run_label: str,
    robustness_config_path: Path | None = None,
) -> int:
    """Train the compact log-Mel CNN (#33).

    Lazy imports keep cli.py importable without torch.
    """

    import yaml

    from voice_detector.cnn_train import train_and_save_cnn

    config = load_config(config_path)
    if manifest_dir is None:
        manifest_dir = config.paths.artifacts_root / "manifest"
    if out_dir is None:
        out_dir = config.paths.artifacts_root / "cnn_study" / run_label
    if seed is None:
        seed = config.seed

    catalog_path = manifest_dir / "manifest.sqlite"
    rows, meta = read_manifest_sqlite(catalog_path)
    if not rows:
        return _refuse(
            "empty_catalog",
            catalog=str(catalog_path),
            hint="rebuild the manifest with build-manifest",
        )

    study_raw = yaml.safe_load(Path(study_config_path).read_text(encoding="utf-8"))
    frozen_record_raw = json.loads(
        Path(frozen_record_path).read_text(encoding="utf-8")
    )

    robustness_raw = None
    if robustness_config_path is not None:
        robustness_raw = yaml.safe_load(
            Path(robustness_config_path).read_text(encoding="utf-8")
        )

    train_rows = [r for r in rows if r["official_split"] == "train"]
    dev_rows = [r for r in rows if r["official_split"] == "development"]

    run_record, outputs = train_and_save_cnn(
        train_rows, dev_rows, frozen_record_raw, study_raw,
        seed=seed, run_label=run_label, out_dir=out_dir,
        robustness_config=robustness_raw,
    )

    print(
        json.dumps(
            {
                "status": "trained",
                "run_label": run_label,
                "best_epoch": run_record["best_epoch"],
                "total_epochs": run_record["total_epochs"],
                "threshold": run_record["threshold"]["value"],
                "dev_full_metrics": {
                    k: run_record["dev_full_metrics"][k]
                    for k in (
                        "eer", "roc_auc", "macro_f1", "balanced_accuracy",
                        "human_false_positive_rate", "synthetic_detection_rate",
                    )
                },
                "outputs": {k: str(v) for k, v in outputs.items()},
                "total_seconds": run_record["total_seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cnn_pilot(
    out_dir: Path | None,
    config_path: Path,
    batch_sizes: list[int],
    n_warmup: int,
    n_timed: int,
) -> int:
    """Run the CNN timing pilot on CPU (#10).

    Lazy import keeps cli.py importable without torch.
    """

    from voice_detector.cnn_pilot import run_timing_pilot

    config = load_config(config_path)
    if out_dir is None:
        out_dir = config.paths.artifacts_root / "cnn_pilot"

    record = run_timing_pilot(
        batch_sizes=batch_sizes,
        n_warmup=n_warmup,
        n_timed=n_timed,
        seed=config.seed,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    record_path = out_dir / "cnn_pilot_record.json"
    payload = json.dumps(record, indent=2, sort_keys=True)
    _atomic_write_text(record_path, payload)
    print(payload)
    return 0


def _train_lr(
    manifest_dir: Path | None,
    features_dir: Path | None,
    config_path: Path,
    protocol_path: Path,
    frozen_record_path: Path,
    out_dir: Path | None,
) -> int:
    """Train the MFCC + Logistic Regression baseline and produce a dev report (#8).

    Lazy imports keep cli.py importable without sklearn/joblib on dataset-free
    environments (CI, laptop smoke tests).
    """

    from voice_detector.baseline import save_bundle, train_and_select
    from voice_detector.features import read_split_features
    from voice_detector.pretuning import load_pre_tuning_protocol

    config = load_config(config_path)
    if manifest_dir is None:
        manifest_dir = config.paths.artifacts_root / "manifest"
    if features_dir is None:
        features_dir = config.paths.artifacts_root / "features"
    if out_dir is None:
        out_dir = config.paths.artifacts_root / "lr_baseline"

    catalog_path = manifest_dir / "manifest.sqlite"
    rows, meta = read_manifest_sqlite(catalog_path)
    if not rows:
        return _refuse(
            "empty_catalog",
            catalog=str(catalog_path),
            hint="rebuild the manifest with build-manifest",
        )

    frozen_record_raw = json.loads(
        Path(frozen_record_path).read_text(encoding="utf-8")
    )

    train_features = read_split_features(features_dir / "features_train.npz")
    dev_features = read_split_features(features_dir / "features_development.npz")

    protocol = load_pre_tuning_protocol(protocol_path)
    dev_rows = [row for row in rows if row["official_split"] == "development"]

    run_record = train_and_select(
        train_features, dev_features, dev_rows, protocol, frozen_record_raw,
        seed=config.seed,
    )

    from voice_detector.baseline import train_candidate

    best_C = run_record["selected_C"]
    best_pipeline = train_candidate(
        train_features, best_C,
        class_weight=protocol.lr_budget.class_weight,
        seed=config.seed,
    )

    outputs = save_bundle(best_pipeline, run_record, out_dir)
    print(
        json.dumps(
            {
                "status": "trained",
                "selected_C": best_C,
                "threshold": run_record["threshold"]["value"],
                "dev_full_metrics": {
                    k: run_record["dev_full_metrics"][k]
                    for k in (
                        "eer", "roc_auc", "macro_f1", "balanced_accuracy",
                        "human_false_positive_rate", "synthetic_detection_rate",
                    )
                },
                "outputs": {k: str(v) for k, v in outputs.items()},
                "total_seconds": run_record["total_seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="voice-detector")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke", help="validate the package and runtime")
    smoke.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="path to the project YAML configuration",
    )

    audit = subparsers.add_parser(
        "protocol-audit",
        help="summarise an official ASVspoof 2019 LA CM protocol file (reads no audio)",
    )
    audit.add_argument(
        "--protocol",
        type=Path,
        required=True,
        help="path to an official CM protocol file",
    )
    audit.add_argument(
        "--split",
        choices=sorted(VALID_SPLITS),
        required=True,
        help="official partition the protocol file belongs to",
    )

    build = subparsers.add_parser(
        "build-manifest",
        help="build the per-clip LA manifest by reading audio headers (run where the data lives)",
    )
    build.add_argument(
        "--la-root",
        type=Path,
        default=os.environ.get("VOICE_DETECTOR_LA_ROOT"),
        required="VOICE_DETECTOR_LA_ROOT" not in os.environ,
        help=(
            "extracted ASVspoof 2019 LA root (contains ASVspoof2019_LA_cm_protocols/ etc.); "
            "defaults to the VOICE_DETECTOR_LA_ROOT environment variable"
        ),
    )
    build.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="project YAML, used to resolve the default output directory",
    )
    build.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="output directory for the manifest (default: <artifacts_root>/manifest)",
    )
    build.add_argument(
        "--dry-run",
        action="store_true",
        help="validate structure without hashing or writing any catalog",
    )
    build.add_argument(
        "--allow-missing",
        action="store_true",
        help="continue when some referenced audio files are absent",
    )
    build.add_argument(
        "--no-expected-counts",
        action="store_true",
        help="skip the published ASVspoof 2019 LA count integrity guard",
    )

    reserve = subparsers.add_parser(
        "reserve-development",
        help=(
            "freeze the speaker-grouped development reservation into the built manifest "
            "(run where the manifest lives; changes the manifest hash)"
        ),
    )
    reserve.add_argument(
        "--manifest-dir",
        type=Path,
        default=None,
        help="directory holding manifest.sqlite (default: <artifacts_root>/manifest)",
    )
    reserve.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="project YAML, used to resolve the default manifest directory",
    )
    reserve.add_argument(
        "--reservation-config",
        type=Path,
        default=Path("configs/dev_reservation.yaml"),
        help="versioned reservation specification (fraction, seed, method, coverage)",
    )
    reserve.add_argument(
        "--record-out",
        type=Path,
        default=Path("reports/dev_reservation_frozen.json"),
        help="path for the committed frozen membership record (speaker IDs only)",
    )
    reserve.add_argument(
        "--note",
        default=None,
        help="note stored in the frozen record, e.g. why a fraction was adjusted",
    )
    reserve.add_argument(
        "--dry-run",
        action="store_true",
        help="derive and print the record without writing anything",
    )
    reserve.add_argument(
        "--force",
        action="store_true",
        help="allow replacing an existing, different frozen reservation",
    )

    extract = subparsers.add_parser(
        "extract-features",
        help=(
            "extract and cache pooled MFCC features for train/development via the manifest "
            "(run where the audio lives; evaluation is locked)"
        ),
    )
    extract.add_argument(
        "--manifest-dir",
        type=Path,
        default=None,
        help="directory holding manifest.sqlite (default: <artifacts_root>/manifest)",
    )
    extract.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="project YAML (preprocessing settings + default artifact roots)",
    )
    extract.add_argument(
        "--splits",
        nargs="+",
        default=["train", "development"],
        help="official splits to extract (default: train development); evaluation is refused",
    )
    extract.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="output directory for the feature cache (default: <artifacts_root>/features)",
    )
    extract.add_argument(
        "--dry-run",
        action="store_true",
        help="report planned splits, clip counts and cache keys without extracting",
    )
    extract.add_argument(
        "--force",
        action="store_true",
        help="re-extract even when a matching cache already exists",
    )

    train_lr = subparsers.add_parser(
        "train-lr",
        help=(
            "train the MFCC + Logistic Regression baseline on training features, "
            "select the best C on dev_select, threshold on dev_threshold, and "
            "produce a versioned dev report (run where features are cached)"
        ),
    )
    train_lr.add_argument(
        "--manifest-dir",
        type=Path,
        default=None,
        help="directory holding manifest.sqlite (default: <artifacts_root>/manifest)",
    )
    train_lr.add_argument(
        "--features-dir",
        type=Path,
        default=None,
        help="directory holding features_train.npz and features_development.npz",
    )
    train_lr.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="project YAML (seed, artifact roots)",
    )
    train_lr.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/pre_tuning_protocol.yaml"),
        help="pre-tuning protocol record",
    )
    train_lr.add_argument(
        "--frozen-record",
        type=Path,
        default=Path("reports/dev_reservation_frozen.json"),
        help="committed frozen reservation record",
    )
    train_lr.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="output directory for the LR bundle (default: <artifacts_root>/lr_baseline)",
    )
    cnn_pilot = subparsers.add_parser(
        "cnn-pilot",
        help=(
            "run a forward/backward timing pilot for the compact log-Mel CNN "
            "on CPU with synthetic data; produces a timing record with "
            "extrapolated full-run estimates"
        ),
    )
    cnn_pilot.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="project YAML (seed)",
    )
    cnn_pilot.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="output directory (default: <artifacts_root>/cnn_pilot)",
    )
    cnn_pilot.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[16, 32, 64],
        help="batch sizes to benchmark (default: 16 32 64)",
    )
    cnn_pilot.add_argument(
        "--n-warmup",
        type=int,
        default=2,
        help="warmup steps per batch size (default: 2)",
    )
    cnn_pilot.add_argument(
        "--n-timed",
        type=int,
        default=10,
        help="timed steps per batch size (default: 10)",
    )

    train_cnn = subparsers.add_parser(
        "train-cnn",
        help=(
            "train the compact log-Mel CNN on training audio, select the best "
            "epoch on dev_select by clip-level EER, threshold on dev_threshold, "
            "and produce a versioned run record (run where audio and manifest live)"
        ),
    )
    train_cnn.add_argument(
        "--manifest-dir",
        type=Path,
        default=None,
        help="directory holding manifest.sqlite (default: <artifacts_root>/manifest)",
    )
    train_cnn.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="project YAML (seed, artifact roots)",
    )
    train_cnn.add_argument(
        "--study-config",
        type=Path,
        default=Path("configs/cnn_study.yaml"),
        help="CNN study configuration (mel, architecture, training, budget)",
    )
    train_cnn.add_argument(
        "--frozen-record",
        type=Path,
        default=Path("reports/dev_reservation_frozen.json"),
        help="committed frozen reservation record",
    )
    train_cnn.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="output directory (default: <artifacts_root>/cnn_study/<run_label>)",
    )
    train_cnn.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed (default: from study config seeds[0] or project config)",
    )
    train_cnn.add_argument(
        "--run-label",
        default="clean",
        help="label for this run (default: clean)",
    )
    train_cnn.add_argument(
        "--augmented",
        action="store_true",
        help="enable training augmentation from the robustness spec",
    )
    train_cnn.add_argument(
        "--robustness-config",
        type=Path,
        default=Path("configs/robustness_spec.yaml"),
        help="robustness specification YAML (used with --augmented)",
    )

    eval_freeze = subparsers.add_parser(
        "eval-freeze",
        help=(
            "create and freeze an evaluation protocol record from declared "
            "candidate bundles (#35)"
        ),
    )
    eval_freeze.add_argument(
        "--manifest-dir",
        type=Path,
        default=None,
        help="directory holding manifest.sqlite (default: <artifacts_root>/manifest)",
    )
    eval_freeze.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="project YAML (artifact roots)",
    )
    eval_freeze.add_argument(
        "--evaluation-config",
        type=Path,
        default=Path("configs/evaluation_protocol.yaml"),
        help="evaluation protocol specification",
    )
    eval_freeze.add_argument(
        "--bundle",
        type=Path,
        action="append",
        dest="bundles",
        default=[],
        help="candidate bundle directory (repeat for multiple candidates)",
    )
    eval_freeze.add_argument(
        "--out",
        type=Path,
        default=Path("reports/evaluation_freeze.json"),
        help="output path for the frozen evaluation record",
    )
    eval_freeze.add_argument(
        "--dry-run",
        action="store_true",
        help="print the draft record without freezing",
    )

    eval_score = subparsers.add_parser(
        "eval-score",
        help=(
            "score clips under all conditions for a frozen evaluation record (#35); "
            "currently supports LR candidates only"
        ),
    )
    eval_score.add_argument(
        "--manifest-dir",
        type=Path,
        default=None,
        help="directory holding manifest.sqlite",
    )
    eval_score.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="project YAML",
    )
    eval_score.add_argument(
        "--freeze-record",
        type=Path,
        default=Path("reports/evaluation_freeze.json"),
        help="frozen evaluation record from eval-freeze",
    )
    eval_score.add_argument(
        "--store",
        type=Path,
        default=Path("artifacts/evaluation/predictions.db"),
        help="output path for the prediction store (SQLite)",
    )
    eval_score.add_argument(
        "--partition",
        default="development",
        help="partition to score (default: development for rehearsal)",
    )

    eval_report = subparsers.add_parser(
        "eval-report",
        help="generate the evaluation report from scored predictions (#35)",
    )
    eval_report.add_argument(
        "--freeze-record",
        type=Path,
        default=Path("reports/evaluation_freeze.json"),
        help="frozen evaluation record",
    )
    eval_report.add_argument(
        "--store",
        type=Path,
        default=Path("artifacts/evaluation/predictions.db"),
        help="prediction store from eval-score",
    )
    eval_report.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/evaluation"),
        help="output directory for the report JSON",
    )
    eval_report.add_argument(
        "--bootstrap",
        action="store_true",
        help="compute speaker-grouped bootstrap confidence intervals",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "smoke":
        return _smoke(args.config)
    if args.command == "protocol-audit":
        return _protocol_audit(args.protocol, args.split)
    if args.command == "build-manifest":
        return _build_manifest(
            args.la_root,
            args.config,
            args.out_dir,
            dry_run=args.dry_run,
            allow_missing=args.allow_missing,
            enforce_counts=not args.no_expected_counts,
        )
    if args.command == "reserve-development":
        return _reserve_development(
            args.manifest_dir,
            args.config,
            args.reservation_config,
            args.record_out,
            note=args.note,
            dry_run=args.dry_run,
            force=args.force,
        )
    if args.command == "extract-features":
        return _extract_features(
            args.manifest_dir,
            args.config,
            args.splits,
            args.out_dir,
            force=args.force,
            dry_run=args.dry_run,
        )
    if args.command == "train-lr":
        return _train_lr(
            args.manifest_dir,
            args.features_dir,
            args.config,
            args.protocol,
            args.frozen_record,
            args.out_dir,
        )
    if args.command == "cnn-pilot":
        return _cnn_pilot(
            args.out_dir,
            args.config,
            args.batch_sizes,
            args.n_warmup,
            args.n_timed,
        )
    if args.command == "train-cnn":
        return _train_cnn(
            args.manifest_dir,
            args.config,
            args.study_config,
            args.frozen_record,
            args.out_dir,
            args.seed,
            args.run_label,
            robustness_config_path=args.robustness_config if args.augmented else None,
        )
    if args.command == "eval-freeze":
        return _eval_freeze(
            args.manifest_dir,
            args.config,
            args.evaluation_config,
            args.bundles,
            args.out,
            dry_run=args.dry_run,
        )
    if args.command == "eval-score":
        return _eval_score(
            args.manifest_dir,
            args.config,
            args.freeze_record,
            args.store,
            args.partition,
        )
    if args.command == "eval-report":
        return _eval_report(
            args.freeze_record,
            args.store,
            args.out_dir,
            with_bootstrap=args.bootstrap,
        )
    return 2


if __name__ == "__main__":
    sys.exit(main())
