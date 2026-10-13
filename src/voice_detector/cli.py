"""Command-line utilities for repository and environment validation."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import sys
from collections.abc import Sequence
from pathlib import Path

from voice_detector import __version__
from voice_detector.config import load_config
from voice_detector.manifest import (
    VALID_SPLITS,
    build_catalog_meta,
    build_la_manifest,
    write_manifest_csv,
    write_manifest_sqlite,
)
from voice_detector.protocol import parse_protocol_file, summarize_protocol

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
        required=True,
        help="extracted ASVspoof 2019 LA root (contains ASVspoof2019_LA_cm_protocols/ etc.)",
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
    return 2


if __name__ == "__main__":
    sys.exit(main())
