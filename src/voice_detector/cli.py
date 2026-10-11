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
from voice_detector.manifest import VALID_SPLITS
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "smoke":
        return _smoke(args.config)
    if args.command == "protocol-audit":
        return _protocol_audit(args.protocol, args.split)
    return 2


if __name__ == "__main__":
    sys.exit(main())
