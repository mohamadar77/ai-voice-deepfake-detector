"""``python -m voice_detector.serving``: a module-pure entry for the service/container."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from voice_detector.serving.cli import DEFAULT_CONFIG, DEFAULT_SERVING_CONFIG, run_serve


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m voice_detector.serving")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--serving-config", type=Path, default=DEFAULT_SERVING_CONFIG)
    parser.add_argument("--check", action="store_true", help="verify and warm up, then exit")
    parser.add_argument("--freeze-record", type=Path, default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args(argv)
    return run_serve(
        config_path=args.config,
        serving_config_path=args.serving_config,
        check=args.check,
        freeze_record=args.freeze_record,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    sys.exit(main())
