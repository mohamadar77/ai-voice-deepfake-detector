from voice_detector import __version__
from voice_detector.cli import build_parser


def test_package_imports() -> None:
    assert __version__ == "0.1.0"


def test_smoke_command_parses() -> None:
    args = build_parser().parse_args(["smoke", "--config", "configs/default.yaml"])
    assert args.command == "smoke"
