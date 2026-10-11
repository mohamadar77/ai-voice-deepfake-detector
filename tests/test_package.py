from voice_detector import __version__
from voice_detector.cli import build_parser
from voice_detector.manifest import validate_manifest


def test_package_imports() -> None:
    assert __version__ == "0.1.0"


def test_smoke_command_parses() -> None:
    args = build_parser().parse_args(["smoke", "--config", "configs/default.yaml"])
    assert args.command == "smoke"


def test_manifest_contract_accepts_a_valid_clip() -> None:
    validate_manifest(
        [
            {
                "clip_id": "LA_T_0001",
                "label": "bonafide",
                "official_split": "development",
                "development_role": "select",
                "duration_seconds": 2.5,
                "sample_rate_hz": 16000,
                "channels": 1,
                "checksum_sha256": "a" * 64,
                "provenance": "ASVspoof 2019 LA protocol; local fixture",
            }
        ]
    )


def test_manifest_contract_rejects_duplicate_clip() -> None:
    row = {
        "clip_id": "duplicate",
        "label": "spoof",
        "official_split": "train",
        "duration_seconds": 1.0,
        "sample_rate_hz": 16000,
        "channels": 1,
        "checksum_sha256": "b" * 64,
        "provenance": "fixture",
    }

    try:
        validate_manifest([row, row])
    except ValueError as error:
        assert "Duplicate clip_id" in str(error)
    else:
        raise AssertionError("duplicate clip IDs must be rejected")
