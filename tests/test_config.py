from pathlib import Path

import pytest

from voice_detector.config import load_config


def test_default_config_uses_repository_local_paths() -> None:
    root = Path(__file__).parents[1]
    config = load_config(root / "configs/default.yaml", project_root=root)

    assert config.name == "ai-voice-deepfake-detector"
    assert config.paths.data_root == root / "data"
    assert config.paths.artifacts_root == root / "artifacts"
    assert config.raw["evaluation"]["preserve_official_splits"] is True
    assert config.raw["evaluation"]["evaluation_locked"] is True


def test_environment_overrides_storage_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data_root = tmp_path / "external-data"
    artifact_root = tmp_path / "external-artifacts"
    monkeypatch.setenv("VOICE_DETECTOR_DATA_DIR", str(data_root))
    monkeypatch.setenv("VOICE_DETECTOR_ARTIFACTS_DIR", str(artifact_root))

    root = Path(__file__).parents[1]
    config = load_config(root / "configs/default.yaml", project_root=root)

    assert config.paths.data_root == data_root
    assert config.paths.artifacts_root == artifact_root
