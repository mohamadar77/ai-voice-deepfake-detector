"""Serving configuration: the committed YAML is valid and strict."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError
from serving_fakes import _BASE_SETTINGS, make_settings

from voice_detector.serving.settings import ServingSettings, load_serving_settings

REPO = Path(__file__).resolve().parents[1]
SERVING_YAML = REPO / "configs" / "serving.yaml"


def test_committed_config_loads_with_the_deployed_pins():
    settings = load_serving_settings(SERVING_YAML)
    assert settings.model.candidate_id == "cnn_augmented_seed1"
    assert settings.model.expected_bundle_hash == (
        "999bd10a831fcdff43829fbc197c3d4991491153fcead7cf594c1a971e3fe8a0"
    )
    assert settings.model.expected_threshold == -1.251954287290573
    assert len(settings.config_sha256) == 64
    assert settings.input.min_duration_seconds == 1.0
    assert settings.runtime.max_concurrent_scoring == 1
    assert settings.hints.provisional is True


def test_threshold_pin_matches_the_documented_value_to_seven_decimals():
    settings = load_serving_settings(SERVING_YAML)
    assert round(settings.model.expected_threshold, 7) == -1.2519543


def test_unknown_keys_are_rejected():
    data = copy.deepcopy(_BASE_SETTINGS)
    data["input"]["max_upload_byte"] = 1  # typo must not silently disable a guard
    with pytest.raises(ValidationError):
        ServingSettings.model_validate(data)


def test_there_is_no_threshold_override_key():
    data = copy.deepcopy(_BASE_SETTINGS)
    data["model"]["threshold"] = 0.0
    with pytest.raises(ValidationError):
        ServingSettings.model_validate(data)


@pytest.mark.parametrize("bad", ["", "abc", "A" * 64, "g" * 64, "a" * 63])
def test_bundle_hash_must_be_lowercase_sha256(bad: str):
    with pytest.raises(ValidationError):
        make_settings(model={"expected_bundle_hash": bad})


def test_non_finite_threshold_is_rejected():
    with pytest.raises(ValidationError):
        make_settings(model={"expected_threshold": float("nan")})


def test_duration_and_rate_bounds_must_be_ordered():
    with pytest.raises(ValidationError):
        make_settings(input={"min_duration_seconds": 30.0, "max_duration_seconds": 1.0})
    with pytest.raises(ValidationError):
        make_settings(input={"min_source_sample_rate_hz": 96000, "max_source_sample_rate_hz": 8000})


def test_environment_variables_are_expanded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data = yaml.safe_load(SERVING_YAML.read_text(encoding="utf-8"))
    monkeypatch.setenv("VOICE_DETECTOR_API_HOST", "0.0.0.0")
    monkeypatch.setenv("VOICE_DETECTOR_API_ALLOWED_HOST", "api")
    path = tmp_path / "serving.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    settings = load_serving_settings(path)
    assert settings.server.host == "0.0.0.0"
    assert "api" in settings.server.allowed_hosts


def test_config_hash_changes_with_the_file(tmp_path: Path):
    text = SERVING_YAML.read_text(encoding="utf-8")
    first = tmp_path / "a.yaml"
    second = tmp_path / "b.yaml"
    first.write_text(text, encoding="utf-8")
    second.write_text(text + "\n# comment\n", encoding="utf-8")
    assert load_serving_settings(first).config_sha256 != load_serving_settings(second).config_sha256
