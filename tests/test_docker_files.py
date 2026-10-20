"""Static checks on the container files: no Docker, network or dataset needed.

They guard the data and privacy rules (nothing restricted in the image, bundle read-only,
localhost-only ports) so a later edit cannot quietly weaken them.
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


def _compose() -> dict:
    return yaml.safe_load((REPO / "compose.yaml").read_text(encoding="utf-8"))


def test_dockerignore_excludes_restricted_material() -> None:
    entries = {
        line.strip()
        for line in (REPO / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    for required in ("data", "artifacts", "reports", ".env", ".git", "*.flac", "*.wav", "*.pt"):
        assert required in entries, required


def test_dockerfile_copies_only_code_and_configs_and_runs_unprivileged() -> None:
    text = (REPO / "Dockerfile").read_text(encoding="utf-8")
    copies = [line for line in text.splitlines() if line.startswith("COPY ")]
    for line in copies:
        for forbidden in ("data", "artifacts", "reports", ".env"):
            assert forbidden not in line.split(), line
    assert "uv sync --frozen --no-dev" in text
    assert "USER 10001" in text
    assert "HEALTHCHECK" in text and "/ready" in text


def test_compose_hardening() -> None:
    compose = _compose()
    services = compose["services"]
    assert set(services) == {"api", "ui"}
    for service in services.values():
        assert service["read_only"] is True
        assert service["cap_drop"] == ["ALL"]
        assert "no-new-privileges:true" in service["security_opt"]
        assert any(str(entry).startswith("/tmp") for entry in service["tmpfs"])
        for port in service["ports"]:
            assert str(port).startswith("127.0.0.1:"), port


def test_compose_mounts_only_the_bundle_read_only() -> None:
    api = _compose()["services"]["api"]
    assert len(api["volumes"]) == 1
    volume = api["volumes"][0]
    assert volume["read_only"] is True
    assert volume["target"] == "/models/cnn_augmented_seed1"
    assert "volumes" not in _compose()["services"]["ui"]


def test_bundle_mount_matches_the_serving_candidate() -> None:
    serving = yaml.safe_load((REPO / "configs" / "serving.yaml").read_text(encoding="utf-8"))
    target = _compose()["services"]["api"]["volumes"][0]["target"]
    assert target == f"/models/{serving['model']['candidate_id']}"
