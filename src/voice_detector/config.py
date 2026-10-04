"""Configuration loading with explicit environment-variable expansion."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?}")


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved project storage paths."""

    data_root: Path
    artifacts_root: Path


@dataclass(frozen=True)
class ProjectConfig:
    """Minimal setup-time configuration shared by future pipeline modules."""

    name: str
    seed: int
    paths: ProjectPaths
    raw: dict[str, Any]


def _expand_environment(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        variable, default = match.groups()
        if variable in os.environ:
            return os.environ[variable]
        if default is not None:
            return default
        raise ValueError(f"Environment variable {variable!r} is required but not set")

    return _ENV_PATTERN.sub(replace, value)


def _expand_values(value: Any) -> Any:
    if isinstance(value, str):
        return _expand_environment(value)
    if isinstance(value, list):
        return [_expand_values(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_values(item) for key, item in value.items()}
    return value


def load_config(path: Path, *, project_root: Path | None = None) -> ProjectConfig:
    """Load a YAML configuration and resolve local storage paths."""

    config_path = path.expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    expanded = _expand_values(raw)

    try:
        project = expanded["project"]
        paths = expanded["paths"]
        name = str(project["name"])
        seed = int(project["seed"])
        data_root = Path(paths["data_root"]).expanduser()
        artifacts_root = Path(paths["artifacts_root"]).expanduser()
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid project configuration: {config_path}") from error

    root = (project_root or Path.cwd()).resolve()
    if not data_root.is_absolute():
        data_root = root / data_root
    if not artifacts_root.is_absolute():
        artifacts_root = root / artifacts_root

    return ProjectConfig(
        name=name,
        seed=seed,
        paths=ProjectPaths(
            data_root=data_root.resolve(),
            artifacts_root=artifacts_root.resolve(),
        ),
        raw=expanded,
    )
