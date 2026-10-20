"""Process-level runtime settings shared by evaluation scoring and serving.

A leaf module: torch is imported lazily, so importing it costs nothing without
torch installed.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
from typing import Any

_TRACKED_PACKAGES = ("numpy", "scipy", "soundfile", "librosa", "torch", "torchaudio")


def configure_torch(
    threads: int, *, interop_threads: int = 1, deterministic: bool = True
) -> None:
    """Fix torch execution settings for scoring; a no-op when torch is absent.

    The defaults are the evaluation settings (``eval-score`` calls this with a
    thread count only). Serving passes the same values from its configuration so
    a clip scores the same way in both.
    """

    try:
        import torch
    except ImportError:
        return
    torch.set_num_threads(threads)
    with contextlib.suppress(RuntimeError):  # already set, or parallel work has started
        torch.set_num_interop_threads(interop_threads)
    torch.use_deterministic_algorithms(deterministic)


def runtime_versions() -> dict[str, str]:
    """Installed versions of the packages that determine scores."""

    versions: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def effective_torch_settings() -> dict[str, Any]:
    """Torch threading and determinism as currently set, or empty without torch."""

    try:
        import torch
    except ImportError:
        return {}
    return {
        "torch_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
