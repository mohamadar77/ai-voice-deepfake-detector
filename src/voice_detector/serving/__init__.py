"""Serving layer: a FastAPI service around the shared, versioned CNN scorer.

Importing this package pulls in neither torch nor any training or evaluation
code; the torch-backed scorer is loaded lazily by ``serving.loader``.
"""

from __future__ import annotations

__all__ = ["create_app", "load_serving_settings"]


def __getattr__(name: str):  # lazy, so ``import voice_detector.serving`` stays cheap
    if name == "create_app":
        from voice_detector.serving.app import create_app

        return create_app
    if name == "load_serving_settings":
        from voice_detector.serving.settings import load_serving_settings

        return load_serving_settings
    raise AttributeError(name)
