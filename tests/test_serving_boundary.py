"""Serving must not import training, evaluation or metrics code, even transitively."""

from __future__ import annotations

import subprocess
import sys

import pytest

FORBIDDEN = (
    "voice_detector.cnn_train",
    "voice_detector.metrics",
    "voice_detector.reservation",
    "voice_detector.evaluation",
    "voice_detector.eval_scoring",
    "voice_detector.eval_report",
    "voice_detector.prediction_store",
    "voice_detector.baseline",
    "voice_detector.pretuning",
    "sklearn",
)


def _leaked(import_line: str) -> list[str]:
    code = (
        f"{import_line}\n"
        "import sys\n"
        f"print(','.join(m for m in {FORBIDDEN!r} if m in sys.modules))\n"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True
    )
    return [name for name in result.stdout.strip().split(",") if name]


def test_the_serving_app_imports_no_training_or_evaluation_code():
    assert _leaked("import voice_detector.serving.app") == []


def test_the_serving_cli_module_imports_no_training_or_evaluation_code():
    assert _leaked("import voice_detector.serving.cli") == []


def test_the_shared_cnn_scorer_imports_no_training_or_evaluation_code():
    pytest.importorskip("torch")
    assert _leaked("import voice_detector.cnn_scoring") == []
