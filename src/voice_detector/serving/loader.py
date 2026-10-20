"""Load and verify the deployed model bundle. The only serving module that touches torch.

The bundle is the one the evaluation scored: the same ``load_cnn_scorer`` checks its
SHA-256 against the pin before anything is deserialized, requires the run record's
threshold to equal the pinned one, and loads weights with ``weights_only=True``.
"""

from __future__ import annotations

from pathlib import Path

from voice_detector.scoring import ServableScorer
from voice_detector.serving.settings import ServingSettings


class LoaderError(Exception):
    """The model bundle cannot be served; ``reason`` is a stable machine-readable code."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


def build_scorer(settings: ServingSettings, artifacts_root: Path) -> ServableScorer:
    """Apply the evaluation torch settings, then load the pinned bundle."""

    from voice_detector.runtime import configure_torch

    runtime = settings.runtime
    configure_torch(
        runtime.torch_threads,
        interop_threads=runtime.interop_threads,
        deterministic=runtime.deterministic_algorithms,
    )

    from voice_detector.cnn_scoring import CnnBundleError, load_cnn_scorer

    bundle_dir = Path(artifacts_root) / settings.model.candidate_id
    if not bundle_dir.is_dir():
        raise LoaderError(
            "bundle_missing", "no model bundle directory for the configured candidate"
        )
    try:
        return load_cnn_scorer(
            bundle_dir,
            expected_bundle_hash=settings.model.expected_bundle_hash,
            expected_threshold=settings.model.expected_threshold,
        )
    except CnnBundleError as error:
        message = str(error)
        if "bundle hash mismatch" in message:
            reason = "bundle_hash_mismatch"
        elif "threshold" in message:
            reason = "threshold_mismatch"
        else:
            reason = "bundle_invalid"
        raise LoaderError(reason, message) from error
