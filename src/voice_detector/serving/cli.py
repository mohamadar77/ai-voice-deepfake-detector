"""Entry point for the serving process: ``python -m voice_detector.serving`` or
``voice-detector serve``.

Defaults are file paths only. Every limit, pin and policy comes from the serving
YAML, and a host or port override is echoed in the startup log.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from voice_detector.config import load_config
from voice_detector.serving.loader import LoaderError, build_scorer
from voice_detector.serving.observability import configure_logging, log_event
from voice_detector.serving.service import ScoringService
from voice_detector.serving.settings import ServingSettings, load_serving_settings

DEFAULT_CONFIG = Path("configs/default.yaml")
DEFAULT_SERVING_CONFIG = Path("configs/serving.yaml")


def _refuse(reason: str, **fields: Any) -> int:
    print(json.dumps({"status": "refused", "reason": reason, **fields}, indent=2, sort_keys=True))
    return 1


def _check_freeze_record(settings: ServingSettings, freeze_record: Path) -> str | None:
    """Return a reason string if the serving pins disagree with the freeze record."""

    record = json.loads(Path(freeze_record).read_text(encoding="utf-8"))
    for candidate in record.get("candidates", []):
        if candidate.get("candidate_id") == settings.model.candidate_id:
            if candidate.get("bundle_hash") != settings.model.expected_bundle_hash:
                return "freeze_bundle_hash_mismatch"
            if float(candidate.get("threshold")) != settings.model.expected_threshold:
                return "freeze_threshold_mismatch"
            return None
    return "candidate_not_in_freeze_record"


def run_serve(
    *,
    config_path: Path = DEFAULT_CONFIG,
    serving_config_path: Path = DEFAULT_SERVING_CONFIG,
    check: bool = False,
    freeze_record: Path | None = None,
    host: str | None = None,
    port: int | None = None,
) -> int:
    """Verify the pinned bundle, warm up, then serve (or just report with ``check``)."""

    settings = load_serving_settings(serving_config_path)
    configure_logging(
        settings.observability.log_level,
        debug_tracebacks=settings.observability.debug_tracebacks,
    )
    if freeze_record is not None:
        mismatch = _check_freeze_record(settings, freeze_record)
        if mismatch is not None:
            return _refuse(mismatch, candidate_id=settings.model.candidate_id)

    project = load_config(config_path)
    try:
        scorer = build_scorer(settings, project.paths.artifacts_root)
    except LoaderError as error:
        return _refuse(error.reason, detail=error.detail)

    service = ScoringService(scorer, settings)
    try:
        warmup_ms = service.warmup() if settings.runtime.warmup.enabled else []
    except Exception as error:  # noqa: BLE001 - report the class only, never decoder text
        return _refuse("warmup_failed", error_type=type(error).__name__)

    if check:
        print(
            json.dumps(
                {
                    "status": "ok",
                    "model": service.model_info().model_dump(mode="json"),
                    "warmup_ms": [round(v, 1) for v in warmup_ms],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    import uvicorn

    from voice_detector.serving.app import create_app

    server = settings.server
    bind_host = host or server.host
    bind_port = port or server.port
    log_event(
        "startup",
        model_version=scorer.model_version,
        bundle_hash=str(scorer.bundle_hash)[:12],
        config_sha256=settings.config_sha256,
        host=bind_host,
        port=bind_port,
    )
    app = create_app(settings, scorer, warmup=False)  # warmed up above
    uvicorn.run(
        app,
        host=bind_host,
        port=bind_port,
        log_config=None,
        access_log=False,
        limit_concurrency=server.limit_concurrency,
        timeout_keep_alive=server.timeout_keep_alive_seconds,
        timeout_graceful_shutdown=server.timeout_graceful_shutdown_seconds,
    )
    return 0
