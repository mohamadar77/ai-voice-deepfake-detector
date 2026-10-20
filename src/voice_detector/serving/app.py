"""FastAPI application for clip scoring (issue #9).

``POST /predict`` takes exactly one multipart ``file`` part and returns a model
score with a decision at the frozen threshold. The response is never a probability,
a confidence or a statement about authenticity. Uploaded audio exists only as bytes
for the duration of the request: nothing is written to disk by this code, nothing
is cached, and nothing derived from the upload is logged.
"""

from __future__ import annotations

import asyncio
import http
import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.requests import ClientDisconnect
from starlette.responses import JSONResponse, Response

from voice_detector.scoring import ServableScorer
from voice_detector.serving.decode import InputRejected
from voice_detector.serving.limits import BodyLimitMiddleware
from voice_detector.serving.observability import (
    REQUEST_ID_KEY,
    RequestContextMiddleware,
    log_event,
)
from voice_detector.serving.schema import ErrorCode, ErrorResponse, ReadyResponse, ScoreResponse
from voice_detector.serving.service import ScoringService
from voice_detector.serving.settings import ServingSettings

PREDICT_PATH = "/predict"
_KNOWN_PATHS = frozenset({"/health", "/ready", "/model", PREDICT_PATH, "/docs", "/openapi.json"})
_UPLOAD_BODY = {
    "required": True,
    "content": {
        "multipart/form-data": {
            "schema": {
                "type": "object",
                "properties": {"file": {"type": "string", "format": "binary"}},
                "required": ["file"],
            }
        }
    },
}
_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    code: {"model": ErrorResponse} for code in (400, 408, 413, 415, 422, 500, 503)
}


def _request_id(request: Request) -> str:
    return request.scope.get(REQUEST_ID_KEY) or str(uuid.uuid4())


def _error(request_id: str, status: int, code: ErrorCode, detail: str) -> JSONResponse:
    body = ErrorResponse(detail=detail, code=code, request_id=request_id)
    return JSONResponse(body.model_dump(mode="json"), status_code=status)


def create_app(
    settings: ServingSettings,
    scorer: ServableScorer | None = None,
    *,
    warmup: bool = True,
) -> FastAPI:
    """Build the app. With no ``scorer`` it is alive but never ready (503 on scoring).

    With a scorer the service is ready once the warmup (if enabled) has scored the
    seeded synthetic clips through the whole decode, resample and score path. A
    failed warmup leaves the app not ready.
    """

    app = FastAPI(title="voice-detector", version="0.1.0", redoc_url=None, debug=False)
    app.state.settings = settings
    app.state.service = ScoringService(scorer, settings) if scorer is not None else None
    app.state.ready = False
    app.state.semaphore = asyncio.Semaphore(settings.runtime.max_concurrent_scoring)

    if app.state.service is not None:
        app.state.ready = True
        if warmup and settings.runtime.warmup.enabled:
            try:
                timings = app.state.service.warmup()
                log_event("warmup_done", warmup_ms=round(sum(timings), 3))
            except Exception as error:  # noqa: BLE001 - never forward decoder text to logs
                app.state.ready = False
                log_event("warmup_failed", error_type=type(error).__name__)

    # Last added is outermost: request context wraps everything, including host checks.
    app.add_middleware(
        BodyLimitMiddleware,
        path=PREDICT_PATH,
        max_body_bytes=settings.input.max_upload_bytes + settings.input.multipart_overhead_bytes,
        upload_timeout_seconds=settings.server.upload_timeout_seconds,
    )
    if settings.server.cors_allow_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.server.cors_allow_origins,
            allow_methods=["GET", "POST"],
        )
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.server.allowed_hosts)
    app.add_middleware(RequestContextMiddleware, known_paths=_KNOWN_PATHS)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # The default handler echoes the offending input; ours never does.
        return _error(_request_id(request), 400, "invalid_request", "The request is invalid.")

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        known = {status.value: status.phrase for status in http.HTTPStatus}
        phrase = known.get(exc.status_code, "Error")
        code: ErrorCode = "invalid_request"
        return _error(_request_id(request), exc.status_code, code, f"{phrase}.")

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Liveness only: trivial, and never blocked by scoring."""

        return {"status": "alive"}

    @app.get("/ready", response_model=ReadyResponse, responses=_ERROR_RESPONSES)
    def ready(request: Request) -> Response:
        service: ScoringService | None = app.state.service
        if service is None or not app.state.ready:
            return _error(
                _request_id(request), 503, "model_not_ready", "The model is not ready."
            )
        scorer_ = service.scorer
        return JSONResponse(
            ReadyResponse(
                model_id=service.model_info().model_id,
                model_version=scorer_.model_version,
                preprocessing_version=scorer_.preprocessing_version,
            ).model_dump(mode="json")
        )

    @app.get("/model", responses=_ERROR_RESPONSES)
    def model(request: Request) -> Response:
        service: ScoringService | None = app.state.service
        if service is None or not app.state.ready:
            return _error(
                _request_id(request), 503, "model_not_ready", "The model is not ready."
            )
        return JSONResponse(service.model_info().model_dump(mode="json"))

    @app.post(
        PREDICT_PATH,
        response_model=ScoreResponse,
        responses=_ERROR_RESPONSES,
        openapi_extra={"requestBody": _UPLOAD_BODY},
    )
    async def predict(request: Request) -> Response:
        request_id = _request_id(request)
        service: ScoringService | None = app.state.service
        if service is None or not app.state.ready:
            return _error(request_id, 503, "model_not_ready", "The model is not ready.")

        max_bytes = settings.input.max_upload_bytes
        data = b""
        try:
            async with request.form(max_files=1, max_fields=0) as form:
                parts = form.getlist("file")
                if len(parts) != 1 or not isinstance(parts[0], UploadFile):
                    return _error(
                        request_id, 400, "invalid_request",
                        "Send exactly one multipart part named 'file'.",
                    )
                data = await parts[0].read(max_bytes + 1)
        except ClientDisconnect:
            log_event("client_disconnect", request_id=request_id)
            return Response(status_code=499)
        # The upload spool is closed here; only the bytes remain.
        if len(data) > max_bytes:
            return _error(request_id, 413, "payload_too_large", "The upload is too large.")

        try:
            await asyncio.wait_for(
                app.state.semaphore.acquire(), timeout=settings.runtime.queue_timeout_seconds
            )
        except TimeoutError:
            return _error(request_id, 503, "server_busy", "The service is busy; retry shortly.")
        try:
            outcome = await run_in_threadpool(service.score_bytes, data, request_id=request_id)
        except InputRejected as rejected:
            return _error(request_id, rejected.status, rejected.code, rejected.message)
        except Exception as error:  # noqa: BLE001 - map everything to a fixed, content-free error
            log_event("score_failed", request_id=request_id, error_type=type(error).__name__)
            return _error(request_id, 500, "internal_error", "Scoring failed.")
        finally:
            app.state.semaphore.release()
            data = b""

        response = outcome.response
        log_event(
            "scored",
            request_id=request_id,
            decode_ms=round(outcome.decode_ms, 3),
            inference_ms=round(outcome.inference_ms, 3),
            duration_s=response.duration_seconds,
            n_windows=response.n_windows,
            warning_codes=[w.code for w in response.warnings],
            model_version=response.model_version,
            bundle_hash=response.bundle_hash[:12],
        )
        return JSONResponse(response.model_dump(mode="json"))

    return app
