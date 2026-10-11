"""Thin FastAPI serving layer for clip-level deepfake scoring.

The API owns input validation and decoding limits; the actual decode, feature
extraction and model live behind the :class:`~voice_detector.scoring.Scorer`
interface, so this module imports neither the feature code nor torch. A scorer
is injected at app creation; with none loaded the service is alive but not
ready, and ``/predict`` returns 503.

Privacy: uploaded audio is transient. Bytes are held only for the request,
decoded in memory, scored and dropped; filenames, paths and audio content are
never logged.
"""

from __future__ import annotations

import io
import time
import uuid
from typing import Annotated

import soundfile as sf
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from voice_detector.scoring import ClipScore, Scorer

DEFAULT_MAX_UPLOAD_BYTES = 20 * 1024 * 1024
DEFAULT_MIN_DURATION_SECONDS = 1.0
DEFAULT_MAX_DURATION_SECONDS = 30.0
DEFAULT_ALLOWED_SUFFIXES = (".wav", ".flac")


class PredictResponse(BaseModel):
    """Typed score response. Deliberately has no probability/confidence field."""

    request_id: str
    label: str = Field(description="bonafide or spoof, by the operating threshold")
    score: float
    score_direction: str
    threshold: float
    model_version: str
    preprocessing_version: str
    duration_seconds: float
    processing_ms: float


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    status: str
    model_version: str | None = None
    preprocessing_version: str | None = None


def _suffix(filename: str | None) -> str:
    if not filename or "." not in filename:
        return ""
    return filename[filename.rindex(".") :].lower()


def create_app(
    scorer: Scorer | None = None,
    *,
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
    min_duration_seconds: float = DEFAULT_MIN_DURATION_SECONDS,
    max_duration_seconds: float = DEFAULT_MAX_DURATION_SECONDS,
    allowed_suffixes: tuple[str, ...] = DEFAULT_ALLOWED_SUFFIXES,
) -> FastAPI:
    """Build the FastAPI app, optionally with a loaded ``scorer``.

    Limits are injectable so tests can exercise the guards cheaply. When
    ``scorer`` is ``None`` the service reports alive-but-not-ready and rejects
    scoring requests with 503.
    """

    app = FastAPI(title="voice-detector", version="0.1.0")
    app.state.scorer = scorer
    app.state.max_upload_bytes = max_upload_bytes
    app.state.min_duration_seconds = min_duration_seconds
    app.state.max_duration_seconds = max_duration_seconds
    app.state.allowed_suffixes = tuple(s.lower() for s in allowed_suffixes)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Liveness: the process is up. Says nothing about a loaded model."""

        return HealthResponse(status="alive")

    @app.get("/ready", response_model=ReadyResponse)
    def ready(request: Request) -> ReadyResponse:
        """Readiness: a valid scorer is loaded. 503 until one is."""

        active: Scorer | None = request.app.state.scorer
        if active is None:
            raise HTTPException(status_code=503, detail="no model bundle loaded")
        return ReadyResponse(
            status="ready",
            model_version=active.model_version,
            preprocessing_version=active.preprocessing_version,
        )

    @app.post("/predict", response_model=PredictResponse)
    async def predict(request: Request, file: Annotated[UploadFile, File()]) -> PredictResponse:
        state = request.app.state
        active: Scorer | None = state.scorer
        if active is None:
            raise HTTPException(status_code=503, detail="no model bundle loaded")

        if _suffix(file.filename) not in state.allowed_suffixes:
            raise HTTPException(
                status_code=415,
                detail=f"unsupported file type; allowed: {', '.join(state.allowed_suffixes)}",
            )

        # Read one byte past the cap so an oversize upload is detectable without
        # loading an unbounded body into memory.
        data = await file.read(state.max_upload_bytes + 1)
        if len(data) > state.max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"file exceeds {state.max_upload_bytes} bytes",
            )

        try:
            info = sf.info(io.BytesIO(data))
            duration = float(info.duration)
        except Exception as error:  # noqa: BLE001 - any decode failure is a bad upload
            raise HTTPException(status_code=400, detail="could not decode audio") from error

        if not state.min_duration_seconds <= duration <= state.max_duration_seconds:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"duration {duration:.3f}s outside "
                    f"[{state.min_duration_seconds}, {state.max_duration_seconds}] s"
                ),
            )

        started = time.perf_counter()
        result: ClipScore = active.score_source(io.BytesIO(data))
        processing_ms = (time.perf_counter() - started) * 1000.0

        return PredictResponse(
            request_id=str(uuid.uuid4()),
            label=result.label,
            score=result.score,
            score_direction=result.score_direction,
            threshold=result.threshold,
            model_version=active.model_version,
            preprocessing_version=active.preprocessing_version,
            duration_seconds=duration,
            processing_ms=processing_ms,
        )

    return app
