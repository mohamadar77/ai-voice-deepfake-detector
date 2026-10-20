"""Body-size and upload-time limits, enforced while the request body streams.

Starlette spools multipart file parts to disk once they pass 1 MiB and applies its
part-size limit only to non-file fields, so the cap must be enforced before the
form parser sees the bytes. This is a pure ASGI wrapper around ``receive``: it
rejects a declared oversize body up front, counts streamed bytes (so chunked
bodies with no length are covered), and applies a deadline to the body only.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

from starlette.responses import JSONResponse

from voice_detector.serving.observability import REQUEST_ID_KEY


class _BodyTooLarge(Exception):
    pass


class _UploadTimeout(Exception):
    pass


def _declared_length(scope: Any) -> int | None:
    for name, value in scope.get("headers", []):
        if name == b"content-length":
            try:
                return int(value)
            except ValueError:
                return None
    return None


class BodyLimitMiddleware:
    def __init__(
        self, app: Any, *, path: str, max_body_bytes: int, upload_timeout_seconds: float
    ) -> None:
        self.app = app
        self.path = path
        self.max_body_bytes = max_body_bytes
        self.upload_timeout_seconds = upload_timeout_seconds

    async def _reject(
        self, scope: Any, receive: Any, send: Any, status: int, code: str, detail: str
    ) -> None:
        body = {
            "detail": detail,
            "code": code,
            "request_id": scope.get(REQUEST_ID_KEY) or str(uuid.uuid4()),
        }
        await JSONResponse(body, status_code=status)(scope, receive, send)

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if (
            scope["type"] != "http"
            or scope["method"] != "POST"
            or scope["path"] != self.path
        ):
            await self.app(scope, receive, send)
            return

        declared = _declared_length(scope)
        if declared is not None and declared > self.max_body_bytes:
            await self._reject(
                scope, receive, send, 413, "payload_too_large", "The upload is too large."
            )
            return

        total = 0
        body_complete = False
        response_started = False
        deadline = time.monotonic() + self.upload_timeout_seconds

        async def limited_receive() -> Any:
            nonlocal total, body_complete
            if body_complete:
                return await receive()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise _UploadTimeout
            try:
                message = await asyncio.wait_for(receive(), timeout=remaining)
            except TimeoutError:
                raise _UploadTimeout from None
            if message["type"] == "http.request":
                total += len(message.get("body", b""))
                if total > self.max_body_bytes:
                    raise _BodyTooLarge
                if not message.get("more_body", False):
                    body_complete = True
            return message

        async def tracking_send(message: Any) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except _BodyTooLarge:
            if not response_started:
                await self._reject(
                    scope, receive, send, 413, "payload_too_large", "The upload is too large."
                )
        except _UploadTimeout:
            if not response_started:
                await self._reject(
                    scope, receive, send, 408, "upload_timeout", "The upload took too long."
                )
