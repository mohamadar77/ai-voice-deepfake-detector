"""The body limiter works on the ASGI stream, before any multipart parsing."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from voice_detector.serving.limits import BodyLimitMiddleware

PATH = "/predict"
CAP = 1000


def _scope(*, method: str = "POST", path: str = PATH, length: int | None = None) -> dict[str, Any]:
    headers = [] if length is None else [(b"content-length", str(length).encode())]
    return {"type": "http", "method": method, "path": path, "headers": headers}


class _Recorder:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    async def __call__(self, message: dict[str, Any]) -> None:
        self.messages.append(message)

    @property
    def status(self) -> int | None:
        for m in self.messages:
            if m["type"] == "http.response.start":
                return m["status"]
        return None

    @property
    def body(self) -> dict[str, Any]:
        parts = [m.get("body", b"") for m in self.messages if m["type"] == "http.response.body"]
        raw = b"".join(parts)
        return json.loads(raw)


def _chunks(sizes: list[int]):
    queue = [
        {"type": "http.request", "body": b"x" * n, "more_body": i < len(sizes) - 1}
        for i, n in enumerate(sizes)
    ]

    async def receive() -> dict[str, Any]:
        return queue.pop(0) if queue else {"type": "http.disconnect"}

    return receive


async def _drain_app(scope: Any, receive: Any, send: Any) -> None:
    """A downstream app that reads the whole body and answers 200."""

    while True:
        message = await receive()
        if message["type"] != "http.request" or not message.get("more_body"):
            break
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"{}"})


def _run(scope: dict[str, Any], receive: Any, *, timeout: float = 5.0) -> _Recorder:
    middleware = BodyLimitMiddleware(
        _drain_app, path=PATH, max_body_bytes=CAP, upload_timeout_seconds=timeout
    )
    send = _Recorder()
    asyncio.run(middleware(scope, receive, send))
    return send


def test_declared_oversize_is_rejected_without_reading_the_body():
    read = {"called": False}

    async def receive() -> dict[str, Any]:
        read["called"] = True
        return {"type": "http.request", "body": b"", "more_body": False}

    send = _run(_scope(length=CAP + 1), receive)
    assert send.status == 413
    assert send.body["code"] == "payload_too_large"
    assert read["called"] is False


def test_chunked_body_without_a_length_stops_at_the_cap():
    send = _run(_scope(), _chunks([400, 400, 400]))
    assert send.status == 413


def test_a_body_exactly_at_the_cap_passes():
    send = _run(_scope(length=CAP), _chunks([500, 500]))
    assert send.status == 200


def test_other_paths_and_methods_are_not_limited():
    assert _run(_scope(path="/health"), _chunks([5000])).status == 200
    assert _run(_scope(method="GET"), _chunks([5000])).status == 200


def test_a_stalled_upload_times_out_with_408():
    async def stalled() -> dict[str, Any]:
        await asyncio.sleep(10)
        return {"type": "http.request", "body": b"", "more_body": False}

    send = _run(_scope(), stalled, timeout=0.05)
    assert send.status == 408
    assert send.body["code"] == "upload_timeout"
