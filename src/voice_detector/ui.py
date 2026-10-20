"""Streamlit upload UI that calls the FastAPI scorer over HTTP.

The UI never loads a model: it uploads the clip to the API's ``/predict`` and
renders the typed response. The HTTP logic lives in small functions that take
any httpx-style client (a real ``httpx.Client`` at runtime, the FastAPI
``TestClient`` in tests), so it is testable in-process against the ASGI app
without a running server. Streamlit itself is imported lazily inside
:func:`render`, keeping these functions importable in a headless test.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol

DEFAULT_API_URL = os.environ.get("VOICE_DETECTOR_API_URL", "http://127.0.0.1:8000")
_CONTENT_TYPES = {".wav": "audio/wav", ".flac": "audio/flac"}


class HttpClient(Protocol):
    """The subset of the httpx client interface the UI relies on."""

    def get(self, url: str, **kwargs: Any) -> Any: ...
    def post(self, url: str, **kwargs: Any) -> Any: ...


class ApiError(RuntimeError):
    """Raised when the API returns a non-success status for a scoring request."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"API returned {status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


@dataclass(frozen=True)
class Readiness:
    ready: bool
    detail: str
    model_version: str | None = None
    preprocessing_version: str | None = None
    model_id: str | None = None


def content_type_for(filename: str) -> str:
    """Return the audio MIME type for a filename, defaulting to octet-stream."""

    for suffix, mime in _CONTENT_TYPES.items():
        if filename.lower().endswith(suffix):
            return mime
    return "application/octet-stream"


def check_ready(client: HttpClient) -> Readiness:
    """Query ``/ready`` and report whether a scorer bundle is loaded."""

    response = client.get("/ready")
    if response.status_code == 200:
        body = response.json()
        return Readiness(
            ready=True,
            detail="A model bundle is loaded.",
            model_version=body.get("model_version"),
            preprocessing_version=body.get("preprocessing_version"),
            model_id=body.get("model_id"),
        )
    detail = _detail(response, default="service not ready")
    return Readiness(ready=False, detail=detail)


def score_clip(client: HttpClient, filename: str, data: bytes) -> dict[str, Any]:
    """POST one clip to ``/predict`` and return the parsed response.

    Raises :class:`ApiError` with the status and server detail on any
    non-200 response, so the caller can show a clear message.
    """

    response = client.post(
        "/predict",
        files={"file": (filename, data, content_type_for(filename))},
    )
    if response.status_code != 200:
        raise ApiError(response.status_code, _detail(response, default="request failed"))
    return response.json()


def fetch_model_info(client: HttpClient) -> dict[str, Any] | None:
    """Fetch ``/model`` (notice, limitations, threshold basis); ``None`` if unavailable."""

    response = client.get("/model")
    if response.status_code != 200:
        return None
    payload = response.json()
    return payload if isinstance(payload, dict) else None


DECISION_TEXT = {
    "flagged_spoof_like": "Flagged: spoof-like score",
    "not_flagged": "Not flagged at this threshold",
}


def decision_text(decision: str) -> str:
    """Neutral display text for a decision; never says authentic or fake."""

    return DECISION_TEXT.get(decision, "Unknown decision")


def _detail(response: Any, *, default: str) -> str:
    try:
        payload = response.json()
    except Exception:  # noqa: BLE001 - a non-JSON error body is still informative-less
        return default
    if isinstance(payload, dict) and "detail" in payload:
        return str(payload["detail"])
    return default


def render(api_url: str = DEFAULT_API_URL) -> None:  # pragma: no cover - UI glue
    """Render the Streamlit page. Imports streamlit lazily."""

    import httpx
    import streamlit as st

    st.set_page_config(page_title="Voice Deepfake Detector", page_icon="🎙️")
    st.title("AI Voice Deepfake Detector")
    st.caption(
        "Upload a short clip (WAV/FLAC, 1–30 s). The score reflects how "
        "spoof-like the audio is at a fixed threshold. It is not a probability, "
        "does not prove a recording is authentic or synthetic, and does not "
        "identify or verify a speaker."
    )

    with httpx.Client(base_url=api_url, timeout=30.0) as client:
        try:
            readiness = check_ready(client)
        except httpx.HTTPError:
            st.error(f"Cannot reach the scoring API at {api_url}.")
            return

        if readiness.ready:
            st.success(
                f"Model ready — model `{readiness.model_version}`, "
                f"preprocessing `{readiness.preprocessing_version}`."
            )
        else:
            st.warning(f"Scoring API is not ready: {readiness.detail}")

        upload = st.file_uploader("Audio clip", type=["wav", "flac"])
        if upload is not None and st.button("Score clip", disabled=not readiness.ready):
            try:
                result = score_clip(client, upload.name, upload.getvalue())
            except ApiError as error:
                st.error(error.detail)
                return
            except httpx.HTTPError:
                st.error(f"Cannot reach the scoring API at {api_url}.")
                return

            st.metric("Decision", decision_text(result["decision"]))
            st.caption(result["notice"])
            for warning in result["warnings"]:
                st.warning(warning["message"])
            st.write(
                {
                    "score": result["score"],
                    "score_direction": result["score_direction"],
                    "threshold": result["threshold"],
                    "decision_rule": result["decision_rule"],
                    "n_windows": result["n_windows"],
                    "duration_seconds": result["duration_seconds"],
                    "processing_ms": result["processing_ms"],
                    "model_id": result["model_id"],
                    "model_version": result["model_version"],
                    "preprocessing_version": result["preprocessing_version"],
                    "request_id": result["request_id"],
                }
            )
        info = fetch_model_info(client)
        if info is not None:
            with st.expander("Limitations of this model"):
                for item in info["limitations"]:
                    st.markdown(f"- {item}")


def main() -> None:  # pragma: no cover - entry point
    render()


if __name__ == "__main__":
    main()
