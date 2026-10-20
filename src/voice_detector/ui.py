"""Streamlit upload UI that calls the FastAPI scorer over HTTP.

The UI never loads a model: it uploads the clip to the API's ``/predict`` and
renders the typed response. The HTTP logic lives in small functions that take
any httpx-style client (a real ``httpx.Client`` at runtime, the FastAPI
``TestClient`` in tests), so it is testable in-process against the ASGI app
without a running server. Streamlit itself is imported lazily inside
:func:`render`, keeping these functions importable in a headless test.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:
    from voice_detector.cnn_dataset import MelSpec
    from voice_detector.preprocessing import AudioConfig

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


DEFAULT_SUMMARY_PATH = Path(
    os.environ.get(
        "VOICE_DETECTOR_UI_SUMMARY",
        str(Path(__file__).resolve().parents[2] / "configs" / "deployed_model_summary.yaml"),
    )
)


def load_evaluation_summary(path: Path = DEFAULT_SUMMARY_PATH) -> dict[str, Any]:
    """Load the committed measured-evaluation summary for the deployed model."""

    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("version") != "ui-summary-v1":
        raise ValueError(f"{path} is not a ui-summary-v1 evaluation summary")
    return payload


def headline_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Headline EER rows with interval and class counts."""

    return [
        {
            "Slice": item["label"],
            "EER %": f"{item['value']:.2f}",
            "95% CI": f"[{item['ci'][0]:.2f}, {item['ci'][1]:.2f}]",
            "Human clips": item["bona_fide"],
            "Synthetic clips": item["spoof"],
        }
        for item in summary["headline"]
    ]


def robustness_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per robustness condition: EER, human false-positive and detection rates."""

    block = summary["robustness"]
    return [
        {
            "Condition": condition,
            "EER %": f"{eer:.2f}",
            "Human flagged % (FPR)": f"{fpr:.2f}",
            "Synthetic flagged % (SDR)": f"{sdr:.2f}",
        }
        for condition, eer, fpr, sdr in zip(
            block["conditions"],
            block["eer"],
            block["human_false_positive_rate"],
            block["synthetic_detection_rate"],
            strict=True,
        )
    ]


def display_spectrogram(
    data: bytes,
    mel: MelSpec | None = None,
    audio: AudioConfig | None = None,
) -> np.ndarray:
    """Un-normalized log-Mel of the first window, ``(n_mels, n_frames)``, for display.

    Uses the evaluation decoder and ``segment_to_spectrogram`` so the picture comes
    from the same numerics the scorer uses. The bundle's per-channel standardization
    is skipped on purpose: it is model-specific and would hide the real energy.
    The audio is decoded from memory only; nothing is written to disk.
    """

    from voice_detector.cnn_dataset import MelSpec as _MelSpec
    from voice_detector.cnn_dataset import repeat_pad, segment_to_spectrogram
    from voice_detector.preprocessing import AudioConfig as _AudioConfig
    from voice_detector.preprocessing import load_waveform

    mel = mel or _MelSpec()
    audio = audio or _AudioConfig(sample_rate_hz=mel.sample_rate_hz)
    waveform = load_waveform(io.BytesIO(data), audio)
    segment = repeat_pad(waveform, mel.window_samples).astype(np.float32)
    spectrogram = segment_to_spectrogram(segment, mel.transform(), None, None)
    return spectrogram.squeeze(0).numpy()


def spectrogram_figure(spectrogram: np.ndarray, mel: MelSpec | None = None) -> Any:
    """Matplotlib figure of a log-Mel array; the caller closes it."""

    from matplotlib.figure import Figure

    from voice_detector.cnn_dataset import MelSpec as _MelSpec

    mel = mel or _MelSpec()
    figure = Figure(figsize=(7, 3), layout="constrained")
    axis = figure.subplots()
    seconds = spectrogram.shape[1] * mel.hop_length / mel.sample_rate_hz
    image = axis.imshow(
        spectrogram,
        origin="lower",
        aspect="auto",
        extent=(0.0, seconds, 0.0, mel.n_mels),
    )
    axis.set_xlabel("Time (s), first 4 s window")
    axis.set_ylabel("Mel bin")
    axis.set_title("Log-Mel spectrogram (display only, not normalized)")
    figure.colorbar(image, ax=axis, label="log energy")
    return figure


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
            data = upload.getvalue()
            try:
                result = score_clip(client, upload.name, data)
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
            try:
                figure = spectrogram_figure(display_spectrogram(data))
                st.pyplot(figure)
                figure.clear()
            except Exception:  # noqa: BLE001 - the score stands even if the picture fails
                st.info("The spectrogram could not be drawn for this clip.")
            del data  # the page holds no other copy; Streamlit drops the upload on rerun

        _render_evaluation_summary(st)
        info = fetch_model_info(client)
        if info is not None:
            with st.expander("Limitations of this model"):
                for item in info["limitations"]:
                    st.markdown(f"- {item}")


def _render_evaluation_summary(st: Any) -> None:  # pragma: no cover - UI glue
    """Show the measured evaluation of the deployed model, with its caveats."""

    try:
        summary = load_evaluation_summary()
    except (OSError, ValueError):
        st.info("The measured evaluation summary is not available in this deployment.")
        return
    with st.expander("How well does this model work? (measured, locked evaluation)"):
        st.caption(summary["framing"].strip())
        st.markdown("**Headline (clean audio, 95% speaker-grouped interval)**")
        st.table(headline_rows(summary))
        st.markdown("**Unseen attacks under degraded audio, at the frozen threshold**")
        st.table(robustness_rows(summary))
        st.caption(
            f"Frozen threshold {summary['threshold']}. EER does not depend on the threshold; "
            "the flagged rates do. Source: " + summary["source_doc"]
        )
        for item in summary["limitations"]:
            st.markdown(f"- {item}")


def main() -> None:  # pragma: no cover - entry point
    render()


if __name__ == "__main__":
    main()
