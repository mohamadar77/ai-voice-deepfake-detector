# Application entry points

This directory is intentionally empty. The application code lives in the package so that evaluation and serving share it:

- FastAPI service: `src/voice_detector/serving/` (start with `uv run voice-detector serve`).
- Streamlit UI: `src/voice_detector/ui.py` (start with `uv run streamlit run src/voice_detector/ui.py`).

See [`docs/pipeline.md`](../docs/pipeline.md) for the serving path.
