# One image, two commands (issue #11). linux/amd64, CPU.
#   API: voice-detector serve        (default)
#   UI:  streamlit run src/voice_detector/ui.py
# The image holds code and configs only. The pinned model bundle is mounted read-only at
# /models/<candidate_id>; data, artifacts, reports and uploaded audio are never baked in.
# Dependencies come from uv.lock unchanged (same torch as evaluation). The lock resolves
# CPU-only torch wheels (torch==2.14.1+cpu) via [tool.uv.sources]; image is ~4.8 GB.
FROM python:3.12-slim-bookworm

ARG UV_VERSION=0.12.0
RUN pip install --no-cache-dir "uv==${UV_VERSION}"

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Dependency layer: cached until the lock or pyproject changes.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY configs ./configs
RUN uv sync --frozen --no-dev

# The root filesystem is read-only at run time; caches go to the tmpfs /tmp.
ENV HOME=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib \
    NUMBA_CACHE_DIR=/tmp/numba \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    VOICE_DETECTOR_ARTIFACTS_DIR=/models \
    VOICE_DETECTOR_API_HOST=0.0.0.0 \
    VOICE_DETECTOR_API_URL=http://api:8000

RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin app
USER 10001

EXPOSE 8000 8501
HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=3 \
    CMD ["python", "-c", "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/ready', timeout=4)"]

CMD ["voice-detector", "serve"]
