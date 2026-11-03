# syntax=docker/dockerfile:1
# PP-OCR ML Backend for Label Studio (CPU version)
#
# Installs PaddlePaddle and PaddleX from PyPI on python:3.11-slim.
# (The PaddleX CPU image ships Python 3.10, which label-studio-sdk no longer supports.)
#
# Build arguments:
#   TEST_ENV: Set to "true" to install test dependencies

ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim AS python-base

ARG TEST_ENV
ARG PADDLEPADDLE_VERSION=3.0.0

RUN --mount=type=cache,target="/var/cache/apt",sharing=locked \
    --mount=type=cache,target="/var/lib/apt/lists",sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 libgomp1

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=9090 \
    PIP_CACHE_DIR=/.cache \
    WORKERS=1 \
    THREADS=8 \
    DEVICE=cpu

# Install base requirements (label-studio-ml and gunicorn)
COPY requirements-base.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install -r requirements-base.txt

# Install PaddlePaddle (CPU)
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install "paddlepaddle==${PADDLEPADDLE_VERSION}"

# Install PaddleX OCR extra dependencies
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install "paddlex[ocr]"

# Install custom requirements (boto3, opencv, etc.)
COPY requirements.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install -r requirements.txt

# Install test requirements if needed
ARG TEST_ENV
COPY requirements-test.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    if [ "$TEST_ENV" = "true" ]; then \
        pip install -r requirements-test.txt; \
    fi

# Copy application code
COPY . .

EXPOSE 9090

CMD gunicorn --preload --bind :$PORT --workers $WORKERS --threads $THREADS --timeout 0 _wsgi:app
