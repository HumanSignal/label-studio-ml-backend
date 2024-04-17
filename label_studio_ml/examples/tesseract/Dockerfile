# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS python-base
ARG TEST_ENV

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=${PORT:-9090} \
    PIP_CACHE_DIR=/.cache

# Update the base OS and install Tesseract
RUN --mount=type=cache,target="/var/cache/apt",sharing=locked \
    --mount=type=cache,target="/var/lib/apt/lists",sharing=locked \
    set -eux; \
    apt-get update; \
    apt-get upgrade -y; \
    apt install --no-install-recommends -y  \
        tesseract-ocr tesseract-ocr-chi-sim tesseract-ocr-chi-tra tesseract-ocr-deu git; \
    apt-get autoremove -y

# Install base requirements
COPY requirements-base.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install -r requirements-base.txt

# Install custom requirements
COPY requirements.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install -r requirements.txt

# Install test requirements if needed
COPY requirements-test.txt .
# build only when TEST_ENV="true"
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    if [ "$TEST_ENV" = "true" ]; then \
      pip install -r requirements-test.txt; \
    fi

COPY . .

EXPOSE 9090

CMD gunicorn --preload --bind :$PORT --workers 1 --threads 8 --timeout 0 _wsgi:app
