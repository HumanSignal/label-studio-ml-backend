# Runs on CPU in dev; torch wheels from PyPI bundle the CUDA runtime for GPU hosts.
ARG PYTHON_VERSION=3.14
FROM python:${PYTHON_VERSION}-slim AS python-base
ARG DEBIAN_FRONTEND=noninteractive
ARG TEST_ENV

WORKDIR /app

RUN --mount=type=cache,target="/var/cache/apt",sharing=locked \
    --mount=type=cache,target="/var/lib/apt/lists",sharing=locked \
    apt-get -y update \
    && apt-get install -y --no-install-recommends \
        git wget curl build-essential \
        ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 \
        libraqm0 libfribidi0  # Pillow text layout used when ultralytics draws labels

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_CACHE_DIR=/.cache \
    PORT=9090 \
    WORKERS=2 \
    THREADS=4

COPY requirements-base.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install -r requirements-base.txt

# install model requirements
COPY requirements.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip3 install -r requirements.txt

# install test requirements if needed
COPY requirements-test.txt .
# build only when TEST_ENV="true"
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    if [ "$TEST_ENV" = "true" ]; then \
      pip3 install -r requirements-test.txt; \
    fi

WORKDIR /app

COPY . ./

# Download the YOLO models
RUN /bin/sh -c 'if [ ! -f /app/models/yolov8m.pt ]; then \
    yolo predict model=/app/models/yolov8m.pt source=/app/tests/car.jpg \
    && yolo predict model=/app/models/yolov8n.pt source=/app/tests/car.jpg \
    && yolo predict model=/app/models/yolov8n-cls.pt source=/app/tests/car.jpg \
    && yolo predict model=/app/models/yolov8n-seg.pt source=/app/tests/car.jpg; \
    fi'

ENV PYTHONPATH=/app

CMD ["/app/start.sh"]
