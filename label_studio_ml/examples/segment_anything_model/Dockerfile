# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.10

FROM python:${PYTHON_VERSION}-slim AS python-base
ARG TEST_ENV

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=${PORT:-9090} \
    PIP_CACHE_DIR=/.cache \
    WORKERS=1 \
    THREADS=8 \
    VITH_CHECKPOINT=/app/models/sam_vit_h_4b8939.pth \
    MOBILESAM_CHECKPOINT=/app/models/mobile_sam.pt \
    ONNX_CHECKPOINT=/app/models/sam_onnx_quantized_example.onnx

# Update the base OS
RUN --mount=type=cache,target="/var/cache/apt",sharing=locked \
    --mount=type=cache,target="/var/lib/apt/lists",sharing=locked \
    set -eux; \
    apt-get update; \
    apt-get upgrade -y; \
    apt install --no-install-recommends -y  \
        wget git libopencv-dev cmake protobuf-compiler binutils patchelf; \
    apt-get autoremove -y

# Copy and run the model download script
COPY download_models.sh .
RUN bash /app/download_models.sh

# Install numpy first to avoid conflicts with system numpy
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install --upgrade pip && \
    pip install "numpy>=2,<2.3.0"

# install base requirements
COPY requirements-base.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install -r requirements-base.txt

# install custom requirements
COPY requirements.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install -r requirements.txt

# Fix executable stack issue with onnxruntime shared library using patchelf
RUN PYTHON_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')") && \
    find /usr/local/lib/python${PYTHON_VER}/site-packages/onnxruntime/capi -name "onnxruntime_pybind11_state*.so" -exec sh -c 'patchelf --clear-execstack "$1" 2>/dev/null || true' _ {} \; || true

# install test requirements if needed
COPY requirements-test.txt .
# build only when TEST_ENV="true"
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    if [ "$TEST_ENV" = "true" ]; then \
      pip install -r requirements-test.txt; \
    fi

COPY . .

# Add ONNX model (skip if it fails - not critical for basic functionality)
RUN python3 onnxconverter.py || echo "Warning: ONNX conversion failed, but continuing build"

EXPOSE 9090

CMD gunicorn --bind :${PORT:-9090} --workers $WORKERS --threads $THREADS --timeout 0 _wsgi:app

