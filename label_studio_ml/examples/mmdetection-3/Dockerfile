# mmcv 2.1 ships prebuilt wheels only up to torch 2.1 / Python 3.11, so this example stays on 3.11.
ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim AS python-base

ARG TEST_ENV
ARG TORCH_VERSION=2.1.2
ARG TORCHVISION_VERSION=0.16.2

WORKDIR /app

# Update the base OS
RUN --mount=type=cache,target="/var/cache/apt",sharing=locked \
    --mount=type=cache,target="/var/lib/apt/lists",sharing=locked \
    set -eux; \
    apt-get update; \
    apt-get upgrade -y; \
    apt install --no-install-recommends -y  \
        git; \
    apt-get autoremove -y

# Install system dependencies for opencv-python and git
RUN apt-get update  \
    && apt-get install -y libgl1 libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=${PORT:-9090} \
    PIP_CACHE_DIR=/.cache \
    WORKERS=1 \
    THREADS=8

 # Upgrade pip
RUN --mount=type=cache,target=$PIP_CACHE_DIR \
    pip install -U pip

# CPU torch; mim then resolves the matching mmcv wheel from the OpenMMLab index (cpu/torch2.1).
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} \
        --index-url https://download.pytorch.org/whl/cpu

# Install numpy early to avoid dependency conflicts
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install "numpy~=1.26"

# Install base requirements
COPY requirements-base.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install -r requirements-base.txt

# Install custom requirements
COPY requirements.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install -r requirements.txt

# Install test requirements if needed (install before mim to ensure numpy is available)
COPY requirements-test.txt .
# build only when TEST_ENV="true"
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    if [ "$TEST_ENV" = "true" ]; then \
      pip install -r requirements-test.txt; \
    fi

# Ensure numpy is available and pinned before mim installs (mim packages may depend on it)
RUN python -c "import numpy; print(f'numpy version: {numpy.__version__}')" || pip install "numpy~=1.26"

# Install mim packages, but prevent numpy from being upgraded
RUN mim install mmengine==0.10.3 && \
    mim install mmcv==2.1.0 && \
    mim install mmdet==3.3.0 && \
    pip install --force-reinstall --no-deps "numpy~=1.26" || true 
RUN mim download mmdet --config yolov3_mobilenetv2_8xb24-320-300e_coco --dest . 

COPY . .

# Final verification that numpy is available (important for tests)
RUN python -c "import numpy; print(f'✓ numpy {numpy.__version__} is available')" || (echo "ERROR: numpy is not available!" && exit 1)

EXPOSE 9090

CMD gunicorn --preload --bind :$PORT --workers $WORKERS --threads $THREADS --timeout 0 _wsgi:app
