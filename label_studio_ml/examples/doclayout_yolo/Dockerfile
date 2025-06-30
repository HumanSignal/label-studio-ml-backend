# Use a PyTorch base image with CUDA support compatible with DocLayout-YOLO's requirements
# Check DocLayout-YOLO/YOLOv10 documentation for compatible CUDA/PyTorch versions
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime
ARG DEBIAN_FRONTEND=noninteractive
ARG TEST_ENV

WORKDIR /app

# Update conda if necessary (often not needed if base image is sufficient)
# RUN conda update conda -y

# Install system dependencies (keep common ones from YOLOv8 example)
RUN --mount=type=cache,target="/var/cache/apt",sharing=locked \
    --mount=type=cache,target="/var/lib/apt/lists",sharing=locked \
    apt-get -y update \
    && apt-get install -y --no-install-recommends git wget curl \
    # Build tools often needed for Python packages
    g++ build-essential \
    # Common image/video processing libs
    ffmpeg libsm6 libxext6 \
    # Python dev headers and pip
    libffi-dev python3-dev python3-pip gcc \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_CACHE_DIR=/.cache \
    PORT=9090 \
    WORKERS=2 \
    THREADS=4 \
    # Specify CUDA Home if needed by dependencies, adjust path based on base image
    CUDA_HOME=/usr/local/cuda

# Set Torch CUDA Arch List if compiling custom ops (check if DocLayout-YOLO needs this)
# ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6+PTX;8.9;9.0"

# --- Install Python Dependencies ---
COPY requirements-base.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install --no-cache-dir -r requirements-base.txt

# Install model requirements (doclayout-yolo and its dependencies)
COPY requirements.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install --no-cache-dir -r requirements.txt

# install test requirements if needed
COPY requirements-test.txt .
# build only when TEST_ENV="true"
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    if [ "$TEST_ENV" = "true" ]; then \
      pip install --no-cache-dir -r requirements-test.txt; \
    fi

WORKDIR /app

COPY . ./

# --- Model Download ---
# Remove the YOLOv8 model download section.
# Models for DocLayout-YOLO should be downloaded manually or via custom script/entrypoint
# and mounted into the /app/models directory using docker-compose volumes.
# Add a placeholder directory
RUN mkdir -p /app/models

# --- END Model Download ---

ENV PYTHONPATH=/app

CMD ["/app/start.sh"]