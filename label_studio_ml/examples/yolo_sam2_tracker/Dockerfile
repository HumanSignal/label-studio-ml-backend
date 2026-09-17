FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive
ARG TEST_ENV

WORKDIR /app

# Update Conda
RUN conda update conda -y

# Install system dependencies
RUN --mount=type=cache,target="/var/cache/apt",sharing=locked \
    --mount=type=cache,target="/var/lib/apt/lists",sharing=locked \
    apt-get -y update \
    && apt-get install -y git wget g++ freeglut3-dev build-essential \
    libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev \
    libfreeimage-dev ffmpeg libsm6 libxext6 libffi-dev python3-dev \
    python3-pip gcc

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_CACHE_DIR=/.cache \
    PORT=9090 \
    WORKERS=2 \
    THREADS=4 \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6+PTX;8.9;9.0" \
    SEGMENT_ANYTHING_2_REPO_PATH=/segment-anything-2 \
    PYTHONPATH=/app

# Install Python dependencies
COPY requirements-base.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install -r requirements-base.txt

COPY requirements.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install -r requirements.txt

# Install segment-anything-2
RUN cd / && git clone --depth 1 --branch main --single-branch https://github.com/facebookresearch/sam2.git
WORKDIR /sam2
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install -e .
RUN cd checkpoints && ./download_ckpts.sh

# Clone and install label-studio-ml-backend
WORKDIR /app
RUN git clone https://github.com/HumanSignal/label-studio-ml-backend.git \
    && cd label-studio-ml-backend/ \
    && pip install -e .

# Return to app working directory
WORKDIR /app

# Install test dependencies (optional)
COPY requirements-test.txt .
RUN if [ "$TEST_ENV" = "true" ]; then \
      pip install -r requirements-test.txt; \
    else \
      echo "Skipping test dependencies installation"; \
    fi

# Download YOLO models (only if the image exists)
RUN /bin/sh -c 'if [ -f /app/tests/car.jpg ]; then \
    yolo predict model=/app/models/yolov8m.pt source=/app/tests/car.jpg \
    && yolo predict model=/app/models/yolov8n.pt source=/app/tests/car.jpg \
    && yolo predict model=/app/models/yolov8n-cls.pt source=/app/tests/car.jpg \
    && yolo predict model=/app/models/yolov8n-seg.pt source=/app/tests/car.jpg; \
    else \
      echo "Image not found, skipping YOLO model tests"; \
    fi'

# Copy app files
COPY . ./

# Ensure the script has executable permissions
RUN chmod +x /app/start.sh

# Default command
CMD ["/app/start.sh"]
