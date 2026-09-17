# Use a PyTorch base image with CUDA support
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime
ARG DEBIAN_FRONTEND=noninteractive
ARG TEST_ENV

WORKDIR /app

# Install essential packages and D-FINE dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    # For OpenCV if needed by D-FINE or its deps, though D-FINE requirements.txt might handle it via pip
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_CACHE_DIR=/.cache \
    PORT=9090 \
    WORKERS=1 \
    THREADS=4 \
    CUDA_HOME=/usr/local/cuda \
    DFINE_CODE_DIR=/app/d-fine-code

# Set Conda's CUDA_HOME if it's a conda based PyTorch image, otherwise system CUDA_HOME
# For official pytorch/pytorch images, system CUDA_HOME is usually fine.
# ENV CUDA_HOME=/opt/conda

ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6+PTX;8.9;9.0"

# Install base requirements for Label Studio ML Backend
COPY requirements-base.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install --no-cache-dir -r requirements-base.txt

# --- D-FINE specific setup ---
# 1. Copy D-FINE's requirements.txt
COPY d_fine_requirements.txt .
# 2. Install D-FINE's Python dependencies
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install --no-cache-dir -r d_fine_requirements.txt

# 3. Copy D-FINE's 'src' and 'configs' directories
COPY d-fine-code/src ${DFINE_CODE_DIR}/src
COPY d-fine-code/configs ${DFINE_CODE_DIR}/configs
# --- End D-FINE specific setup ---

# Install ML backend example specific requirements (if any, usually empty for this setup)
#COPY requirements.txt .
#RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
#    pip install --no-cache-dir -r requirements.txt

# install test requirements if needed
COPY requirements-test.txt .
# build only when TEST_ENV="true"
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    if [ "$TEST_ENV" = "true" ]; then \
      pip3 install -r requirements-test.txt; \
    fi

# Copy the rest of the ML backend example files
COPY . ./

# Set PYTHONPATH to include the D-FINE source code directory
ENV PYTHONPATH=${DFINE_CODE_DIR}:${PYTHONPATH}

EXPOSE ${PORT}

CMD gunicorn --preload --bind :${PORT} --workers ${WORKERS} --threads ${THREADS} --timeout 0 _wsgi:app