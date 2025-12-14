FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel
ARG DEBIAN_FRONTEND=noninteractive
ARG TEST_ENV

WORKDIR /app

# Clean any stray conda torch libs that conflict with pip torch
# The base image has conda torch which can leave conflicting libs
RUN rm -f /opt/conda/lib/libc10*.so /opt/conda/lib/libtorch*.so 2>/dev/null || true && \
    rm -rf /opt/conda/pkgs/pytorch-* /opt/conda/pkgs/libtorch-* 2>/dev/null || true

RUN --mount=type=cache,target="/var/cache/apt",sharing=locked \
    --mount=type=cache,target="/var/lib/apt/lists",sharing=locked \
    apt-get -y update \
    && apt-get install -y git \
    && apt-get install -y wget \
    && apt-get install -y g++ gcc-10 g++-10 ninja-build curl \
    && apt-get install -y freeglut3-dev build-essential libx11-dev \
    libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev libfreeimage-dev \
    && apt-get -y install ffmpeg libsm6 libxext6 libffi-dev python3-dev python3-pip gcc

RUN pip install --upgrade pip "setuptools>=62.3.0,<75.9" wheel

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_CACHE_DIR=/.cache \
    PORT=9090 \
    WORKERS=2 \
    THREADS=4 \
    CUDA_HOME=/usr/local/cuda-12.1 \
    SAM2_BUILD_CUDA=1 \
    GROUNDINGDINO_REPO_PATH=/GroundingDINO \
    PYTHONPATH=/sam2:/GroundingDINO:${PYTHONPATH}

ENV PATH="/usr/local/cuda-12.1/bin:${PATH}"
# add torch shared libs to runtime path so sam2 custom ops can load libc10_cuda/libtorch
ENV TORCH_LIB_PATH="/opt/conda/lib/python3.10/site-packages/torch/lib"
ENV LD_LIBRARY_PATH="/usr/local/cuda-12.1/lib64:${TORCH_LIB_PATH}:${LD_LIBRARY_PATH}"
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9"
ENV CC=gcc-10
ENV CXX=g++-10

# install base requirements
COPY requirements-base.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install -r requirements-base.txt

# Re-pin CUDA-enabled torch/torchvision (avoid CPU wheel replacement by upstream deps)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.3.1 torchvision==0.18.1

COPY requirements.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip3 install -r requirements.txt

# install SAM 2.1 (sam2 repo)
ARG SAM2_REF=main
RUN cd / && git clone --depth 1 --branch ${SAM2_REF} --single-branch https://github.com/facebookresearch/sam2.git || \
    (cd / && git clone --depth 1 --branch main --single-branch https://github.com/facebookresearch/sam2.git)
WORKDIR /sam2
# Use --no-deps to prevent SAM2 from upgrading torch
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip3 install --no-build-isolation --no-deps -e .
RUN cd checkpoints && ./download_ckpts.sh

WORKDIR /app

COPY requirements-test.txt .
# build only when TEST_ENV="true"
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    if [ "$TEST_ENV" = "true" ]; then \
      pip3 install -r requirements-test.txt; \
    fi

COPY . ./

# Prefer vendored GroundingDINO snapshot if provided in build context (e.g., ./grounding_dino)
ARG GROUNDINGDINO_VENDOR_PATH=/app/grounding_dino
RUN if [ -d "${GROUNDINGDINO_VENDOR_PATH}" ]; then \
        echo "Using vendored GroundingDINO from ${GROUNDINGDINO_VENDOR_PATH}"; \
        rm -rf ${GROUNDINGDINO_REPO_PATH} && cp -r ${GROUNDINGDINO_VENDOR_PATH} ${GROUNDINGDINO_REPO_PATH}; \
    else \
        echo "Vendored GroundingDINO not found; cloning from GitHub"; \
        git clone https://github.com/IDEA-Research/GroundingDINO.git ${GROUNDINGDINO_REPO_PATH}; \
    fi

# Use --no-deps to prevent GroundingDINO from upgrading torch
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip3 install --no-build-isolation --no-deps -e ${GROUNDINGDINO_REPO_PATH}

RUN mkdir -p ${GROUNDINGDINO_REPO_PATH}/weights

RUN if [ -f "/app/models/gdino_swint_darpa-ir-v1-1k_20_1.pth" ]; then \
        cp /app/models/gdino_swint_darpa-ir-v1-1k_20_1.pth ${GROUNDINGDINO_REPO_PATH}/weights/; \
        echo "Copied local gdino weight into ${GROUNDINGDINO_REPO_PATH}/weights/"; \
    else \
        echo "Warning: /app/models/gdino_swint_darpa-ir-v1-1k_20_1.pth not found; skipping copy"; \
    fi

# Normalize line endings and ensure executable for start script (fix Windows CRLF issues)
RUN sed -i 's/\r$//' /app/start.sh && chmod +x /app/start.sh

CMD ["/app/start.sh"]
