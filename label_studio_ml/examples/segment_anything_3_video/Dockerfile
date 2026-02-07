# Build args — override for DGX Spark via .env.dgx-spark or inline env vars
ARG CUDA_VERSION=12.6.0
ARG UBUNTU_VERSION=24.04
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/root/.cache/huggingface
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_CACHE_DIR=/.cache PORT=9090

ARG TEST_ENV
ARG TORCH_CUDA_INDEX=cu126

# Ubuntu 24.04 ships Python 3.12 natively — no deadsnakes PPA needed
RUN apt-get update && apt-get install -y \
    python3.12 python3.12-dev python3.12-venv \
    git curl ffmpeg \
    libavformat-dev libavcodec-dev libavdevice-dev libavfilter-dev \
    libavutil-dev libpostproc-dev libswresample-dev libswscale-dev \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12
RUN pip install --upgrade pip setuptools wheel

WORKDIR /app

COPY requirements-base.txt .
RUN pip install --no-cache-dir --ignore-installed -r requirements-base.txt

# Install PyTorch with the correct CUDA index for target architecture:
#   cu126 for amd64/RTX 6000 Ada (default)
#   cu130 for arm64/DGX Spark
RUN pip install --no-cache-dir \
    torch==2.9.1 torchvision==0.24.1 \
    --index-url https://download.pytorch.org/whl/${TORCH_CUDA_INDEX}

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY requirements-test.txt .
RUN if [ "$TEST_ENV" = "true" ]; then pip install --no-cache-dir -r requirements-test.txt; fi

COPY . ./
RUN sed -i 's/\r$//' /app/start.sh && chmod +x /app/start.sh

CMD ["/app/start.sh"]
