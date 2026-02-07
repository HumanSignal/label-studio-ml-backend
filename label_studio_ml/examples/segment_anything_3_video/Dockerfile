FROM nvidia/cuda:12.6.0-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/root/.cache/huggingface
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_CACHE_DIR=/.cache PORT=9090

ARG TEST_ENV

RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
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
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY requirements-test.txt .
RUN if [ "$TEST_ENV" = "true" ]; then pip install --no-cache-dir -r requirements-test.txt; fi

COPY . ./
RUN sed -i 's/\r$//' /app/start.sh && chmod +x /app/start.sh

CMD ["/app/start.sh"]
