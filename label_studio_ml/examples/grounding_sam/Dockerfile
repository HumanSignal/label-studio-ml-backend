FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel
ARG DEBIAN_FRONTEND=noninteractive
ARG TEST_ENV

WORKDIR /app

RUN conda update conda -y

RUN --mount=type=cache,target="/var/cache/apt",sharing=locked \
    --mount=type=cache,target="/var/lib/apt/lists",sharing=locked \
    apt-get -y update \
    && apt-get install -y git \
    && apt-get install -y wget \
    && apt-get install -y g++ freeglut3-dev build-essential libx11-dev \
    libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev libfreeimage-dev ninja-build \
    && apt-get -y install ffmpeg libsm6 libxext6 libffi-dev python3-dev python3-pip gcc

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_CACHE_DIR=/.cache \
    PORT=9090 \
    WORKERS=2 \
    THREADS=4 \
    CUDA_HOME=/usr/local/cuda \
    GROUNDINGDINO_REPO_PATH=/GroundingDINO
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6+PTX;8.9;9.0"

COPY requirements.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip3 install --no-cache-dir -r requirements.txt

# 1) Make sure sentence-transformers is in your requirements.txt
#    (or install it here explicitly)
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip3 install --no-cache-dir sentence-transformers huggingface-hub

# pre-cache the bert tokenizer and config so runtime is offline-safe
RUN python3 - <<EOF
from transformers import AutoTokenizer, AutoConfig, AutoModel
for model in ["bert-base-uncased", "sentence-transformers/all-MiniLM-L6-v2"]:
    AutoTokenizer.from_pretrained(model)
    AutoConfig.from_pretrained(model)
    AutoModel.from_pretrained(model)
EOF


# 2) Pre‐cache the 'all-MiniLM-L6-v2' model into the HF cache
#    so that SentenceTransformer(...) never hits the network.
RUN python3 - <<'EOF'
from sentence_transformers import SentenceTransformer
# this will download weights, tokenizer, config into ~/.cache/huggingface/hub
SentenceTransformer('all-MiniLM-L6-v2')
EOF

# 3) Enable Transformers “offline” mode at runtime
ENV TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1


# install GroundingDINO
RUN cd / && git clone https://github.com/IDEA-Research/GroundingDINO.git
WORKDIR /GroundingDINO
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip3 install -e .
RUN mkdir weights
WORKDIR /GroundingDINO/weights
RUN wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

WORKDIR /app
RUN wget -q https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt
RUN wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# install test requirements if needed
COPY requirements-test.txt .
# build only when TEST_ENV="true"
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    if [ "$TEST_ENV" = "true" ]; then \
      pip3 install -r requirements-test.txt; \
    fi

COPY . ./

CMD ["/app/start.sh"]
