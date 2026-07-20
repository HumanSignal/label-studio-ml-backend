FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ARG DEBIAN_FRONTEND=noninteractive
ARG TEST_ENV

WORKDIR /app

# System dependencies:
#   git, wget                 - clone SAM 2 and fetch the checkpoint
#   build-essential/g++/gcc   - compile SAM 2's optional C++/CUDA extension during `pip install -e .`
#   python3-dev               - Python headers for that build
#   ffmpeg                    - video decoding
#   libgl1/libglib2.0-0/...   - OpenCV (opencv-python) runtime shared libraries
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        git wget \
        build-essential g++ gcc python3-dev \
        ffmpeg libgl1 libglib2.0-0 libsm6 libxext6

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_CACHE_DIR=/.cache \
    PORT=9090 \
    WORKERS=1 \
    THREADS=8 \
    TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6+PTX;8.9;9.0" \
    CUDA_HOME=/opt/conda \
    LD_LIBRARY_PATH=/opt/conda/lib \
    SEGMENT_ANYTHING_2_REPO_PATH=/segment-anything-2

# Install Python requirements first so these layers cache independently of the app source.
COPY requirements-base.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install -r requirements-base.txt

COPY requirements.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install -r requirements.txt

# Install segment-anything-2. Override SAM2_REF with a branch/tag/commit for reproducible builds.
ARG SAM2_REF=main
RUN git clone --depth 1 --branch "${SAM2_REF}" --single-branch \
        https://github.com/facebookresearch/segment-anything-2.git "${SEGMENT_ANYTHING_2_REPO_PATH}"
WORKDIR ${SEGMENT_ANYTHING_2_REPO_PATH}
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install -e .

# Download only the checkpoint the server actually uses (MODEL_CHECKPOINT in docker-compose.yml /
# model.py defaults to sam2.1_hiera_large). Override SAM2_CHECKPOINT_URL for a different size, or
# run `checkpoints/download_ckpts.sh` to grab every checkpoint.
ARG SAM2_CHECKPOINT_URL=https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
RUN wget -nv -P checkpoints "${SAM2_CHECKPOINT_URL}"

WORKDIR /app

# Install test requirements only when building the test image (TEST_ENV=true).
COPY requirements-test.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    if [ "$TEST_ENV" = "true" ]; then \
      pip install -r requirements-test.txt; \
    fi

COPY . ./

CMD ["/app/start.sh"]
