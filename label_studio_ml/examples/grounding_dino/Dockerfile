FROM python:3.11-slim

WORKDIR /app

RUN apt-get -y update \
    && apt-get install -y git \
    && apt-get install -y wget \
    && apt-get -y install ffmpeg libsm6 libxext6 libffi-dev python3-dev gcc


ENV PYTHONUNBUFFERED=True \
    PORT=9090 \
    WORKERS=2 \
    THREADS=4

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN git clone https://github.com/yuwenmichael/Grounding-DINO-Batch-Inference.git
WORKDIR /app/Grounding-DINO-Batch-Inference/GroundingDINO
RUN pip -q install -e .
RUN mkdir weights
WORKDIR /app/Grounding-DINO-Batch-Inference/GroundingDINO/weights

RUN wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

WORKDIR /app
RUN wget -q https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt
RUN wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

RUN pip install olefile
COPY . ./


CMD gunicorn --preload --bind :$PORT --workers $WORKERS --threads $THREADS --timeout 0 _wsgi:app
