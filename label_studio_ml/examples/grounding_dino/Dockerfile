FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get -y update \
    && apt-get install -y git \
    && apt-get install -y wget \
    && apt-get install -y g++ freeglut3-dev build-essential libx11-dev \
    libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev libfreeimage-dev \
    && apt-get -y install ffmpeg libsm6 libxext6 libffi-dev python3-dev python3-pip gcc

ENV PYTHONUNBUFFERED=True \
    PORT=9090 \
    WORKERS=2 \
    THREADS=4 \
    CUDA_HOME=/usr/local/cuda

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# install GroundingDINO
RUN git clone https://github.com/IDEA-Research/GroundingDINO.git
WORKDIR /app/GroundingDINO
RUN pip3 install -e .
RUN mkdir weights
WORKDIR /app/GroundingDINO/weights
RUN wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
RUN pip3 install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

WORKDIR /app
RUN wget -q https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt
RUN wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

COPY . ./

CMD ["/app/start.sh"]
