ARG PYTORCH="1.13.0"
ARG CUDA="11.6"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# To fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# Install system dependencies for opencv-python
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp
COPY requirements.txt .

ENV PYTHONUNBUFFERED=True \
    PORT=${PORT:-9090} \
    PIP_CACHE_DIR=/.cache

RUN --mount=type=cache,target=$PIP_CACHE_DIR \
    pip install -r requirements.txt
RUN pip install label_studio_ml
WORKDIR /app

RUN mim install mmengine
RUN mim download mmdet --config yolov3_mobilenetv2_8xb24-320-300e_coco --dest .
RUN mim install mmcv==2.0.0rc3


COPY _wsgi.py .
COPY mmdetection.py .
EXPOSE 9090

ENV checkpoint_file=yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth
ENV config_file=yolov3_mobilenetv2_8xb24-320-300e_coco.py
ENV LABEL_STUDIO_ML_BACKEND_V2=true

# RUN mkdir /label-studio-ml-backend  # TODO: remove me!
# COPY ./dist /label-studio-ml-backend
# RUN cd /label-studio-ml-backend && pip install label_studio_ml-1.0.8rc3-py3-none-any.whl

CMD exec gunicorn --preload --bind :$PORT --workers 1 --threads 8 --timeout 0 _wsgi:app
