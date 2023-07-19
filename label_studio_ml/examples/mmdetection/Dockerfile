FROM python:3.7

WORKDIR /tmp
COPY requirements.txt .

ENV PYTHONUNBUFFERED=True \
    PORT=${PORT:-9090} \
    PIP_CACHE_DIR=/.cache

RUN --mount=type=cache,target=$PIP_CACHE_DIR \
    pip install -r requirements.txt

RUN pip install mmdet==2.28.2 "mmcv-full>=1.3.17,<=1.8.0"

COPY uwsgi.ini /etc/uwsgi/
COPY supervisord.conf /etc/supervisor/conf.d/

WORKDIR /app

RUN curl -O https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
COPY * /app/
COPY /_base_/ /app/_base_/


EXPOSE 9090

ENV checkpoint_file=faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
ENV config_file=faster_rcnn_r50_fpn_1x_coco.py

CMD ["/usr/local/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
